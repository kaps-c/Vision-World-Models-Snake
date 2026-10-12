"""DINOv2-WM: train an action-conditioned predictor on frozen DINOv2 features.

Reports, per epoch, the predictor's validation MSE and cosine similarity next
to the identity world model's (``z_hat_{t+1} = z_t``), then linear-probes the
final predictions for next-head position, head displacement and change.
"""

from __future__ import annotations

import math
from dataclasses import asdict

import torch
import torch.nn.functional as F  # noqa: N812

from snekwm.config import WorldModelConfig
from snekwm.data.features import FeatureTransitions
from snekwm.eval.report import Panel, metrics_table, plot_curves, probe_table
from snekwm.experiments.common import (
    Predictor,
    autocast,
    load_split,
    one_cycle,
    predict,
    probe_dynamics,
)
from snekwm.models.predictor import ActionConditionedPredictor
from snekwm.results import RunDir

PANELS = (
    Panel("train_loss", "Train loss (MSE)", log=True),
    Panel(
        "val_mse",
        "Validation MSE (lower is better)",
        baseline="identity_mse",
        baseline_label="identity",
    ),
)


def run_dinov2_wm(cfg: WorldModelConfig) -> dict:
    """Train, evaluate and save a DINOv2-WM run; return the probe results."""
    torch.manual_seed(cfg.seed)
    run = RunDir(cfg.run_dir)
    run.write_json("config.json", cfg)

    episodes, train, val = load_split(cfg.data_dir, cfg.val_frac, cfg.seed)
    feats = torch.load(cfg.features_path)
    train_set = FeatureTransitions(feats, episodes, train, cfg.device)
    val_set = FeatureTransitions(feats, episodes, val, cfg.device)

    # Token count and width come from the cached features, not the config.
    _, num_tokens, in_dim = next(iter(feats.values())).shape
    model = ActionConditionedPredictor(
        in_dim=in_dim,
        dim=cfg.dim,
        depth=cfg.depth,
        heads=cfg.heads,
        num_tokens=num_tokens,
        residual=cfg.residual,
        use_action=cfg.use_action,
    ).to(cfg.device)
    steps = cfg.epochs * math.ceil(len(train_set) / cfg.batch_size)
    opt, sched = one_cycle(model.parameters(), cfg.lr, cfg.weight_decay, steps)
    gen = torch.Generator().manual_seed(cfg.seed)
    run.log.info(
        "%d train / %d val transitions, %.1fM params",
        len(train_set),
        len(val_set),
        sum(p.numel() for p in model.parameters()) / 1e6,
    )

    id_mse, id_cos = _evaluate(lambda z, a: z, val_set, cfg)
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for z, a, z1 in train_set.batches(cfg.batch_size, generator=gen):
            with autocast(cfg.device):
                loss = F.mse_loss(model(z, a).float(), z1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(z)
            count += len(z)
        model.eval()
        val_mse, val_cos = _evaluate(model, val_set, cfg)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / count,
                "val_mse": val_mse,
                "val_cos": val_cos,
                "identity_mse": id_mse,
                "identity_cos": id_cos,
            }
        )
        run.log.info(
            "epoch %d  train %.4f  val mse %.4f (identity %.4f)  cos %.4f (%.4f)",
            epoch,
            total / count,
            val_mse,
            id_mse,
            val_cos,
            id_cos,
        )
    run.write_json("history.json", history)
    torch.save(
        {"model": model.state_dict(), "config": asdict(cfg)}, run.path / "final.pt"
    )

    preds = predict(model, feats, episodes, cfg.batch_size, cfg.device)
    identity = {k: v[:-1] for k, v in feats.items()}
    results = probe_dynamics(preds, feats, episodes, train, val)
    baseline = probe_dynamics(identity, feats, episodes, train, val)
    for name, r in results.items():
        run.log.info("%s probe: R2_joint=%.3f", name, r["r2_joint"])
    run.write_json("probe.json", results)
    _write_summary(run, cfg, history, {"Identity WM": baseline, cfg.run_dir: results})
    run.close()
    return results


def _write_summary(
    run: RunDir, cfg: WorldModelConfig, history: list[dict], probes: dict
) -> None:
    """Write ``summary.md`` (error + probe tables) and ``curves.png``."""
    last = history[-1]
    errors = metrics_table(
        {
            "Identity WM": {
                "val MSE": last["identity_mse"],
                "val cos": last["identity_cos"],
            },
            cfg.run_dir: {"val MSE": last["val_mse"], "val cos": last["val_cos"]},
        }
    )
    plot_curves(history, run.path / "curves.png", cfg.run_dir, PANELS)
    run.write_text(
        "summary.md",
        f"# {cfg.run_dir}\n\n"
        f"residual={cfg.residual}, use_action={cfg.use_action}, "
        f"epochs={cfg.epochs}, depth={cfg.depth}, dim={cfg.dim}\n\n"
        "## Feature-space error (final epoch, validation)\n\n"
        f"{errors}\n"
        "## Linear probes (held-out R²_joint)\n\n"
        f"{probe_table(probes)}\n"
        "![training curves](curves.png)\n",
    )


@torch.no_grad()
def _evaluate(
    model: Predictor, data: FeatureTransitions, cfg: WorldModelConfig
) -> tuple[float, float]:
    """Return mean per-element MSE and mean per-token cosine similarity."""
    mse, cos, n = 0.0, 0.0, 0
    for z, a, z1 in data.batches(cfg.batch_size, shuffle=False):
        with autocast(cfg.device):
            pred = model(z, a).float()
        mse += F.mse_loss(pred, z1, reduction="sum").item() / pred[0].numel()
        cos += F.cosine_similarity(pred, z1, dim=-1).mean(-1).sum().item()
        n += len(z)
    return mse / n, cos / n
