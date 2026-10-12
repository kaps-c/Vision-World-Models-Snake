"""Strict-JEPA training with collapse monitoring and linear-probe evaluation.

Per epoch: train loss (and its invariance / variance / covariance parts),
validation loss, and the online/target effective-rank ratio. The rank-ratio
kill-switch stops a dimensionally collapsed run early and records it.

After training, four probes:

* encoder: online ``z_t -> head_t`` (does the encoder know where the snake is?)
* predictor / displacement / change: on predictions in the target space,
  with ``change = z_hat_{t+1} - target(o_t)``.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from functools import partial

import torch

from snekwm.config import JEPAConfig
from snekwm.data.features import FeatureTransitions
from snekwm.eval.collapse import CollapseError, check_rank_ratio
from snekwm.eval.probe import encoder_probe_data, evaluate
from snekwm.eval.report import Panel, metrics_table, plot_curves, probe_table
from snekwm.experiments.common import (
    autocast,
    encode_all,
    grids_to_images,
    load_split,
    one_cycle,
    predict,
    probe_dynamics,
)
from snekwm.models.jepa import JEPA, JEPAOutput, SmallViT
from snekwm.models.predictor import ActionConditionedPredictor
from snekwm.results import RunDir

PANELS = (
    Panel("train_loss", "Train loss", log=True),
    Panel("val_loss", "Validation loss (lower is better)", log=True),
    Panel(
        "rank_ratio",
        "Online / target effective rank",
        baseline="rank_ratio_min",
        baseline_label="kill-switch",
    ),
)
RANK_SAMPLE = 4096  # token rows used for the effective-rank SVD


def run_jepa(cfg: JEPAConfig) -> dict:
    """Train, evaluate and save one JEPA run; return probe results."""
    torch.manual_seed(cfg.seed)
    run = RunDir(cfg.run_dir)
    run.write_json("config.json", cfg)
    episodes, train, val = load_split(cfg.data_dir, cfg.val_frac, cfg.seed)
    inputs, prep, jepa = _build(cfg, episodes)
    train_set = FeatureTransitions(inputs, episodes, train, cfg.device)
    val_set = FeatureTransitions(inputs, episodes, val, cfg.device)

    params = [p for p in jepa.parameters() if p.requires_grad]
    steps = cfg.epochs * math.ceil(len(train_set) / cfg.batch_size)
    opt, sched = one_cycle(params, cfg.lr, cfg.weight_decay, steps)
    gen = torch.Generator().manual_seed(cfg.seed)
    run.log.info(
        "%s: %d train / %d val transitions, %.1fM trainable params",
        cfg.run_dir,
        len(train_set),
        len(val_set),
        sum(p.numel() for p in params) / 1e6,
    )

    history, collapsed = [], False
    for epoch in range(1, cfg.epochs + 1):
        jepa.train()
        sums = dict.fromkeys(("loss", "inv", "var", "cov"), 0.0)
        count = 0
        for x, a, x1 in train_set.batches(cfg.batch_size, generator=gen):
            with autocast(cfg.device):
                out = jepa(prep(x), a, prep(x1))
            loss = _total(out, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            jepa.update_target()
            n = len(x)
            for key, value in zip(
                sums,
                (loss, out.invariance, out.variance, out.covariance),
                strict=True,
            ):
                sums[key] += value.item() * n
            count += n

        val_loss, val_inv, online, target = _validate(jepa, val_set, prep, cfg)
        row = {
            "epoch": epoch,
            "train_loss": sums["loss"] / count,
            "train_invariance": sums["inv"] / count,
            "train_variance": sums["var"] / count,
            "train_covariance": sums["cov"] / count,
            "val_loss": val_loss,
            "val_invariance": val_inv,
            "rank_ratio_min": cfg.rank_ratio_min,
        }
        try:
            row["rank_ratio"] = check_rank_ratio(online, target, cfg.rank_ratio_min)
        except CollapseError as err:
            row["rank_ratio"] = err.ratio
            collapsed = True
        history.append(row)
        run.log.info(
            "epoch %d  train %.4f (inv %.4f var %.4f cov %.4f)  val %.4f  rank %.3f",
            epoch,
            row["train_loss"],
            row["train_invariance"],
            row["train_variance"],
            row["train_covariance"],
            val_loss,
            row["rank_ratio"],
        )
        if collapsed:
            run.log.info(
                "COLLAPSE: rank ratio below %.2f, stopping", cfg.rank_ratio_min
            )
            break

    run.write_json("history.json", history)
    torch.save(
        {"model": jepa.state_dict(), "config": asdict(cfg)}, run.path / "final.pt"
    )

    results = _probe(jepa, inputs, prep, episodes, train, val, cfg)
    results["collapsed"] = collapsed
    results["rank_ratio"] = history[-1]["rank_ratio"]
    for name in ("encoder", "predictor", "displacement", "change"):
        run.log.info("%s probe: R2_joint=%.3f", name, results[name]["r2_joint"])
    run.write_json("probe.json", results)
    _write_summary(run, cfg, history, results)
    run.close()
    return results


def _build(cfg: JEPAConfig, episodes) -> tuple[dict, callable, JEPA]:
    """Return per-episode inputs, a batch pre-processing fn, and the model."""
    if cfg.encoder == "dinov2":
        inputs = torch.load(cfg.features_path)
        _, num_tokens, width = next(iter(inputs.values())).shape
        encoder, prep = None, (lambda x: x)
    elif cfg.encoder == "small_vit":
        inputs = {k: torch.from_numpy(e["frames"]) for k, e in episodes.items()}
        size = next(iter(inputs.values())).shape[1] * cfg.cell_px
        encoder = SmallViT(cfg.patch, cfg.enc_dim, cfg.enc_depth, cfg.enc_heads, size)
        num_tokens, width = (size // cfg.patch) ** 2, cfg.enc_dim
        prep = partial(grids_to_images, cell_px=cfg.cell_px)
    else:
        raise ValueError(f"unknown encoder {cfg.encoder!r}")
    predictor = ActionConditionedPredictor(
        in_dim=width,
        dim=cfg.pred_dim,
        depth=cfg.pred_depth,
        heads=cfg.pred_heads,
        num_tokens=num_tokens,
        use_action=cfg.use_action,
    )
    jepa = JEPA(encoder, predictor, cfg.momentum, cfg.projector).to(cfg.device)
    return inputs, prep, jepa


def _total(out: JEPAOutput, cfg: JEPAConfig) -> torch.Tensor:
    reg = cfg.mu * out.variance + cfg.nu * out.covariance
    return cfg.lam * out.invariance + cfg.beta * reg


@torch.no_grad()
def _validate(jepa: JEPA, data, prep, cfg: JEPAConfig):
    """Return val loss, val invariance, and sampled online/target token rows."""
    jepa.eval()
    loss, inv, n = 0.0, 0.0, 0
    online, target = [], []
    for x, a, x1 in data.batches(cfg.batch_size, shuffle=False):
        with autocast(cfg.device):
            out = jepa(prep(x), a, prep(x1))
        loss += _total(out, cfg).item() * len(x)
        inv += out.invariance.item() * len(x)
        n += len(x)
        if sum(len(o) for o in online) < RANK_SAMPLE:
            online.append(out.online.flatten(0, 1).float().cpu())
            target.append(out.target.flatten(0, 1).float().cpu())
    online = torch.cat(online)[:RANK_SAMPLE]
    target = torch.cat(target)[:RANK_SAMPLE]
    return loss / n, inv / n, online, target


def _probe(jepa: JEPA, inputs, prep, episodes, train, val, cfg) -> dict:
    jepa.eval()
    bs, dev = cfg.batch_size, cfg.device
    online = encode_all(lambda x: jepa.online(prep(x)), inputs, bs, dev)
    target = encode_all(lambda x: jepa.target(prep(x)), inputs, bs, dev)

    def step(x, a):
        return jepa.projector(jepa.predictor(jepa.online(prep(x)), a))

    preds = predict(step, inputs, episodes, bs, dev)
    encoder = evaluate(encoder_probe_data, online, episodes, train, val).to_dict()
    return {"encoder": encoder, **probe_dynamics(preds, target, episodes, train, val)}


def _write_summary(run: RunDir, cfg: JEPAConfig, history: list, results: dict) -> None:
    last = history[-1]
    plot_curves(history, run.path / "curves.png", cfg.run_dir, PANELS)
    status = (
        f"**COLLAPSE** — rank ratio {last['rank_ratio']:.3f} fell below "
        f"{cfg.rank_ratio_min} at epoch {last['epoch']}; training stopped."
        if results["collapsed"]
        else f"No dimensional collapse (final rank ratio {last['rank_ratio']:.3f})."
    )
    train_table = metrics_table(
        {
            f"epoch {h['epoch']}": {
                "train loss": h["train_loss"],
                "invariance": h["train_invariance"],
                "variance": h["train_variance"],
                "covariance": h["train_covariance"],
                "val loss": h["val_loss"],
                "rank ratio": h["rank_ratio"],
            }
            for h in history
        },
        label="Epoch",
    )
    probes = {k: results[k] for k in ("encoder", "predictor", "displacement", "change")}
    run.write_text(
        "summary.md",
        f"# {cfg.run_dir}\n\n"
        f"encoder={cfg.encoder}, projector={cfg.projector}, beta={cfg.beta}, "
        f"momentum={cfg.momentum}, use_action={cfg.use_action}, "
        f"epochs={cfg.epochs}\n\n"
        f"{status}\n\n"
        "## Linear probes (held-out R²_joint)\n\n"
        f"{probe_table({cfg.run_dir: probes})}\n"
        "## Training\n\n"
        f"{train_table}\n"
        "![training curves](curves.png)\n",
    )
