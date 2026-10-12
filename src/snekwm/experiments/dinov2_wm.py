"""DINOv2-WM: train an action-conditioned predictor on frozen DINOv2 features.

Reports, per epoch, the predictor's validation MSE and cosine similarity next
to the identity world model's (``z_hat_{t+1} = z_t``), then linear-probes the
final predictions for next-head position and head displacement.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812

from snekwm.config import WorldModelConfig
from snekwm.data.collect import load_episode
from snekwm.data.dataset import split_episodes
from snekwm.data.features import FeatureTransitions
from snekwm.eval.probe import displacement_probe_data, evaluate, predictor_probe_data
from snekwm.models.predictor import ActionConditionedPredictor
from snekwm.results import RunDir

Predictor = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def run_dinov2_wm(cfg: WorldModelConfig) -> dict:
    """Train, evaluate and save a DINOv2-WM run; return the probe results."""
    torch.manual_seed(cfg.seed)
    run = RunDir(cfg.run_dir)
    run.write_json("config.json", cfg)

    paths = sorted(Path(cfg.data_dir).glob("*.npz"))
    episodes = {p.stem: load_episode(p) for p in paths}
    train_paths, val_paths = split_episodes(paths, cfg.val_frac, cfg.seed)
    train, val = [p.stem for p in train_paths], [p.stem for p in val_paths]
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
    opt = torch.optim.AdamW(model.parameters(), cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * math.ceil(len(train_set) / cfg.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=steps)
    gen = torch.Generator().manual_seed(cfg.seed)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    run.log.info(
        "%d train / %d val transitions, %.1fM params",
        len(train_set),
        len(val_set),
        n_params,
    )

    id_mse, id_cos = _evaluate(lambda z, a: z, val_set, cfg)
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for z, a, z1 in train_set.batches(cfg.batch_size, generator=gen):
            with _autocast(cfg.device):
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

    preds = _predict(model, feats, episodes, cfg)
    results = {
        "predictor": evaluate(predictor_probe_data, preds, episodes, train, val),
        "displacement": evaluate(displacement_probe_data, preds, episodes, train, val),
    }
    results = {k: v.to_dict() for k, v in results.items()}
    for name, r in results.items():
        run.log.info("%s probe: R2_joint=%.3f", name, r["r2_joint"])
    run.write_json("probe.json", results)
    run.close()
    return results


def _autocast(device: str) -> torch.autocast:
    return torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda")


@torch.no_grad()
def _evaluate(
    model: Predictor, data: FeatureTransitions, cfg: WorldModelConfig
) -> tuple[float, float]:
    """Return mean per-element MSE and mean per-token cosine similarity."""
    mse, cos, n = 0.0, 0.0, 0
    for z, a, z1 in data.batches(cfg.batch_size, shuffle=False):
        with _autocast(cfg.device):
            pred = model(z, a).float()
        mse += F.mse_loss(pred, z1, reduction="sum").item() / pred[0].numel()
        cos += F.cosine_similarity(pred, z1, dim=-1).mean(-1).sum().item()
        n += len(z)
    return mse / n, cos / n


@torch.no_grad()
def _predict(
    model: Predictor, feats: dict, episodes: dict, cfg: WorldModelConfig
) -> dict[str, torch.Tensor]:
    """Predict ``z_hat_{t+1}`` for every transition: ``{name: (T, N, D)}``."""
    out = {}
    bs = cfg.batch_size
    for name, f in feats.items():
        actions = torch.as_tensor(episodes[name]["actions"]).long()
        chunks = []
        for i in range(0, len(actions), bs):
            z = f[i : i + bs][: len(actions[i : i + bs])].to(cfg.device).float()
            with _autocast(cfg.device):
                pred = model(z, actions[i : i + bs].to(cfg.device))
            chunks.append(pred.half().cpu())
        out[name] = torch.cat(chunks)
    return out
