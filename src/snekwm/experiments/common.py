"""Pieces shared by the world-model experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from snekwm.data.collect import load_episode
from snekwm.data.dataset import split_episodes
from snekwm.eval.probe import (
    change_tensors,
    displacement_probe_data,
    evaluate,
    latent_cosine,
    predictor_probe_data,
    transfer_probe,
)

Predictor = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Episodes = dict[str, dict[str, np.ndarray]]


def load_split(
    data_dir: str, val_frac: float, seed: int
) -> tuple[Episodes, list[str], list[str]]:
    """Load all episodes and split their names into train/val by episode."""
    paths = sorted(Path(data_dir).glob("*.npz"))
    episodes = {p.stem: load_episode(p) for p in paths}
    train, val = split_episodes(paths, val_frac, seed)
    return episodes, [p.stem for p in train], [p.stem for p in val]


def autocast(device: str) -> torch.autocast:
    """Return bf16 autocast on CUDA, a no-op elsewhere."""
    return torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda")


@torch.no_grad()
def predict(
    model: Predictor,
    inputs: Mapping[str, torch.Tensor],
    episodes: Episodes,
    batch_size: int,
    device: str,
) -> dict[str, torch.Tensor]:
    """Apply ``model(x_t, a_t)`` to every transition: ``{name: (T, N, D)}``.

    ``inputs`` holds one ``(T+1, ...)`` tensor per episode (features or
    frames); the last frame has no action and is skipped.
    """
    out = {}
    for name, x in inputs.items():
        actions = torch.as_tensor(episodes[name]["actions"]).long()
        chunks = []
        for i in range(0, len(actions), batch_size):
            a = actions[i : i + batch_size]
            with autocast(device):
                pred = model(x[i : i + len(a)].to(device).float(), a.to(device))
            chunks.append(pred.half().cpu())
        out[name] = torch.cat(chunks)
    return out


def probe_dynamics(
    preds: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    episodes: Episodes,
    train: Sequence[str],
    val: Sequence[str],
    normalize_transfer: bool = False,
) -> dict[str, dict]:
    """All prediction probes (predictor, displacement, change, transfer).

    ``reference`` holds the per-frame embeddings the predictions live next to
    (the target space); the change probe uses ``preds - reference[:-1]``.
    """
    deltas = change_tensors(preds, reference)
    probes = {
        "predictor": evaluate(predictor_probe_data, preds, episodes, train, val),
        "displacement": evaluate(displacement_probe_data, preds, episodes, train, val),
        "change": evaluate(displacement_probe_data, deltas, episodes, train, val),
        **transfer_probe(
            preds, reference, episodes, train, val, normalize=normalize_transfer
        ),
    }
    out = {k: v.to_dict() for k, v in probes.items()}
    out["latent_cosine"] = latent_cosine(preds, reference, episodes, val)
    return out


def one_cycle(
    params, lr: float, weight_decay: float, total_steps: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """AdamW with a one-cycle schedule (warm-up then cosine decay)."""
    opt = torch.optim.AdamW(params, lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=total_steps)
    return opt, sched


@torch.no_grad()
def encode_all(
    encoder: Callable[[torch.Tensor], torch.Tensor],
    inputs: Mapping[str, torch.Tensor],
    batch_size: int,
    device: str,
) -> dict[str, torch.Tensor]:
    """Apply ``encoder`` to every frame: ``{name: (T+1, N, D)}`` fp16 on CPU."""
    out = {}
    for name, x in inputs.items():
        chunks = []
        for i in range(0, len(x), batch_size):
            with autocast(device):
                chunks.append(
                    encoder(x[i : i + batch_size].to(device).float()).half().cpu()
                )
        out[name] = torch.cat(chunks)
    return out


def grids_to_images(grids: torch.Tensor, cell_px: int) -> torch.Tensor:
    """Upsample ``(B, L, L, 3)`` grids in ``[0, 255]`` to ``(B, 3, H, W)`` in [0, 1]."""
    x = grids.repeat_interleave(cell_px, 1).repeat_interleave(cell_px, 2)
    return x.permute(0, 3, 1, 2).float().div(255.0)


def log_probes(log, prefix: str, results: Mapping[str, dict]) -> None:
    """Log every probe R^2 (and the latent-cosine skill) in ``results``."""
    for name, r in results.items():
        if "r2_joint" in r:
            log.info("%s%s: R2_joint=%.3f", prefix, name, r["r2_joint"])
        elif "skill" in r:
            log.info(
                "%s%s: cos %.3f vs identity %.3f (skill %.3f)",
                prefix,
                name,
                r["prediction"],
                r["identity"],
                r["skill"],
            )
