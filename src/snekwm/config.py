"""Typed experiment configs loaded from TOML sections."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class CollectConfig:
    """Settings for :func:`snekwm.data.collect.collect`."""

    size: int = 14
    episodes: int = 32
    max_steps: int = 300
    epsilon: float = 0.1
    seed: int = 0
    out_dir: str = "data/episodes"


@dataclass(frozen=True)
class ProbeConfig:
    """Settings for frozen-encoder probing."""

    data_dir: str = "data/episodes"
    features_path: str = "data/features/dinov2_vits14.pt"
    run_dir: str = "results/dinov2_frozen"
    val_frac: float = 0.2
    seed: int = 0
    cell_px: int = 16
    device: str = "cuda"


@dataclass(frozen=True)
class WorldModelConfig:
    """Settings for training a predictor on frozen-encoder features."""

    data_dir: str = "data/episodes"
    features_path: str = "data/features/dinov2_vits14.pt"
    run_dir: str = "results/dinov2_wm"
    val_frac: float = 0.2
    seed: int = 0
    epochs: int = 5
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.05
    dim: int = 384
    depth: int = 4
    heads: int = 6
    residual: bool = False
    use_action: bool = True
    device: str = "cuda"


@dataclass(frozen=True)
class JEPAConfig:
    """Settings for strict-JEPA training (see :mod:`snekwm.models.jepa`).

    Loss: ``lam * invariance + beta * (mu * variance + nu * covariance)``.
    ``encoder="dinov2"`` swaps the scratch encoder for cached frozen DINOv2
    features (online = target = frozen); only predictor/projector train.
    """

    data_dir: str = "data/episodes"
    features_path: str = "data/features/dinov2_vits14.pt"
    run_dir: str = "results/jepa"
    val_frac: float = 0.2
    seed: int = 0
    epochs: int = 5
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.05
    encoder: str = "small_vit"
    cell_px: int = 16
    patch: int = 16
    enc_dim: int = 192
    enc_depth: int = 6
    enc_heads: int = 3
    pred_dim: int = 384
    pred_depth: int = 4
    pred_heads: int = 6
    momentum: float = 0.996
    projector: bool = True
    lam: float = 25.0
    mu: float = 25.0
    nu: float = 1.0
    beta: float = 1.0
    use_action: bool = True
    rank_ratio_min: float = 0.5
    # "record": flag collapse and keep training (equal budget across ablations);
    # "stop": abort the run at the first collapsed epoch.
    kill_switch: str = "record"
    device: str = "cuda"


def load_config(path: str | Path, section: str, cls: type[T]) -> T:
    """Build ``cls`` from ``[section]`` of a TOML file.

    Unknown keys raise ``TypeError`` so typos fail loudly instead of
    silently falling back to defaults.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return cls(**data.get(section, {}))
