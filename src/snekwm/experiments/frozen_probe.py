"""Frozen-encoder baselines: encoder-probe ceiling and identity-predictor floor."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from snekwm.config import ProbeConfig
from snekwm.data.collect import load_episode
from snekwm.data.dataset import split_episodes
from snekwm.data.features import encode_episodes
from snekwm.eval.probe import encoder_probe_data, evaluate, predictor_probe_data
from snekwm.models.encoders import FrozenDinov2
from snekwm.results import RunDir


def load_or_encode(cfg: ProbeConfig, encoder: nn.Module | None) -> dict:
    """Load cached features, or encode all episodes and cache them."""
    path = Path(cfg.features_path)
    if path.exists():
        return torch.load(path)
    paths = sorted(Path(cfg.data_dir).glob("*.npz"))
    feats = encode_episodes(
        encoder or FrozenDinov2(), paths, cell_px=cfg.cell_px, device=cfg.device
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feats, path)
    return feats


def run_frozen_probe(cfg: ProbeConfig, encoder: nn.Module | None = None) -> dict:
    """Probe frozen features and the identity world model ``z_hat_{t+1} = z_t``.

    The identity predictor is a necessary floor: the head moves one cell per
    step, so most patches are unchanged and "predict no change" is strong
    in feature space.
    """
    run = RunDir(cfg.run_dir)
    run.write_json("config.json", cfg)
    paths = sorted(Path(cfg.data_dir).glob("*.npz"))
    episodes = {p.stem: load_episode(p) for p in paths}
    train, val = split_episodes(paths, cfg.val_frac, cfg.seed)
    train, val = [p.stem for p in train], [p.stem for p in val]
    run.log.info("episodes: %d train / %d val", len(train), len(val))

    feats = load_or_encode(cfg, encoder)
    identity = {k: v[:-1] for k, v in feats.items()}
    results = {
        "encoder": evaluate(encoder_probe_data, feats, episodes, train, val),
        "identity_predictor": evaluate(
            predictor_probe_data, identity, episodes, train, val
        ),
    }
    results = {k: v.to_dict() for k, v in results.items()}
    for name, r in results.items():
        run.log.info("%s probe: R2_joint=%.3f", name, r["r2_joint"])
    run.write_json("probe.json", results)
    run.close()
    return results
