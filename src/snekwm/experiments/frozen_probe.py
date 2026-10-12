"""Frozen-encoder baselines: encoder-probe ceiling and identity-predictor floor."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from snekwm.config import ProbeConfig
from snekwm.data.features import encode_episodes
from snekwm.eval.probe import action_only_control, encoder_probe_data, evaluate
from snekwm.eval.report import probe_table
from snekwm.experiments.common import load_split, probe_dynamics
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
    episodes, train, val = load_split(cfg.data_dir, cfg.val_frac, cfg.seed)
    run.log.info("episodes: %d train / %d val", len(train), len(val))

    feats = load_or_encode(cfg, encoder)
    identity = {k: v[:-1] for k, v in feats.items()}
    results = {
        "encoder": evaluate(encoder_probe_data, feats, episodes, train, val).to_dict(),
        "identity": probe_dynamics(identity, feats, episodes, train, val),
        "action_only": {
            "displacement": action_only_control(episodes, train, val).to_dict()
        },
    }
    run.log.info("encoder probe: R2_joint=%.3f", results["encoder"]["r2_joint"])
    for name, r in results["identity"].items():
        run.log.info("identity %s: R2_joint=%.3f", name, r["r2_joint"])
    run.log.info(
        "action-only displacement: R2_joint=%.3f",
        results["action_only"]["displacement"]["r2_joint"],
    )
    run.write_json("probe.json", results)
    table = probe_table(
        {
            "Frozen DINOv2 encoder": {"encoder": results["encoder"]},
            "Identity WM": results["identity"],
            "Action only (no image)": results["action_only"],
        }
    )
    run.write_text(
        "summary.md",
        "# Frozen DINOv2 probes\n\n"
        "Held-out linear-probe R²_joint (snake-head position). "
        "`transfer*` decode predictions with a probe fit on real encodings only.\n\n"
        + table,
    )
    run.close()
    return results
