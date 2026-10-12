#!/usr/bin/env bash
# Regenerate every result in results/ from scratch (about 45 min on an RTX 3070 Ti Laptop).
set -euo pipefail
snek collect --config configs/collect.toml
snek probe --config configs/probe_dinov2.toml
snek wm --config configs/dinov2_wm.toml
snek wm --config configs/dinov2_wm_residual.toml
snek jepa --config configs/jepa/*.toml
