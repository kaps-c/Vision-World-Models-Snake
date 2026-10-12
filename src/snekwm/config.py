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


def load_config(path: str | Path, section: str, cls: type[T]) -> T:
    """Build ``cls`` from ``[section]`` of a TOML file.

    Unknown keys raise ``TypeError`` so typos fail loudly instead of
    silently falling back to defaults.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return cls(**data.get(section, {}))
