"""Transition dataset over collected episodes."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from snekwm.data.collect import load_episode


def split_episodes(
    paths: Sequence[Path], val_frac: float, seed: int = 0
) -> tuple[list[Path], list[Path]]:
    """Split episode files into train/val.

    Splitting by episode (not by transition) keeps near-duplicate consecutive
    frames from leaking between train and validation.
    """
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    n_val = round(len(shuffled) * val_frac)
    return shuffled[n_val:], shuffled[:n_val]


class TransitionDataset(Dataset):
    """``(obs_t, a_t, obs_t+1)`` transitions with head-position labels.

    Items are dicts with ``obs``/``next_obs`` (3, H, W) float in ``[0, 1]``,
    ``action`` (long), ``head``/``next_head`` (2,) float, and ``next_alive``.

    Args:
        paths: Episode files from :func:`snekwm.data.collect.collect`.
        cell_px: Upsampling factor (16 gives 224 px for a 14x14 grid).
    """

    def __init__(self, paths: Sequence[Path], cell_px: int = 16):
        self.cell_px = cell_px
        self.episodes = [load_episode(p) for p in paths]
        self.index = [
            (e, t)
            for e, ep in enumerate(self.episodes)
            for t in range(len(ep["actions"]))
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        e, t = self.index[i]
        ep = self.episodes[e]
        return {
            "obs": self._image(ep["frames"][t]),
            "next_obs": self._image(ep["frames"][t + 1]),
            "action": torch.tensor(ep["actions"][t], dtype=torch.long),
            "head": torch.tensor(ep["heads"][t], dtype=torch.float32),
            "next_head": torch.tensor(ep["heads"][t + 1], dtype=torch.float32),
            "next_alive": torch.tensor(ep["alive"][t + 1]),
        }

    def _image(self, frame: np.ndarray) -> torch.Tensor:
        up = frame.repeat(self.cell_px, axis=0).repeat(self.cell_px, axis=1)
        return torch.from_numpy(up).permute(2, 0, 1).float().div_(255.0)
