"""Pre-compute frozen-encoder features so experiments don't re-encode images."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from snekwm.data.collect import load_episode


@torch.no_grad()
def encode_episodes(
    encoder: nn.Module,
    paths: Sequence[Path],
    *,
    cell_px: int = 16,
    batch_size: int = 64,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Encode every frame of every episode.

    Returns:
        ``{episode_stem: (T+1, N, D) float16}`` on CPU.
    """
    encoder = encoder.to(device).eval()
    out = {}
    for path in paths:
        frames = torch.from_numpy(load_episode(path)["frames"])
        chunks = []
        for i in range(0, len(frames), batch_size):
            x = frames[i : i + batch_size].to(device)
            x = x.repeat_interleave(cell_px, 1).repeat_interleave(cell_px, 2)
            x = x.permute(0, 3, 1, 2).float().div_(255.0)
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                chunks.append(encoder(x).half().cpu())
        out[Path(path).stem] = torch.cat(chunks)
    return out
