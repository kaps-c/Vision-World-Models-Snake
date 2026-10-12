"""Pre-compute frozen-encoder features so experiments don't re-encode images."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
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


class FeatureTransitions:
    """``(z_t, a_t, z_{t+1})`` minibatches drawn from cached episode features.

    All frames of the selected episodes are concatenated into one tensor on
    ``device`` (about 1.4 GB fp16 for the L=14 train split), which avoids
    DataLoader overhead entirely; batches are cast to float32 on the fly.
    """

    def __init__(
        self,
        feats: Mapping[str, torch.Tensor],
        episodes: Mapping[str, dict[str, np.ndarray]],
        names: Sequence[str],
        device: str = "cuda",
    ):
        frames, src, dst, actions, offset = [], [], [], [], 0
        for name in names:
            f = feats[name]
            t = len(episodes[name]["actions"])
            frames.append(f)
            src.append(torch.arange(offset, offset + t))
            dst.append(torch.arange(offset + 1, offset + t + 1))
            actions.append(torch.as_tensor(episodes[name]["actions"]))
            offset += len(f)
        self.frames = torch.cat(frames).to(device)
        self.src = torch.cat(src).to(device)
        self.dst = torch.cat(dst).to(device)
        self.actions = torch.cat(actions).long().to(device)

    def __len__(self) -> int:
        return len(self.src)

    def batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Yield ``(z_t, a_t, z_{t+1})`` float32 batches."""
        if shuffle:
            order = torch.randperm(len(self), generator=generator)
        else:
            order = torch.arange(len(self))
        order = order.to(self.src.device)
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            yield (
                self.frames[self.src[idx]].float(),
                self.actions[idx],
                self.frames[self.dst[idx]].float(),
            )
