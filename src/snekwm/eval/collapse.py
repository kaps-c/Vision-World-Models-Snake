"""Representation-collapse diagnostics.

Effective rank (Roy & Vetterli, 2007) is ``exp(H(p))`` where ``p`` are the
normalised singular values of the centred embedding matrix: about ``D`` for
isotropic embeddings and 1 when everything lies on a line.

The rank-ratio kill-switch compares the online encoder's effective rank with
the target's. It catches *dimensional* collapse only: embeddings can span
many directions and still carry nothing a probe can decode (*informational*
collapse), so it is necessary but not sufficient.
"""

from __future__ import annotations

import torch


class CollapseError(RuntimeError):
    """Raised when the online representation has dimensionally collapsed.

    Attributes:
        ratio: The offending online/target effective-rank ratio.
    """

    def __init__(self, ratio: float, minimum: float):
        super().__init__(f"rank ratio {ratio:.3f} < {minimum}")
        self.ratio = ratio


def effective_rank(x: torch.Tensor) -> float:
    """Return the effective rank of ``(n, d)`` embeddings ``x``."""
    x = x.float()
    s = torch.linalg.svdvals(x - x.mean(0, keepdim=True))
    p = s / s.sum().clamp_min(1e-12)
    entropy = -(p * torch.log(p.clamp_min(1e-12))).sum()
    return float(torch.exp(entropy))


def check_rank_ratio(
    online: torch.Tensor, target: torch.Tensor, minimum: float = 0.5
) -> float:
    """Return ``erank(online) / erank(target)``; raise if below ``minimum``."""
    ratio = effective_rank(online) / max(effective_rank(target), 1e-12)
    if ratio < minimum:
        raise CollapseError(ratio, minimum)
    return ratio
