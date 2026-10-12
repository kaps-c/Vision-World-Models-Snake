"""Strict-JEPA world model: online encoder, EMA target, action-conditioned predictor.

Training signal: predict the *target encoder's* embedding of the next frame
from the *online encoder's* embedding of the current frame and the action::

    z_t      = online(o_t)
    z_hat    = projector(predictor(z_t, a_t))       # projector: online branch only
    z_tgt    = stopgrad(target(o_{t+1}))            # target = EMA(online)
    loss     = lam * invariance(z_hat, z_tgt) + beta * (mu * var(z_t) + nu * cov(z_t))

Anti-collapse levers: the EMA target, the asymmetric projector, and the
VICReg variance/covariance regulariser (Bardes et al., 2022).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from snekwm.models.predictor import ActionConditionedPredictor


def vicreg_terms(
    x: torch.Tensor, eps: float = 1e-4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return VICReg (variance, covariance) penalties for ``(n, d)`` embeddings.

    Variance: hinge pushing each dimension's std up to 1 (prevents collapse to
    a point). Covariance: squared off-diagonal covariances, scaled by ``1/d``
    (prevents dimensions from duplicating each other).
    """
    x = x.float()
    n, d = x.shape
    std = torch.sqrt(x.var(dim=0) + eps)
    variance = F.relu(1.0 - std).mean()
    xc = x - x.mean(dim=0)
    cov = (xc.T @ xc) / (n - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    covariance = off_diag.pow(2).sum() / d
    return variance, covariance


def invariance_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Smooth-L1 between L2-normalised token embeddings (scale-invariant)."""
    return F.smooth_l1_loss(
        F.normalize(pred.float(), dim=-1), F.normalize(target.float(), dim=-1)
    )


class SmallViT(nn.Module):
    """Scratch ViT encoder: ``(B, 3, H, W)`` in ``[0, 1]`` -> ``(B, N, dim)``.

    With ``patch=16`` on a 224 px render of a 14x14 grid, each token is
    exactly one grid cell.
    """

    def __init__(
        self,
        patch: int = 16,
        dim: int = 192,
        depth: int = 6,
        heads: int = 3,
        image_size: int = 224,
    ):
        super().__init__()
        self.embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        num_tokens = (image_size // patch) ** 2
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to patch tokens."""
        x = self.embed((images - 0.5) / 0.5).flatten(2).transpose(1, 2)
        return self.norm(self.blocks(x + self.pos))


class Projector(nn.Module):
    """Bottleneck MLP ``d -> d/4 -> d`` applied to the online branch only."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.GELU(), nn.Linear(dim // 4, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project token embeddings."""
        return self.net(x)


@dataclass
class JEPAOutput:
    """Embeddings and loss terms from one JEPA forward pass."""

    online: torch.Tensor  # z_t from the online encoder (B, N, D)
    target: torch.Tensor  # stop-grad target embedding of o_{t+1}
    prediction: torch.Tensor  # projector(predictor(z_t, a_t))
    invariance: torch.Tensor
    variance: torch.Tensor
    covariance: torch.Tensor


class JEPA(nn.Module):
    """Online/EMA-target JEPA around an :class:`ActionConditionedPredictor`.

    Args:
        encoder: Online encoder. ``None`` means inputs already are frozen
            features (e.g. cached DINOv2 tokens): online and target are both
            the identity and nothing is learned on the encoder side.
        predictor: Action-conditioned predictor in the encoder's token space.
        momentum: EMA momentum for the target. ``0`` makes the target a
            stop-grad copy of the online encoder (the "no EMA" ablation).
        projector: Whether to add the asymmetric projector.
    """

    def __init__(
        self,
        encoder: nn.Module | None,
        predictor: ActionConditionedPredictor,
        momentum: float = 0.996,
        projector: bool = True,
    ):
        super().__init__()
        self.online = encoder if encoder is not None else nn.Identity()
        self.target = copy.deepcopy(self.online).requires_grad_(False)
        self.predictor = predictor
        self.momentum = momentum
        width = predictor.out.out_features
        self.projector = Projector(width) if projector else nn.Identity()

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor
    ) -> JEPAOutput:
        """Encode, predict and compute the loss terms for one batch."""
        z = self.online(obs)
        with torch.no_grad():
            z_tgt = self.target(next_obs)
        pred = self.projector(self.predictor(z, action))
        variance, covariance = vicreg_terms(z.reshape(-1, z.shape[-1]))
        return JEPAOutput(
            online=z,
            target=z_tgt,
            prediction=pred,
            invariance=invariance_loss(pred, z_tgt),
            variance=variance,
            covariance=covariance,
        )

    @torch.no_grad()
    def update_target(self) -> None:
        """``target <- m * target + (1 - m) * online``."""
        for t, o in zip(
            self.target.parameters(), self.online.parameters(), strict=True
        ):
            t.mul_(self.momentum).add_(o.detach(), alpha=1.0 - self.momentum)
