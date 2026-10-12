"""Image encoders mapping ``(B, 3, H, W)`` images in ``[0, 1]`` to patch tokens."""

from __future__ import annotations

import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FrozenDinov2(nn.Module):
    """Frozen DINOv2 ViT-S/14 returning normalised patch tokens ``(B, N, 384)``.

    At 224 px input, ``N = 16 * 16 = 256``.

    Args:
        backbone: Optional pre-built backbone exposing ``forward_features``
            (used by tests); by default ``dinov2_vits14`` is fetched via
            ``torch.hub``.
    """

    def __init__(self, backbone: nn.Module | None = None):
        super().__init__()
        if backbone is None:
            backbone = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14", verbose=False
            )
        self.backbone = backbone.eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True) -> FrozenDinov2:
        """Keep the backbone in eval mode regardless of ``mode``."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to patch tokens."""
        x = (images - self.mean) / self.std
        return self.backbone.forward_features(x)["x_norm_patchtokens"]
