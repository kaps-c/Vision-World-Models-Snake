"""Action-conditioned latent predictor: ``(z_t, a_t) -> z_hat_{t+1}``."""

from __future__ import annotations

import torch
from torch import nn


class ActionConditionedPredictor(nn.Module):
    """ViT over patch tokens with one prepended learned action token.

    The input window is Markovian (one frame), so no temporal (block-causal)
    mask is needed: every patch token attends to all patches and the action.

    Args:
        in_dim: Token width of the encoder features (384 for ViT-S/14).
        dim: Transformer width.
        depth: Number of transformer blocks.
        heads: Attention heads.
        num_tokens: Patch tokens per frame (256 for 224 px / 14).
        num_actions: Size of the discrete action space.
        residual: Predict ``z_t + f(z_t, a_t)`` with ``f`` zero-initialised, so
            the model starts as the identity world model.
        use_action: If ``False`` the action token is a constant (ablation).
    """

    def __init__(
        self,
        in_dim: int = 384,
        dim: int = 384,
        depth: int = 4,
        heads: int = 6,
        num_tokens: int = 256,
        num_actions: int = 5,
        residual: bool = False,
        use_action: bool = True,
    ):
        super().__init__()
        self.residual = residual
        self.use_action = use_action
        self.inp = nn.Linear(in_dim, dim)
        self.action = nn.Embedding(num_actions, dim)
        self.pos = nn.Parameter(torch.zeros(1, num_tokens + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, in_dim)
        if residual:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next-step tokens ``(B, N, in_dim)`` from ``z`` and ``action``."""
        if not self.use_action:
            action = torch.zeros_like(action)
        tokens = torch.cat([self.action(action)[:, None], self.inp(z)], dim=1)
        h = self.norm(self.blocks(tokens + self.pos))
        delta = self.out(h[:, 1:])
        return z + delta if self.residual else delta
