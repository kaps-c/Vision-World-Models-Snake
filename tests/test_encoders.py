import pytest
import torch
from torch import nn

from snekwm.data.collect import collect
from snekwm.data.features import encode_episodes
from snekwm.models.encoders import IMAGENET_MEAN, IMAGENET_STD, FrozenDinov2


class FakeBackbone(nn.Module):
    """Mimics DINOv2's forward_features: 4x4 patches -> (B, N, 3) tokens."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))
        self.seen = None

    def forward_features(self, x):
        self.seen = x
        tokens = nn.functional.avg_pool2d(x, 4).flatten(2).transpose(1, 2)
        return {"x_norm_patchtokens": tokens * self.w, "x_norm_clstoken": None}


def test_frozen_encoder_normalises_input_and_has_no_trainable_params():
    bb = FakeBackbone()
    enc = FrozenDinov2(backbone=bb)
    x = torch.rand(2, 3, 8, 8)
    out = enc(x)
    assert out.shape == (2, 4, 3)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    torch.testing.assert_close(bb.seen, (x - mean) / std)
    assert not any(p.requires_grad for p in enc.parameters())


def test_encode_episodes_caches_one_token_tensor_per_frame(tmp_path):
    paths = collect(tmp_path, size=4, episodes=2, max_steps=10, epsilon=0.3, seed=0)
    enc = FrozenDinov2(backbone=FakeBackbone())
    feats = encode_episodes(enc, paths, cell_px=2, batch_size=4, device="cpu")
    assert set(feats) == {p.stem for p in paths}
    for p in paths:
        assert feats[p.stem].shape[1:] == (4, 3)
        assert feats[p.stem].dtype == torch.float16


@pytest.mark.network
def test_real_dinov2_gives_256_patches_of_384_dims():
    enc = FrozenDinov2()
    with torch.no_grad():
        assert enc(torch.rand(1, 3, 224, 224)).shape == (1, 256, 384)
