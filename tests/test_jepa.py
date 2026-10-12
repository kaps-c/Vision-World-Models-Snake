import pytest
import torch

from snekwm.models.jepa import (
    JEPA,
    Projector,
    SmallViT,
    invariance_loss,
    vicreg_terms,
)
from snekwm.models.predictor import ActionConditionedPredictor


def test_vicreg_variance_term_is_zero_for_spread_and_positive_for_collapsed():
    torch.manual_seed(0)
    var_spread, _ = vicreg_terms(torch.randn(2000, 8) * 2)
    var_collapsed, _ = vicreg_terms(torch.ones(2000, 8))
    assert var_spread == pytest.approx(0.0, abs=1e-3)
    assert var_collapsed == pytest.approx(0.99, abs=1e-3)  # std = sqrt(eps)


def test_vicreg_covariance_term_penalises_redundant_dimensions():
    torch.manual_seed(0)
    _, cov_indep = vicreg_terms(torch.randn(4000, 8))
    col = torch.randn(4000, 1)
    _, cov_dup = vicreg_terms(col.repeat(1, 8))
    assert cov_indep < 0.05 < cov_dup


def test_invariance_loss_ignores_scale():
    x = torch.randn(4, 5, 6)
    assert invariance_loss(3 * x, x) == pytest.approx(0.0, abs=1e-6)
    assert invariance_loss(-x, x) > 0


def test_small_vit_gives_one_token_per_patch():
    vit = SmallViT(patch=16, dim=24, depth=1, heads=2, image_size=32)
    assert vit(torch.rand(2, 3, 32, 32)).shape == (2, 4, 24)


def test_projector_bottleneck_keeps_width():
    proj = Projector(16)
    assert proj.net[0].out_features == 4
    assert proj(torch.randn(3, 5, 16)).shape == (3, 5, 16)


def make_jepa(momentum=0.9, projector=True):
    enc = SmallViT(patch=16, dim=8, depth=1, heads=2, image_size=32)
    pred = ActionConditionedPredictor(
        in_dim=8, dim=8, depth=1, heads=2, num_tokens=4, num_actions=5
    )
    return JEPA(enc, pred, momentum=momentum, projector=projector)


def test_target_encoder_gets_no_gradient_but_online_does():
    torch.manual_seed(0)
    jepa = make_jepa()
    obs, nxt = torch.rand(4, 3, 32, 32), torch.rand(4, 3, 32, 32)
    out = jepa(obs, torch.tensor([0, 1, 2, 3]), nxt)
    out.invariance.backward()
    assert all(p.grad is None for p in jepa.target.parameters())
    assert any(p.grad is not None for p in jepa.online.parameters())


def test_ema_step_moves_target_toward_online():
    torch.manual_seed(0)
    jepa = make_jepa(momentum=0.5)
    with torch.no_grad():
        for p in jepa.online.parameters():
            p.add_(1.0)
    before = [p.clone() for p in jepa.target.parameters()]
    jepa.update_target()
    for b, t, o in zip(
        before, jepa.target.parameters(), jepa.online.parameters(), strict=True
    ):
        torch.testing.assert_close(t, 0.5 * b + 0.5 * o)


def test_zero_momentum_makes_target_a_copy_of_online():
    jepa = make_jepa(momentum=0.0)
    with torch.no_grad():
        for p in jepa.online.parameters():
            p.add_(1.0)
    jepa.update_target()
    for t, o in zip(jepa.target.parameters(), jepa.online.parameters(), strict=True):
        torch.testing.assert_close(t, o)


def test_frozen_feature_mode_uses_inputs_as_embeddings():
    pred = ActionConditionedPredictor(
        in_dim=6, dim=8, depth=1, heads=2, num_tokens=3, num_actions=5
    )
    jepa = JEPA(None, pred, momentum=0.0, projector=False)
    z, z1 = torch.randn(2, 3, 6), torch.randn(2, 3, 6)
    out = jepa(z, torch.tensor([0, 1]), z1)
    torch.testing.assert_close(out.online, z)
    torch.testing.assert_close(out.target, z1)
