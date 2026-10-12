import torch

from snekwm.models.predictor import ActionConditionedPredictor


def small(**kw):
    return ActionConditionedPredictor(
        in_dim=8, dim=16, depth=2, heads=2, num_tokens=4, num_actions=5, **kw
    )


def test_output_matches_input_token_shape():
    z = torch.randn(3, 4, 8)
    assert small()(z, torch.tensor([0, 1, 2])).shape == (3, 4, 8)


def test_prediction_depends_on_action():
    torch.manual_seed(0)
    model = small().eval()
    z = torch.randn(1, 4, 8)
    a0 = model(z, torch.tensor([0]))
    a1 = model(z, torch.tensor([3]))
    assert not torch.allclose(a0, a1)


def test_residual_predictor_is_identity_at_init():
    model = small(residual=True).eval()
    z = torch.randn(2, 4, 8)
    torch.testing.assert_close(model(z, torch.tensor([1, 4])), z)


def test_without_action_the_action_is_ignored():
    torch.manual_seed(0)
    model = small(use_action=False).eval()
    z = torch.randn(1, 4, 8)
    torch.testing.assert_close(model(z, torch.tensor([0])), model(z, torch.tensor([3])))
