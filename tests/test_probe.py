import numpy as np
import pytest
import torch

from snekwm.data.collect import collect, load_episode
from snekwm.eval.probe import (
    encoder_probe_data,
    fit_probe,
    predictor_probe_data,
)


def test_probe_recovers_linear_targets():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 16))
    y = X[:, :2] @ np.array([[2.0, 0.0], [0.0, -3.0]]) + 5
    res = fit_probe(X[:400], y[:400], X[400:], y[400:])
    assert res.r2_joint > 0.99
    assert res.r2_x > 0.99 and res.r2_y > 0.99


def test_probe_on_noise_scores_near_zero_or_below():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 16))
    y = rng.normal(size=(500, 2))
    assert fit_probe(X[:400], y[:400], X[400:], y[400:]).r2_joint < 0.05


@pytest.fixture
def episode(tmp_path):
    (p,) = collect(tmp_path, size=4, episodes=1, max_steps=40, epsilon=0.5, seed=3)
    return load_episode(p)


def test_encoder_probe_data_pairs_mean_pooled_frames_with_alive_heads(episode):
    t1 = len(episode["frames"])
    feats = torch.randn(t1, 5, 8)
    X, y = encoder_probe_data(feats, episode)
    n = episode["alive"].sum()
    assert X.shape == (n, 8) and y.shape == (n, 2)
    np.testing.assert_allclose(X[0], feats[0].mean(0).numpy(), rtol=1e-5)
    np.testing.assert_array_equal(y[0], episode["heads"][0])


def test_predictor_probe_data_targets_next_head(episode):
    t = len(episode["actions"])
    preds = torch.randn(t, 5, 8)
    X, y = predictor_probe_data(preds, episode)
    n = episode["alive"][1:].sum()
    assert X.shape == (n, 8)
    np.testing.assert_array_equal(y[0], episode["heads"][1])


def test_evaluate_fits_on_train_episodes_and_scores_val(tmp_path):
    from snekwm.eval.probe import evaluate

    paths = collect(tmp_path, size=6, episodes=6, max_steps=60, epsilon=0.3, seed=0)
    eps = {p.stem: load_episode(p) for p in paths}
    # Features that linearly encode the head: perfect probe.
    feats = {}
    for k, ep in eps.items():
        h = torch.tensor(ep["heads"], dtype=torch.float32)
        feats[k] = torch.cat([h, torch.randn(len(h), 3)], 1).unsqueeze(1)
    names = sorted(eps)
    res = evaluate(encoder_probe_data, feats, eps, names[:4], names[4:])
    assert res.r2_joint > 0.99
    assert res.n_val == sum(eps[k]["alive"].sum() for k in names[4:])


def test_displacement_probe_data_targets_head_delta(episode):
    from snekwm.eval.probe import displacement_probe_data

    t = len(episode["actions"])
    X, y = displacement_probe_data(torch.randn(t, 5, 8), episode)
    mask = episode["alive"][1:]
    delta = (episode["heads"][1:] - episode["heads"][:-1])[mask]
    assert X.shape == (mask.sum(), 8)
    np.testing.assert_array_equal(y, delta)
    assert set(np.abs(y).sum(1)) == {1}  # one cell per step


def test_change_tensors_subtract_current_frame():
    from snekwm.eval.probe import change_tensors

    feats = {"e": torch.randn(4, 3, 2)}
    preds = {"e": feats["e"][:-1] + 1.0}
    torch.testing.assert_close(change_tensors(preds, feats)["e"], torch.ones(3, 3, 2))


def test_identity_world_model_scores_zero_on_change_probe(tmp_path):
    from snekwm.eval.probe import change_tensors, displacement_probe_data, evaluate

    paths = collect(tmp_path, size=6, episodes=6, max_steps=60, epsilon=0.3, seed=0)
    eps = {p.stem: load_episode(p) for p in paths}
    feats = {k: torch.randn(len(e["frames"]), 3, 4) for k, e in eps.items()}
    identity = {k: v[:-1] for k, v in feats.items()}
    names = sorted(eps)
    deltas = change_tensors(identity, feats)
    res = evaluate(displacement_probe_data, deltas, eps, names[:4], names[4:])
    assert abs(res.r2_joint) < 0.05


def _linear_world(tmp_path):
    """Episodes plus features that encode the head linearly in one token."""
    paths = collect(tmp_path, size=6, episodes=8, max_steps=80, epsilon=0.3, seed=1)
    eps = {p.stem: load_episode(p) for p in paths}
    feats = {}
    for k, ep in eps.items():
        h = torch.tensor(ep["heads"], dtype=torch.float32)
        feats[k] = torch.cat([h, torch.zeros(len(h), 2)], 1).unsqueeze(1)
    names = sorted(eps)
    return eps, feats, names[:6], names[6:]


def test_transfer_scores_a_perfect_world_model_at_the_ceiling(tmp_path):
    from snekwm.eval.probe import transfer_probe

    eps, feats, train, val = _linear_world(tmp_path)
    perfect = {k: v[1:] for k, v in feats.items()}  # predicts the true next latent
    res = transfer_probe(perfect, feats, eps, train, val)
    assert res["transfer"].r2_joint > 0.99
    assert res["transfer_displacement"].r2_joint > 0.99


def test_transfer_displacement_is_zero_for_the_identity_world_model(tmp_path):
    from snekwm.eval.probe import transfer_probe

    eps, feats, train, val = _linear_world(tmp_path)
    identity = {k: v[:-1] for k, v in feats.items()}
    res = transfer_probe(identity, feats, eps, train, val)
    assert res["transfer"].r2_joint > 0.8  # copy-the-head is a strong floor
    assert abs(res["transfer_displacement"].r2_joint) < 0.05


def test_transfer_ignores_action_passthrough(tmp_path):
    from snekwm.eval.probe import transfer_probe

    eps, feats, train, val = _linear_world(tmp_path)
    # Identity prediction plus the action one-hot written into unused dims:
    # a pooled probe could read the action, the frozen transfer probe cannot.
    leaky = {}
    for k, v in feats.items():
        onehot = torch.nn.functional.one_hot(
            torch.as_tensor(eps[k]["actions"]), 5
        ).float()
        leaky[k] = torch.cat([v[:-1, :, :2], onehot[:, None, :2] * 100], -1)
    res = transfer_probe(leaky, feats, eps, train, val)
    assert res["transfer_displacement"].r2_joint < 0.05


def test_action_only_control_reads_absolute_actions(tmp_path):
    from snekwm.eval.probe import action_only_control

    eps, _, train, val = _linear_world(tmp_path)
    assert action_only_control(eps, train, val).r2_joint > 0.7
