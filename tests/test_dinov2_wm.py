import json

import torch

from snekwm.config import WorldModelConfig
from snekwm.data.collect import collect, load_episode
from snekwm.data.features import FeatureTransitions
from snekwm.experiments.dinov2_wm import run_dinov2_wm


def fake_features(paths, n=4, d=8):
    """Tokens that encode the head position, so dynamics are learnable."""
    out = {}
    for p in paths:
        heads = torch.tensor(load_episode(p)["heads"], dtype=torch.float32)
        tok = torch.zeros(len(heads), n, d)
        tok[:, :, :2] = heads[:, None, :] / 4
        out[p.stem] = tok.half()
    return out


def test_feature_transitions_pair_consecutive_frames(tmp_path):
    paths = collect(tmp_path, size=4, episodes=2, max_steps=10, epsilon=0.3, seed=0)
    feats = fake_features(paths)
    eps = {p.stem: load_episode(p) for p in paths}
    ft = FeatureTransitions(feats, eps, [p.stem for p in paths], device="cpu")
    assert len(ft) == sum(len(e["actions"]) for e in eps.values())
    z, a, z1 = next(ft.batches(5, shuffle=False))
    stem = paths[0].stem
    torch.testing.assert_close(z[0], feats[stem][0].float())
    torch.testing.assert_close(z1[0], feats[stem][1].float())
    assert a[0] == eps[stem]["actions"][0]


def test_run_writes_history_checkpoint_and_probes(tmp_path):
    data = tmp_path / "eps"
    paths = collect(data, size=4, episodes=6, max_steps=30, epsilon=0.3, seed=0)
    feats_path = tmp_path / "f.pt"
    torch.save(fake_features(paths), feats_path)
    cfg = WorldModelConfig(
        data_dir=str(data),
        features_path=str(feats_path),
        run_dir=str(tmp_path / "run"),
        epochs=3,
        batch_size=16,
        dim=16,
        depth=1,
        heads=2,
        device="cpu",
    )
    run_dinov2_wm(cfg)
    run = tmp_path / "run"
    history = json.loads((run / "history.json").read_text())
    assert len(history) == 3
    assert {"train_loss", "val_mse", "val_cos", "identity_mse"} <= set(history[0])
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    probe = json.loads((run / "probe.json").read_text())
    assert {"predictor", "displacement", "change"} <= set(probe)
    for name in ("final.pt", "train.log", "summary.md", "curves.png"):
        assert (run / name).exists()
    assert "Identity WM" in (run / "summary.md").read_text(encoding="utf-8")
