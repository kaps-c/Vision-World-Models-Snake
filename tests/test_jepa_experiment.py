import json

import torch

from snekwm.config import JEPAConfig
from snekwm.data.collect import collect
from snekwm.experiments.jepa import run_jepa
from tests.test_dinov2_wm import fake_features

TINY = dict(
    epochs=2,
    batch_size=16,
    cell_px=8,  # 4x4 grid -> 32 px -> 4 tokens of 16 px
    enc_dim=8,
    enc_depth=1,
    enc_heads=2,
    pred_dim=8,
    pred_depth=1,
    pred_heads=2,
    device="cpu",
)


def run_tiny(tmp_path, **overrides):
    data = tmp_path / "eps"
    paths = collect(data, size=4, episodes=6, max_steps=30, epsilon=0.3, seed=0)
    feats = tmp_path / "f.pt"
    torch.save(fake_features(paths), feats)
    cfg = JEPAConfig(
        data_dir=str(data),
        features_path=str(feats),
        run_dir=str(tmp_path / "run"),
        **{**TINY, **overrides},
    )
    return run_jepa(cfg), tmp_path / "run"


def test_pixel_jepa_writes_results_with_all_probes_and_rank_history(tmp_path):
    results, run = run_tiny(tmp_path)
    assert {"encoder", "change", "transfer", "transfer_displacement"} <= set(results)
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    assert len(history) == 2
    assert {"train_loss", "val_loss", "val_invariance", "rank_ratio"} <= set(history[0])
    for name in ("final.pt", "train.log", "summary.md", "curves.png", "probe.json"):
        assert (run / name).exists()


def test_frozen_feature_mode_trains_only_the_predictor(tmp_path):
    results, _ = run_tiny(tmp_path, encoder="dinov2", projector=False, beta=0.0)
    assert results["collapsed"] is False
    assert "encoder" in results


def test_kill_switch_stop_mode_stops_training_and_is_recorded(tmp_path):
    results, run = run_tiny(tmp_path, rank_ratio_min=1e9, kill_switch="stop")
    assert results["collapsed"] is True
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    assert len(history) == 1
    assert "collapse" in (run / "summary.md").read_text(encoding="utf-8").lower()


def test_kill_switch_record_mode_trains_full_budget_and_marks_first_collapse(tmp_path):
    results, run = run_tiny(tmp_path, rank_ratio_min=1e9, kill_switch="record")
    assert results["collapsed"] is True
    assert results["first_collapse_epoch"] == 1
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    assert len(history) == 2
    assert all(h["collapsed"] for h in history)


def test_reprobe_reproduces_probes_from_the_saved_checkpoint(tmp_path):
    from snekwm.experiments.jepa import reprobe_jepa

    results, run = run_tiny(tmp_path)
    (run / "probe.json").unlink()
    again = reprobe_jepa(run)
    assert again["encoder"]["r2_joint"] == results["encoder"]["r2_joint"]
    assert again["collapsed"] == results["collapsed"]
    assert (run / "probe.json").exists() and (run / "summary.md").exists()
