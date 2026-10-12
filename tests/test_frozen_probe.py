import json

from snekwm.config import ProbeConfig
from snekwm.data.collect import collect
from snekwm.experiments.frozen_probe import run_frozen_probe
from snekwm.models.encoders import FrozenDinov2
from tests.test_encoders import FakeBackbone


def test_frozen_probe_writes_encoder_and_identity_results(tmp_path):
    data = tmp_path / "eps"
    collect(data, size=4, episodes=5, max_steps=30, epsilon=0.3, seed=0)
    cfg = ProbeConfig(
        data_dir=str(data),
        features_path=str(tmp_path / "f.pt"),
        run_dir=str(tmp_path / "run"),
        cell_px=2,
        device="cpu",
    )
    out = run_frozen_probe(cfg, encoder=FrozenDinov2(backbone=FakeBackbone()))
    saved = json.loads((tmp_path / "run" / "probe.json").read_text())
    assert saved == out
    assert {"encoder", "identity_predictor"} <= set(saved)
    assert "r2_joint" in saved["encoder"]
    assert (tmp_path / "f.pt").exists()
    assert (tmp_path / "run" / "train.log").exists()
