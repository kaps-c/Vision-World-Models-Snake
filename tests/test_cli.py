import pytest

from snekwm.cli import main
from snekwm.config import CollectConfig, load_config


def write(tmp_path, text):
    p = tmp_path / "c.toml"
    p.write_text(text)
    return p


def test_load_config_fills_dataclass_and_keeps_defaults(tmp_path):
    p = write(tmp_path, "[collect]\nsize = 4\nepisodes = 2\n")
    cfg = load_config(p, "collect", CollectConfig)
    assert cfg.size == 4 and cfg.episodes == 2
    assert cfg.max_steps == CollectConfig().max_steps


def test_unknown_key_is_rejected(tmp_path):
    p = write(tmp_path, "[collect]\nsizee = 4\n")
    with pytest.raises(TypeError):
        load_config(p, "collect", CollectConfig)


def test_snek_collect_writes_episodes(tmp_path):
    out = tmp_path / "eps"
    toml = "[collect]\nsize = 4\nepisodes = 3\nmax_steps = 10\n"
    p = write(tmp_path, toml + f'out_dir = "{out.as_posix()}"\n')
    main(["collect", "--config", str(p)])
    assert len(list(out.glob("*.npz"))) == 3
