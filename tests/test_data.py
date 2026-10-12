import numpy as np
import torch

from snekwm.data.collect import collect, load_episode
from snekwm.data.dataset import TransitionDataset, split_episodes
from snekwm.env.simulator import SnakeEnv


def test_collect_writes_one_consistent_file_per_episode(tmp_path):
    paths = collect(tmp_path, size=4, episodes=3, max_steps=50, epsilon=0.3, seed=0)
    assert len(paths) == 3
    for p in paths:
        ep = load_episode(p)
        t = len(ep["actions"])
        assert 1 <= t <= 50
        assert ep["frames"].shape == (t + 1, 4, 4, 3)
        assert ep["heads"].shape == (t + 1, 2)


def test_recorded_head_matches_rendered_head_pixel(tmp_path):
    (path,) = collect(tmp_path, size=4, episodes=1, max_steps=30, epsilon=0.5, seed=1)
    ep = load_episode(path)
    alive = ep["alive"]
    for frame, (r, c), ok in zip(ep["frames"], ep["heads"], alive, strict=True):
        if ok:
            assert tuple(frame[r, c]) == SnakeEnv.HEAD


def test_collect_is_seed_deterministic(tmp_path):
    a = collect(tmp_path / "a", size=4, episodes=2, max_steps=40, epsilon=0.5, seed=7)
    b = collect(tmp_path / "b", size=4, episodes=2, max_steps=40, epsilon=0.5, seed=7)
    for pa, pb in zip(a, b, strict=True):
        np.testing.assert_array_equal(
            load_episode(pa)["actions"], load_episode(pb)["actions"]
        )


def test_dataset_yields_upsampled_transitions(tmp_path):
    paths = collect(tmp_path, size=4, episodes=2, max_steps=20, epsilon=0.3, seed=0)
    ds = TransitionDataset(paths, cell_px=3)
    total = sum(len(load_episode(p)["actions"]) for p in paths)
    assert len(ds) == total
    item = ds[0]
    assert item["obs"].shape == (3, 12, 12) and item["obs"].dtype == torch.float32
    assert item["next_obs"].shape == (3, 12, 12)
    assert item["obs"].min() >= 0.0 and item["obs"].max() <= 1.0
    assert item["action"].dtype == torch.long
    assert item["head"].shape == (2,) and item["next_head"].shape == (2,)


def test_split_is_by_episode_so_no_trajectory_leaks(tmp_path):
    paths = collect(tmp_path, size=4, episodes=10, max_steps=5, epsilon=0.3, seed=0)
    train, val = split_episodes(paths, val_frac=0.2, seed=0)
    assert len(val) == 2 and len(train) == 8
    assert not set(train) & set(val)
