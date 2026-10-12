"""Roll out the data policy and save episodes as compressed ``.npz`` files.

Frames are stored at grid resolution (one pixel per cell); the dataset
upsamples at load time. Rendering is pure pixel repetition, so this is
lossless and ~``cell_px**2`` times smaller on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from snekwm.env.hamiltonian import PerturbedHamiltonian
from snekwm.env.simulator import SnakeEnv


def collect(
    out_dir: str | Path,
    *,
    size: int,
    episodes: int,
    max_steps: int,
    epsilon: float,
    seed: int,
) -> list[Path]:
    """Collect ``episodes`` trajectories and return the written file paths.

    Each file holds ``frames`` (T+1, L, L, 3) uint8, ``actions`` (T,) int64,
    ``heads`` (T+1, 2) int64 and ``alive`` (T+1,) bool. A dead snake's head
    may lie outside the grid, so consumers should mask with ``alive``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(episodes):
        ep_seed = seed * 100_003 + i
        env = SnakeEnv(size=size, cell_px=1, seed=ep_seed)
        policy = PerturbedHamiltonian(size=size, epsilon=epsilon, seed=ep_seed)
        state = env.reset()
        frames, heads, alive, actions = [env.render()], [state.head], [True], []
        for _ in range(max_steps):
            action = policy.act(state)
            result = env.step(action)
            state = result.state
            actions.append(action)
            frames.append(env.render())
            heads.append(state.head)
            alive.append(state.alive or state.won)
            if result.done:
                break
        path = out_dir / f"episode_{i:05d}.npz"
        np.savez_compressed(
            path,
            frames=np.stack(frames),
            actions=np.array(actions, dtype=np.int64),
            heads=np.array(heads, dtype=np.int64),
            alive=np.array(alive, dtype=bool),
        )
        paths.append(path)
    return paths


def load_episode(path: str | Path) -> dict[str, np.ndarray]:
    """Load one episode file into a dict of arrays."""
    with np.load(path) as f:
        return {k: f[k] for k in f.files}
