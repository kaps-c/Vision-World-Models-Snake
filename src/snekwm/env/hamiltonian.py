"""Hamiltonian-cycle data policy for collecting off-policy Snake trajectories.

Following a Hamiltonian cycle (a closed path through every cell exactly once)
never dies, so it is a strong prior that yields long episodes. Random
perturbations add action diversity so world models see more than one move
per state.
"""

from __future__ import annotations

import random

from snekwm.env.simulator import ACTIONS, Cell, State, would_die


def hamiltonian_cycle(size: int) -> list[Cell]:
    """Return a Hamiltonian cycle on a ``size x size`` grid.

    Path: right along row 0, zig-zag down/up columns ``size-1 .. 1``, then
    back up column 0. Requires an even ``size`` (odd grids have none).
    """
    if size < 2 or size % 2:
        raise ValueError(f"Hamiltonian cycle needs an even size >= 2, got {size}")
    cycle = [(0, c) for c in range(size)]
    for i, col in enumerate(range(size - 1, 0, -1)):
        rows = range(1, size) if i % 2 == 0 else range(size - 1, 0, -1)
        cycle.extend((r, col) for r in rows)
    cycle.extend((r, 0) for r in range(size - 1, 0, -1))
    return cycle


class PerturbedHamiltonian:
    """Follow the Hamiltonian cycle, but take a random safe action w.p. ``epsilon``.

    If the cycle move would be fatal (possible after a perturbation knocked the
    snake off the cycle), a random safe action is taken instead when one exists.

    Args:
        size: Grid side length (even).
        epsilon: Perturbation probability in ``[0, 1]``.
        seed: RNG seed.
    """

    def __init__(self, size: int, epsilon: float = 0.0, seed: int | None = None):
        self.size = size
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        cycle = hamiltonian_cycle(size)
        self._succ = dict(zip(cycle, cycle[1:] + cycle[:1], strict=True))
        self._pred = {b: a for a, b in self._succ.items()}

    def act(self, state: State) -> int:
        """Return an action index for ``state``."""
        safe = [a for a in range(len(ACTIONS)) if not would_die(state, a, self.size)]
        cycle_action = self._cycle_action(state)
        perturb = self.rng.random() < self.epsilon
        if safe and (perturb or cycle_action not in safe):
            return self.rng.choice(safe)
        return cycle_action

    def _cycle_action(self, state: State) -> int:
        # Walk the cycle in whichever direction keeps the neck behind the head.
        step = self._pred if state.snake[1] == self._succ[state.head] else self._succ
        target = step[state.head]
        return ACTIONS.index((target[0] - state.head[0], target[1] - state.head[1]))
