"""Hamiltonian-cycle data policy for collecting off-policy Snake trajectories.

Following a Hamiltonian cycle (a closed path through every cell exactly once)
never dies, so it is a strong prior that yields long episodes. Three sources
of diversity keep the data from being one memorised loop:

1. a **random backbone**: a fresh random Hamiltonian cycle per episode;
2. **safe shortcuts**: skip ahead along the cycle toward food when doing so
   keeps the body in cycle order (so the snake still can't trap itself);
3. **epsilon perturbation**: a random non-fatal action with probability eps.
"""

from __future__ import annotations

import random

from snekwm.env.simulator import ACTIONS, Cell, State, would_die


def hamiltonian_cycle(size: int) -> list[Cell]:
    """Return the canonical zig-zag Hamiltonian cycle on a ``size x size`` grid.

    Path: right along row 0, zig-zag down/up columns ``size-1 .. 1``, then
    back up column 0. Requires an even ``size`` (odd grids have none).
    """
    _check_size(size)
    cycle = [(0, c) for c in range(size)]
    for i, col in enumerate(range(size - 1, 0, -1)):
        rows = range(1, size) if i % 2 == 0 else range(size - 1, 0, -1)
        cycle.extend((r, col) for r in rows)
    cycle.extend((r, 0) for r in range(size - 1, 0, -1))
    return cycle


def random_hamiltonian_cycle(size: int, rng: random.Random) -> list[Cell]:
    """Return a random Hamiltonian cycle via the 2x2-block spanning-tree trick.

    Tile the grid with 2x2 blocks, each a 4-cell loop. Draw a random spanning
    tree over the blocks; for every tree edge, cut the two facing sides of the
    adjacent loops and bridge them. Merging loops along a tree never creates a
    second cycle, so the result is one loop through every cell.
    """
    _check_size(size)
    n = size // 2
    adj: dict[Cell, set[Cell]] = {}

    def link(a: Cell, b: Cell) -> None:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    def unlink(a: Cell, b: Cell) -> None:
        adj[a].discard(b)
        adj[b].discard(a)

    for i in range(n):
        for j in range(n):
            r, c = 2 * i, 2 * j
            for a, b in [
                ((r, c), (r, c + 1)),
                ((r, c + 1), (r + 1, c + 1)),
                ((r + 1, c + 1), (r + 1, c)),
                ((r + 1, c), (r, c)),
            ]:
                link(a, b)

    for (i, j), (k, _) in _random_spanning_tree(n, rng):
        if k == i:  # horizontal neighbours: block (i, j) left of (i, j + 1)
            r, c = 2 * i, 2 * j + 1
            unlink((r, c), (r + 1, c))
            unlink((r, c + 1), (r + 1, c + 1))
            link((r, c), (r, c + 1))
            link((r + 1, c), (r + 1, c + 1))
        else:  # vertical neighbours: block (i, j) above (i + 1, j)
            r, c = 2 * i + 1, 2 * j
            unlink((r, c), (r, c + 1))
            unlink((r + 1, c), (r + 1, c + 1))
            link((r, c), (r + 1, c))
            link((r, c + 1), (r + 1, c + 1))

    cycle, prev, cur = [(0, 0)], None, (0, 0)
    while True:
        nxt = next(x for x in adj[cur] if x != prev)
        if nxt == (0, 0):
            return cycle
        cycle.append(nxt)
        prev, cur = cur, nxt


def _random_spanning_tree(n: int, rng: random.Random) -> list[tuple[Cell, Cell]]:
    """Return tree edges ``(a, b)`` over an ``n x n`` grid, with ``a < b``."""
    start = (rng.randrange(n), rng.randrange(n))
    seen = {start}
    frontier = [(start, nb) for nb in _neighbours(start, n)]
    edges = []
    while frontier:
        a, b = frontier.pop(rng.randrange(len(frontier)))
        if b in seen:
            continue
        seen.add(b)
        edges.append((min(a, b), max(a, b)))
        frontier.extend((b, nb) for nb in _neighbours(b, n) if nb not in seen)
    return edges


def _neighbours(cell: Cell, n: int) -> list[Cell]:
    r, c = cell
    cand = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [(a, b) for a, b in cand if 0 <= a < n and 0 <= b < n]


def _check_size(size: int) -> None:
    if size < 2 or size % 2:
        raise ValueError(f"Hamiltonian cycle needs an even size >= 2, got {size}")


class PerturbedHamiltonian:
    """Random-backbone Hamiltonian policy with safe shortcuts and eps-noise.

    If the chosen move would be fatal (possible after a perturbation knocked
    the snake off cycle order), a random safe action is taken instead.

    Args:
        size: Grid side length (even).
        epsilon: Probability of a random non-fatal action, in ``[0, 1]``.
        seed: RNG seed; also selects the episode's backbone cycle.
    """

    def __init__(self, size: int, epsilon: float = 0.0, seed: int | None = None):
        self.size = size
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.cycle = random_hamiltonian_cycle(size, self.rng)
        self._index = {cell: i for i, cell in enumerate(self.cycle)}

    def act(self, state: State) -> int:
        """Return an action index for ``state``."""
        safe = [a for a in range(len(ACTIONS)) if not would_die(state, a, self.size)]
        planned = self._planned_action(state)
        perturb = self.rng.random() < self.epsilon
        if safe and (perturb or planned not in safe):
            return self.rng.choice(safe)
        return planned

    def _planned_action(self, state: State) -> int:
        n = len(self.cycle)
        head = self._index[state.head]
        # Walk the cycle in whichever direction keeps the neck behind the head.
        sign = -1 if (self._index[state.snake[1]] - head) % n == 1 else 1

        def ahead(cell: Cell) -> int:
            return (sign * (self._index[cell] - head)) % n

        tail_gap = ahead(state.snake[-1])  # free stretch is (0, tail_gap)
        limit = ahead(state.food) if state.food is not None else 1
        best, best_d = None, 0
        for a in range(1, len(ACTIONS)):
            d_row, d_col = ACTIONS[a]
            cell = (state.head[0] + d_row, state.head[1] + d_col)
            if cell not in self._index or cell in state.snake:
                continue
            d = ahead(cell)
            if 0 < d < tail_gap and d <= limit and d > best_d:
                best, best_d = a, d
        if best is not None and self._ordered(state, sign):
            return best
        target = self.cycle[(head + sign) % n]
        return ACTIONS.index((target[0] - state.head[0], target[1] - state.head[1]))

    def _ordered(self, state: State, sign: int) -> bool:
        """Whether the body follows the cycle order behind the head."""
        n = len(self.cycle)
        head = self._index[state.head]
        dists = [(sign * (head - self._index[c])) % n for c in state.snake]
        return all(a < b for a, b in zip(dists, dists[1:], strict=False))
