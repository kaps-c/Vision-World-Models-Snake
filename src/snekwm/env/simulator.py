"""Single-player Snake on an L x L grid with absolute-direction actions.

Coordinates are ``(row, col)``, zero-indexed, with ``(0, 0)`` top-left.
The snake is a tuple of cells ordered head first.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

import numpy as np

Cell = tuple[int, int]

#: Action index -> (d_row, d_col). ``(0, 0)`` keeps the current heading.
ACTIONS: tuple[Cell, ...] = ((0, 0), (0, 1), (0, -1), (-1, 0), (1, 0))


@dataclass(frozen=True)
class State:
    """Immutable game state."""

    snake: tuple[Cell, ...]
    food: Cell | None
    heading: Cell
    alive: bool = True
    won: bool = False
    steps: int = 0

    @property
    def head(self) -> Cell:
        """Return the head cell."""
        return self.snake[0]


@dataclass(frozen=True)
class StepResult:
    """Outcome of one :meth:`SnakeEnv.step`."""

    state: State
    ate: bool
    done: bool

    @property
    def won(self) -> bool:
        """Return whether the snake filled the grid."""
        return self.state.won


class SnakeEnv:
    """Snake simulator with RGB rendering.

    Args:
        size: Grid side length ``L``.
        cell_px: Pixels per cell when rendering; ``14 * 16 = 224`` matches
            DINOv2's pre-training resolution.
        seed: Seed for spawn positions and food placement.
    """

    BACKGROUND = (0, 0, 0)
    FOOD = (0, 200, 0)
    HEAD = (110, 170, 255)
    BODY = (20, 60, 200)

    def __init__(self, size: int = 14, cell_px: int = 16, seed: int | None = None):
        self.size = size
        self.cell_px = cell_px
        self.rng = random.Random(seed)
        self.state: State | None = None

    def reset(self) -> State:
        """Spawn a length-2 snake with a random heading and place food."""
        while True:
            head = (self.rng.randrange(self.size), self.rng.randrange(self.size))
            heading = ACTIONS[self.rng.randrange(1, len(ACTIONS))]
            neck = (head[0] - heading[0], head[1] - heading[1])
            if self._inside(neck):
                break
        snake = (head, neck)
        self.state = State(snake=snake, food=self._spawn_food(snake), heading=heading)
        return self.state

    def step(self, action: int) -> StepResult:
        """Advance one tick.

        A turn straight back into the neck is ignored (the snake keeps going),
        as in most Snake implementations.
        """
        s = self.state
        if s is None or not s.alive:
            raise RuntimeError("call reset() before stepping a finished episode")

        heading = self._resolve_heading(s.heading, ACTIONS[action])
        new_head = (s.head[0] + heading[0], s.head[1] + heading[1])
        ate = new_head == s.food
        body = s.snake if ate else s.snake[:-1]
        snake = (new_head, *body)
        steps = s.steps + 1

        if not self._inside(new_head) or new_head in body:
            self.state = replace(s, heading=heading, alive=False, steps=steps)
            return StepResult(self.state, ate=False, done=True)

        food = self._spawn_food(snake) if ate else s.food
        won = food is None
        self.state = State(snake, food, heading, alive=not won, won=won, steps=steps)
        return StepResult(self.state, ate=ate, done=won)

    def render(self, state: State | None = None) -> np.ndarray:
        """Render ``state`` (default: current) as a ``(H, W, 3)`` uint8 image."""
        s = state or self.state
        grid = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        if s.food is not None:
            grid[s.food] = self.FOOD
        for cell in s.snake[1:]:
            if self._inside(cell):
                grid[cell] = self.BODY
        if self._inside(s.head):
            grid[s.head] = self.HEAD
        return grid.repeat(self.cell_px, axis=0).repeat(self.cell_px, axis=1)

    def _inside(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.size and 0 <= cell[1] < self.size

    def _spawn_food(self, snake: tuple[Cell, ...]) -> Cell | None:
        occupied = set(snake)
        free = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) not in occupied
        ]
        return self.rng.choice(free) if free else None

    @staticmethod
    def _resolve_heading(current: Cell, requested: Cell) -> Cell:
        if requested == (0, 0) or requested == (-current[0], -current[1]):
            return current
        return requested
