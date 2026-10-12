import numpy as np
import pytest

from snekwm.env.simulator import ACTIONS, SnakeEnv, State

KEEP, RIGHT, LEFT, UP, DOWN = range(5)


def make(snake, food, heading=(0, 1), size=5):
    env = SnakeEnv(size=size, cell_px=4, seed=0)
    env.state = State(snake=tuple(snake), food=food, heading=heading)
    return env


def test_actions_are_five_absolute_directions():
    assert ACTIONS == ((0, 0), (0, 1), (0, -1), (-1, 0), (1, 0))


def test_reset_gives_length_two_snake_inside_grid_with_food_elsewhere():
    env = SnakeEnv(size=14, seed=1)
    s = env.reset()
    assert len(s.snake) == 2
    assert all(0 <= r < 14 and 0 <= c < 14 for r, c in s.snake)
    assert s.food not in s.snake
    assert s.alive


def test_keep_moves_along_heading():
    env = make([(2, 2), (2, 1)], food=(0, 0))
    s = env.step(KEEP).state
    assert s.snake == ((2, 3), (2, 2))


def test_turn_changes_heading():
    env = make([(2, 2), (2, 1)], food=(0, 0))
    s = env.step(DOWN).state
    assert s.snake == ((3, 2), (2, 2))
    assert s.heading == (1, 0)


def test_reverse_is_ignored():
    env = make([(2, 2), (2, 1)], food=(0, 0))
    assert env.step(LEFT).state.snake == ((2, 3), (2, 2))


def test_eating_grows_and_respawns_food():
    env = make([(2, 2), (2, 1)], food=(2, 3))
    r = env.step(KEEP)
    assert r.ate
    assert r.state.snake == ((2, 3), (2, 2), (2, 1))
    assert r.state.food not in r.state.snake


def test_wall_kills():
    env = make([(2, 4), (2, 3)], food=(0, 0))
    r = env.step(KEEP)
    assert r.done and not r.state.alive


def test_self_collision_kills():
    snake = [(2, 2), (2, 3), (3, 3), (3, 2), (3, 1)]
    env = make(snake, food=(0, 0), heading=(0, -1))
    r = env.step(DOWN)
    assert r.done and not r.state.alive


def test_moving_into_vacating_tail_is_legal():
    snake = [(2, 2), (2, 3), (3, 3), (3, 2)]
    env = make(snake, food=(0, 0), heading=(0, -1))
    assert env.step(DOWN).state.alive


def test_step_after_done_raises():
    env = make([(2, 4), (2, 3)], food=(0, 0))
    env.step(KEEP)
    with pytest.raises(RuntimeError):
        env.step(KEEP)


def test_filling_grid_wins():
    env = make([(0, 0), (0, 1)], food=(1, 0), heading=(0, -1), size=2)
    env.state = State(((0, 0), (0, 1), (1, 1)), (1, 0), (0, -1))
    r = env.step(DOWN)
    assert r.done and r.won and r.state.food is None


def test_render_shape_and_colours():
    env = make([(2, 2), (2, 1)], food=(0, 0))
    img = env.render()
    assert img.shape == (20, 20, 3) and img.dtype == np.uint8
    px = lambda r, c: tuple(img[r * 4 + 1, c * 4 + 1])  # noqa: E731
    assert px(0, 0) == SnakeEnv.FOOD
    assert px(2, 2) == SnakeEnv.HEAD
    assert px(2, 1) == SnakeEnv.BODY
    assert px(4, 4) == SnakeEnv.BACKGROUND


def test_default_render_is_224_for_14_grid():
    env = SnakeEnv(size=14, seed=0)
    env.reset()
    assert env.render().shape == (224, 224, 3)


def test_seeded_runs_are_deterministic():
    def roll(seed):
        env = SnakeEnv(size=6, seed=seed)
        env.reset()
        out = []
        for a in [KEEP, DOWN, KEEP, RIGHT, KEEP]:
            r = env.step(a)
            out.append(r.state)
            if r.done:
                break
        return out

    assert roll(3) == roll(3)
