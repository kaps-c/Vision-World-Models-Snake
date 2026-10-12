import random

import pytest

from snekwm.env.hamiltonian import (
    PerturbedHamiltonian,
    hamiltonian_cycle,
    random_hamiltonian_cycle,
)
from snekwm.env.simulator import ACTIONS, SnakeEnv


@pytest.mark.parametrize("size", [2, 4, 14])
def test_cycle_visits_every_cell_once_through_adjacent_steps(size):
    cycle = hamiltonian_cycle(size)
    assert sorted(cycle) == [(r, c) for r in range(size) for c in range(size)]
    for a, b in zip(cycle, cycle[1:] + cycle[:1], strict=True):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _assert_hamiltonian(cycle, size):
    assert sorted(cycle) == [(r, c) for r in range(size) for c in range(size)]
    for a, b in zip(cycle, cycle[1:] + cycle[:1], strict=True):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


@pytest.mark.parametrize("size", [2, 4, 14])
@pytest.mark.parametrize("seed", range(5))
def test_random_cycle_is_hamiltonian(size, seed):
    _assert_hamiltonian(random_hamiltonian_cycle(size, random.Random(seed)), size)


def test_random_cycles_differ_across_seeds():
    cycles = {tuple(random_hamiltonian_cycle(8, random.Random(s))) for s in range(10)}
    assert len(cycles) > 1


def test_odd_grid_has_no_cycle():
    with pytest.raises(ValueError):
        hamiltonian_cycle(5)


@pytest.mark.parametrize("size", [4, 6])
@pytest.mark.parametrize("seed", range(20))
def test_unperturbed_policy_never_dies_and_eventually_wins(size, seed):
    env = SnakeEnv(size=size, cell_px=1, seed=seed)
    policy = PerturbedHamiltonian(size=size, epsilon=0.0, seed=seed)
    state = env.reset()
    for _ in range(size**4):
        result = env.step(policy.act(state))
        state = result.state
        if result.done:
            break
    assert state.won


def _fatal(env, state, action):
    env.state = state
    r = env.step(action)
    return r.done and not r.won


@pytest.mark.parametrize("seed", range(20))
def test_perturbed_policy_avoids_death_whenever_a_safe_move_exists(seed):
    env = SnakeEnv(size=6, cell_px=1, seed=seed)
    policy = PerturbedHamiltonian(size=6, epsilon=1.0, seed=seed)
    state = env.reset()
    for _ in range(200):
        action = policy.act(state)
        safe = [a for a in range(len(ACTIONS)) if not _fatal(env, state, a)]
        if safe:
            assert action in safe
        env.state = state
        result = env.step(action)
        if result.done:
            break
        state = result.state


def test_shortcuts_reach_first_food_much_faster_than_following_the_cycle():
    # Pure cycle-following averages ~L*L/2 = 32 steps to the first food on 8x8.
    total = 0
    for seed in range(30):
        env = SnakeEnv(size=8, cell_px=1, seed=seed)
        policy = PerturbedHamiltonian(size=8, epsilon=0.0, seed=seed)
        state = env.reset()
        while not env.step(policy.act(state)).ate:
            state = env.state
            total += 1
    assert total / 30 < 16
