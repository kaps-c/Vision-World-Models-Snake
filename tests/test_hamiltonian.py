import pytest

from snekwm.env.hamiltonian import PerturbedHamiltonian, hamiltonian_cycle
from snekwm.env.simulator import ACTIONS, SnakeEnv


@pytest.mark.parametrize("size", [2, 4, 14])
def test_cycle_visits_every_cell_once_through_adjacent_steps(size):
    cycle = hamiltonian_cycle(size)
    assert sorted(cycle) == [(r, c) for r in range(size) for c in range(size)]
    for a, b in zip(cycle, cycle[1:] + cycle[:1], strict=True):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def test_odd_grid_has_no_cycle():
    with pytest.raises(ValueError):
        hamiltonian_cycle(5)


@pytest.mark.parametrize("seed", range(20))
def test_unperturbed_policy_never_dies_and_eventually_wins(seed):
    env = SnakeEnv(size=4, cell_px=1, seed=seed)
    policy = PerturbedHamiltonian(size=4, epsilon=0.0, seed=seed)
    state = env.reset()
    for _ in range(2000):
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
