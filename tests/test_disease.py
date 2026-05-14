import numpy as np

from sir.config import default_config
from sir.constants import DiseaseState
from sir.disease import step_recovery
from sir.world import build_world


def test_recovery_moves_infected_to_recovered():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    # Infect 100 agents
    world.state[:100] = DiseaseState.I
    world.days_infected[:100] = 1
    # With gamma=1 (force recovery), all should recover
    step_recovery(world, gamma=1.0, rng=rng)
    assert (world.state[:100] == DiseaseState.R).all()


def test_recovery_zero_when_gamma_zero():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    world.state[:100] = DiseaseState.I
    world.days_infected[:100] = 1
    step_recovery(world, gamma=0.0, rng=rng)
    assert (world.state[:100] == DiseaseState.I).all()


def test_recovery_does_not_touch_susceptible_or_vaccinated():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    world.state[:50] = DiseaseState.S
    world.state[50:100] = DiseaseState.V
    world.state[100:200] = DiseaseState.I
    world.days_infected[100:200] = 1
    step_recovery(world, gamma=1.0, rng=rng)
    assert (world.state[:50] == DiseaseState.S).all()
    assert (world.state[50:100] == DiseaseState.V).all()
    assert (world.state[100:200] == DiseaseState.R).all()


def test_days_infected_increments():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    world.state[:100] = DiseaseState.I
    world.days_infected[:100] = 1
    step_recovery(world, gamma=0.0, rng=rng)
    assert (world.days_infected[:100] == 2).all()


def test_recovery_rate_approximately_gamma():
    cfg = default_config()
    rng = np.random.default_rng(42)
    world = build_world(cfg, rng)
    n_infected = 5000
    world.state[:n_infected] = DiseaseState.I
    world.days_infected[:n_infected] = 1
    step_recovery(world, gamma=0.2, rng=rng)
    recovered = (world.state[:n_infected] == DiseaseState.R).sum()
    # ~20% should recover; allow generous tolerance for 5000 draws
    assert 800 < recovered < 1200
