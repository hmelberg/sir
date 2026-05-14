import numpy as np

from sir.config import default_config
from sir.constants import DiseaseState, GroupType
from sir.transmission import step_transmission
from sir.world import build_world


def test_no_infected_means_no_new_infections():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    # All susceptible by default
    theta = np.ones(7, dtype=np.float64)
    e = np.zeros(7, dtype=np.float64)
    new_infections = step_transmission(world, cfg, theta, e, rng)
    assert new_infections.sum() == 0


def test_zero_transmission_probability_means_no_new_infections():
    cfg_zero = default_config()
    cfg = type(cfg_zero)(
        **{**cfg_zero.__dict__,
           "group_p_baseline": {gt: 0.0 for gt in GroupType}}
    )
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    # Infect 10% of agents
    n = world.N
    world.state[: n // 10] = DiseaseState.I
    theta = np.ones(7, dtype=np.float64)
    e = np.zeros(7, dtype=np.float64)
    new_infections = step_transmission(world, cfg, theta, e, rng)
    assert new_infections.sum() == 0


def test_transmission_produces_some_infections_when_p_high_and_prevalence_high():
    cfg_orig = default_config()
    cfg = type(cfg_orig)(
        **{**cfg_orig.__dict__,
           "group_p_baseline": {gt: 0.5 for gt in GroupType}}
    )
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    # Infect half the population
    n = world.N
    world.state[: n // 2] = DiseaseState.I
    theta = np.ones(7, dtype=np.float64)
    e = np.zeros(7, dtype=np.float64)
    new_infections = step_transmission(world, cfg, theta, e, rng)
    assert new_infections.sum() > 0


def test_full_precaution_blocks_transmission():
    cfg_orig = default_config()
    cfg = type(cfg_orig)(
        **{**cfg_orig.__dict__,
           "group_p_baseline": {gt: 0.5 for gt in GroupType}}
    )
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    n = world.N
    world.state[: n // 2] = DiseaseState.I
    theta = np.ones(7, dtype=np.float64)
    e = np.ones(7, dtype=np.float64)  # full precaution
    new_infections = step_transmission(world, cfg, theta, e, rng)
    assert new_infections.sum() == 0


def test_vaccinated_agents_have_reduced_infection_rate():
    cfg_orig = default_config()
    cfg = type(cfg_orig)(
        **{**cfg_orig.__dict__,
           "group_p_baseline": {gt: 0.3 for gt in GroupType},
           "v_eff": 1.0}  # perfect vaccine
    )
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    n = world.N
    world.state[: n // 4] = DiseaseState.I
    world.state[n // 4 : n // 2] = DiseaseState.V
    theta = np.ones(7, dtype=np.float64)
    e = np.zeros(7, dtype=np.float64)
    new_infections = step_transmission(world, cfg, theta, e, rng)
    # No vaccinated agent should be newly infected
    vax_idx = np.where(world.state == DiseaseState.V)[0]
    assert not new_infections[vax_idx].any()
