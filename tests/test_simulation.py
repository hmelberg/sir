import numpy as np

from sir.config import default_config
from sir.constants import DiseaseState
from sir.simulation import SimResult, simulate


def test_simulation_runs_and_returns_result():
    cfg = default_config()
    rng = np.random.default_rng(0)
    result = simulate(cfg, interventions=[], rng=rng, initial_infected=10)
    assert isinstance(result, SimResult)
    assert result.S_history.shape == (cfg.T + 1,)
    assert result.I_history.shape == (cfg.T + 1,)
    assert result.R_history.shape == (cfg.T + 1,)
    assert result.V_history.shape == (cfg.T + 1,)


def test_initial_state_correct():
    cfg = default_config()
    rng = np.random.default_rng(0)
    result = simulate(cfg, interventions=[], rng=rng, initial_infected=10)
    assert result.S_history[0] == cfg.N - 10
    assert result.I_history[0] == 10
    assert result.R_history[0] == 0
    assert result.V_history[0] == 0


def test_population_conservation():
    cfg = default_config()
    rng = np.random.default_rng(0)
    result = simulate(cfg, interventions=[], rng=rng, initial_infected=10)
    totals = (
        result.S_history
        + result.I_history
        + result.R_history
        + result.V_history
    )
    assert (totals == cfg.N).all()


def test_no_initial_infected_means_no_epidemic():
    cfg = default_config()
    rng = np.random.default_rng(0)
    result = simulate(cfg, interventions=[], rng=rng, initial_infected=0)
    assert (result.I_history == 0).all()


def test_welfare_ledger_populated():
    cfg = default_config()
    rng = np.random.default_rng(0)
    result = simulate(cfg, interventions=[], rng=rng, initial_infected=10)
    assert result.welfare.totals["attendance_utility"] > 0
    assert len(result.welfare.by_day) == cfg.T
