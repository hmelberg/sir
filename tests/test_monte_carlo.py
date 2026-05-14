import numpy as np

from sir.config import default_config
from sir.monte_carlo import MCResult, run_mc


def test_mc_returns_n_runs():
    cfg = default_config()
    result = run_mc(cfg, interventions=[], n_runs=5, base_seed=0, initial_infected=10)
    assert isinstance(result, MCResult)
    assert result.S_history.shape == (5, cfg.T + 1)
    assert result.welfare_totals.shape == (5,)


def test_mc_runs_are_distinct():
    cfg = default_config()
    result = run_mc(cfg, interventions=[], n_runs=5, base_seed=0, initial_infected=10)
    # At least one run should differ from another (stochastic)
    assert not np.all(result.I_history[0] == result.I_history[1])


def test_mc_summary_statistics():
    cfg = default_config()
    result = run_mc(cfg, interventions=[], n_runs=5, base_seed=0, initial_infected=10)
    mean_S = result.S_history.mean(axis=0)
    assert mean_S.shape == (cfg.T + 1,)
