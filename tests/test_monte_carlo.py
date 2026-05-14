import numpy as np

from sir.config import default_config
from sir.healthcare import default_healthcare_config
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


def test_mc_returns_healthcare_when_requested():
    cfg = default_config()
    hc = default_healthcare_config()
    result = run_mc(
        cfg, interventions=[], n_runs=3, base_seed=0,
        initial_infected=10, healthcare=hc, parallel=False,
    )
    assert result.healthcare_outcomes is not None
    assert len(result.healthcare_outcomes) == 3
    assert result.healthcare_outcomes[0].total_deaths >= 0


def test_mc_healthcare_default_none():
    cfg = default_config()
    result = run_mc(cfg, interventions=[], n_runs=2, base_seed=0, parallel=False)
    assert result.healthcare_outcomes is None


def test_mc_new_infections_by_age_shape():
    cfg = default_config()
    cfg = type(cfg)(**{**cfg.__dict__, "N": 500, "T": 30})
    result = run_mc(cfg, interventions=[], n_runs=3, base_seed=0, parallel=False)
    assert result.new_infections_by_age.shape == (3, cfg.T + 1, 7)


def test_mc_new_infections_by_age_sums_to_total():
    cfg = default_config()
    cfg = type(cfg)(**{**cfg.__dict__, "N": 500, "T": 30})
    result = run_mc(cfg, interventions=[], n_runs=2, base_seed=0, parallel=False)
    per_run_total = result.new_infections_by_age.sum(axis=2)
    np.testing.assert_array_equal(per_run_total, result.new_infections_history)
