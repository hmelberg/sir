import numpy as np
import pytest

from sir.config import default_config
from sir.healthcare import (
    HealthcareConfig,
    HealthcareOutcomes,
    compute_healthcare_outcomes,
    default_healthcare_config,
)
from sir.simulation import simulate


def test_default_has_seven_rates_per_array():
    cfg = default_healthcare_config()
    assert len(cfg.hosp_rate_by_age) == 7
    assert len(cfg.icu_rate_by_age) == 7
    assert len(cfg.death_rate_by_age) == 7
    assert len(cfg.yll_per_death_by_age) == 7


def test_default_rates_increase_with_age():
    cfg = default_healthcare_config()
    # Older ages should have higher hospitalization and death rates
    assert cfg.hosp_rate_by_age[6] > cfg.hosp_rate_by_age[0]
    assert cfg.death_rate_by_age[6] > cfg.death_rate_by_age[0]


def test_default_yll_decreases_with_age():
    cfg = default_healthcare_config()
    # Younger deaths cost more life-years
    assert cfg.yll_per_death_by_age[0] > cfg.yll_per_death_by_age[6]


def test_default_delays_and_los_are_positive():
    cfg = default_healthcare_config()
    assert cfg.hosp_delay > 0
    assert cfg.icu_delay > 0
    assert cfg.death_delay > 0
    assert cfg.hosp_los > 0
    assert cfg.icu_los > 0


def test_rejects_wrong_length_arrays():
    with pytest.raises(ValueError):
        HealthcareConfig(
            hosp_rate_by_age=(0.01,) * 5,  # wrong length
            icu_rate_by_age=(0.001,) * 7,
            death_rate_by_age=(0.001,) * 7,
            yll_per_death_by_age=(10.0,) * 7,
        )


def _run_sim(initial_infected=10):
    cfg = default_config()
    rng = np.random.default_rng(42)
    return simulate(cfg, interventions=[], rng=rng, initial_infected=initial_infected), cfg


def test_outcomes_have_expected_shapes():
    result, cfg = _run_sim()
    hc = default_healthcare_config()
    out = compute_healthcare_outcomes(result, hc)
    assert isinstance(out, HealthcareOutcomes)
    T1 = cfg.T + 1
    assert out.hosp_admit.shape == (T1,)
    assert out.hosp_prev.shape == (T1,)
    assert out.icu_admit.shape == (T1,)
    assert out.icu_prev.shape == (T1,)
    assert out.daily_deaths.shape == (T1,)
    assert out.cum_deaths.shape == (T1,)
    assert out.deaths_by_age.shape == (T1, 7)
    assert out.total_deaths_by_age.shape == (7,)
    assert out.total_yll_by_age.shape == (7,)


def test_zero_rates_give_zero_outcomes():
    result, _ = _run_sim()
    hc = HealthcareConfig(
        hosp_rate_by_age=(0,) * 7,
        icu_rate_by_age=(0,) * 7,
        death_rate_by_age=(0,) * 7,
        yll_per_death_by_age=(10,) * 7,
    )
    out = compute_healthcare_outcomes(result, hc)
    assert out.hosp_admit.sum() == 0
    assert out.icu_admit.sum() == 0
    assert out.total_deaths == 0
    assert out.total_yll == 0


def test_zero_infections_give_zero_outcomes():
    cfg = default_config()
    rng = np.random.default_rng(0)
    result = simulate(cfg, interventions=[], rng=rng, initial_infected=0)
    hc = default_healthcare_config()
    out = compute_healthcare_outcomes(result, hc)
    assert out.total_deaths == 0
    assert out.hosp_admit.sum() == 0


def test_yll_equals_deaths_times_yll_per_age():
    result, _ = _run_sim()
    hc = default_healthcare_config()
    out = compute_healthcare_outcomes(result, hc)
    expected_yll = sum(
        out.total_deaths_by_age[a] * hc.yll_per_death_by_age[a]
        for a in range(7)
    )
    assert np.isclose(out.total_yll, expected_yll)


def test_total_deaths_sums_daily_deaths():
    result, _ = _run_sim()
    hc = default_healthcare_config()
    out = compute_healthcare_outcomes(result, hc)
    assert np.isclose(out.total_deaths, out.daily_deaths.sum())
    assert np.isclose(out.cum_deaths[-1], out.total_deaths)


def test_hospital_prevalence_is_sum_of_admissions_in_los_window():
    result, _ = _run_sim()
    hc = default_healthcare_config()
    out = compute_healthcare_outcomes(result, hc)
    # Check at t = hosp_delay + hosp_los - 1: prev should equal sum of admissions on [hosp_delay, t]
    t = hc.hosp_delay + hc.hosp_los - 1
    expected = out.hosp_admit[hc.hosp_delay:t + 1].sum()
    assert np.isclose(out.hosp_prev[t], expected)


def test_admissions_are_lagged_by_delay():
    # With a single seed infection, no infections happen for ~hosp_delay days
    cfg = default_config()
    rng = np.random.default_rng(123)
    result = simulate(cfg, interventions=[], rng=rng, initial_infected=1)
    hc = default_healthcare_config()
    out = compute_healthcare_outcomes(result, hc)
    # In the first few days, admissions should be tiny since infections haven't grown
    assert out.hosp_admit[0] == 0
