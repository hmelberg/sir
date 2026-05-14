import pytest

from sir.healthcare import HealthcareConfig, default_healthcare_config


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
