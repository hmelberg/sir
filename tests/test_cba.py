import pytest

from sir.cba import CBAConfig, default_cba_config


def test_default_cba_config_constructs():
    cfg = default_cba_config()
    assert cfg.discount_rate >= 0
    assert cfg.hospital_cost_per_day > 0
    assert cfg.icu_cost_per_day > cfg.hospital_cost_per_day


def test_default_cba_has_seven_wage_entries():
    cfg = default_cba_config()
    assert len(cfg.wage_per_day_by_age) == 7


def test_default_cba_has_seven_working_years_entries():
    cfg = default_cba_config()
    assert len(cfg.remaining_working_years_by_age) == 7


def test_cbaconfig_rejects_negative_discount_rate():
    with pytest.raises(ValueError):
        CBAConfig(
            discount_rate=-0.01,
            hospital_cost_per_day=1500.0,
            icu_cost_per_day=4000.0,
            outpatient_cost_per_case=200.0,
            vax_dose_cost=25.0,
            vax_admin_cost=30.0,
            vax_sideeffect_value=50.0,
            wage_per_day_by_age=(0,)*7,
            sick_days_per_infection=7.0,
            sick_days_hospitalized=14.0,
            sick_days_icu=21.0,
            value_per_yll=100_000.0,
            remaining_working_years_by_age=(45,)*7,
            disability_weight_symptomatic=0.15,
            disability_weight_hospitalized=0.5,
            disability_weight_icu=0.8,
            value_of_time_per_hour=25.0,
            hours_per_meeting=1.0,
            monetize_health=False,
        )


def test_cbaconfig_rejects_wrong_length_wage_tuple():
    with pytest.raises(ValueError):
        CBAConfig(
            discount_rate=0.03,
            hospital_cost_per_day=1500.0,
            icu_cost_per_day=4000.0,
            outpatient_cost_per_case=200.0,
            vax_dose_cost=25.0,
            vax_admin_cost=30.0,
            vax_sideeffect_value=50.0,
            wage_per_day_by_age=(0,)*5,  # wrong length
            sick_days_per_infection=7.0,
            sick_days_hospitalized=14.0,
            sick_days_icu=21.0,
            value_per_yll=100_000.0,
            remaining_working_years_by_age=(45,)*7,
            disability_weight_symptomatic=0.15,
            disability_weight_hospitalized=0.5,
            disability_weight_icu=0.8,
            value_of_time_per_hour=25.0,
            hours_per_meeting=1.0,
            monetize_health=False,
        )
