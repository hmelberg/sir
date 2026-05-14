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


import numpy as np

from sir.cba import discount_factors


def test_discount_factor_zero_rate_returns_ones():
    factors = discount_factors(annual_rate=0.0, T=10)
    np.testing.assert_array_equal(factors, np.ones(10))


def test_discount_factor_positive_rate_decreases_over_time():
    factors = discount_factors(annual_rate=0.03, T=365)
    assert factors[0] == 1.0
    assert factors[364] < 1.0
    # After 1 year at 3%, factor should be approximately 1/(1+0.03) ≈ 0.971
    assert 0.965 < factors[364] < 0.975


def test_discount_factor_compounds_daily():
    # After 730 days (2 years) at 3% annual rate
    factors = discount_factors(annual_rate=0.03, T=730)
    # (1+0.03)^-2 ≈ 0.9426
    assert 0.94 < factors[729] < 0.945


from sir.cba import CBAReport


def test_cbareport_constructible():
    rpt = CBAReport(
        streams={"direct_medical": 1000.0},
        units={"direct_medical": "$"},
        total_monetary_cost=1000.0,
        total_health_burden_yll=0.0,
        total_health_burden_qaly=0.0,
        total_cost_including_health=None,
        per_day={"direct_medical": np.zeros(10)},
    )
    assert rpt.streams["direct_medical"] == 1000.0
    assert rpt.total_monetary_cost == 1000.0


from sir.cba import compute_direct_medical


def test_direct_medical_zero_when_no_hospitalizations():
    hosp_prev = np.zeros(10)
    icu_prev = np.zeros(10)
    new_inf = np.zeros(10)
    cfg = default_cba_config()
    total, per_day = compute_direct_medical(hosp_prev, icu_prev, new_inf, cfg, discount_rate=0.0)
    assert total == 0.0
    np.testing.assert_array_equal(per_day, np.zeros(10))


def test_direct_medical_sums_streams():
    hosp_prev = np.array([10.0, 20.0])
    icu_prev = np.array([2.0, 4.0])
    new_inf = np.array([0.0, 100.0])
    cfg = default_cba_config()
    total, per_day = compute_direct_medical(hosp_prev, icu_prev, new_inf, cfg, discount_rate=0.0)
    # Day 0: 10*1500 + 2*4000 + 0*200 = 23000; Day 1: 20*1500 + 4*4000 + 100*200 = 66000
    assert np.isclose(per_day[0], 23000.0)
    assert np.isclose(per_day[1], 66000.0)
    assert np.isclose(total, 89000.0)


def test_direct_medical_discounts_future():
    hosp_prev = np.array([0.0, 100.0])
    icu_prev = np.zeros(2)
    new_inf = np.zeros(2)
    cfg = default_cba_config()
    total_disc, _ = compute_direct_medical(hosp_prev, icu_prev, new_inf, cfg, discount_rate=0.5)
    total_undisc, _ = compute_direct_medical(hosp_prev, icu_prev, new_inf, cfg, discount_rate=0.0)
    assert total_disc < total_undisc
