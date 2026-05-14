"""Cost-benefit analysis: two-layer welfare framework.

The behavioral layer (FOC) uses V_a and c_vax as bundled taste parameters
that drive agent choice. This module — the social layer — computes a
9-stream cost-benefit report by reading SimResult + HealthcareOutcomes
post-hoc. Streams are independent and additive: V_a is NOT in any CBA
stream; the report computes mortality cost directly from deaths × YLL
or VSL. See docs/superpowers/specs/2026-05-15-v1.2-cba-framework-design.md
for the design rationale.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CBAConfig:
    # Discount rate (annual) applied to time-distributed costs
    discount_rate: float

    # Direct medical costs (per day or per case)
    hospital_cost_per_day: float
    icu_cost_per_day: float
    outpatient_cost_per_case: float

    # Vaccination program costs (per dose)
    vax_dose_cost: float
    vax_admin_cost: float
    vax_sideeffect_value: float

    # Productivity from illness
    wage_per_day_by_age: tuple[float, ...]
    sick_days_per_infection: float
    sick_days_hospitalized: float
    sick_days_icu: float

    # Mortality monetization
    value_per_yll: float
    remaining_working_years_by_age: tuple[float, ...]

    # Acute morbidity (QALY)
    disability_weight_symptomatic: float
    disability_weight_hospitalized: float
    disability_weight_icu: float

    # Behavioral monetization
    value_of_time_per_hour: float
    hours_per_meeting: float

    # Whether to monetize health outcomes
    monetize_health: bool

    def __post_init__(self) -> None:
        if self.discount_rate < 0:
            raise ValueError(f"discount_rate must be >= 0, got {self.discount_rate}")
        for name, arr in [
            ("wage_per_day_by_age", self.wage_per_day_by_age),
            ("remaining_working_years_by_age", self.remaining_working_years_by_age),
        ]:
            if len(arr) != 7:
                raise ValueError(f"{name} must have 7 entries, got {len(arr)}")


def default_cba_config() -> CBAConfig:
    """Illustrative defaults loosely based on COVID-era US/EU literature.

    These are NOT calibrated to any specific pathogen or country. Users
    running real policy analysis should override every value.
    """
    return CBAConfig(
        discount_rate=0.03,
        hospital_cost_per_day=1500.0,
        icu_cost_per_day=4000.0,
        outpatient_cost_per_case=200.0,
        vax_dose_cost=25.0,
        vax_admin_cost=30.0,
        vax_sideeffect_value=50.0,
        # 0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60+
        wage_per_day_by_age=(0.0, 0.0, 200.0, 250.0, 280.0, 260.0, 50.0),
        sick_days_per_infection=7.0,
        sick_days_hospitalized=14.0,
        sick_days_icu=21.0,
        value_per_yll=100_000.0,
        remaining_working_years_by_age=(45.0, 40.0, 35.0, 25.0, 15.0, 5.0, 0.0),
        disability_weight_symptomatic=0.15,
        disability_weight_hospitalized=0.50,
        disability_weight_icu=0.80,
        value_of_time_per_hour=25.0,
        hours_per_meeting=1.0,
        monetize_health=False,
    )


import numpy as np


def discount_factors(annual_rate: float, T: int) -> np.ndarray:
    """Return per-day discount factors for T days, given an annual rate.

    Factor at day t is (1 + annual_rate)^(-t/365). Day 0 has factor 1.0.
    Used to discount time-distributed costs and benefits.
    """
    if T <= 0:
        return np.zeros(0, dtype=np.float64)
    days = np.arange(T, dtype=np.float64)
    return np.power(1.0 + annual_rate, -days / 365.0)


def compute_direct_medical(
    hosp_prev: np.ndarray,
    icu_prev: np.ndarray,
    new_inf: np.ndarray,
    cfg: CBAConfig,
    discount_rate: float,
) -> tuple[float, np.ndarray]:
    """Stream 1: Direct medical costs (hospital + ICU + outpatient)."""
    T = hosp_prev.size
    raw = (
        hosp_prev * cfg.hospital_cost_per_day
        + icu_prev * cfg.icu_cost_per_day
        + new_inf * cfg.outpatient_cost_per_case
    )
    factors = discount_factors(discount_rate, T)
    discounted = raw * factors
    return float(discounted.sum()), discounted


def compute_vaccination_program(
    new_vax_by_age: np.ndarray,
    cfg: CBAConfig,
    discount_rate: float,
) -> tuple[float, np.ndarray]:
    """Stream 2: Vaccination program costs (dose + admin + side-effect WTP)."""
    T = new_vax_by_age.shape[0]
    new_vax_per_day = new_vax_by_age.sum(axis=1)
    per_dose_cost = cfg.vax_dose_cost + cfg.vax_admin_cost + cfg.vax_sideeffect_value
    raw = new_vax_per_day * per_dose_cost
    factors = discount_factors(discount_rate, T)
    discounted = raw * factors
    return float(discounted.sum()), discounted


def compute_productivity_illness(
    new_inf_by_age: np.ndarray,
    hosp_rate_by_age: tuple[float, ...],
    icu_rate_by_age: tuple[float, ...],
    cfg: CBAConfig,
    discount_rate: float,
) -> tuple[float, np.ndarray]:
    """Stream 3: Lost productivity from non-fatal illness."""
    T = new_inf_by_age.shape[0]
    wage = np.asarray(cfg.wage_per_day_by_age, dtype=np.float64)
    hr = np.asarray(hosp_rate_by_age, dtype=np.float64)
    ir = np.asarray(icu_rate_by_age, dtype=np.float64)
    days_per_case = (
        cfg.sick_days_per_infection
        + hr * cfg.sick_days_hospitalized
        + ir * cfg.sick_days_icu
    )
    loss_per_case = days_per_case * wage
    raw = new_inf_by_age @ loss_per_case
    factors = discount_factors(discount_rate, T)
    discounted = raw * factors
    return float(discounted.sum()), discounted


WORKING_DAYS_PER_YEAR = 260


def compute_productivity_death(
    daily_deaths_by_age: np.ndarray,
    cfg: CBAConfig,
    discount_rate: float,
) -> tuple[float, np.ndarray]:
    """Stream 4: Lost productivity from death."""
    T = daily_deaths_by_age.shape[0]
    wage = np.asarray(cfg.wage_per_day_by_age, dtype=np.float64)
    years = np.asarray(cfg.remaining_working_years_by_age, dtype=np.float64)
    loss_per_death = years * wage * WORKING_DAYS_PER_YEAR
    raw = daily_deaths_by_age @ loss_per_death
    factors = discount_factors(discount_rate, T)
    discounted = raw * factors
    return float(discounted.sum()), discounted


def compute_yll(
    daily_deaths_by_age: np.ndarray,
    yll_per_death_by_age: tuple[float, ...],
    discount_rate: float,
) -> tuple[float, np.ndarray]:
    """Stream 5: Years of life lost from mortality, PV-discounted within YLL."""
    T = daily_deaths_by_age.shape[0]
    yll = np.asarray(yll_per_death_by_age, dtype=np.float64)
    if discount_rate > 0:
        pv_yll = (1.0 - np.exp(-discount_rate * yll)) / discount_rate
    else:
        pv_yll = yll
    raw = daily_deaths_by_age @ pv_yll
    factors = discount_factors(discount_rate, T)
    discounted = raw * factors
    return float(discounted.sum()), discounted


@dataclass
class CBAReport:
    streams: dict[str, float]
    units: dict[str, str]
    total_monetary_cost: float
    total_health_burden_yll: float
    total_health_burden_qaly: float
    total_cost_including_health: float | None
    per_day: dict[str, np.ndarray]
