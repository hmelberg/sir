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


def compute_morbidity_qaly(
    new_inf_by_age: np.ndarray,
    hosp_rate_by_age: tuple[float, ...],
    icu_rate_by_age: tuple[float, ...],
    cfg: CBAConfig,
    discount_rate: float,
) -> tuple[float, np.ndarray]:
    """Stream 6: Acute morbidity QALY loss."""
    T = new_inf_by_age.shape[0]
    hr = np.asarray(hosp_rate_by_age, dtype=np.float64)
    ir = np.asarray(icu_rate_by_age, dtype=np.float64)
    qaly_per_case = (
        cfg.disability_weight_symptomatic * cfg.sick_days_per_infection / 365.0
        + hr * cfg.disability_weight_hospitalized * cfg.sick_days_hospitalized / 365.0
        + ir * cfg.disability_weight_icu * cfg.sick_days_icu / 365.0
    )
    raw = new_inf_by_age @ qaly_per_case
    factors = discount_factors(discount_rate, T)
    discounted = raw * factors
    return float(discounted.sum()), discounted


def compute_behavioral_loss(
    theta_hist: np.ndarray,
    m_bar_total: float,
    cfg: CBAConfig,
    discount_rate: float,
) -> tuple[float, np.ndarray]:
    """Stream 7: Behavioral utility loss from reduced attendance.

    Value EXCLUDES wages (stream 3) — only intrinsic/social contact value.
    """
    T = theta_hist.size
    lost_meetings_per_day = (1.0 - theta_hist) * m_bar_total
    raw = lost_meetings_per_day * cfg.hours_per_meeting * cfg.value_of_time_per_hour
    factors = discount_factors(discount_rate, T)
    discounted = raw * factors
    return float(discounted.sum()), discounted


def compute_precaution_cost(
    e_hist: np.ndarray,
    N: int,
    cfg: CBAConfig,
    discount_rate: float,
) -> tuple[float, np.ndarray]:
    """Stream 8: Precaution effort cost, monetized via VOT."""
    T = e_hist.size
    raw = N * (e_hist ** 2) * cfg.value_of_time_per_hour * cfg.hours_per_meeting
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


import pandas as pd

from sir.simulation import SimResult
from sir.healthcare import HealthcareConfig, HealthcareOutcomes


def compute_cba(
    sim_result: SimResult,
    hc_outcomes: HealthcareOutcomes,
    hc_config: HealthcareConfig,
    cba_config: CBAConfig,
    policy_cost_per_day: float = 0.0,
) -> CBAReport:
    """Compute the 9-stream CBA report.

    Streams 7 (behavioral_loss) and 8 (precaution_cost) are set to 0
    in v1.2a because SimResult does not yet expose per-day average theta
    and e by population. Their formulas are implemented in
    compute_behavioral_loss / compute_precaution_cost; the orchestrator
    will wire them up in v1.2b once per-day averages are stored.

    policy_cost_per_day is the constant daily policy cost from active
    interventions, provided by the caller (run_comparison).
    """
    r = cba_config.discount_rate
    T = sim_result.new_infections_by_age.shape[0]
    new_inf = sim_result.new_infections_history.astype(np.float64)
    new_inf_by_age = sim_result.new_infections_by_age.astype(np.float64)

    # Stream 1: Direct medical
    s1, s1_pd = compute_direct_medical(
        hc_outcomes.hosp_prev, hc_outcomes.icu_prev, new_inf, cba_config, r,
    )

    # Stream 2: Vaccination program — diff V_history; allocate all to age bin 6
    # (eldest-first policy per v0.2). Clamp negative diffs (state can transition V→I).
    new_vax_per_day_total = np.diff(
        sim_result.V_history.astype(np.float64), prepend=sim_result.V_history[0]
    )
    new_vax_per_day_total = np.clip(new_vax_per_day_total, 0.0, None)
    new_vax_by_age = np.zeros((T, 7), dtype=np.float64)
    new_vax_by_age[:, 6] = new_vax_per_day_total
    s2, s2_pd = compute_vaccination_program(new_vax_by_age, cba_config, r)

    # Stream 3: Productivity loss from illness
    s3, s3_pd = compute_productivity_illness(
        new_inf_by_age,
        hc_config.hosp_rate_by_age, hc_config.icu_rate_by_age,
        cba_config, r,
    )

    # Convert cumulative deaths_by_age → daily incidence
    deaths_cum = hc_outcomes.deaths_by_age
    daily_deaths_by_age = np.zeros_like(deaths_cum)
    daily_deaths_by_age[1:] = deaths_cum[1:] - deaths_cum[:-1]
    daily_deaths_by_age[0] = deaths_cum[0]

    # Stream 4: Productivity loss from death
    s4, s4_pd = compute_productivity_death(daily_deaths_by_age, cba_config, r)

    # Stream 5: YLL
    s5, s5_pd = compute_yll(daily_deaths_by_age, hc_config.yll_per_death_by_age, r)

    # Stream 6: Acute morbidity QALY
    s6, s6_pd = compute_morbidity_qaly(
        new_inf_by_age,
        hc_config.hosp_rate_by_age, hc_config.icu_rate_by_age,
        cba_config, r,
    )

    # Streams 7 & 8: behavioral_loss + precaution_cost — see docstring.
    s7 = 0.0
    s7_pd = np.zeros(T, dtype=np.float64)
    s8 = 0.0
    s8_pd = np.zeros(T, dtype=np.float64)

    # Stream 9: Policy costs (caller supplies constant daily rate)
    raw_policy = np.full(T, policy_cost_per_day, dtype=np.float64)
    factors = discount_factors(r, T)
    s9_pd = raw_policy * factors
    s9 = float(s9_pd.sum())

    streams = {
        "direct_medical": s1,
        "vaccination_program": s2,
        "productivity_illness": s3,
        "productivity_death": s4,
        "yll": s5,
        "morbidity_qaly": s6,
        "behavioral_loss": s7,
        "precaution_cost": s8,
        "policy_costs": s9,
    }
    units = {
        "direct_medical": "$",
        "vaccination_program": "$",
        "productivity_illness": "$",
        "productivity_death": "$",
        "yll": "YLL",
        "morbidity_qaly": "QALY",
        "behavioral_loss": "$",
        "precaution_cost": "$",
        "policy_costs": "$",
    }
    per_day = {
        "direct_medical": s1_pd,
        "vaccination_program": s2_pd,
        "productivity_illness": s3_pd,
        "productivity_death": s4_pd,
        "yll": s5_pd,
        "morbidity_qaly": s6_pd,
        "behavioral_loss": s7_pd,
        "precaution_cost": s8_pd,
        "policy_costs": s9_pd,
    }

    monetary = [
        "direct_medical", "vaccination_program",
        "productivity_illness", "productivity_death",
        "behavioral_loss", "precaution_cost", "policy_costs",
    ]
    total_monetary_cost = sum(streams[k] for k in monetary)
    total_health_yll = streams["yll"]
    total_health_qaly = streams["morbidity_qaly"]

    total_cost_including_health: float | None = None
    if cba_config.monetize_health:
        health_in_money = (
            total_health_yll * cba_config.value_per_yll
            + total_health_qaly * cba_config.value_per_yll
        )
        total_cost_including_health = total_monetary_cost + health_in_money

    return CBAReport(
        streams=streams,
        units=units,
        total_monetary_cost=total_monetary_cost,
        total_health_burden_yll=total_health_yll,
        total_health_burden_qaly=total_health_qaly,
        total_cost_including_health=total_cost_including_health,
        per_day=per_day,
    )


def _cbareport_summary(self: CBAReport) -> "pd.DataFrame":
    rows = [
        {"stream": k, "value": v, "unit": self.units[k]}
        for k, v in self.streams.items()
    ]
    df = pd.DataFrame(rows).set_index("stream")
    return df


def _cbareport_cost_effectiveness_vs(
    self: CBAReport, baseline: CBAReport
) -> dict[str, float]:
    delta_cost = self.total_monetary_cost - baseline.total_monetary_cost
    delta_yll = self.total_health_burden_yll - baseline.total_health_burden_yll
    delta_qaly = self.total_health_burden_qaly - baseline.total_health_burden_qaly
    yll_averted = -delta_yll
    qaly_gained = -delta_qaly
    cost_per_yll = delta_cost / yll_averted if yll_averted != 0 else float("inf")
    cost_per_qaly = delta_cost / qaly_gained if qaly_gained != 0 else float("inf")
    return {
        "delta_monetary_cost": delta_cost,
        "yll_averted": yll_averted,
        "qaly_gained": qaly_gained,
        "cost_per_yll_averted": cost_per_yll,
        "cost_per_qaly_gained": cost_per_qaly,
    }


CBAReport.summary = _cbareport_summary
CBAReport.cost_effectiveness_vs = _cbareport_cost_effectiveness_vs
