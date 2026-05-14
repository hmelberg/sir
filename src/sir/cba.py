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
