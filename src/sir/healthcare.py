"""Healthcare outcomes: post-processing of SimResult into hospital, ICU, death, and YLL series.

This module computes derived series — not part of the disease dynamics. The model
does not transition agents through hospital or ICU states; those are post-hoc
estimates from the new-infections-by-age time series.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HealthcareConfig:
    hosp_rate_by_age: tuple[float, ...]
    icu_rate_by_age: tuple[float, ...]
    death_rate_by_age: tuple[float, ...]
    yll_per_death_by_age: tuple[float, ...]
    hosp_delay: int = 8
    icu_delay: int = 11
    death_delay: int = 18
    hosp_los: int = 7
    icu_los: int = 10

    def __post_init__(self) -> None:
        for name, arr in [
            ("hosp_rate_by_age", self.hosp_rate_by_age),
            ("icu_rate_by_age", self.icu_rate_by_age),
            ("death_rate_by_age", self.death_rate_by_age),
            ("yll_per_death_by_age", self.yll_per_death_by_age),
        ]:
            if len(arr) != 7:
                raise ValueError(f"{name} must have 7 entries, got {len(arr)}")


def default_healthcare_config() -> HealthcareConfig:
    """Plausible defaults for a moderately severe respiratory pathogen."""
    return HealthcareConfig(
        # 0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60+
        hosp_rate_by_age=(0.005, 0.005, 0.01, 0.02, 0.03, 0.06, 0.20),
        icu_rate_by_age=(0.001, 0.001, 0.002, 0.004, 0.008, 0.020, 0.060),
        death_rate_by_age=(0.0001, 0.0001, 0.0005, 0.001, 0.003, 0.01, 0.05),
        yll_per_death_by_age=(75.0, 65.0, 55.0, 45.0, 35.0, 25.0, 8.0),
    )


from sir.simulation import SimResult


@dataclass
class HealthcareOutcomes:
    hosp_admit: np.ndarray             # (T+1,) total admissions per day
    hosp_prev: np.ndarray              # (T+1,) currently in hospital
    icu_admit: np.ndarray
    icu_prev: np.ndarray
    daily_deaths: np.ndarray
    cum_deaths: np.ndarray
    deaths_by_age: np.ndarray          # (T+1, 7) cumulative per age bin
    total_deaths: float
    total_deaths_by_age: np.ndarray    # (7,)
    total_yll: float
    total_yll_by_age: np.ndarray       # (7,)


def _lagged_rate(
    new_inf_by_age: np.ndarray, rate_by_age: np.ndarray, delay: int
) -> np.ndarray:
    """Return per-day admissions: sum_a rate_a * new_inf_by_age[t - delay, a]."""
    T1 = new_inf_by_age.shape[0]
    out = np.zeros(T1, dtype=np.float64)
    if delay >= T1:
        return out
    # For t >= delay: out[t] = (new_inf_by_age[t - delay] * rate_by_age).sum()
    contributions = (new_inf_by_age * rate_by_age).sum(axis=1)
    out[delay:] = contributions[: T1 - delay]
    return out


def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """Sum of the last `window` entries at each index (inclusive)."""
    if window <= 1:
        return arr.copy()
    cs = np.concatenate(([0.0], np.cumsum(arr)))
    T1 = arr.size
    out = np.empty(T1, dtype=np.float64)
    for t in range(T1):
        lo = max(0, t - window + 1)
        out[t] = cs[t + 1] - cs[lo]
    return out


def compute_healthcare_outcomes(
    result: SimResult, hc_cfg: HealthcareConfig
) -> HealthcareOutcomes:
    """Compute hospital/ICU/death series from SimResult.new_infections_by_age."""
    new_inf = result.new_infections_by_age.astype(np.float64)  # (T+1, 7)
    T1 = new_inf.shape[0]

    hosp_rate = np.asarray(hc_cfg.hosp_rate_by_age, dtype=np.float64)
    icu_rate = np.asarray(hc_cfg.icu_rate_by_age, dtype=np.float64)
    death_rate = np.asarray(hc_cfg.death_rate_by_age, dtype=np.float64)
    yll_per_death = np.asarray(hc_cfg.yll_per_death_by_age, dtype=np.float64)

    hosp_admit = _lagged_rate(new_inf, hosp_rate, int(hc_cfg.hosp_delay))
    icu_admit = _lagged_rate(new_inf, icu_rate, int(hc_cfg.icu_delay))
    daily_deaths = _lagged_rate(new_inf, death_rate, int(hc_cfg.death_delay))

    hosp_prev = _rolling_sum(hosp_admit, int(hc_cfg.hosp_los))
    icu_prev = _rolling_sum(icu_admit, int(hc_cfg.icu_los))
    cum_deaths = np.cumsum(daily_deaths)

    # Per-age cumulative deaths
    deaths_by_age = np.zeros((T1, 7), dtype=np.float64)
    dd = int(hc_cfg.death_delay)
    if dd < T1:
        per_day_age = new_inf * death_rate  # (T1, 7)
        # Cumulative per age, then shifted by death_delay
        cum_per_age = np.cumsum(per_day_age, axis=0)
        deaths_by_age[dd:] = cum_per_age[: T1 - dd]

    total_deaths_by_age = deaths_by_age[-1]
    total_deaths = float(total_deaths_by_age.sum())
    total_yll_by_age = total_deaths_by_age * yll_per_death
    total_yll = float(total_yll_by_age.sum())

    return HealthcareOutcomes(
        hosp_admit=hosp_admit,
        hosp_prev=hosp_prev,
        icu_admit=icu_admit,
        icu_prev=icu_prev,
        daily_deaths=daily_deaths,
        cum_deaths=cum_deaths,
        deaths_by_age=deaths_by_age,
        total_deaths=total_deaths,
        total_deaths_by_age=total_deaths_by_age,
        total_yll=total_yll,
        total_yll_by_age=total_yll_by_age,
    )
