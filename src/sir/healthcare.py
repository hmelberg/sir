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
