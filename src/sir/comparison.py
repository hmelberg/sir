"""High-level comparison wrapper: baseline (no interventions) vs treatment.

Wraps run_mc to produce a paired with-vs-without comparison. Supports
intervention-parameter uncertainty via PSA over distributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from sir.config import ScenarioConfig
from sir.healthcare import HealthcareConfig, HealthcareOutcomes
from sir.interventions import Intervention
from sir.monte_carlo import MCResult


@dataclass
class ComparisonResult:
    cfg: ScenarioConfig | None
    interventions_resolved: list[list[Intervention]]
    baseline_results: list[MCResult]
    treatment_results: list[MCResult]
    baseline_healthcare: list[list[HealthcareOutcomes]] | None
    treatment_healthcare: list[list[HealthcareOutcomes]] | None
    intervention_samples: list[dict[str, float]] | None
