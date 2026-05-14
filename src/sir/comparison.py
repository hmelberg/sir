"""High-level comparison wrapper: baseline (no interventions) vs treatment.

Wraps run_mc to produce a paired with-vs-without comparison. Supports
intervention-parameter uncertainty via PSA over distributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from sir.config import ScenarioConfig
from sir.constants import GroupType
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


def transmission_reduction(
    p_factor: float,
    window: tuple[int, int],
    targets: set[GroupType] | None = None,
    direct_cost_per_day: float = 0.0,
) -> Intervention:
    """Reduce per-meeting transmission by factor `p_factor` during the window.

    If `targets` is None, applies to all group types. Otherwise, only the
    specified types (e.g., {SCHOOL, WORKPLACE, COMMUNITY} for masks indoors).
    """
    if targets is None:
        target_ints = np.array([int(gt) for gt in GroupType])
    else:
        target_ints = np.array([int(gt) for gt in targets])

    def target_filter(world):
        return np.isin(world.group_type, target_ints)

    def apply(world, mask):
        world.group_p_mult[mask] = p_factor

    return Intervention(
        name=f"transmission_reduction_{p_factor:.2f}",
        start_day=window[0],
        end_day=window[1],
        target_filter=target_filter,
        apply_to_groups=apply,
        direct_cost_per_day=direct_cost_per_day,
    )
