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


def contact_reduction(
    people: float,
    events_per_week: float,
    encounters_per_event: float,
    window: tuple[int, int],
    target: GroupType | None = None,
    spillover_cost_per_person_per_day: float = 0.0,
) -> Intervention:
    """X*Y*Z formulation of a contact-reduction intervention.

    Removes (people * events_per_week * encounters_per_event) / 7 person-meetings
    per day from the population pool. If `target` is None, applies the reduction
    as a uniform multiplier across all groups. If `target` is a GroupType, only
    that group type is affected.
    """
    meetings_per_day = people * events_per_week * encounters_per_event / 7.0

    if target is None:
        target_ints = np.array([int(gt) for gt in GroupType])
    else:
        target_ints = np.array([int(target)])

    def target_filter(world):
        return np.isin(world.group_type, target_ints)

    def apply(world, mask):
        if not mask.any() or meetings_per_day == 0.0:
            return
        target_groups = np.where(mask)[0]
        in_target = np.isin(world.membership_group_id, target_groups)
        baseline_target_meetings = float(in_target.sum())
        if baseline_target_meetings == 0.0:
            return
        factor = max(0.0, 1.0 - meetings_per_day / baseline_target_meetings)
        world.group_attendance_mult[mask] *= factor

    def spillover(world, cfg):
        return spillover_cost_per_person_per_day * people

    return Intervention(
        name=f"contact_reduction_{int(people)}x{events_per_week}x{encounters_per_event}",
        start_day=window[0],
        end_day=window[1],
        target_filter=target_filter,
        apply_to_groups=apply,
        spillover_cost_fn=spillover if spillover_cost_per_person_per_day > 0 else None,
    )
