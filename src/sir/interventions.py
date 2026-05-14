"""Intervention layer: group-modifying and agent-action policies.

Each Intervention has a time window and applies its effect when called
during that window. Group-modifying interventions adjust group properties
(active flag, attendance multiplier, transmission multiplier).
Agent-action interventions move agents between disease states or apply
direct utility costs (e.g., vaccination, mass testing).
"""

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from sir.config import ScenarioConfig
from sir.constants import DiseaseState, GroupType
from sir.world import World


@dataclass
class Intervention:
    name: str
    start_day: int
    end_day: int
    target_filter: Callable[[World], np.ndarray] | None = None
    apply_to_groups: Callable[[World, np.ndarray], None] | None = None
    agent_action: Callable[[World, ScenarioConfig, np.random.Generator], np.ndarray] | None = None
    direct_cost_per_day: float = 0.0
    spillover_cost_fn: Callable[[World, ScenarioConfig], float] | None = None

    def is_active(self, day: int) -> bool:
        return self.start_day <= day < self.end_day


# ---------- Standard group-modifying interventions ----------


def close_schools(start_day: int, end_day: int) -> Intervention:
    def target(world: World) -> np.ndarray:
        return world.group_type == GroupType.SCHOOL

    def apply(world: World, target_mask: np.ndarray) -> None:
        world.group_active[target_mask] = False

    def spillover(world: World, cfg: ScenarioConfig) -> float:
        # Cost = learning_loss_per_kid_per_day * number of kids whose schools are closed
        school_mask = world.group_type == GroupType.SCHOOL
        closed_schools = np.where(school_mask & ~world.group_active)[0]
        if closed_schools.size == 0:
            return 0.0
        in_closed = np.isin(world.membership_group_id, closed_schools)
        affected_agents = np.unique(world.membership_agent_id[in_closed])
        return cfg.learning_loss_per_kid_per_day * affected_agents.size

    return Intervention(
        name="close_schools",
        start_day=start_day,
        end_day=end_day,
        target_filter=target,
        apply_to_groups=apply,
        spillover_cost_fn=spillover,
    )


def wfh_mandate(start_day: int, end_day: int, attendance_factor: float) -> Intervention:
    def target(world: World) -> np.ndarray:
        return world.group_type == GroupType.WORKPLACE

    def apply(world: World, target_mask: np.ndarray) -> None:
        world.group_attendance_mult[target_mask] = attendance_factor

    return Intervention(
        name="wfh_mandate",
        start_day=start_day,
        end_day=end_day,
        target_filter=target,
        apply_to_groups=apply,
    )


def mask_mandate(
    start_day: int,
    end_day: int,
    target_types: set[GroupType],
    p_factor: float,
) -> Intervention:
    target_ints = np.array([int(gt) for gt in target_types])

    def target(world: World) -> np.ndarray:
        return np.isin(world.group_type, target_ints)

    def apply(world: World, target_mask: np.ndarray) -> None:
        world.group_p_mult[target_mask] = p_factor

    return Intervention(
        name="mask_mandate",
        start_day=start_day,
        end_day=end_day,
        target_filter=target,
        apply_to_groups=apply,
    )


def gathering_limit(start_day: int, end_day: int, max_size: int) -> Intervention:
    def target(world: World) -> np.ndarray:
        # Affects any group with size > max_size
        return world.group_size > max_size

    def apply(world: World, target_mask: np.ndarray) -> None:
        world.group_attendance_mult[target_mask] = 0.1

    return Intervention(
        name=f"gathering_limit_{max_size}",
        start_day=start_day,
        end_day=end_day,
        target_filter=target,
        apply_to_groups=apply,
    )


def event_ban(start_day: int, end_day: int) -> Intervention:
    def target(world: World) -> np.ndarray:
        return world.group_type == GroupType.ONE_OFF_EVENT

    def apply(world: World, target_mask: np.ndarray) -> None:
        world.group_active[target_mask] = False

    return Intervention(
        name="event_ban",
        start_day=start_day,
        end_day=end_day,
        target_filter=target,
        apply_to_groups=apply,
    )


# ---------- Agent-action: vaccination ----------


def vaccinate_eldest_first(
    start_day: int, end_day: int, doses_per_day: int
) -> Intervention:
    def action(world: World, cfg: ScenarioConfig, rng: np.random.Generator) -> np.ndarray:
        new_vax = np.zeros(world.N, dtype=bool)
        eligible = (world.state == DiseaseState.S) & ~world.vaccinated
        if not eligible.any():
            return new_vax
        eligible_idx = np.where(eligible)[0]
        # Sort by age descending
        order = np.argsort(-world.age[eligible_idx])
        chosen = eligible_idx[order[:doses_per_day]]
        world.state[chosen] = DiseaseState.V
        world.vaccinated[chosen] = True
        new_vax[chosen] = True
        return new_vax

    return Intervention(
        name="vaccinate_eldest_first",
        start_day=start_day,
        end_day=end_day,
        agent_action=action,
        direct_cost_per_day=doses_per_day * 0.001,
    )


# ---------- Reset and apply ----------


def _reset_group_overrides(world: World) -> None:
    world.group_active[:] = True
    world.group_attendance_mult[:] = 1.0
    world.group_p_mult[:] = 1.0


def apply_interventions(
    world: World,
    cfg: ScenarioConfig,
    day: int,
    interventions: Iterable[Intervention],
    rng: np.random.Generator,
) -> np.ndarray:
    """Reset group overrides, apply all active interventions in order.

    Returns a bool mask of new vaccinations this step.
    """
    _reset_group_overrides(world)
    new_vax_total = np.zeros(world.N, dtype=bool)
    for inter in interventions:
        if not inter.is_active(day):
            continue
        if inter.apply_to_groups is not None and inter.target_filter is not None:
            mask = inter.target_filter(world)
            inter.apply_to_groups(world, mask)
        if inter.agent_action is not None:
            new_vax_total |= inter.agent_action(world, cfg, rng)
    return new_vax_total
