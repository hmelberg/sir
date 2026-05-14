"""Transmission engine: group-level hazard aggregation and per-agent draws.

Hazard for agent i in group g on day t:
    hazard_{i,g} = a_i,g * p_g(1-e_i) * I_g/N_g
where a_i,g is attendance at group g (theta * m_bar_g for voluntary,
m_bar_g otherwise, scaled by sick_attendance_multiplier if infected).

Total hazard for agent i is the sum over their groups; infection
probability is 1 - exp(-total_hazard).

Vaccinated agents have an additional factor (1 - v_eff).
"""

import numpy as np

from sir.config import ScenarioConfig
from sir.constants import DiseaseState, GroupType
from sir.world import World


def _voluntary_mask(cfg: ScenarioConfig) -> np.ndarray:
    """Return a bool array of length 7 (one per GroupType) marking voluntary groups."""
    mask = np.zeros(len(GroupType), dtype=bool)
    for gt in cfg.voluntary_groups:
        mask[gt] = True
    return mask


def step_transmission(
    world: World,
    cfg: ScenarioConfig,
    theta_by_age: np.ndarray,   # shape (7,)
    e_by_age: np.ndarray,       # shape (7,)
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a bool mask of length N: True where agent newly infected this step.

    Does NOT mutate world.state. Caller is responsible for applying transitions.
    """
    N = world.N
    G = world.G
    M = world.membership_agent_id.size

    # Precompute per-group-type arrays
    p_baseline_by_gt = np.array(
        [cfg.group_p_baseline[GroupType(gt)] for gt in range(len(GroupType))],
        dtype=np.float64,
    )
    m_bar_by_gt = np.array(
        [cfg.group_m_bar[GroupType(gt)] for gt in range(len(GroupType))],
        dtype=np.float64,
    )
    vol_mask_by_gt = _voluntary_mask(cfg)

    # Per-group p effective (after policy multiplier)
    p_eff_by_group = (
        p_baseline_by_gt[world.group_type] * world.group_p_mult * world.group_active
    )

    # Effective per-meeting transmission probability per membership row
    # depends on the agent's e (precaution). e is indexed by age bin.
    m_group_id = world.membership_group_id
    m_agent_id = world.membership_agent_id
    agent_age_bin = world.age_bin[m_agent_id]
    e_for_row = e_by_age[agent_age_bin]
    p_eff_for_row = p_eff_by_group[m_group_id] * (1.0 - e_for_row)

    # Attendance per membership row
    group_type_for_row = world.group_type[m_group_id]
    m_bar_for_row = m_bar_by_gt[group_type_for_row]
    attendance_mult_for_row = world.group_attendance_mult[m_group_id]

    is_voluntary_row = vol_mask_by_gt[group_type_for_row]
    theta_for_row = theta_by_age[agent_age_bin]
    voluntary_scaling = np.where(is_voluntary_row, theta_for_row, 1.0)

    # Sick attendance reduction
    is_sick = world.state[m_agent_id] == DiseaseState.I
    sick_scaling = np.where(is_sick, cfg.sick_attendance_multiplier, 1.0)

    attendance_for_row = (
        m_bar_for_row * attendance_mult_for_row * voluntary_scaling * sick_scaling
    )

    # Group-level prevalence: count of infected attendees / total attendees (weighted by attendance)
    # We approximate by using head-count prevalence per group, weighted by attendance.
    # Sum attended agents per group:
    attended_per_group = np.bincount(m_group_id, weights=attendance_for_row, minlength=G)
    # Sum attended infected per group (only I, not V):
    is_infected_row = (world.state[m_agent_id] == DiseaseState.I).astype(np.float64)
    infected_attended_per_group = np.bincount(
        m_group_id, weights=attendance_for_row * is_infected_row, minlength=G
    )
    # Compute prevalence only where attended > 0, leaving zero elsewhere.
    # Using np.divide with `where=` avoids the RuntimeWarning from dividing by zero.
    prevalence_per_group = np.divide(
        infected_attended_per_group,
        attended_per_group,
        out=np.zeros_like(attended_per_group, dtype=np.float64),
        where=attended_per_group > 0,
    )

    # Per-membership hazard contribution
    prevalence_for_row = prevalence_per_group[m_group_id]
    hazard_for_row = attendance_for_row * p_eff_for_row * prevalence_for_row

    # Sum per-agent
    hazard_per_agent = np.bincount(m_agent_id, weights=hazard_for_row, minlength=N)

    # Vaccinated agents get an additional (1 - v_eff) factor
    is_vaccinated = world.state == DiseaseState.V
    hazard_per_agent = np.where(
        is_vaccinated, hazard_per_agent * (1.0 - cfg.v_eff), hazard_per_agent
    )

    # Susceptible AND vaccinated agents can be infected; I and R cannot
    eligible = (world.state == DiseaseState.S) | (world.state == DiseaseState.V)
    p_infect = 1.0 - np.exp(-hazard_per_agent)
    draws = rng.random(N)
    new_infections = eligible & (draws < p_infect)
    return new_infections


def apply_new_infections(world: World, new_infections: np.ndarray) -> None:
    """Move newly-infected agents to I state. Mutates world."""
    from sir.constants import DiseaseState
    idx = np.where(new_infections)[0]
    world.state[idx] = DiseaseState.I
    world.days_infected[idx] = 1
