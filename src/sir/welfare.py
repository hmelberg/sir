"""Welfare accounting: per-day utility and decomposed ledger.

Components (signed sum gives total welfare):
  + attendance_utility       (sum alpha_g log(1 + m_g) over all agents, groups)
  - precaution_cost          (0.5 * kappa * e^2 per agent)
  - infection_cost           (V_a per new infection)
  - vaccination_cost         (c_vax per new vaccinee)
  - direct_policy_cost       (set by intervention layer)
  - spillover_policy_cost    (set by intervention layer)
"""

from dataclasses import dataclass, field

import numpy as np

from sir.config import ScenarioConfig
from sir.constants import DiseaseState, GroupType
from sir.world import World


@dataclass
class WelfareLedger:
    totals: dict[str, float] = field(default_factory=lambda: {
        "attendance_utility": 0.0,
        "precaution_cost": 0.0,
        "infection_cost": 0.0,
        "vaccination_cost": 0.0,
        "direct_policy_cost": 0.0,
        "spillover_policy_cost": 0.0,
    })
    by_day: list[dict[str, float]] = field(default_factory=list)

    def total_welfare(self) -> float:
        return (
            self.totals["attendance_utility"]
            - self.totals["precaution_cost"]
            - self.totals["infection_cost"]
            - self.totals["vaccination_cost"]
            - self.totals["direct_policy_cost"]
            - self.totals["spillover_policy_cost"]
        )


def step_welfare(
    world: World,
    cfg: ScenarioConfig,
    theta_by_age: np.ndarray,
    e_by_age: np.ndarray,
    new_infections: np.ndarray,
    new_vaccinations: np.ndarray,
    ledger: WelfareLedger,
    day: int,
) -> None:
    """Compute today's welfare components and accumulate into the ledger."""
    N = world.N
    m_agent_id = world.membership_agent_id
    m_group_id = world.membership_group_id

    # Per-row attendance utility
    group_type_for_row = world.group_type[m_group_id]
    m_bar_by_gt = np.array(
        [cfg.group_m_bar[GroupType(gt)] for gt in range(len(GroupType))],
        dtype=np.float64,
    )
    alpha_by_gt = np.array(
        [cfg.group_alpha[GroupType(gt)] for gt in range(len(GroupType))],
        dtype=np.float64,
    )
    is_vol_by_gt = np.zeros(len(GroupType), dtype=bool)
    for gt in cfg.voluntary_groups:
        is_vol_by_gt[gt] = True

    agent_age_bin = world.age_bin[m_agent_id]
    theta_for_row = theta_by_age[agent_age_bin]
    voluntary_scaling = np.where(is_vol_by_gt[group_type_for_row], theta_for_row, 1.0)
    is_sick = world.state[m_agent_id] == DiseaseState.I
    sick_scaling = np.where(is_sick, cfg.sick_attendance_multiplier, 1.0)
    m_for_row = (
        m_bar_by_gt[group_type_for_row]
        * world.group_attendance_mult[m_group_id]
        * world.group_active[m_group_id]
        * voluntary_scaling
        * sick_scaling
    )
    utility_for_row = alpha_by_gt[group_type_for_row] * np.log1p(m_for_row)
    attendance_utility = utility_for_row.sum()

    # Precaution cost
    e_for_agent = e_by_age[world.age_bin]
    precaution_cost = 0.5 * cfg.kappa * (e_for_agent ** 2).sum()

    # Infection cost (lump-sum on day of new infection)
    V_a_arr = np.array(cfg.V_a, dtype=np.float64)
    new_inf_age_bins = world.age_bin[new_infections]
    infection_cost = V_a_arr[new_inf_age_bins].sum() if new_inf_age_bins.size > 0 else 0.0

    # Vaccination cost
    vaccination_cost = cfg.c_vax * int(new_vaccinations.sum())

    ledger.totals["attendance_utility"] += attendance_utility
    ledger.totals["precaution_cost"] += precaution_cost
    ledger.totals["infection_cost"] += infection_cost
    ledger.totals["vaccination_cost"] += vaccination_cost

    ledger.by_day.append({
        "day": day,
        "attendance_utility": attendance_utility,
        "precaution_cost": precaution_cost,
        "infection_cost": infection_cost,
        "vaccination_cost": vaccination_cost,
    })
