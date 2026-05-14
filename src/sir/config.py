"""Scenario configuration: all parameters in one immutable dataclass."""

from dataclasses import dataclass, field
from typing import Mapping

from sir.constants import DEFAULT_V_A, GroupType


@dataclass(frozen=True)
class ScenarioConfig:
    N: int
    T: int
    gamma: float
    kappa: float
    v_eff: float
    c_vax: float
    sick_attendance_multiplier: float
    V_a: tuple[float, ...]
    group_alpha: Mapping[GroupType, float]
    group_p_baseline: Mapping[GroupType, float]
    group_m_bar: Mapping[GroupType, float]
    voluntary_groups: frozenset[GroupType]
    learning_loss_per_kid_per_day: float = 50.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        if self.N <= 0:
            raise ValueError(f"N must be positive, got {self.N}")
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}")
        if not 0 <= self.v_eff <= 1:
            raise ValueError(f"v_eff must be in [0,1], got {self.v_eff}")
        if len(self.V_a) != 7:
            raise ValueError(f"V_a must have 7 entries, got {len(self.V_a)}")


def default_config() -> ScenarioConfig:
    return ScenarioConfig(
        N=10_000,
        T=365,
        gamma=1 / 7,
        kappa=1.0,
        v_eff=0.8,
        c_vax=0.01,
        sick_attendance_multiplier=0.3,
        V_a=DEFAULT_V_A,
        group_alpha={
            GroupType.HOUSEHOLD: 2.0,
            GroupType.KIN: 0.5,
            GroupType.SCHOOL: 1.5,
            GroupType.WORKPLACE: 2.0,
            GroupType.RECURRING_LEISURE: 0.8,
            GroupType.ONE_OFF_EVENT: 0.2,
            GroupType.COMMUNITY: 1.0,
        },
        group_p_baseline={
            GroupType.HOUSEHOLD: 0.05,
            GroupType.KIN: 0.02,
            GroupType.SCHOOL: 0.01,
            GroupType.WORKPLACE: 0.008,
            GroupType.RECURRING_LEISURE: 0.01,
            GroupType.ONE_OFF_EVENT: 0.02,
            GroupType.COMMUNITY: 0.001,
        },
        group_m_bar={
            GroupType.HOUSEHOLD: 1.0,
            GroupType.KIN: 0.1,
            GroupType.SCHOOL: 1.0,
            GroupType.WORKPLACE: 1.0,
            GroupType.RECURRING_LEISURE: 0.3,
            GroupType.ONE_OFF_EVENT: 0.0,
            GroupType.COMMUNITY: 1.0,
        },
        voluntary_groups=frozenset({
            GroupType.KIN,
            GroupType.RECURRING_LEISURE,
            GroupType.ONE_OFF_EVENT,
            GroupType.COMMUNITY,
        }),
    )
