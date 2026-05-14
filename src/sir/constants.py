"""Project-wide constants: age bins, V_a defaults, enums."""

from enum import IntEnum

AGE_BIN_EDGES: tuple[float, ...] = (0, 10, 20, 30, 40, 50, 60, float("inf"))
AGE_BIN_LABELS: tuple[str, ...] = ("0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60+")

DEFAULT_V_A: tuple[float, ...] = (0.5, 0.5, 1.0, 2.0, 3.0, 5.0, 30.0)


class GroupType(IntEnum):
    HOUSEHOLD = 0
    KIN = 1
    SCHOOL = 2
    WORKPLACE = 3
    RECURRING_LEISURE = 4
    ONE_OFF_EVENT = 5
    COMMUNITY = 6


class DiseaseState(IntEnum):
    S = 0
    V = 1
    I = 2
    R = 3
