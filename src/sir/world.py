"""World state container and builder.

The world holds all agent attributes (1-D arrays of length N), all group
properties (1-D arrays of length G), and the bipartite membership relation
as two parallel arrays (agent_id, group_id) of length M.
"""

from dataclasses import dataclass, field
from typing import Self

import numpy as np

from sir.config import ScenarioConfig
from sir.constants import AGE_BIN_EDGES, DiseaseState, GroupType


# Age distribution roughly matching developed-country demographics
AGE_BIN_WEIGHTS = np.array([0.11, 0.12, 0.13, 0.13, 0.13, 0.13, 0.25])


@dataclass
class World:
    # Agent attributes (length N)
    age: np.ndarray              # int — actual age in years
    age_bin: np.ndarray          # int — index into AGE_BIN_LABELS (0..6)
    state: np.ndarray            # int — DiseaseState value
    days_infected: np.ndarray    # int — days since infection (0 if not I)
    vaccinated: np.ndarray       # bool — has received vaccine

    # Group properties (length G)
    group_type: np.ndarray       # int — GroupType value
    group_size: np.ndarray       # int
    group_active: np.ndarray     # bool — open or closed by policy
    group_attendance_mult: np.ndarray  # float — intervention attendance scaling
    group_p_mult: np.ndarray     # float — intervention transmission scaling

    # Memberships (length M, parallel arrays)
    membership_agent_id: np.ndarray  # int
    membership_group_id: np.ndarray  # int

    @property
    def N(self) -> int:
        return self.age.size

    @property
    def G(self) -> int:
        return self.group_type.size


def _draw_ages(N: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw an age (in years) and an age_bin (0..6) for each of N agents."""
    bins = rng.choice(7, size=N, p=AGE_BIN_WEIGHTS)
    # Within bin, draw uniform on [edge_lo, edge_hi); for the open 60+ bin use [60, 90)
    edges = list(AGE_BIN_EDGES)
    edges[-1] = 90.0
    ages = np.empty(N, dtype=np.int32)
    for b in range(7):
        mask = bins == b
        lo, hi = edges[b], edges[b + 1]
        ages[mask] = rng.integers(int(lo), int(hi), size=mask.sum())
    return ages, bins.astype(np.int8)


def _build_households(
    age: np.ndarray, age_bin: np.ndarray, rng: np.random.Generator
) -> tuple[list[list[int]], np.ndarray]:
    """Greedy household assignment.

    Strategy: shuffle agents and pack into households of size drawn from a
    plausible distribution. Returns list of households (lists of agent ids)
    plus a per-agent household_id array.
    """
    N = age.size
    perm = rng.permutation(N)
    households: list[list[int]] = []
    household_id = np.full(N, -1, dtype=np.int32)
    size_dist = np.array([0.30, 0.30, 0.15, 0.15, 0.07, 0.03])  # sizes 1..6
    sizes = np.arange(1, 7)
    i = 0
    while i < N:
        s = int(rng.choice(sizes, p=size_dist))
        s = min(s, N - i)
        members = perm[i : i + s].tolist()
        for m in members:
            household_id[m] = len(households)
        households.append(members)
        i += s
    return households, household_id


def _build_schools(
    age_bin: np.ndarray, rng: np.random.Generator
) -> list[list[int]]:
    """Assign all kids (age_bin 0 and 1) to schools of size ~400."""
    kids = np.where(age_bin <= 1)[0]
    rng.shuffle(kids)
    schools: list[list[int]] = []
    target_size = 400
    for i in range(0, len(kids), target_size):
        schools.append(kids[i : i + target_size].tolist())
    return schools


def _build_workplaces(
    age_bin: np.ndarray, rng: np.random.Generator
) -> list[list[int]]:
    """Assign working-age agents (age_bin 2..5) to workplaces.

    Firm sizes drawn from a heavy-tailed distribution: many small firms,
    few large. ~80% employed.
    """
    workers = np.where((age_bin >= 2) & (age_bin <= 5))[0]
    rng.shuffle(workers)
    employed = workers[: int(0.8 * len(workers))]
    workplaces: list[list[int]] = []
    i = 0
    while i < len(employed):
        # Mix of small (10), medium (50), large (200) firms
        size = int(rng.choice([10, 50, 200], p=[0.6, 0.3, 0.1]))
        size = min(size, len(employed) - i)
        workplaces.append(employed[i : i + size].tolist())
        i += size
    return workplaces


def _build_leisure(
    age_bin: np.ndarray, rng: np.random.Generator
) -> list[list[int]]:
    """RecurringLeisure venues: clubs, gyms, etc., size 10-50."""
    # 50% of population participates in some recurring leisure
    N = age_bin.size
    participants = rng.choice(N, size=N // 2, replace=False)
    rng.shuffle(participants)
    venues: list[list[int]] = []
    i = 0
    while i < len(participants):
        size = int(rng.integers(10, 51))
        size = min(size, len(participants) - i)
        venues.append(participants[i : i + size].tolist())
        i += size
    return venues


def _build_kin_links(
    households: list[list[int]], rng: np.random.Generator
) -> list[list[int]]:
    """Linked-household kin groups. Each group is a small set of households
    pooled together; their members all interact at low frequency."""
    H = len(households)
    n_kin_groups = H // 3
    kin_groups: list[list[int]] = []
    for _ in range(n_kin_groups):
        n_linked = int(rng.integers(2, 4))  # 2 or 3 linked households
        chosen_hh = rng.choice(H, size=min(n_linked, H), replace=False)
        members: list[int] = []
        for h in chosen_hh:
            members.extend(households[h])
        kin_groups.append(members)
    return kin_groups


def _build_community(N: int, rng: np.random.Generator) -> list[list[int]]:
    """Single global community group with all agents as members."""
    return [list(range(N))]


def _assemble_memberships(
    *group_lists: list[list[int]],
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Flatten per-type group lists into:
      - group_type (length G)
      - group_size (length G)
      - membership_agent_id (length M)
      - membership_group_id (length M)
    """
    group_types: list[int] = []
    group_sizes: list[int] = []
    agent_ids: list[int] = []
    group_ids: list[int] = []
    next_group_id = 0
    for type_idx, groups in enumerate(group_lists):
        for g in groups:
            for a in g:
                agent_ids.append(a)
                group_ids.append(next_group_id)
            group_types.append(type_idx)
            group_sizes.append(len(g))
            next_group_id += 1
    return (
        np.array(group_types, dtype=np.int8),
        np.array(group_sizes, dtype=np.int32),
        np.array(agent_ids, dtype=np.int32),
        np.array(group_ids, dtype=np.int32),
    )


def build_world(cfg: ScenarioConfig, rng: np.random.Generator) -> World:
    age, age_bin = _draw_ages(cfg.N, rng)
    households, _ = _build_households(age, age_bin, rng)
    schools = _build_schools(age_bin, rng)
    workplaces = _build_workplaces(age_bin, rng)
    leisure = _build_leisure(age_bin, rng)
    kin = _build_kin_links(households, rng)
    community = _build_community(cfg.N, rng)
    one_off: list[list[int]] = []  # populated on demand by interventions/events

    # Order must match GroupType enum: HOUSEHOLD=0, KIN=1, SCHOOL=2,
    # WORKPLACE=3, RECURRING_LEISURE=4, ONE_OFF_EVENT=5, COMMUNITY=6
    group_type, group_size, m_agent, m_group = _assemble_memberships(
        households, kin, schools, workplaces, leisure, one_off, community
    )

    G = group_type.size
    return World(
        age=age,
        age_bin=age_bin,
        state=np.full(cfg.N, DiseaseState.S, dtype=np.int8),
        days_infected=np.zeros(cfg.N, dtype=np.int16),
        vaccinated=np.zeros(cfg.N, dtype=bool),
        group_type=group_type,
        group_size=group_size,
        group_active=np.ones(G, dtype=bool),
        group_attendance_mult=np.ones(G, dtype=np.float32),
        group_p_mult=np.ones(G, dtype=np.float32),
        membership_agent_id=m_agent,
        membership_group_id=m_group,
    )
