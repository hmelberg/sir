# Microfounded SIR ABM — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a microfounded agent-based SIRV epidemic model with rational behavioral response, bipartite group structure, and welfare-decomposed cost-benefit reporting, per the design at `docs/superpowers/specs/2026-05-14-sir-microfoundations-abm-design.md`.

**Architecture:** Numpy-first ABM. Agents and groups are 1-D structured arrays; memberships are a sparse CSR `(group × agent)` matrix. Daily step does (1) per-type FOC solve, (2) per-group hazard aggregation via `bincount`, (3) per-agent stochastic transitions. Interventions modify group properties or move agents between disease states. Welfare is a decomposed per-day accumulator.

**Tech Stack:** Python 3.11+, numpy, scipy (sparse + brentq), pandas, matplotlib, pytest.

---

## File Structure

```
sir/
├── pyproject.toml                              # packaging
├── .gitignore
├── docs/
│   └── superpowers/
│       ├── specs/2026-05-14-sir-microfoundations-abm-design.md
│       └── plans/2026-05-14-sir-implementation-plan.md
├── src/sir/
│   ├── __init__.py
│   ├── constants.py        # AGE_BINS, GroupType enum, DiseaseState enum
│   ├── config.py           # ScenarioConfig + factory for defaults
│   ├── world.py            # World dataclass + builder
│   ├── disease.py          # SIRV state transitions
│   ├── transmission.py     # hazard + infection draws
│   ├── foc.py              # per-type (θ*, e*) solver
│   ├── welfare.py          # per-day utility + decomposed welfare ledger
│   ├── interventions.py    # Intervention dataclass + standard library + vax
│   ├── simulation.py       # daily loop
│   ├── monte_carlo.py      # parallel MC runs
│   ├── psa.py              # PSA: TOML spec, sampling, runner
│   └── plots.py            # standard plots
├── configs/
│   └── psa_example.toml    # example PSA spec
└── tests/
    ├── __init__.py
    ├── test_config.py
    ├── test_world.py
    ├── test_disease.py
    ├── test_transmission.py
    ├── test_foc.py
    ├── test_welfare.py
    ├── test_interventions.py
    ├── test_simulation.py
    ├── test_monte_carlo.py
    └── test_psa.py
```

Each `src/sir/<module>.py` has one clear responsibility. Tests sit alongside in `tests/test_<module>.py`. Imports flow only "upward" (e.g., `simulation.py` may import `transmission`, `disease`, `foc`, `welfare`, `interventions`; none of those import `simulation`).

---

## Task 1: Project skeleton

**Files:**
- Create: `/Users/hom/Documents/GitHub/sir/pyproject.toml`
- Create: `/Users/hom/Documents/GitHub/sir/.gitignore`
- Create: `/Users/hom/Documents/GitHub/sir/src/sir/__init__.py`
- Create: `/Users/hom/Documents/GitHub/sir/tests/__init__.py`

- [ ] **Step 1: Initialize git and verify directory layout**

```bash
cd /Users/hom/Documents/GitHub/sir
git init
mkdir -p src/sir tests
```

Expected: `.git/` exists; `src/sir/` and `tests/` exist.

- [ ] **Step 2: Write `pyproject.toml`**

Path: `/Users/hom/Documents/GitHub/sir/pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "sir"
version = "0.1.0"
description = "Microfounded ABM for epidemic cost-benefit analysis"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "pandas>=2.1",
    "matplotlib>=3.8",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-xdist>=3.5",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

- [ ] **Step 3: Write `.gitignore`**

Path: `/Users/hom/Documents/GitHub/sir/.gitignore`

```
__pycache__/
*.py[cod]
*$py.class
*.so
.pytest_cache/
.coverage
htmlcov/
.eggs/
*.egg-info/
build/
dist/
.venv/
venv/
.DS_Store
*.ipynb_checkpoints
```

- [ ] **Step 4: Create empty package init files**

Path: `/Users/hom/Documents/GitHub/sir/src/sir/__init__.py`

```python
"""Microfounded ABM for epidemic cost-benefit analysis."""

__version__ = "0.1.0"
```

Path: `/Users/hom/Documents/GitHub/sir/tests/__init__.py`

```python
```

- [ ] **Step 5: Install in editable mode and verify pytest runs**

```bash
cd /Users/hom/Documents/GitHub/sir
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Expected: `pytest` exits 5 ("no tests collected") cleanly.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore src/sir/__init__.py tests/__init__.py
git commit -m "chore: initialize project skeleton"
```

---

## Task 2: Core constants — age bins, group types, disease states

**Files:**
- Create: `src/sir/constants.py`
- Test: `tests/test_constants.py`

- [ ] **Step 1: Write the failing test**

Path: `/Users/hom/Documents/GitHub/sir/tests/test_constants.py`

```python
from sir.constants import (
    AGE_BIN_EDGES,
    AGE_BIN_LABELS,
    DEFAULT_V_A,
    GroupType,
    DiseaseState,
)


def test_age_bins_have_seven_decadal_groups():
    assert len(AGE_BIN_LABELS) == 7
    assert AGE_BIN_LABELS == ("0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60+")
    # edges: 0, 10, 20, 30, 40, 50, 60, +inf
    assert AGE_BIN_EDGES[0] == 0
    assert AGE_BIN_EDGES[-1] == float("inf")
    assert len(AGE_BIN_EDGES) == 8


def test_default_v_a_has_one_value_per_age_bin():
    assert len(DEFAULT_V_A) == 7
    assert DEFAULT_V_A == (0.5, 0.5, 1.0, 2.0, 3.0, 5.0, 30.0)


def test_group_type_enum_has_seven_members():
    assert {gt.name for gt in GroupType} == {
        "HOUSEHOLD", "KIN", "SCHOOL", "WORKPLACE",
        "RECURRING_LEISURE", "ONE_OFF_EVENT", "COMMUNITY",
    }


def test_disease_state_enum_has_sirv():
    assert {ds.name for ds in DiseaseState} == {"S", "V", "I", "R"}
    assert DiseaseState.S.value == 0
    assert DiseaseState.V.value == 1
    assert DiseaseState.I.value == 2
    assert DiseaseState.R.value == 3
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_constants.py -v
```

Expected: ImportError — `sir.constants` does not exist.

- [ ] **Step 3: Implement the constants**

Path: `/Users/hom/Documents/GitHub/sir/src/sir/constants.py`

```python
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
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_constants.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/sir/constants.py tests/test_constants.py
git commit -m "feat: add constants for age bins, group types, disease states"
```

---

## Task 3: ScenarioConfig dataclass

**Files:**
- Create: `src/sir/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Path: `/Users/hom/Documents/GitHub/sir/tests/test_config.py`

```python
import pytest

from sir.config import ScenarioConfig, default_config
from sir.constants import GroupType


def test_default_config_has_expected_population():
    cfg = default_config()
    assert cfg.N == 10_000
    assert cfg.T == 365


def test_default_config_has_all_seven_group_types():
    cfg = default_config()
    assert set(cfg.group_p_baseline.keys()) == set(GroupType)
    assert set(cfg.group_alpha.keys()) == set(GroupType)
    assert set(cfg.group_m_bar.keys()) == set(GroupType)


def test_default_config_voluntary_groups_listed():
    cfg = default_config()
    assert GroupType.RECURRING_LEISURE in cfg.voluntary_groups
    assert GroupType.COMMUNITY in cfg.voluntary_groups
    assert GroupType.KIN in cfg.voluntary_groups
    assert GroupType.HOUSEHOLD not in cfg.voluntary_groups
    assert GroupType.WORKPLACE not in cfg.voluntary_groups
    assert GroupType.SCHOOL not in cfg.voluntary_groups


def test_default_config_has_seven_v_a_values():
    cfg = default_config()
    assert len(cfg.V_a) == 7


def test_default_config_vaccine_efficacy_is_scalar():
    cfg = default_config()
    assert 0 <= cfg.v_eff <= 1


def test_config_rejects_negative_gamma():
    with pytest.raises(ValueError):
        ScenarioConfig(
            N=100, T=10, gamma=-0.1, kappa=1.0, v_eff=0.8, c_vax=0.01,
            sick_attendance_multiplier=0.3, V_a=(1.0,)*7,
            group_alpha={gt: 1.0 for gt in GroupType},
            group_p_baseline={gt: 0.01 for gt in GroupType},
            group_m_bar={gt: 1.0 for gt in GroupType},
            voluntary_groups=frozenset({GroupType.COMMUNITY}),
        )
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_config.py -v
```

Expected: ImportError on `sir.config`.

- [ ] **Step 3: Implement ScenarioConfig**

Path: `/Users/hom/Documents/GitHub/sir/src/sir/config.py`

```python
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
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_config.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/sir/config.py tests/test_config.py
git commit -m "feat: add ScenarioConfig with v1 defaults"
```

---

## Task 4: World dataclass and demographic generation

**Files:**
- Create: `src/sir/world.py`
- Test: `tests/test_world.py`

- [ ] **Step 1: Write the failing test**

Path: `/Users/hom/Documents/GitHub/sir/tests/test_world.py`

```python
import numpy as np

from sir.config import default_config
from sir.constants import DiseaseState, GroupType
from sir.world import World, build_world


def test_world_has_n_agents():
    cfg = default_config()
    rng = np.random.default_rng(cfg.seed)
    world = build_world(cfg, rng)
    assert world.age.shape == (cfg.N,)
    assert world.age_bin.shape == (cfg.N,)
    assert world.state.shape == (cfg.N,)


def test_age_bins_in_valid_range():
    cfg = default_config()
    rng = np.random.default_rng(cfg.seed)
    world = build_world(cfg, rng)
    assert world.age_bin.min() >= 0
    assert world.age_bin.max() <= 6


def test_initial_state_is_all_susceptible():
    cfg = default_config()
    rng = np.random.default_rng(cfg.seed)
    world = build_world(cfg, rng)
    assert (world.state == DiseaseState.S).all()


def test_every_agent_has_a_household():
    cfg = default_config()
    rng = np.random.default_rng(cfg.seed)
    world = build_world(cfg, rng)
    # Each agent must appear in exactly one HOUSEHOLD group membership
    hh_mask = world.group_type[world.membership_group_id] == GroupType.HOUSEHOLD
    hh_agent_ids = world.membership_agent_id[hh_mask]
    counts = np.bincount(hh_agent_ids, minlength=cfg.N)
    assert (counts == 1).all()


def test_population_age_distribution_reasonable():
    cfg = default_config()
    rng = np.random.default_rng(cfg.seed)
    world = build_world(cfg, rng)
    # 60+ should be 15-25% of population for realistic demographics
    elderly_share = (world.age_bin == 6).mean()
    assert 0.10 < elderly_share < 0.30


def test_kids_assigned_to_school_workers_to_workplace():
    cfg = default_config()
    rng = np.random.default_rng(cfg.seed)
    world = build_world(cfg, rng)
    # Kids (age_bin 0, 1) attend school
    school_mask = world.group_type[world.membership_group_id] == GroupType.SCHOOL
    school_agent_ids = world.membership_agent_id[school_mask]
    school_ages = world.age_bin[school_agent_ids]
    # At least 80% of school attendees should be age_bin 0 or 1
    assert (school_ages <= 1).mean() > 0.8
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_world.py -v
```

Expected: ImportError on `sir.world`.

- [ ] **Step 3: Implement world generation**

Path: `/Users/hom/Documents/GitHub/sir/src/sir/world.py`

```python
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
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_world.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/sir/world.py tests/test_world.py
git commit -m "feat: add World dataclass and builder with seven group types"
```

---

## Task 5: Disease state transitions (recovery)

**Files:**
- Create: `src/sir/disease.py`
- Test: `tests/test_disease.py`

- [ ] **Step 1: Write the failing test**

Path: `/Users/hom/Documents/GitHub/sir/tests/test_disease.py`

```python
import numpy as np

from sir.config import default_config
from sir.constants import DiseaseState
from sir.disease import step_recovery
from sir.world import build_world


def test_recovery_moves_infected_to_recovered():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    # Infect 100 agents
    world.state[:100] = DiseaseState.I
    world.days_infected[:100] = 1
    # With gamma=1 (force recovery), all should recover
    step_recovery(world, gamma=1.0, rng=rng)
    assert (world.state[:100] == DiseaseState.R).all()


def test_recovery_zero_when_gamma_zero():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    world.state[:100] = DiseaseState.I
    world.days_infected[:100] = 1
    step_recovery(world, gamma=0.0, rng=rng)
    assert (world.state[:100] == DiseaseState.I).all()


def test_recovery_does_not_touch_susceptible_or_vaccinated():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    world.state[:50] = DiseaseState.S
    world.state[50:100] = DiseaseState.V
    world.state[100:200] = DiseaseState.I
    world.days_infected[100:200] = 1
    step_recovery(world, gamma=1.0, rng=rng)
    assert (world.state[:50] == DiseaseState.S).all()
    assert (world.state[50:100] == DiseaseState.V).all()
    assert (world.state[100:200] == DiseaseState.R).all()


def test_days_infected_increments():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    world.state[:100] = DiseaseState.I
    world.days_infected[:100] = 1
    step_recovery(world, gamma=0.0, rng=rng)
    assert (world.days_infected[:100] == 2).all()


def test_recovery_rate_approximately_gamma():
    cfg = default_config()
    rng = np.random.default_rng(42)
    world = build_world(cfg, rng)
    n_infected = 5000
    world.state[:n_infected] = DiseaseState.I
    world.days_infected[:n_infected] = 1
    step_recovery(world, gamma=0.2, rng=rng)
    recovered = (world.state[:n_infected] == DiseaseState.R).sum()
    # ~20% should recover; allow generous tolerance for 5000 draws
    assert 800 < recovered < 1200
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_disease.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement disease step**

Path: `/Users/hom/Documents/GitHub/sir/src/sir/disease.py`

```python
"""SIRV disease state transitions.

This module is responsible for I -> R recovery only. S -> I and V -> I
transitions are owned by `transmission.py`. S -> V transitions are owned
by `interventions.py` (vaccination is a policy action, not a natural event).
"""

import numpy as np

from sir.constants import DiseaseState
from sir.world import World


def step_recovery(world: World, gamma: float, rng: np.random.Generator) -> None:
    """Move each infected agent to R with probability gamma. Mutates world."""
    infected_mask = world.state == DiseaseState.I
    n_infected = int(infected_mask.sum())
    if n_infected == 0:
        return
    draws = rng.random(n_infected)
    will_recover = draws < gamma
    infected_idx = np.where(infected_mask)[0]
    recovering_idx = infected_idx[will_recover]
    world.state[recovering_idx] = DiseaseState.R
    world.days_infected[recovering_idx] = 0
    # Increment days_infected for remaining infected
    still_infected_idx = infected_idx[~will_recover]
    world.days_infected[still_infected_idx] += 1
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_disease.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/sir/disease.py tests/test_disease.py
git commit -m "feat: add SIRV recovery step"
```

---

## Task 6: Transmission engine — group-level hazard

**Files:**
- Create: `src/sir/transmission.py`
- Test: `tests/test_transmission.py`

- [ ] **Step 1: Write the failing test**

Path: `/Users/hom/Documents/GitHub/sir/tests/test_transmission.py`

```python
import numpy as np

from sir.config import default_config
from sir.constants import DiseaseState, GroupType
from sir.transmission import step_transmission
from sir.world import build_world


def test_no_infected_means_no_new_infections():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    # All susceptible by default
    theta = np.ones(7, dtype=np.float64)
    e = np.zeros(7, dtype=np.float64)
    new_infections = step_transmission(world, cfg, theta, e, rng)
    assert new_infections.sum() == 0


def test_zero_transmission_probability_means_no_new_infections():
    cfg_zero = default_config()
    cfg = type(cfg_zero)(
        **{**cfg_zero.__dict__,
           "group_p_baseline": {gt: 0.0 for gt in GroupType}}
    )
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    # Infect 10% of agents
    n = world.N
    world.state[: n // 10] = DiseaseState.I
    theta = np.ones(7, dtype=np.float64)
    e = np.zeros(7, dtype=np.float64)
    new_infections = step_transmission(world, cfg, theta, e, rng)
    assert new_infections.sum() == 0


def test_transmission_produces_some_infections_when_p_high_and_prevalence_high():
    cfg_orig = default_config()
    cfg = type(cfg_orig)(
        **{**cfg_orig.__dict__,
           "group_p_baseline": {gt: 0.5 for gt in GroupType}}
    )
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    # Infect half the population
    n = world.N
    world.state[: n // 2] = DiseaseState.I
    theta = np.ones(7, dtype=np.float64)
    e = np.zeros(7, dtype=np.float64)
    new_infections = step_transmission(world, cfg, theta, e, rng)
    assert new_infections.sum() > 0


def test_full_precaution_blocks_transmission():
    cfg_orig = default_config()
    cfg = type(cfg_orig)(
        **{**cfg_orig.__dict__,
           "group_p_baseline": {gt: 0.5 for gt in GroupType}}
    )
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    n = world.N
    world.state[: n // 2] = DiseaseState.I
    theta = np.ones(7, dtype=np.float64)
    e = np.ones(7, dtype=np.float64)  # full precaution
    new_infections = step_transmission(world, cfg, theta, e, rng)
    assert new_infections.sum() == 0


def test_vaccinated_agents_have_reduced_infection_rate():
    cfg_orig = default_config()
    cfg = type(cfg_orig)(
        **{**cfg_orig.__dict__,
           "group_p_baseline": {gt: 0.3 for gt in GroupType},
           "v_eff": 1.0}  # perfect vaccine
    )
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    n = world.N
    world.state[: n // 4] = DiseaseState.I
    world.state[n // 4 : n // 2] = DiseaseState.V
    theta = np.ones(7, dtype=np.float64)
    e = np.zeros(7, dtype=np.float64)
    new_infections = step_transmission(world, cfg, theta, e, rng)
    # No vaccinated agent should be newly infected
    vax_idx = np.where(world.state == DiseaseState.V)[0]
    assert not new_infections[vax_idx].any()
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_transmission.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement transmission**

Path: `/Users/hom/Documents/GitHub/sir/src/sir/transmission.py`

```python
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
    # Avoid divide-by-zero
    prevalence_per_group = np.where(
        attended_per_group > 0,
        infected_attended_per_group / attended_per_group,
        0.0,
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
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_transmission.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/sir/transmission.py tests/test_transmission.py
git commit -m "feat: add bipartite transmission engine with vaccinated and sick handling"
```

---

## Task 7: FOC solver — per-type (θ*, e*)

**Files:**
- Create: `src/sir/foc.py`
- Test: `tests/test_foc.py`

- [ ] **Step 1: Write the failing test**

Path: `/Users/hom/Documents/GitHub/sir/tests/test_foc.py`

```python
import numpy as np

from sir.config import default_config
from sir.constants import GroupType
from sir.foc import solve_foc_by_age


def test_zero_prevalence_gives_theta_one_e_zero():
    cfg = default_config()
    prevalence_by_group_type = {gt: 0.0 for gt in GroupType}
    theta, e = solve_foc_by_age(cfg, prevalence_by_group_type)
    assert np.allclose(theta, 1.0, atol=1e-6)
    assert np.allclose(e, 0.0, atol=1e-6)


def test_zero_V_gives_theta_one_e_zero():
    cfg_orig = default_config()
    cfg = type(cfg_orig)(**{**cfg_orig.__dict__, "V_a": (0.0,) * 7})
    prevalence_by_group_type = {gt: 0.5 for gt in GroupType}
    theta, e = solve_foc_by_age(cfg, prevalence_by_group_type)
    assert np.allclose(theta, 1.0, atol=1e-6)
    assert np.allclose(e, 0.0, atol=1e-6)


def test_high_V_drives_theta_down_and_e_up():
    cfg_orig = default_config()
    cfg = type(cfg_orig)(**{**cfg_orig.__dict__, "V_a": (1000.0,) * 7})
    prevalence_by_group_type = {gt: 0.1 for gt in GroupType}
    theta, e = solve_foc_by_age(cfg, prevalence_by_group_type)
    assert (theta < 0.5).all()
    assert (e > 0.5).all()


def test_e_capped_at_one():
    cfg_orig = default_config()
    cfg = type(cfg_orig)(**{**cfg_orig.__dict__, "V_a": (1e10,) * 7})
    prevalence_by_group_type = {gt: 0.5 for gt in GroupType}
    theta, e = solve_foc_by_age(cfg, prevalence_by_group_type)
    assert (e <= 1.0).all()
    assert (e >= 0.0).all()


def test_older_agents_react_more_at_same_prevalence():
    cfg = default_config()
    # Higher V for older => stronger response
    prevalence_by_group_type = {gt: 0.05 for gt in GroupType}
    theta, e = solve_foc_by_age(cfg, prevalence_by_group_type)
    # Age bin 6 (60+) has V_a = 30; age bin 0 (0-9) has V_a = 0.5
    assert theta[6] < theta[0]
    assert e[6] > e[0]
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_foc.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement FOC solver**

Path: `/Users/hom/Documents/GitHub/sir/src/sir/foc.py`

```python
"""First-order-condition solver for rational behavioral response.

For each age type a, choose (theta, e) in [0,1] x [0,1] to maximize:
  U(theta, e) = sum_g alpha_g log(1 + m_g(theta))
              - 0.5 * kappa * e^2
              - V_a * (1-e) * sum_g p_g * m_g(theta) * I_g/N_g

Voluntary groups: m_g(theta) = theta * m_bar_g
Institutional groups: m_g = m_bar_g (independent of theta)

This decomposes into:
  A = sum_{vol} p_g * m_bar_g * (I_g/N_g)
  B = sum_{inst} p_g * m_bar_g * (I_g/N_g)
  C = sum_{vol} alpha_g log(1 + theta * m_bar_g)  (function of theta)

FOC for e:   e* = V_a * (1-0) * (theta*A + B) / kappa  ... wait
                  d/de: -kappa*e + V_a * (theta*A + B) = 0
                  e* = V_a * (theta*A + B) / kappa
FOC for theta: sum_{vol} alpha_g * m_bar_g / (1 + theta * m_bar_g)
              = V_a * (1-e*) * A

We solve the coupled system by 1-D search over theta with e* substituted.
"""

import numpy as np
from scipy.optimize import brentq

from sir.config import ScenarioConfig
from sir.constants import GroupType


def _compute_AB(
    cfg: ScenarioConfig, prevalence_by_group_type: dict[GroupType, float]
) -> tuple[float, float]:
    """Return (A, B): exposure intensity from voluntary and institutional groups."""
    A = 0.0
    B = 0.0
    for gt in GroupType:
        p = cfg.group_p_baseline[gt]
        m_bar = cfg.group_m_bar[gt]
        prev = prevalence_by_group_type[gt]
        contribution = p * m_bar * prev
        if gt in cfg.voluntary_groups:
            A += contribution
        else:
            B += contribution
    return A, B


def _solve_one_type(cfg: ScenarioConfig, V_a: float, A: float, B: float) -> tuple[float, float]:
    """Solve (theta*, e*) for one age type. Returns clamped values in [0,1]."""
    if V_a == 0.0 or A == 0.0:
        # No risk from voluntary groups => no reason to reduce theta below 1
        # but may still want precaution if B > 0
        if V_a == 0.0:
            return 1.0, 0.0
        # V > 0 but A = 0: theta is unconstrained (no voluntary risk reduction available)
        # e is determined by B only
        e_star = min(1.0, V_a * B / cfg.kappa)
        return 1.0, e_star

    # alpha_g, m_bar_g for voluntary groups
    vol_alpha_m: list[tuple[float, float]] = []
    for gt in cfg.voluntary_groups:
        vol_alpha_m.append((cfg.group_alpha[gt], cfg.group_m_bar[gt]))

    def theta_foc(theta: float) -> float:
        """LHS - RHS of theta FOC. Want zero."""
        e_star = V_a * (theta * A + B) / cfg.kappa
        e_star = min(max(e_star, 0.0), 1.0)
        lhs = sum(
            alpha * m_bar / (1.0 + theta * m_bar)
            for alpha, m_bar in vol_alpha_m
        )
        rhs = V_a * (1.0 - e_star) * A
        return lhs - rhs

    # Check corners
    f_at_1 = theta_foc(1.0)
    f_at_0 = theta_foc(1e-6)
    if f_at_1 >= 0:
        # Even at theta=1, marginal benefit > marginal cost; corner solution
        theta_star = 1.0
    elif f_at_0 <= 0:
        # Even at theta near 0, marginal cost > benefit
        theta_star = 1e-6
    else:
        theta_star = brentq(theta_foc, 1e-6, 1.0, xtol=1e-6)

    e_star = V_a * (theta_star * A + B) / cfg.kappa
    e_star = min(max(e_star, 0.0), 1.0)
    return theta_star, e_star


def solve_foc_by_age(
    cfg: ScenarioConfig, prevalence_by_group_type: dict[GroupType, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (theta_by_age, e_by_age), each of shape (7,)."""
    A, B = _compute_AB(cfg, prevalence_by_group_type)
    theta = np.empty(7, dtype=np.float64)
    e = np.empty(7, dtype=np.float64)
    for a in range(7):
        V_a = cfg.V_a[a]
        theta[a], e[a] = _solve_one_type(cfg, V_a, A, B)
    return theta, e
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_foc.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/sir/foc.py tests/test_foc.py
git commit -m "feat: add FOC solver for rational (theta, e) by age"
```

---

## Task 8: Welfare ledger — per-day utility and decomposition

**Files:**
- Create: `src/sir/welfare.py`
- Test: `tests/test_welfare.py`

- [ ] **Step 1: Write the failing test**

Path: `/Users/hom/Documents/GitHub/sir/tests/test_welfare.py`

```python
import numpy as np

from sir.config import default_config
from sir.constants import DiseaseState, GroupType
from sir.welfare import WelfareLedger, step_welfare
from sir.world import build_world


def test_zero_infections_zero_precaution_gives_positive_utility():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    ledger = WelfareLedger()
    theta = np.ones(7)
    e = np.zeros(7)
    new_infections = np.zeros(world.N, dtype=bool)
    new_vaccinations = np.zeros(world.N, dtype=bool)
    step_welfare(world, cfg, theta, e, new_infections, new_vaccinations, ledger, day=0)
    # No infection cost, no precaution cost; attendance utility positive
    assert ledger.totals["attendance_utility"] > 0
    assert ledger.totals["precaution_cost"] == 0
    assert ledger.totals["infection_cost"] == 0


def test_full_precaution_costs_kappa_over_two_times_N():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    ledger = WelfareLedger()
    theta = np.ones(7)
    e = np.ones(7)  # full precaution
    new_infections = np.zeros(world.N, dtype=bool)
    new_vaccinations = np.zeros(world.N, dtype=bool)
    step_welfare(world, cfg, theta, e, new_infections, new_vaccinations, ledger, day=0)
    # Each agent pays 0.5 * kappa * 1^2 = 0.5
    assert np.isclose(ledger.totals["precaution_cost"], 0.5 * cfg.kappa * cfg.N)


def test_new_infections_charge_V_a():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    ledger = WelfareLedger()
    theta = np.ones(7)
    e = np.zeros(7)
    new_infections = np.zeros(world.N, dtype=bool)
    new_vaccinations = np.zeros(world.N, dtype=bool)
    # Infect the first elderly agent
    elderly_idx = np.where(world.age_bin == 6)[0][0]
    new_infections[elderly_idx] = True
    step_welfare(world, cfg, theta, e, new_infections, new_vaccinations, ledger, day=0)
    assert np.isclose(ledger.totals["infection_cost"], cfg.V_a[6])


def test_new_vaccinations_charge_c_vax():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    ledger = WelfareLedger()
    theta = np.ones(7)
    e = np.zeros(7)
    new_infections = np.zeros(world.N, dtype=bool)
    new_vaccinations = np.zeros(world.N, dtype=bool)
    new_vaccinations[:100] = True
    step_welfare(world, cfg, theta, e, new_infections, new_vaccinations, ledger, day=0)
    assert np.isclose(ledger.totals["vaccination_cost"], 100 * cfg.c_vax)


def test_total_welfare_is_signed_sum_of_components():
    ledger = WelfareLedger()
    ledger.totals["attendance_utility"] = 100.0
    ledger.totals["precaution_cost"] = 10.0
    ledger.totals["infection_cost"] = 30.0
    ledger.totals["vaccination_cost"] = 2.0
    ledger.totals["direct_policy_cost"] = 5.0
    ledger.totals["spillover_policy_cost"] = 3.0
    assert np.isclose(ledger.total_welfare(), 100.0 - 10.0 - 30.0 - 2.0 - 5.0 - 3.0)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_welfare.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement welfare ledger**

Path: `/Users/hom/Documents/GitHub/sir/src/sir/welfare.py`

```python
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
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_welfare.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/sir/welfare.py tests/test_welfare.py
git commit -m "feat: add decomposed welfare ledger"
```

---

## Task 9: Intervention layer

**Files:**
- Create: `src/sir/interventions.py`
- Test: `tests/test_interventions.py`

- [ ] **Step 1: Write the failing test**

Path: `/Users/hom/Documents/GitHub/sir/tests/test_interventions.py`

```python
import numpy as np

from sir.config import default_config
from sir.constants import DiseaseState, GroupType
from sir.interventions import (
    Intervention,
    apply_interventions,
    close_schools,
    mask_mandate,
    vaccinate_eldest_first,
    wfh_mandate,
)
from sir.world import build_world


def test_close_schools_deactivates_school_groups_only():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    inter = close_schools(start_day=10, end_day=20)
    apply_interventions(world, cfg, day=15, interventions=[inter], rng=rng)
    school_mask = world.group_type == GroupType.SCHOOL
    other_mask = ~school_mask
    assert not world.group_active[school_mask].any()
    assert world.group_active[other_mask].all()


def test_school_closure_not_active_outside_window():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    inter = close_schools(start_day=10, end_day=20)
    apply_interventions(world, cfg, day=5, interventions=[inter], rng=rng)
    assert world.group_active.all()


def test_wfh_mandate_reduces_workplace_attendance():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    inter = wfh_mandate(start_day=0, end_day=10, attendance_factor=0.3)
    apply_interventions(world, cfg, day=5, interventions=[inter], rng=rng)
    workplace_mask = world.group_type == GroupType.WORKPLACE
    other_mask = ~workplace_mask
    assert np.allclose(world.group_attendance_mult[workplace_mask], 0.3)
    assert np.allclose(world.group_attendance_mult[other_mask], 1.0)


def test_mask_mandate_reduces_transmission_in_targeted_groups():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    inter = mask_mandate(
        start_day=0, end_day=10,
        target_types={GroupType.SCHOOL, GroupType.WORKPLACE, GroupType.COMMUNITY},
        p_factor=0.5,
    )
    apply_interventions(world, cfg, day=5, interventions=[inter], rng=rng)
    target_mask = np.isin(
        world.group_type,
        [GroupType.SCHOOL, GroupType.WORKPLACE, GroupType.COMMUNITY],
    )
    assert np.allclose(world.group_p_mult[target_mask], 0.5)
    assert np.allclose(world.group_p_mult[~target_mask], 1.0)


def test_vaccination_moves_eldest_first():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    inter = vaccinate_eldest_first(start_day=0, end_day=5, doses_per_day=50)
    new_vax = apply_interventions(world, cfg, day=0, interventions=[inter], rng=rng)
    vax_mask = world.state == DiseaseState.V
    assert vax_mask.sum() == 50
    # All vaccinated should be in age_bin 6 (or as old as possible)
    vax_ages = world.age_bin[vax_mask]
    # Should prefer oldest available
    assert vax_ages.min() == vax_ages.max() or vax_ages.min() == 6


def test_vaccination_skips_already_vaccinated():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    inter = vaccinate_eldest_first(start_day=0, end_day=5, doses_per_day=50)
    apply_interventions(world, cfg, day=0, interventions=[inter], rng=rng)
    apply_interventions(world, cfg, day=1, interventions=[inter], rng=rng)
    assert (world.state == DiseaseState.V).sum() == 100  # 50 + 50, no repeats
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_interventions.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement intervention layer**

Path: `/Users/hom/Documents/GitHub/sir/src/sir/interventions.py`

```python
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
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_interventions.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/sir/interventions.py tests/test_interventions.py
git commit -m "feat: add intervention layer with standard library and vaccination"
```

---

## Task 10: Daily simulation loop

**Files:**
- Create: `src/sir/simulation.py`
- Test: `tests/test_simulation.py`

- [ ] **Step 1: Write the failing test**

Path: `/Users/hom/Documents/GitHub/sir/tests/test_simulation.py`

```python
import numpy as np

from sir.config import default_config
from sir.constants import DiseaseState
from sir.simulation import SimResult, simulate


def test_simulation_runs_and_returns_result():
    cfg = default_config()
    rng = np.random.default_rng(0)
    result = simulate(cfg, interventions=[], rng=rng, initial_infected=10)
    assert isinstance(result, SimResult)
    assert result.S_history.shape == (cfg.T + 1,)
    assert result.I_history.shape == (cfg.T + 1,)
    assert result.R_history.shape == (cfg.T + 1,)
    assert result.V_history.shape == (cfg.T + 1,)


def test_initial_state_correct():
    cfg = default_config()
    rng = np.random.default_rng(0)
    result = simulate(cfg, interventions=[], rng=rng, initial_infected=10)
    assert result.S_history[0] == cfg.N - 10
    assert result.I_history[0] == 10
    assert result.R_history[0] == 0
    assert result.V_history[0] == 0


def test_population_conservation():
    cfg = default_config()
    rng = np.random.default_rng(0)
    result = simulate(cfg, interventions=[], rng=rng, initial_infected=10)
    totals = (
        result.S_history
        + result.I_history
        + result.R_history
        + result.V_history
    )
    assert (totals == cfg.N).all()


def test_no_initial_infected_means_no_epidemic():
    cfg = default_config()
    rng = np.random.default_rng(0)
    result = simulate(cfg, interventions=[], rng=rng, initial_infected=0)
    assert (result.I_history == 0).all()


def test_welfare_ledger_populated():
    cfg = default_config()
    rng = np.random.default_rng(0)
    result = simulate(cfg, interventions=[], rng=rng, initial_infected=10)
    assert result.welfare.totals["attendance_utility"] > 0
    assert len(result.welfare.by_day) == cfg.T
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_simulation.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement simulation loop**

Path: `/Users/hom/Documents/GitHub/sir/src/sir/simulation.py`

```python
"""Daily simulation loop: orchestrates FOC, transmission, disease, interventions, welfare."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from sir.config import ScenarioConfig
from sir.constants import DiseaseState, GroupType
from sir.disease import step_recovery
from sir.foc import solve_foc_by_age
from sir.interventions import Intervention, apply_interventions
from sir.transmission import apply_new_infections, step_transmission
from sir.welfare import WelfareLedger, step_welfare
from sir.world import World, build_world


@dataclass
class SimResult:
    S_history: np.ndarray
    I_history: np.ndarray
    R_history: np.ndarray
    V_history: np.ndarray
    new_infections_history: np.ndarray
    welfare: WelfareLedger
    final_world: World


def _prevalence_by_group_type(world: World) -> dict[GroupType, float]:
    """Approximate prevalence per group type (averaged across groups of that type)."""
    out: dict[GroupType, float] = {}
    is_infected = (world.state == DiseaseState.I).astype(np.float64)
    for gt in GroupType:
        gt_mask_per_membership = world.group_type[world.membership_group_id] == gt
        if not gt_mask_per_membership.any():
            out[gt] = 0.0
            continue
        agent_ids = world.membership_agent_id[gt_mask_per_membership]
        if agent_ids.size == 0:
            out[gt] = 0.0
        else:
            out[gt] = float(is_infected[agent_ids].mean())
    return out


def simulate(
    cfg: ScenarioConfig,
    interventions: Iterable[Intervention],
    rng: np.random.Generator,
    initial_infected: int = 10,
    world: World | None = None,
) -> SimResult:
    if world is None:
        world = build_world(cfg, rng)
    # Seed initial infections
    if initial_infected > 0:
        idx = rng.choice(world.N, size=initial_infected, replace=False)
        world.state[idx] = DiseaseState.I
        world.days_infected[idx] = 1

    interventions_list = list(interventions)
    T = cfg.T

    S_hist = np.zeros(T + 1, dtype=np.int32)
    I_hist = np.zeros(T + 1, dtype=np.int32)
    R_hist = np.zeros(T + 1, dtype=np.int32)
    V_hist = np.zeros(T + 1, dtype=np.int32)
    new_inf_hist = np.zeros(T + 1, dtype=np.int32)

    def snapshot(t: int) -> None:
        S_hist[t] = int((world.state == DiseaseState.S).sum())
        I_hist[t] = int((world.state == DiseaseState.I).sum())
        R_hist[t] = int((world.state == DiseaseState.R).sum())
        V_hist[t] = int((world.state == DiseaseState.V).sum())

    snapshot(0)
    welfare = WelfareLedger()

    for t in range(T):
        # 1. Apply interventions (resets group overrides, then applies active ones)
        new_vax = apply_interventions(world, cfg, t, interventions_list, rng)

        # 2. Compute prevalence by group type
        prev_by_gt = _prevalence_by_group_type(world)

        # 3. Solve FOC -> (theta, e) per age
        theta_by_age, e_by_age = solve_foc_by_age(cfg, prev_by_gt)

        # 4. Compute new infections (hazard + draws)
        new_infections = step_transmission(world, cfg, theta_by_age, e_by_age, rng)

        # 5. Welfare for today (uses new_infections for cost, new_vax for vax cost)
        step_welfare(
            world, cfg, theta_by_age, e_by_age,
            new_infections, new_vax, welfare, day=t,
        )

        # 6. Direct + spillover policy costs
        direct = sum(i.direct_cost_per_day for i in interventions_list if i.is_active(t))
        welfare.totals["direct_policy_cost"] += direct
        spillover = 0.0
        for i in interventions_list:
            if i.is_active(t) and i.spillover_cost_fn is not None:
                spillover += i.spillover_cost_fn(world, cfg)
        welfare.totals["spillover_policy_cost"] += spillover

        # 7. Apply transitions
        apply_new_infections(world, new_infections)
        step_recovery(world, cfg.gamma, rng)

        # 8. Record new-infections count and snapshot
        new_inf_hist[t + 1] = int(new_infections.sum())
        snapshot(t + 1)

    return SimResult(
        S_history=S_hist,
        I_history=I_hist,
        R_history=R_hist,
        V_history=V_hist,
        new_infections_history=new_inf_hist,
        welfare=welfare,
        final_world=world,
    )
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_simulation.py -v
```

Expected: all 5 tests pass. The simulation should complete a 365-day run in <30 seconds.

- [ ] **Step 5: Commit**

```bash
git add src/sir/simulation.py tests/test_simulation.py
git commit -m "feat: add daily simulation loop tying components together"
```

---

## Task 11: Monte Carlo driver

**Files:**
- Create: `src/sir/monte_carlo.py`
- Test: `tests/test_monte_carlo.py`

- [ ] **Step 1: Write the failing test**

Path: `/Users/hom/Documents/GitHub/sir/tests/test_monte_carlo.py`

```python
import numpy as np

from sir.config import default_config
from sir.monte_carlo import MCResult, run_mc


def test_mc_returns_n_runs():
    cfg = default_config()
    result = run_mc(cfg, interventions=[], n_runs=5, base_seed=0, initial_infected=10)
    assert isinstance(result, MCResult)
    assert result.S_history.shape == (5, cfg.T + 1)
    assert result.welfare_totals.shape == (5,)


def test_mc_runs_are_distinct():
    cfg = default_config()
    result = run_mc(cfg, interventions=[], n_runs=5, base_seed=0, initial_infected=10)
    # At least one run should differ from another (stochastic)
    assert not np.all(result.I_history[0] == result.I_history[1])


def test_mc_summary_statistics():
    cfg = default_config()
    result = run_mc(cfg, interventions=[], n_runs=5, base_seed=0, initial_infected=10)
    mean_S = result.S_history.mean(axis=0)
    assert mean_S.shape == (cfg.T + 1,)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_monte_carlo.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement MC driver**

Path: `/Users/hom/Documents/GitHub/sir/src/sir/monte_carlo.py`

```python
"""Monte Carlo driver: parallel runs over independent seeds."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from sir.config import ScenarioConfig
from sir.interventions import Intervention
from sir.simulation import simulate


@dataclass
class MCResult:
    S_history: np.ndarray              # (n_runs, T+1)
    I_history: np.ndarray
    R_history: np.ndarray
    V_history: np.ndarray
    new_infections_history: np.ndarray
    welfare_totals: np.ndarray         # (n_runs,)
    welfare_components: dict[str, np.ndarray]  # each (n_runs,)


def _one_run(
    args: tuple[ScenarioConfig, list[Intervention], int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, dict[str, float]]:
    cfg, interventions, seed, initial_infected = args
    rng = np.random.default_rng(seed)
    result = simulate(cfg, interventions, rng, initial_infected=initial_infected)
    return (
        result.S_history,
        result.I_history,
        result.R_history,
        result.V_history,
        result.new_infections_history,
        result.welfare.total_welfare(),
        dict(result.welfare.totals),
    )


def run_mc(
    cfg: ScenarioConfig,
    interventions: Iterable[Intervention],
    n_runs: int,
    base_seed: int = 0,
    initial_infected: int = 10,
    parallel: bool = True,
) -> MCResult:
    interventions_list = list(interventions)
    args_list = [
        (cfg, interventions_list, base_seed + i, initial_infected)
        for i in range(n_runs)
    ]

    if parallel and n_runs > 1:
        with ProcessPoolExecutor() as pool:
            results = list(pool.map(_one_run, args_list))
    else:
        results = [_one_run(a) for a in args_list]

    S = np.stack([r[0] for r in results])
    I = np.stack([r[1] for r in results])
    R = np.stack([r[2] for r in results])
    V = np.stack([r[3] for r in results])
    NI = np.stack([r[4] for r in results])
    W = np.array([r[5] for r in results])

    components_keys = list(results[0][6].keys())
    components = {
        k: np.array([r[6][k] for r in results]) for k in components_keys
    }

    return MCResult(
        S_history=S,
        I_history=I,
        R_history=R,
        V_history=V,
        new_infections_history=NI,
        welfare_totals=W,
        welfare_components=components,
    )
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_monte_carlo.py -v -s
```

Expected: all 3 tests pass. 5 runs in serial mode should complete in <2 minutes.

- [ ] **Step 5: Commit**

```bash
git add src/sir/monte_carlo.py tests/test_monte_carlo.py
git commit -m "feat: add Monte Carlo driver with optional parallelism"
```

---

## Task 12: PSA — probabilistic sensitivity analysis

**Files:**
- Create: `src/sir/psa.py`
- Create: `configs/psa_example.toml`
- Test: `tests/test_psa.py`

- [ ] **Step 1: Create configs directory**

```bash
mkdir -p /Users/hom/Documents/GitHub/sir/configs
```

- [ ] **Step 2: Write the example PSA spec**

Path: `/Users/hom/Documents/GitHub/sir/configs/psa_example.toml`

```toml
# Example PSA spec for the SIR microfoundations ABM.
# Each table key is a parameter path. Path syntax:
#   - "gamma"                       (scalar attribute)
#   - "V_a.60+"                     (tuple element identified by age-bin label)
#   - "group_p_baseline.HOUSEHOLD"  (mapping element by GroupType name)

[gamma]
dist = "lognormal"
mu = -1.946
sigma = 0.15

[kappa]
dist = "uniform"
lo = 0.5
hi = 2.0

[v_eff]
dist = "beta"
a = 16
b = 4

[c_vax]
ci95 = [0.005, 0.020]

[sick_attendance_multiplier]
ci95 = [0.1, 0.5]
family = "beta"

["V_a.60+"]
dist = "lognormal"
mu = 3.40
sigma = 0.5

["V_a.20-29"]
dist = "fixed"
value = 1.0

["group_p_baseline.HOUSEHOLD"]
ci95 = [0.03, 0.07]

["group_p_baseline.COMMUNITY"]
ci95 = [0.0005, 0.002]
family = "lognormal"

["group_alpha.WORKPLACE"]
dist = "triangular"
lo = 1.5
mode = 2.0
hi = 2.5
```

- [ ] **Step 3: Write the failing test**

Path: `/Users/hom/Documents/GitHub/sir/tests/test_psa.py`

```python
from pathlib import Path

import numpy as np
import pytest

from sir.config import default_config
from sir.constants import AGE_BIN_LABELS, GroupType
from sir.psa import (
    ParameterDistribution,
    PSAResult,
    apply_sample,
    ci_to_distribution,
    draw_sample,
    load_psa_spec,
    run_psa,
)


def test_fixed_distribution_returns_value():
    d = ParameterDistribution("gamma", "fixed", {"value": 0.2})
    rng = np.random.default_rng(0)
    assert d.sample(rng) == 0.2


def test_normal_distribution_centered_correctly():
    d = ParameterDistribution("x", "normal", {"mean": 5.0, "sd": 0.0})
    rng = np.random.default_rng(0)
    assert d.sample(rng) == 5.0


def test_uniform_distribution_in_bounds():
    d = ParameterDistribution("x", "uniform", {"lo": 0.0, "hi": 1.0})
    rng = np.random.default_rng(0)
    for _ in range(100):
        assert 0.0 <= d.sample(rng) <= 1.0


def test_beta_distribution_in_unit_interval():
    d = ParameterDistribution("x", "beta", {"a": 2.0, "b": 5.0})
    rng = np.random.default_rng(0)
    for _ in range(100):
        v = d.sample(rng)
        assert 0.0 <= v <= 1.0


def test_lognormal_strictly_positive():
    d = ParameterDistribution("x", "lognormal", {"mu": 0.0, "sigma": 1.0})
    rng = np.random.default_rng(0)
    for _ in range(100):
        assert d.sample(rng) > 0.0


def test_triangular_within_bounds():
    d = ParameterDistribution("x", "triangular", {"lo": 1.0, "mode": 2.0, "hi": 3.0})
    rng = np.random.default_rng(0)
    for _ in range(100):
        v = d.sample(rng)
        assert 1.0 <= v <= 3.0


def test_unknown_distribution_raises():
    d = ParameterDistribution("x", "weibull", {"a": 1.0})
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        d.sample(rng)


def test_ci_to_normal_conversion():
    d = ci_to_distribution("x", [0.0, 10.0], family="normal")
    assert d.dist == "normal"
    assert np.isclose(d.params["mean"], 5.0)
    # 95% CI under normal: half-width = 1.96 * sd
    assert np.isclose(d.params["sd"], (10.0 - 0.0) / (2 * 1.96))


def test_ci_to_lognormal_conversion():
    d = ci_to_distribution("x", [1.0, 100.0], family="lognormal")
    assert d.dist == "lognormal"
    # mu = mean of log-bounds
    assert np.isclose(d.params["mu"], (np.log(1.0) + np.log(100.0)) / 2)


def test_load_psa_spec_parses_example(tmp_path: Path):
    spec_path = Path(__file__).parent.parent / "configs" / "psa_example.toml"
    spec = load_psa_spec(spec_path)
    names = {d.name for d in spec}
    assert "gamma" in names
    assert "kappa" in names
    assert "V_a.60+" in names
    assert "group_p_baseline.HOUSEHOLD" in names
    # CI shorthand was converted
    c_vax = [d for d in spec if d.name == "c_vax"][0]
    assert c_vax.dist in ("normal", "lognormal", "beta")


def test_draw_sample_reproducible_with_same_seed():
    spec = [
        ParameterDistribution("gamma", "lognormal", {"mu": -2.0, "sigma": 0.1}),
        ParameterDistribution("kappa", "uniform", {"lo": 0.5, "hi": 2.0}),
    ]
    s1 = draw_sample(spec, np.random.default_rng(42))
    s2 = draw_sample(spec, np.random.default_rng(42))
    assert s1 == s2


def test_apply_sample_overrides_scalar():
    base = default_config()
    sample = {"gamma": 0.25, "kappa": 1.5}
    new_cfg = apply_sample(base, sample)
    assert new_cfg.gamma == 0.25
    assert new_cfg.kappa == 1.5
    # Untouched parameters preserved
    assert new_cfg.v_eff == base.v_eff


def test_apply_sample_overrides_v_a_element():
    base = default_config()
    # 60+ is age bin 6, label "60+"
    sample = {"V_a.60+": 99.0}
    new_cfg = apply_sample(base, sample)
    assert new_cfg.V_a[6] == 99.0
    # Other bins unchanged
    for i in range(6):
        assert new_cfg.V_a[i] == base.V_a[i]


def test_apply_sample_overrides_group_mapping_element():
    base = default_config()
    sample = {"group_p_baseline.HOUSEHOLD": 0.123}
    new_cfg = apply_sample(base, sample)
    assert new_cfg.group_p_baseline[GroupType.HOUSEHOLD] == 0.123
    # Other group types unchanged
    assert new_cfg.group_p_baseline[GroupType.COMMUNITY] == base.group_p_baseline[GroupType.COMMUNITY]


def test_apply_sample_unknown_path_raises():
    base = default_config()
    with pytest.raises(KeyError):
        apply_sample(base, {"nonexistent": 1.0})


def test_run_psa_returns_correct_shape():
    base = default_config()
    # Make config small for speed
    base = type(base)(**{**base.__dict__, "N": 500, "T": 30})
    spec = [
        ParameterDistribution("gamma", "uniform", {"lo": 0.1, "hi": 0.2}),
    ]
    result = run_psa(
        base, spec, interventions=[],
        n_psa_samples=3, n_mc_per_sample=2,
        base_seed=0, initial_infected=5,
    )
    assert isinstance(result, PSAResult)
    assert result.welfare_per_sample.shape == (3,)
    assert len(result.samples) == 3


def test_psa_result_quantiles():
    res = PSAResult(
        samples=[{"x": 1.0}, {"x": 2.0}, {"x": 3.0}, {"x": 4.0}, {"x": 5.0}],
        welfare_per_sample=np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        peak_I_per_sample=np.array([100, 200, 300, 400, 500]),
        cumulative_infections_per_sample=np.array([1000, 2000, 3000, 4000, 5000]),
        cumulative_60plus_per_sample=np.array([100, 200, 300, 400, 500]),
        welfare_components_per_sample={"infection_cost": np.array([10.0]*5)},
    )
    lo, med, hi = res.welfare_ci(alpha=0.10)
    assert np.isclose(med, 30.0)
    assert np.isclose(lo, np.percentile(res.welfare_per_sample, 5))
    assert np.isclose(hi, np.percentile(res.welfare_per_sample, 95))
```

- [ ] **Step 4: Run to verify failure**

```bash
pytest tests/test_psa.py -v
```

Expected: ImportError on `sir.psa`.

- [ ] **Step 5: Implement the PSA module**

Path: `/Users/hom/Documents/GitHub/sir/src/sir/psa.py`

```python
"""Probabilistic sensitivity analysis (PSA).

Workflow:
  1. Load TOML spec -> list[ParameterDistribution]
  2. For each PSA sample: draw values, apply to base config -> ScenarioConfig
  3. Run n_mc_per_sample stochastic MC runs at that config
  4. Aggregate per-sample summaries -> PSAResult with quantile methods
"""

from __future__ import annotations

import tomllib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

import numpy as np

from sir.config import ScenarioConfig
from sir.constants import AGE_BIN_LABELS, GroupType
from sir.interventions import Intervention
from sir.simulation import simulate


# -------- Parameter distributions --------


@dataclass
class ParameterDistribution:
    name: str
    dist: str
    params: dict

    def sample(self, rng: np.random.Generator) -> float:
        if self.dist == "fixed":
            return float(self.params["value"])
        if self.dist == "normal":
            return float(rng.normal(self.params["mean"], self.params["sd"]))
        if self.dist == "lognormal":
            return float(rng.lognormal(self.params["mu"], self.params["sigma"]))
        if self.dist == "uniform":
            return float(rng.uniform(self.params["lo"], self.params["hi"]))
        if self.dist == "beta":
            return float(rng.beta(self.params["a"], self.params["b"]))
        if self.dist == "triangular":
            return float(rng.triangular(self.params["lo"], self.params["mode"], self.params["hi"]))
        raise ValueError(f"Unknown distribution: {self.dist}")


def ci_to_distribution(name: str, ci95: list[float], family: str = "normal") -> ParameterDistribution:
    """Convert a 95% CI [lo, hi] to a fitted distribution."""
    lo, hi = ci95
    if family == "normal":
        mean = (lo + hi) / 2
        sd = (hi - lo) / (2 * 1.96)
        return ParameterDistribution(name, "normal", {"mean": mean, "sd": sd})
    if family == "lognormal":
        # Fit so that exp(mu ± 1.96 sigma) = lo, hi
        mu = (np.log(lo) + np.log(hi)) / 2
        sigma = (np.log(hi) - np.log(lo)) / (2 * 1.96)
        return ParameterDistribution(name, "lognormal", {"mu": mu, "sigma": sigma})
    if family == "beta":
        # Method of moments: solve for (a, b) given mean and variance
        mean = (lo + hi) / 2
        sd = (hi - lo) / (2 * 1.96)
        var = sd ** 2
        if var <= 0 or mean <= 0 or mean >= 1:
            raise ValueError(f"Beta CI infeasible for {name}: lo={lo}, hi={hi}")
        common = mean * (1 - mean) / var - 1
        a = mean * common
        b = (1 - mean) * common
        return ParameterDistribution(name, "beta", {"a": a, "b": b})
    raise ValueError(f"Unknown family for CI shorthand: {family}")


# -------- Spec loading --------


def load_psa_spec(path: Path | str) -> list[ParameterDistribution]:
    path = Path(path)
    with open(path, "rb") as f:
        data = tomllib.load(f)
    out: list[ParameterDistribution] = []
    for name, body in data.items():
        if "ci95" in body:
            family = body.get("family", "normal")
            out.append(ci_to_distribution(name, body["ci95"], family))
        else:
            dist = body["dist"]
            params = {k: v for k, v in body.items() if k != "dist"}
            out.append(ParameterDistribution(name, dist, params))
    return out


# -------- Sampling and application --------


def draw_sample(spec: list[ParameterDistribution], rng: np.random.Generator) -> dict[str, float]:
    return {d.name: d.sample(rng) for d in spec}


_AGE_BIN_INDEX = {label: i for i, label in enumerate(AGE_BIN_LABELS)}
_GROUP_TYPE_INDEX = {gt.name: gt for gt in GroupType}


def apply_sample(base: ScenarioConfig, sample: dict[str, float]) -> ScenarioConfig:
    """Return a new ScenarioConfig with `sample` overrides applied.

    Path syntax:
      - "gamma"                          -> scalar attribute
      - "V_a.60+"                        -> tuple element by age-bin label
      - "group_p_baseline.HOUSEHOLD"     -> mapping element by GroupType name
    """
    scalar_overrides: dict[str, float] = {}
    v_a_list = list(base.V_a)
    group_alpha = dict(base.group_alpha)
    group_p = dict(base.group_p_baseline)
    group_m = dict(base.group_m_bar)
    v_a_touched = False
    group_alpha_touched = False
    group_p_touched = False
    group_m_touched = False

    for path, value in sample.items():
        if "." not in path:
            if not hasattr(base, path):
                raise KeyError(f"Unknown scalar parameter: {path}")
            scalar_overrides[path] = value
            continue
        parent, child = path.split(".", 1)
        if parent == "V_a":
            if child not in _AGE_BIN_INDEX:
                raise KeyError(f"Unknown age bin: {child}")
            v_a_list[_AGE_BIN_INDEX[child]] = value
            v_a_touched = True
        elif parent == "group_alpha":
            if child not in _GROUP_TYPE_INDEX:
                raise KeyError(f"Unknown group type: {child}")
            group_alpha[_GROUP_TYPE_INDEX[child]] = value
            group_alpha_touched = True
        elif parent == "group_p_baseline":
            if child not in _GROUP_TYPE_INDEX:
                raise KeyError(f"Unknown group type: {child}")
            group_p[_GROUP_TYPE_INDEX[child]] = value
            group_p_touched = True
        elif parent == "group_m_bar":
            if child not in _GROUP_TYPE_INDEX:
                raise KeyError(f"Unknown group type: {child}")
            group_m[_GROUP_TYPE_INDEX[child]] = value
            group_m_touched = True
        else:
            raise KeyError(f"Unknown parameter path: {path}")

    kwargs: dict = dict(scalar_overrides)
    if v_a_touched:
        kwargs["V_a"] = tuple(v_a_list)
    if group_alpha_touched:
        kwargs["group_alpha"] = group_alpha
    if group_p_touched:
        kwargs["group_p_baseline"] = group_p
    if group_m_touched:
        kwargs["group_m_bar"] = group_m
    return replace(base, **kwargs)


# -------- PSA runner --------


@dataclass
class PSAResult:
    samples: list[dict[str, float]]
    welfare_per_sample: np.ndarray
    peak_I_per_sample: np.ndarray
    cumulative_infections_per_sample: np.ndarray
    cumulative_60plus_per_sample: np.ndarray
    welfare_components_per_sample: dict[str, np.ndarray]

    def welfare_ci(self, alpha: float = 0.05) -> tuple[float, float, float]:
        lo = float(np.percentile(self.welfare_per_sample, 100 * alpha / 2))
        med = float(np.percentile(self.welfare_per_sample, 50))
        hi = float(np.percentile(self.welfare_per_sample, 100 * (1 - alpha / 2)))
        return lo, med, hi

    def parameter_correlations(self, outcome: np.ndarray) -> dict[str, float]:
        """Spearman-like rank correlation of each parameter with an outcome array."""
        if not self.samples:
            return {}
        param_names = list(self.samples[0].keys())
        from scipy.stats import spearmanr
        out: dict[str, float] = {}
        for p in param_names:
            vals = np.array([s[p] for s in self.samples])
            if vals.std() == 0:
                out[p] = 0.0
            else:
                rho, _ = spearmanr(vals, outcome)
                out[p] = float(rho)
        return out


def _summarize_run(
    cfg: ScenarioConfig,
    interventions: list[Intervention],
    n_mc: int,
    seed: int,
    initial_infected: int,
) -> dict:
    """Run n_mc stochastic simulations at cfg; return mean summary."""
    welfare_list = []
    peak_I_list = []
    cum_inf_list = []
    cum_60_list = []
    components_acc: dict[str, list[float]] = {}
    for k in range(n_mc):
        rng = np.random.default_rng(seed * 1000 + k)
        result = simulate(cfg, interventions, rng, initial_infected=initial_infected)
        welfare_list.append(result.welfare.total_welfare())
        peak_I_list.append(int(result.I_history.max()))
        cum_inf = int(result.R_history[-1] + result.V_history[-1] - result.V_history[0])
        cum_inf_list.append(cum_inf)
        # Cumulative 60+ infections: count agents in age_bin 6 who are R at end
        ages = result.final_world.age_bin
        is_60plus = ages == 6
        was_infected = (result.final_world.state[is_60plus] == 3) | (result.final_world.state[is_60plus] == 1)
        cum_60_list.append(int(was_infected.sum()))
        for ck, cv in result.welfare.totals.items():
            components_acc.setdefault(ck, []).append(cv)
    return {
        "welfare": float(np.mean(welfare_list)),
        "peak_I": float(np.mean(peak_I_list)),
        "cum_inf": float(np.mean(cum_inf_list)),
        "cum_60": float(np.mean(cum_60_list)),
        "components": {k: float(np.mean(v)) for k, v in components_acc.items()},
    }


def _outer_iteration(args: tuple) -> tuple[dict[str, float], dict]:
    base_cfg, spec, interventions, n_mc, seed, initial_infected = args
    rng = np.random.default_rng(seed)
    sample = draw_sample(spec, rng)
    cfg = apply_sample(base_cfg, sample)
    summary = _summarize_run(cfg, interventions, n_mc, seed, initial_infected)
    return sample, summary


def run_psa(
    base_cfg: ScenarioConfig,
    spec: list[ParameterDistribution],
    interventions: Iterable[Intervention],
    n_psa_samples: int,
    n_mc_per_sample: int,
    base_seed: int = 0,
    initial_infected: int = 10,
    parallel: bool = True,
) -> PSAResult:
    interventions_list = list(interventions)
    args_list = [
        (base_cfg, spec, interventions_list, n_mc_per_sample, base_seed + i, initial_infected)
        for i in range(n_psa_samples)
    ]
    if parallel and n_psa_samples > 1:
        with ProcessPoolExecutor() as pool:
            outputs = list(pool.map(_outer_iteration, args_list))
    else:
        outputs = [_outer_iteration(a) for a in args_list]

    samples = [o[0] for o in outputs]
    welfare = np.array([o[1]["welfare"] for o in outputs])
    peak_I = np.array([o[1]["peak_I"] for o in outputs])
    cum_inf = np.array([o[1]["cum_inf"] for o in outputs])
    cum_60 = np.array([o[1]["cum_60"] for o in outputs])
    component_keys = list(outputs[0][1]["components"].keys())
    components = {
        k: np.array([o[1]["components"][k] for o in outputs])
        for k in component_keys
    }
    return PSAResult(
        samples=samples,
        welfare_per_sample=welfare,
        peak_I_per_sample=peak_I,
        cumulative_infections_per_sample=cum_inf,
        cumulative_60plus_per_sample=cum_60,
        welfare_components_per_sample=components,
    )
```

- [ ] **Step 6: Run to verify pass**

```bash
pytest tests/test_psa.py -v
```

Expected: all 16 tests pass. The end-to-end PSA run uses a small config (N=500, T=30) so it should complete in <30 seconds.

- [ ] **Step 7: Commit**

```bash
git add src/sir/psa.py tests/test_psa.py configs/psa_example.toml
git commit -m "feat: add probabilistic sensitivity analysis with TOML spec"
```

---

## Task 13: Standard plots

**Files:**
- Create: `src/sir/plots.py`
- (No tests — visual outputs verified by running scenario scripts.)

- [ ] **Step 1: Implement plotting utilities**

Path: `/Users/hom/Documents/GitHub/sir/src/sir/plots.py`

```python
"""Standard plots: epidemic curves, welfare decomposition."""

import matplotlib.pyplot as plt
import numpy as np

from sir.monte_carlo import MCResult


def plot_epidemic_curves(result: MCResult, title: str = "Epidemic curves") -> plt.Figure:
    """Plot mean and 10-90 percentile bands for S/I/R/V trajectories."""
    fig, ax = plt.subplots(figsize=(10, 6))
    days = np.arange(result.S_history.shape[1])
    for label, hist, color in [
        ("Susceptible", result.S_history, "tab:blue"),
        ("Infected", result.I_history, "tab:red"),
        ("Recovered", result.R_history, "tab:green"),
        ("Vaccinated", result.V_history, "tab:purple"),
    ]:
        mean = hist.mean(axis=0)
        q10 = np.percentile(hist, 10, axis=0)
        q90 = np.percentile(hist, 90, axis=0)
        ax.plot(days, mean, label=label, color=color)
        ax.fill_between(days, q10, q90, alpha=0.2, color=color)
    ax.set_xlabel("Days")
    ax.set_ylabel("Number of agents")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_welfare_decomposition(
    results_by_scenario: dict[str, MCResult],
    title: str = "Welfare decomposition by scenario",
) -> plt.Figure:
    """Stacked bar chart of welfare components per scenario."""
    component_keys = [
        "attendance_utility",
        "precaution_cost",
        "infection_cost",
        "vaccination_cost",
        "direct_policy_cost",
        "spillover_policy_cost",
    ]
    component_signs = {
        "attendance_utility": +1,
        "precaution_cost": -1,
        "infection_cost": -1,
        "vaccination_cost": -1,
        "direct_policy_cost": -1,
        "spillover_policy_cost": -1,
    }
    scenarios = list(results_by_scenario.keys())
    means = {
        k: [results_by_scenario[s].welfare_components[k].mean() for s in scenarios]
        for k in component_keys
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(scenarios))
    width = 0.8
    bottoms_pos = np.zeros(len(scenarios))
    bottoms_neg = np.zeros(len(scenarios))
    for k in component_keys:
        signed = np.array(means[k]) * component_signs[k]
        for i, v in enumerate(signed):
            if v >= 0:
                ax.bar(x[i], v, width, bottom=bottoms_pos[i], label=k if i == 0 else "")
                bottoms_pos[i] += v
            else:
                ax.bar(x[i], v, width, bottom=bottoms_neg[i], label=k if i == 0 else "")
                bottoms_neg[i] += v
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Welfare (utility units)")
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.3, axis="y")
    return fig


def plot_welfare_totals(
    results_by_scenario: dict[str, MCResult],
    title: str = "Total welfare by scenario",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    scenarios = list(results_by_scenario.keys())
    means = [results_by_scenario[s].welfare_totals.mean() for s in scenarios]
    stds = [results_by_scenario[s].welfare_totals.std() for s in scenarios]
    x = np.arange(len(scenarios))
    ax.bar(x, means, yerr=stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Total welfare")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    return fig
```

- [ ] **Step 2: Smoke-test the imports**

```bash
python -c "from sir.plots import plot_epidemic_curves, plot_welfare_decomposition, plot_welfare_totals"
```

Expected: no output, no errors.

- [ ] **Step 3: Commit**

```bash
git add src/sir/plots.py
git commit -m "feat: add standard plots for epidemic curves and welfare decomposition"
```

---

## Task 14: Demonstration scripts (baseline, comparison, PSA)

**Files:**
- Create: `scripts/run_baseline.py`
- Create: `scripts/run_comparison.py`
- Create: `scripts/run_psa.py`

- [ ] **Step 1: Create scripts directory**

```bash
mkdir -p /Users/hom/Documents/GitHub/sir/scripts
```

- [ ] **Step 2: Write baseline scenario script**

Path: `/Users/hom/Documents/GitHub/sir/scripts/run_baseline.py`

```python
"""Run a no-intervention baseline scenario and save outputs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from sir.config import default_config
from sir.monte_carlo import run_mc
from sir.plots import plot_epidemic_curves


def main() -> None:
    cfg = default_config()
    print(f"Running baseline: N={cfg.N}, T={cfg.T}")
    result = run_mc(cfg, interventions=[], n_runs=10, base_seed=0, initial_infected=10)
    print(f"Mean total welfare: {result.welfare_totals.mean():.1f}")
    print(f"Mean peak infected: {result.I_history.max(axis=1).mean():.0f}")
    print(f"Mean final recovered: {result.R_history[:, -1].mean():.0f}")

    out_dir = Path(__file__).parent.parent / "output"
    out_dir.mkdir(exist_ok=True)
    fig = plot_epidemic_curves(result, title="Baseline (no intervention)")
    fig.savefig(out_dir / "baseline_curves.png", dpi=120)
    plt.close(fig)
    print(f"Saved: {out_dir / 'baseline_curves.png'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run baseline and verify output**

```bash
cd /Users/hom/Documents/GitHub/sir
python scripts/run_baseline.py
```

Expected output (approximate, will vary by run):

```
Running baseline: N=10000, T=365
Mean total welfare: <some number>
Mean peak infected: <some number>
Mean final recovered: <some number>
Saved: /Users/hom/Documents/GitHub/sir/output/baseline_curves.png
```

Inspect `output/baseline_curves.png` and confirm:
- Susceptible curve declines monotonically.
- Infected curve rises then falls (epidemic peak).
- Recovered curve rises monotonically.
- All curves sum to N at every t.

- [ ] **Step 4: Write comparison scenario script**

Path: `/Users/hom/Documents/GitHub/sir/scripts/run_comparison.py`

```python
"""Compare baseline, mask mandate, school closure, and vaccination scenarios."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sir.config import default_config
from sir.constants import GroupType
from sir.interventions import (
    close_schools,
    mask_mandate,
    vaccinate_eldest_first,
)
from sir.monte_carlo import run_mc
from sir.plots import (
    plot_epidemic_curves,
    plot_welfare_decomposition,
    plot_welfare_totals,
)


def main() -> None:
    cfg = default_config()
    n_runs = 10
    out_dir = Path(__file__).parent.parent / "output"
    out_dir.mkdir(exist_ok=True)

    scenarios = {
        "baseline": [],
        "mask_mandate": [
            mask_mandate(
                start_day=30, end_day=180,
                target_types={
                    GroupType.SCHOOL, GroupType.WORKPLACE, GroupType.COMMUNITY,
                },
                p_factor=0.5,
            ),
        ],
        "school_closure": [
            close_schools(start_day=30, end_day=120),
        ],
        "vaccination": [
            vaccinate_eldest_first(start_day=30, end_day=200, doses_per_day=50),
        ],
    }

    results = {}
    for name, interventions in scenarios.items():
        print(f"Running scenario: {name}")
        results[name] = run_mc(cfg, interventions, n_runs=n_runs, base_seed=42, initial_infected=10)
        print(f"  Mean welfare: {results[name].welfare_totals.mean():.1f}")
        print(f"  Mean peak I: {results[name].I_history.max(axis=1).mean():.0f}")
        print(f"  Mean final R: {results[name].R_history[:, -1].mean():.0f}")

    # Per-scenario epidemic curves
    for name, r in results.items():
        fig = plot_epidemic_curves(r, title=f"Scenario: {name}")
        fig.savefig(out_dir / f"curves_{name}.png", dpi=120)
        plt.close(fig)

    # Welfare comparison
    fig = plot_welfare_decomposition(results, title="Welfare decomposition by scenario")
    fig.savefig(out_dir / "welfare_decomposition.png", dpi=120)
    plt.close(fig)
    fig = plot_welfare_totals(results, title="Total welfare by scenario")
    fig.savefig(out_dir / "welfare_totals.png", dpi=120)
    plt.close(fig)
    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run comparison and verify output**

```bash
cd /Users/hom/Documents/GitHub/sir
python scripts/run_comparison.py
```

Expected behavior:

- All four scenarios complete in <5 minutes.
- `output/welfare_decomposition.png` shows stacked-bar comparison.
- `output/welfare_totals.png` shows scenario welfare totals.
- Vaccination scenario should have notably lower infection cost than baseline.
- Mask mandate should reduce peak infected vs. baseline.

- [ ] **Step 6: Write the PSA runner script**

Path: `/Users/hom/Documents/GitHub/sir/scripts/run_psa.py`

```python
"""Run probabilistic sensitivity analysis using configs/psa_example.toml.

Outputs a summary of welfare CIs and a tornado plot of parameter sensitivities.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sir.config import default_config
from sir.psa import load_psa_spec, run_psa


def plot_tornado(
    correlations: dict[str, float],
    outcome_name: str,
    out_path: Path,
) -> None:
    items = sorted(correlations.items(), key=lambda kv: abs(kv[1]), reverse=True)
    names = [k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(names) + 1)))
    colors = ["tab:red" if v < 0 else "tab:blue" for v in values]
    ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(f"Rank correlation with {outcome_name}")
    ax.set_title(f"Parameter sensitivity tornado: {outcome_name}")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    base_cfg = default_config()
    spec_path = Path(__file__).parent.parent / "configs" / "psa_example.toml"
    spec = load_psa_spec(spec_path)
    print(f"Loaded {len(spec)} parameter distributions from {spec_path.name}")

    n_psa = 30
    n_mc = 5
    print(f"Running PSA: {n_psa} samples x {n_mc} MC runs = {n_psa * n_mc} simulations")
    result = run_psa(
        base_cfg, spec,
        interventions=[],
        n_psa_samples=n_psa,
        n_mc_per_sample=n_mc,
        base_seed=0,
        initial_infected=10,
    )

    lo, med, hi = result.welfare_ci(alpha=0.05)
    print(f"Total welfare: median {med:.1f}, 95% CI [{lo:.1f}, {hi:.1f}]")
    peak_med = float(np.median(result.peak_I_per_sample))
    peak_lo = float(np.percentile(result.peak_I_per_sample, 2.5))
    peak_hi = float(np.percentile(result.peak_I_per_sample, 97.5))
    print(f"Peak infected: median {peak_med:.0f}, 95% CI [{peak_lo:.0f}, {peak_hi:.0f}]")
    cum_med = float(np.median(result.cumulative_infections_per_sample))
    print(f"Cumulative infections (median): {cum_med:.0f}")

    out_dir = Path(__file__).parent.parent / "output"
    out_dir.mkdir(exist_ok=True)
    welfare_corr = result.parameter_correlations(result.welfare_per_sample)
    plot_tornado(welfare_corr, "total welfare", out_dir / "psa_tornado_welfare.png")
    peak_corr = result.parameter_correlations(result.peak_I_per_sample)
    plot_tornado(peak_corr, "peak infected", out_dir / "psa_tornado_peak.png")
    print(f"Saved tornado plots to {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run PSA script and verify**

```bash
cd /Users/hom/Documents/GitHub/sir
python scripts/run_psa.py
```

Expected output (approximate):

```
Loaded ~10 parameter distributions from psa_example.toml
Running PSA: 30 samples x 5 MC runs = 150 simulations
Total welfare: median <num>, 95% CI [<lo>, <hi>]
Peak infected: median <num>, 95% CI [<lo>, <hi>]
Cumulative infections (median): <num>
Saved tornado plots to /Users/hom/Documents/GitHub/sir/output
```

Inspect `output/psa_tornado_welfare.png` and confirm:
- High-leverage parameters at the top (longest bars).
- Sign of correlation makes sense (e.g., higher `V_a.60+` should correlate *negatively* with welfare).

Total runtime should be ~3–10 minutes depending on parallelism.

- [ ] **Step 8: Commit**

```bash
git add scripts/run_baseline.py scripts/run_comparison.py scripts/run_psa.py
git commit -m "feat: add baseline, comparison, and PSA scenario scripts"
```

---

## Task 15: Final sanity sweep

**Files:**
- (No new files. Run all tests + scenario scripts together.)

- [ ] **Step 1: Run all tests**

```bash
cd /Users/hom/Documents/GitHub/sir
pytest -v
```

Expected: all tests pass. Total runtime <2 minutes.

- [ ] **Step 2: Run baseline and comparison scripts**

```bash
python scripts/run_baseline.py
python scripts/run_comparison.py
python scripts/run_psa.py
```

Expected: both complete without errors; output PNGs are created.

- [ ] **Step 3: Inspect outputs**

Open each PNG in `output/` and verify:
- Epidemic curves are smooth and stochastically reasonable.
- Welfare decomposition shows the expected components.
- Mask mandate, school closure, and vaccination each reduce infection cost compared to baseline (with their own offsetting costs visible).

- [ ] **Step 4: Tag the v0.1 release**

```bash
git tag v0.1
git log --oneline
```

Expected: clean linear history of feature commits, ~15 commits.

---

## Self-Review notes

Spec coverage check (against `docs/superpowers/specs/2026-05-14-sir-microfoundations-abm-design.md`):

| Spec section | Implementing task(s) |
|---|---|
| 1 Purpose | All |
| 2 Scope | All; out-of-scope items not implemented |
| 3 Model overview | Task 10 (simulation loop) |
| 4.1 Agent state | Task 4 (World) |
| 4.2 Utility | Task 8 (welfare) |
| 4.3 FOC | Task 7 (foc) |
| 4.4 Rational expectations | Task 7 (true I_g/N_g used) |
| 5.1 Bipartite | Task 4 (world memberships) |
| 5.2 Group properties | Task 4 (group_* arrays) |
| 5.3 Group types | Task 4 (seven types built) |
| 5.4 Transmission | Task 6 (transmission) |
| 5.5 Disease transitions | Tasks 5, 6 |
| 6 Functional forms | Tasks 7, 8 |
| 7 Parameters | Task 3 (default_config) |
| 8 Intervention layer | Task 9 |
| 9 Welfare | Task 8 |
| 9.5 PSA | Task 12 |
| 10 Module structure | All; matches layout exactly |
| 11 Implementation notes | Tasks 4, 6 (numpy/bincount throughout) |
| 12 Verification | Tasks 5, 6, 7, 8, 9, 10 (all tests) |
| 13 Confirmed decisions | All defaults baked in (Task 2, 3) |
| 14 Future extensions | Not implemented (v2 scope) |
| 15 Success criteria | Task 13 (comparison script demonstrates) |

No placeholders; types and signatures consistent across tasks. Plan complete.
