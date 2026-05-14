# Microfounded ABM for Epidemic Cost-Benefit Analysis — Design

**Date:** 2026-05-14
**Status:** Draft for review

## 1. Purpose

Build a policy-relevant agent-based epidemic model in which:

- Disease dynamics follow a discrete-time SIR process at the individual level.
- Per-contact transmission and contact rates emerge from **rational agents** optimizing flow utility under perceived infection risk.
- Contact structure is captured by a **bipartite agent-group** representation with categorical group types (household, kin, school/workplace, leisure, etc.).
- **Interventions** are first-class objects that shift either policy constraints (closures, mandates) or agent costs (utility weights, precaution costs).
- **Welfare** is computed as the sum of individual utilities, enabling cost-benefit comparisons across intervention scenarios.

The model is a methodology demonstration first, a calibrated forecast second. It should make qualitative comparative statics legible and provide a foundation that can later be fit to a specific pathogen.

## 2. Scope

### In scope (v1)

- Myopic rational agents (one-period optimization).
- Standard SIRV disease model (S, I, R, plus V for vaccinated; no exposed compartment).
- Single closed population (no geography, no births/deaths).
- Generic respiratory pathogen (no specific calibration target).
- Heterogeneity by age group and group-type membership.
- Bipartite agent-group contact structure with ~6–7 group types.
- Stochastic individual-level disease transitions.
- Welfare accounting in utility units, decomposed by source (work, leisure, illness, precaution, policy spillover).
- Vaccination as an **exogenous policy intervention** (no endogenous vaccine-choice optimization).
- **Spillover costs** of interventions (e.g., learning loss from school closure) modeled as parametric add-ons to the intervention's direct cost.
- **Probabilistic sensitivity analysis** (PSA): a TOML spec of parameter distributions, outer-loop sampling, inner Monte Carlo per draw, and confidence intervals on outcomes.

### Out of scope (v1; potential v2 extensions)

- Forward-looking agents with perfect-foresight equilibrium.
- SEIR / presymptomatic transmission.
- Multiple populations / spatial geography / mobility.
- Pathogen-specific calibration.
- **Endogenous vaccination choice** (vaccination is policy-controlled in v1).
- Dynamic friendship networks (friends absorbed into leisure venues).
- Behavioral fatigue and signal-extraction problems (rational expectations assumed).
- Explicit macroeconomic submodel (income/wages bundled into `α_workplace`).
- Distributional spillover effects attached to individual agents over their lifetime (handled at intervention level in v1).

## 3. Model overview

```
                ┌──────────────────────────────┐
                │   World setup (once)         │
                │   - sample agents, groups    │
                │   - assign memberships       │
                └──────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────┐
│  Daily step (repeat for T days)                          │
│                                                          │
│   1. Compute infection prevalence per group              │
│   2. Per-type FOC solve → (θ_a, e_a) for each type a     │
│   3. Per-agent transmission draws (bipartite groups)     │
│   4. Per-agent disease transitions (Bernoulli recovery)  │
│   5. Accumulate utility per agent                        │
│   6. Apply intervention layer (if active today)          │
└──────────────────────────────────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────────┐
                │   Welfare aggregation        │
                │   W(θ) = Σ_a Σ_t U_a(t)      │
                └──────────────────────────────┘
```

## 4. Microfoundations

### 4.1 Agent state and types

Each agent `i` has:

- **Type** `a_i`: a tuple `(age_group, employment_status, household_type)`, ~20–50 unique types in the population.
- **Disease state** `s_i ∈ {S, V, I, R}` where `V` denotes vaccinated.
- **Group memberships**: a list of group IDs the agent attends.

Vaccinated agents `V` remain at-risk but with reduced per-meeting transmission probability `p_g^{(0)} \cdot (1 - v_{\text{eff}})`. They transition to `I` (still possible at reduced rate), then to `R`. Setting `v_eff = 1` recovers the fully-immunizing-vaccine special case where `V` is absorbing.

### 4.2 Per-period utility (susceptible agent)

$$
U_a(\theta, e \mid I/N) = \sum_{g \in \mathcal{G}(i)} \alpha_g \log(1 + m_g(\theta)) - \tfrac{1}{2} \kappa e^2 - V_a \sum_{g \in \mathcal{G}(i)} p_g^{(0)} (1 - e) \cdot m_g(\theta) \cdot \frac{I_g}{N_g}
$$

where:

- `θ ∈ [0, 1]` — voluntary attendance multiplier (applied to choosable groups).
- `e ∈ [0, 1]` — precaution effort.
- `m_g(θ) = θ · m̄_g` for voluntary groups; `m̄_g` (institutional) otherwise.
- `α_g` — utility weight per group type.
- `p_g^{(0)}` — baseline per-meeting transmission probability per group type.
- `V_a` — value of staying healthy for agent type `a`.
- `I_g / N_g` — current prevalence among members of group `g`.

For infected (`I`) and recovered (`R`) agents, there is no infection risk term; their utility is `Σ α_g log(1+m_g) − c(e)` (typically `e = 0` for `R`; `I` agents may be partly isolated by symptoms — modeled via a fixed attendance multiplier).

### 4.3 First-order conditions

The susceptible agent's interior solution satisfies:

$$
\sum_{g \in \text{vol}} \frac{\alpha_g \, m̄_g}{1 + \theta m̄_g} = V_a (1 - e) \sum_{g \in \text{vol}} p_g^{(0)} m̄_g \frac{I_g}{N_g}
$$

$$
\kappa e = V_a (1 - e_{\text{eq}}) \cdot \Lambda_a(\theta)
\quad\text{where}\quad
\Lambda_a(\theta) = \sum_g p_g^{(0)} m_g(\theta) \frac{I_g}{N_g}
$$

Solve numerically (one rootfind, dim 2) per type per day. ~20–50 FOC solves per day total.

### 4.4 Rational expectations and atomism

Agents use **true** `I_g / N_g` (no lag, no media bias). Each agent treats `I_g / N_g` as exogenous (atomism). The gap between private and social optimum — the infection externality — is the welfare basis for policy.

## 5. Contact structure

### 5.1 Bipartite agent-group representation

The world contains:

- **Agents**: `N` individuals with attributes.
- **Groups**: a set of structures, each with properties.
- **Memberships**: a sparse bipartite relation `agent_id × group_id`.

### 5.2 Group properties

Each group carries:

| Property | Type | Notes |
|---|---|---|
| `group_id` | int | unique |
| `type` | enum | one of `{Household, Kin, School, Workplace, RecurringLeisure, OneOffEvent, Community}` |
| `size` | int | number of members |
| `attendance_base` | float | baseline attendance rate per day per member (m̄_g) |
| `persistent` | bool | same members across days? |
| `p_baseline` | float | per-meeting transmission probability (p_g^{(0)}) |
| `closable_by_policy` | bool | targetable by intervention |
| `active` | bool | currently open (set by intervention layer) |

### 5.3 Group types

| Type | Generation | Targetability |
|---|---|---|
| **Household** | sampled from census composition table | closed → fully connected |
| **Kin** | linked-household graph, λ_kin ~ Poisson(1–2) per household | bubble policies |
| **School** | enrollment by age, ~300–1000 per school | individual school closure |
| **Workplace** | firm-size distribution, sampled assignment | WFH share, sector closures |
| **RecurringLeisure** | gym, choir, club; small (~10–50), stable membership | gathering-size limits |
| **OneOffEvent** | ephemeral; weddings, concerts; one-day lifetime | event bans |
| **Community** | age × neighborhood random mixing pool | gathering limits, distancing |

Diffuse friendship is **absorbed into RecurringLeisure** (people see friends at venues).

### 5.4 Transmission within a group

A single quantity `m_{i,g,t}` represents the agent's expected contact-meetings in group `g` on day `t`:

$$
m_{i,g,t} = \begin{cases}
\theta_i \cdot m̄_g & \text{if } g \text{ is a voluntary group} \\
m̄_g & \text{if } g \text{ is institutional}
\end{cases}
$$

interpreted as expected number of within-group contact events (can be fractional). The per-day infection hazard contribution from group `g` for susceptible agent `i` is:

$$
\text{hazard}_{i,g,t} = 1 - (1 - p_g^{(0)}(1 - e_i))^{m_{i,g,t} \cdot I_{g,t} / N_g} \;\approx\; p_g^{(0)}(1 - e_i) \cdot m_{i,g,t} \cdot \tfrac{I_{g,t}}{N_g}
$$

The total daily hazard for agent `i` is the sum over their group memberships (composing escape probabilities is equivalent to first order at low per-group risk):

$$
\text{hazard}_{i,t} = \sum_{g \in \mathcal{G}(i)} \text{hazard}_{i,g,t}
$$

and infection probability is `1 − exp(−hazard_{i,t})`.

### 5.5 Disease transitions (per agent, per day)

```
For each susceptible agent i (s_i = S):
    total_hazard_i = Σ over groups g in memberships(i) of hazard_{i,g,t}
    p_infect_i = 1 - exp(-total_hazard_i)
    if uniform(0,1) < p_infect_i:
        s_i ← I

For each vaccinated agent i (s_i = V):
    total_hazard_i = (1 - v_eff) · Σ over groups g in memberships(i) of hazard_{i,g,t}
    p_infect_i = 1 - exp(-total_hazard_i)
    if uniform(0,1) < p_infect_i:
        s_i ← I

For each infected agent i (s_i = I):
    if uniform(0,1) < γ:
        s_i ← R

Vaccination (S → V) is handled by the intervention layer; see section 8.
```

## 6. Functional forms

| Quantity | Form | Parameters |
|---|---|---|
| Utility from group attendance | `α_g · log(1 + m)` | `α_g` per group type |
| Cost of precaution | `½ κ e²`, `e ∈ [0,1]` | `κ` global |
| Per-meeting transmission | `p_g^{(0)} · (1 - e)` | `p_g^{(0)}` per group type |
| Vaccinated transmission scaling | `× (1 − v_eff)` | `v_eff` single scalar |
| Cost of infection | `V_a` per age group | 7 age groups, see below |
| Recovery rate | `γ` constant | mean infectious period `1/γ` |
| Sick-agent attendance | `0.3 · m̄_g` (all groups) | uniform 30% reduction while infected |

### V_a defaults (7 decadal age bins, IFR-scaled)

| Age group | `V_a` (relative) | Rationale |
|---|---|---|
| 0–9 | 0.5 | Very low IFR for respiratory diseases |
| 10–19 | 0.5 | Very low IFR |
| 20–29 | 1.0 | Reference |
| 30–39 | 2.0 | Modest IFR increase |
| 40–49 | 3.0 | Modest IFR increase |
| 50–59 | 5.0 | Moderate IFR increase |
| 60+ | 30.0 | Dominant mortality risk |

(60+ is the open upper bin. For absolute monetary CB later, scale by `V_a_real = V_a · (VSL × IFR_reference)`.)

## 7. Parameters

| Symbol | Default | Source |
|---|---|---|
| `N` | 10,000 | confirmed for v1 |
| `T` | 365 days | scenario |
| `γ` | 1/7 day⁻¹ | generic respiratory |
| `p_g^{(0)}` | 0.001 (Community) – 0.05 (Household) | calibrated to R₀ ≈ 2.5 |
| `α_g` | normalized so baseline `θ* ≈ 1` at `I = 0` | one free scalar; relative `α_g` ratios from time-use anchoring |
| `m̄_g` (Household) | 1.0 (always attended) | structural |
| `m̄_g` (Workplace) | 1.0 weekdays, 0 weekends | structural |
| `m̄_g` (School) | 1.0 weekdays during term, 0 otherwise | structural |
| `m̄_g` (Kin) | 0.1 (≈ weekly visits) | survey-anchored |
| `m̄_g` (RecurringLeisure) | 0.3 (a few times/week) | survey-anchored |
| `m̄_g` (Community) | calibrated to close R₀ target | POLYMOD-style |
| `κ` | swept | sensitivity parameter |
| `V_a` | see table above | 7 decadal bins |
| `v_eff` | 0.8 (single scalar) | typical respiratory vaccine |
| `c_vax` | small fixed disutility | per recipient |
| `sick_attendance_multiplier` | 0.3 | uniform across groups |
| `learning_loss_per_kid_per_day` | parameterized; documented source; swept ±50% | spillover sensitivity |

Calibration target: `R₀ = m̄ · p̄ / γ ≈ 2.5` at baseline (no behavioral response, no interventions). `α_g` are normalized so that in the no-disease world `θ* = 1` (every agent chooses baseline attendance).

## 8. Intervention layer

Interventions are objects that modify model parameters or move agents between disease states during specified time windows.

```python
@dataclass
class Intervention:
    name: str
    start_day: int
    end_day: int
    target_filter: Callable[[GroupRecord], bool] | None   # for group-modifying interventions
    apply: Callable[[GroupRecord], GroupRecord] | None    # how it modifies groups

    # Per-agent action (used by vaccination, mass testing, etc.); takes the World
    agent_action: Callable[[World, int], None] | None = None

    # Cost accounting
    direct_cost_per_day: float = 0.0                      # admin, enforcement
    spillover_cost_per_day: Callable[[World, int], float] | None = None
    # spillover_cost is a function so it can scale with the number of affected agents
    # (e.g., learning loss × number of kids in closed schools)
```

### 8.1 Group-targeting interventions

| Intervention | Filter | Effect |
|---|---|---|
| School closure | `type == School` | `active = False` |
| WFH mandate | `type == Workplace` | `attendance_base *= 0.3` |
| Mask mandate | `type ∈ {School, Workplace, Community}` | `p_baseline *= 0.5` |
| Gathering limit (size 10) | `size > 10` | `attendance_base *= 0.1` |
| Bubble rule | `type == Kin` | restrict to one linked household |
| Event ban | `type == OneOffEvent` | `active = False` |

### 8.2 Vaccination as an agent-action intervention

```python
def vaccinate(world: World, t: int) -> None:
    # Pick up to `doses_per_day` susceptibles, prioritized by age
    susceptible = (world.state == S)
    eligible = susceptible & not_already_vaccinated
    priority_order = age_priority(world, eligible)        # oldest first by default
    chosen = priority_order[:doses_per_day]
    world.state[chosen] = V                               # move to vaccinated
    world.utility[chosen] -= vaccine_inconvenience_cost   # one-time disutility
```

The vaccination program is a single `Intervention` with `agent_action = vaccinate` and `direct_cost_per_day = doses_per_day × cost_per_dose`. Multiple programs can run in parallel (e.g., elderly campaign then general campaign) with different priority functions and dose budgets.

### 8.3 Spillover-cost examples

| Intervention | Spillover model |
|---|---|
| School closure | `learning_loss_per_kid_per_day × num_kids_in_closed_schools` |
| Prolonged isolation | `mental_health_cost_per_agent × duration × num_agents` (optional, default 0) |
| WFH mandate | usually no spillover beyond utility loss |
| Mask mandate | small per-mask cost × N_users |

Spillover-cost parameters are part of the scenario config and can be swept as sensitivities. Calibrating their exact values is a separate exercise; the model just needs the magnitudes to be displayed in the CB calculation.

Interventions compose by sequential application each day. Agents re-solve FOC given the current parameter state.

## 9. Welfare accounting

### 9.1 Conventions

- `V_a` is the **total utility cost** of one infection for an agent of age type `a`. It bundles symptom suffering, severity risk, mortality risk × VSL, and long-term sequelae.
- The infection cost is subtracted **once, on the day of infection** (`S → I` or `V → I` transition).
- For time-series plots, the same cost can be displayed as `V_a · γ` allocated per day over the infectious period — this is a presentation choice, not a different model.
- `α_workplace · log(1 + m_workplace)` is interpreted as **bundling intrinsic and economic value of work**. There is no separate income/wage variable in v1.

### 9.2 Per-agent per-day utility

$$
U_{i,t} = \underbrace{\sum_g \alpha_g \log(1 + m_{i,g,t})}_{\text{attendance utility}} \;-\; \underbrace{\tfrac{1}{2}\kappa\, e_{i,t}^2}_{\text{precaution cost}} \;-\; \underbrace{V_{a(i)} \cdot \mathbb{1}\{\text{infection event on day } t\}}_{\text{infection cost}} \;-\; \underbrace{c_{\text{vax}} \cdot \mathbb{1}\{\text{vaccinated on day } t\}}_{\text{vaccination cost}}
$$

Note that the FOC uses the *expected* infection cost `V_a · π_i`, while the realized welfare uses the *event* cost `V_a · 1{infected}`. In expectation these agree; sample-by-sample they differ (this is normal — agents face risk, not certainty).

### 9.3 Aggregate welfare under intervention scenario `θ`

$$
W(\theta) = \sum_{t=0}^{T} \sum_{i=1}^{N} U_{i,t} \;-\; \sum_t \bigl[\text{direct\_cost}(\theta, t) + \text{spillover\_cost}(\theta, t)\bigr]
$$

Cost-benefit of intervention vs. baseline: `ΔW = W(θ) − W(0)`.

### 9.4 Decomposed reporting

Each scenario produces a welfare breakdown:

| Component | Source | Sign |
|---|---|---|
| Workplace attendance utility | `Σ α_workplace · log(1 + m_workplace)` | + |
| School attendance utility | `Σ α_school · log(1 + m_school)` | + |
| Leisure attendance utility | `Σ α_leisure · log(1 + m_leisure)` | + |
| Household/kin attendance utility | `Σ α_household · log(1 + m_household)` | + |
| Precaution cost | `Σ ½ κ e²` | − |
| Infection cost (suffering) | `Σ V_a · {new infections}` | − |
| Vaccination cost (recipients) | `Σ c_vax · {vaccinations}` | − |
| Direct policy cost | `Σ direct_cost(θ, t)` | − |
| Spillover policy cost | `Σ spillover_cost(θ, t)` | − |

Report by:
- Aggregate (single number per scenario).
- By age group (distributional welfare).
- By group type (which sectors lost / gained utility).
- By component (the table above) — makes the CB tradeoff legible.

This decomposition is what makes the model **policy-relevant**: a reader can see precisely which channel drives the welfare comparison between two intervention scenarios.

## 9.5 Probabilistic Sensitivity Analysis (PSA)

PSA propagates parameter uncertainty to outcome uncertainty. Each uncertain parameter is given a distribution; we draw `n_psa` joint samples, run `n_mc` stochastic simulations per draw, and report confidence intervals across draws.

### 9.5.1 File format (TOML)

A PSA spec is a TOML file mapping **parameter paths** to distribution specifications. Two forms are supported per path:

**Form A — explicit distribution:**

```toml
[gamma]
dist = "lognormal"
mu = -1.946
sigma = 0.15

["V_a.60+"]
dist = "lognormal"
mu = 3.40
sigma = 0.5
```

**Form B — CI shorthand:**

```toml
[c_vax]
ci95 = [0.005, 0.020]              # default family: normal

["group_p_baseline.COMMUNITY"]
ci95 = [0.0005, 0.002]
family = "lognormal"               # override default
```

**Form C — fixed (no uncertainty, useful for locking parameters):**

```toml
[kappa]
dist = "fixed"
value = 1.0
```

### 9.5.2 Supported distributions

| Name | Params | Typical use |
|---|---|---|
| `fixed` | `value` | Lock a parameter |
| `normal` | `mean`, `sd` | Unbounded symmetric uncertainty |
| `lognormal` | `mu`, `sigma` (of log) | Strictly-positive parameters |
| `uniform` | `lo`, `hi` | Bounded uniform |
| `beta` | `a`, `b` | Probabilities in [0,1] |
| `triangular` | `lo`, `mode`, `hi` | Point estimate + range |

### 9.5.3 Parameter paths

- **Scalars**: `gamma`, `kappa`, `v_eff`, `c_vax`, `sick_attendance_multiplier`, `learning_loss_per_kid_per_day`
- **V_a tuple elements**: `V_a.0-9`, `V_a.10-19`, ..., `V_a.60+`
- **Mapping elements**: `group_alpha.HOUSEHOLD`, `group_p_baseline.SCHOOL`, `group_m_bar.WORKPLACE`, etc.

Structural parameters (`N`, `T`, group counts, voluntary-group set) are not exposed for PSA.

### 9.5.4 Sampling model

Draws are **independent across parameters** in v1. (Correlations are a v2 extension.) Each PSA outer iteration draws all parameter values once; the resulting `ScenarioConfig` is then passed through `n_mc` MC runs that vary only the RNG seed.

### 9.5.5 Outputs and reporting

`PSAResult` stores, per outer sample:

- Mean total welfare (averaged over inner MC)
- Mean peak `I`
- Mean cumulative infections
- Mean cumulative 60+ infections (mortality proxy)
- Mean welfare components

Aggregated across samples, the report shows median, 95% CI (2.5/97.5 percentiles), and **tornado-style marginal sensitivity** (rank correlation between each parameter draw and an outcome).

### 9.5.6 Run budget

Default: `n_psa = 50`, `n_mc = 10` → 500 simulations. ~5 minutes on 8 cores in parallel.

## 10. Module structure

```
sir/
├── docs/
│   └── superpowers/specs/2026-05-14-sir-microfoundations-abm-design.md
├── pyproject.toml
└── src/sir/
    ├── __init__.py
    ├── world.py              # agent/group generation, sampling
    ├── types.py              # dataclasses: AgentRecord, GroupRecord, Intervention, ScenarioConfig
    ├── foc.py                # solve per-type (θ*, e*) given current I_g/N_g
    ├── transmission.py       # bipartite transmission engine (one loop over groups)
    ├── disease.py            # SIR transitions (S→I, I→R)
    ├── interventions.py      # Intervention dataclass + a library of standard ones
    ├── welfare.py            # utility computation + aggregation
    ├── simulation.py         # daily loop, orchestrates the above
    ├── monte_carlo.py        # parallel runs + summary stats
    ├── psa.py                # PSA: load TOML, sample, apply, run, aggregate
    └── plots.py              # standard outputs (curves, welfare bars, dist'l plots)
└── tests/
    ├── test_foc.py           # FOC reduces to known limits (no risk → θ=1, e=0; high V → θ small, e large)
    ├── test_transmission.py  # well-mixed group reproduces SIR ODE in deterministic limit
    ├── test_disease.py       # R₀ at I₀=1 in fully susceptible matches m·p/γ
    ├── test_welfare.py       # zero infections + zero precaution → max welfare
    ├── test_interventions.py # each intervention has expected sign on welfare components
    └── test_psa.py           # distribution sampling, path overrides, end-to-end PSA run
```

### 10.1 Key interfaces

```python
# types.py
@dataclass
class ScenarioConfig:
    N: int
    T: int
    gamma: float
    age_dist: dict[str, float]
    group_specs: dict[str, GroupSpec]    # one per group type
    kappa: float
    V_by_age: dict[str, float]

@dataclass
class GroupSpec:
    group_type: str
    size_dist: Callable[[], int]
    attendance_base: float
    persistent: bool
    p_baseline: float
    closable_by_policy: bool

# world.py
def build_world(cfg: ScenarioConfig, rng) -> World: ...

# foc.py
def solve_foc_by_type(world: World, prevalence_by_group: np.ndarray, cfg) -> ChoiceTable: ...

# transmission.py
def step_transmission(world: World, choices: ChoiceTable, rng) -> np.ndarray: ...   # new infections mask

# disease.py
def step_disease(world: World, new_infections: np.ndarray, cfg, rng) -> None: ...

# interventions.py
def apply_interventions(world: World, t: int, active_interventions: list[Intervention]) -> None: ...

# welfare.py
def step_welfare(world: World, choices: ChoiceTable, cfg) -> WelfareRecord: ...

# simulation.py
def simulate(cfg: ScenarioConfig, interventions: list[Intervention], rng) -> SimResult: ...

# monte_carlo.py
def run_mc(cfg: ScenarioConfig, interventions: list[Intervention], n_runs: int, seeds) -> MCResult: ...
```

## 11. Implementation notes

- **Numpy-first**, no Mesa. Agent attributes are 1-D arrays of length `N`; group properties are 1-D arrays of length `G`; memberships are a sparse CSR matrix (`G × N`) or two parallel arrays `(agent_id, group_id)`.
- **Group-level aggregations** use `np.bincount(group_id, weights=...)` for `O(N + G)` operations.
- **FOC solver**: SciPy `brentq` on a 1-D reduced problem (substitute `e` into `θ` equation and solve in `θ`). Vectorize over types if profile shows it matters.
- **RNG**: explicit `numpy.random.Generator` plumbed through; one seed per Monte Carlo run.
- **Parallelism**: `multiprocessing.Pool` over MC runs; each run is single-threaded.
- **Storage**: every run produces a tidy `pandas.DataFrame` with `(day, age_group, group_type, S, I, R, utility, hazard_contribution)`.
- **No persistence layer in v1**; results stay in memory.

## 12. Verification strategy

| Test | Expected outcome |
|---|---|
| Set `p = 0` | No infections; θ = 1, e = 0; max welfare |
| Set `V = 0` | No behavioral response; θ = 1, e = 0; uncontrolled SIR |
| Set `V → ∞` | Full abstention; θ → 0, e → 1; no infections |
| Disable all groups except one | Recovers single-group SIR with `R₀ = m̄·p/γ` |
| Deterministic limit (`N → ∞`, average over MC) | Should approach the ODE prediction |
| Mask mandate (lower `p` only) | Decreases infections; θ slightly *rises* (risk compensation) |
| School closure | Decreases infections; younger age groups bear utility loss |

## 13. Confirmed design decisions

| # | Question | Decision |
|---|---|---|
| 1 | Population size for v1 | **10,000 agents** |
| 2 | Number of age groups | **7 decadal bins** (0–9, 10–19, 20–29, 30–39, 40–49, 50–59, 60+) |
| 3 | `α_g` calibration anchor | **Normalize to no-disease baseline** — `θ* = 1` when `I = 0` |
| 4 | Vaccination priority | **Eldest-first only** (single rule for v1) |
| 5 | Infection cost timing | **Lump-sum** on day of infection; allocated per day only for time-series visualization |
| 6 | Infected agents' attendance | **30% uniform reduction** across all groups; `e = 0` |
| 7 | Vaccine efficacy | **Single scalar `v_eff`** (default 0.8) applied uniformly |
| 8 | Spillover-cost defaults | **Non-zero, documented sources**, swept ±50% as sensitivity |

All defaults are written into the parameters table (section 7) and the functional-form table (section 6).

## 14. Future extensions (v2 candidates)

- Forward-looking agents (perfect foresight equilibrium with fixed-point solver).
- SEIR with presymptomatic infectiousness.
- Vaccination compartment and adoption dynamics.
- Multiple linked populations / mobility.
- Pathogen-specific calibration (COVID, influenza).
- Behavioral fatigue (`V` shrinks slowly with cumulative exposure).
- Heterogeneous compliance (fraction non-responders).
- Targeted interventions on the kin layer (bubble policies as data).

## 15. Success criteria

The model is successful if it can answer questions of the form:

- *"What is the welfare cost of an X-week school closure compared to its benefit in averted infections, by age group?"*
- *"Does a mask mandate at level e produce more welfare than a contact restriction at level θ that yields the same infection reduction?"*
- *"How does the optimal intervention mix depend on disease severity `V` and infectiousness `p`?"*

For each of these, the model produces:
- A point estimate (mean over Monte Carlo).
- An uncertainty band (quantiles).
- A distributional breakdown (by age, by group type).
- A decomposition (utility lost from behavioral response vs. from illness vs. from direct policy cost).
