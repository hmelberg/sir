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
    new_infections_by_age: np.ndarray
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
    new_inf_by_age_hist = np.zeros((T + 1, 7), dtype=np.int32)

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
        if new_infections.any():
            new_inf_by_age_hist[t + 1] = np.bincount(
                world.age_bin[new_infections], minlength=7
            )
        snapshot(t + 1)

    return SimResult(
        S_history=S_hist,
        I_history=I_hist,
        R_history=R_hist,
        V_history=V_hist,
        new_infections_history=new_inf_hist,
        new_infections_by_age=new_inf_by_age_hist,
        welfare=welfare,
        final_world=world,
    )
