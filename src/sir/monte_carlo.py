"""Monte Carlo driver: parallel runs over independent seeds."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from sir.config import ScenarioConfig
from sir.healthcare import HealthcareConfig, HealthcareOutcomes, compute_healthcare_outcomes
from sir.interventions import Intervention
from sir.simulation import simulate


@dataclass
class MCResult:
    S_history: np.ndarray              # (n_runs, T+1)
    I_history: np.ndarray
    R_history: np.ndarray
    V_history: np.ndarray
    new_infections_history: np.ndarray
    new_infections_by_age: np.ndarray  # (n_runs, T+1, 7)
    welfare_totals: np.ndarray         # (n_runs,)
    welfare_components: dict[str, np.ndarray]  # each (n_runs,)
    healthcare_outcomes: list[HealthcareOutcomes] | None = None


def _one_run(
    args: tuple,
) -> tuple:
    cfg, interventions, seed, initial_infected, hc_cfg = args
    rng = np.random.default_rng(seed)
    result = simulate(cfg, interventions, rng, initial_infected=initial_infected)
    hc = compute_healthcare_outcomes(result, hc_cfg) if hc_cfg is not None else None
    return (
        result.S_history,
        result.I_history,
        result.R_history,
        result.V_history,
        result.new_infections_history,
        result.new_infections_by_age,
        result.welfare.total_welfare(),
        dict(result.welfare.totals),
        hc,
    )


def run_mc(
    cfg: ScenarioConfig,
    interventions: Iterable[Intervention],
    n_runs: int,
    base_seed: int = 0,
    initial_infected: int = 10,
    parallel: bool = True,
    healthcare: HealthcareConfig | None = None,
) -> MCResult:
    interventions_list = list(interventions)
    args_list = [
        (cfg, interventions_list, base_seed + i, initial_infected, healthcare)
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
    NI_age = np.stack([r[5] for r in results])
    W = np.array([r[6] for r in results])

    components_keys = list(results[0][7].keys())
    components = {
        k: np.array([r[7][k] for r in results]) for k in components_keys
    }
    hc_outcomes = [r[8] for r in results] if healthcare is not None else None

    return MCResult(
        S_history=S,
        I_history=I,
        R_history=R,
        V_history=V,
        new_infections_history=NI,
        new_infections_by_age=NI_age,
        welfare_totals=W,
        welfare_components=components,
        healthcare_outcomes=hc_outcomes,
    )
