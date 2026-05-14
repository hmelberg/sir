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
