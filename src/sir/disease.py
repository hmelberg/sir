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
