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
