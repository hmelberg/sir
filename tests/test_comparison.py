import numpy as np
import pandas as pd

from sir.comparison import (
    ComparisonResult,
    UncertainIntervention,
    contact_reduction,
    run_comparison,
    transmission_reduction,
    vaccination,
)
from sir.config import default_config
from sir.constants import DiseaseState, GroupType
from sir.distributions import dist
from sir.healthcare import default_healthcare_config
from sir.interventions import Intervention, apply_interventions
from sir.world import build_world


def test_comparison_result_is_constructible():
    cr = ComparisonResult(
        cfg=None,
        interventions_resolved=[],
        baseline_results=[],
        treatment_results=[],
        baseline_healthcare=None,
        treatment_healthcare=None,
        intervention_samples=None,
    )
    assert cr.baseline_results == []
    assert cr.treatment_results == []


def test_transmission_reduction_returns_intervention():
    inter = transmission_reduction(p_factor=0.5, window=(10, 30))
    assert isinstance(inter, Intervention)
    assert inter.start_day == 10
    assert inter.end_day == 30


def test_transmission_reduction_applies_p_factor_to_all_groups():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    inter = transmission_reduction(p_factor=0.3, window=(0, 10))
    apply_interventions(world, cfg, day=5, interventions=[inter], rng=rng)
    assert np.allclose(world.group_p_mult, 0.3)


def test_transmission_reduction_with_targets():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    inter = transmission_reduction(
        p_factor=0.4, window=(0, 10),
        targets={GroupType.SCHOOL, GroupType.WORKPLACE},
    )
    apply_interventions(world, cfg, day=5, interventions=[inter], rng=rng)
    school_mask = world.group_type == GroupType.SCHOOL
    workplace_mask = world.group_type == GroupType.WORKPLACE
    other_mask = ~(school_mask | workplace_mask)
    assert np.allclose(world.group_p_mult[school_mask], 0.4)
    assert np.allclose(world.group_p_mult[workplace_mask], 0.4)
    assert np.allclose(world.group_p_mult[other_mask], 1.0)


def test_transmission_reduction_inactive_outside_window():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    inter = transmission_reduction(p_factor=0.3, window=(10, 20))
    apply_interventions(world, cfg, day=5, interventions=[inter], rng=rng)
    assert np.allclose(world.group_p_mult, 1.0)




def test_contact_reduction_returns_intervention():
    inter = contact_reduction(
        people=1000, events_per_week=3, encounters_per_event=5,
        window=(10, 30),
    )
    assert isinstance(inter, Intervention)


def test_contact_reduction_no_target_scales_all_groups_uniformly():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    # Aggressive: X*Y*Z = 8000 * 7 * 8 = 448000 / 7 / 10000 = 6.4 reduction
    inter = contact_reduction(
        people=8000, events_per_week=7, encounters_per_event=8,
        window=(0, 10),
    )
    apply_interventions(world, cfg, day=5, interventions=[inter], rng=rng)
    # All groups should have the same attendance multiplier (uniform scaling)
    mults = world.group_attendance_mult
    assert mults.std() < 1e-9
    # And it should be < 1 (i.e., something was reduced)
    assert mults[0] < 1.0


def test_contact_reduction_targeted_only_affects_one_group_type():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    inter = contact_reduction(
        people=2500, events_per_week=5, encounters_per_event=10,
        window=(0, 10), target=GroupType.SCHOOL,
    )
    apply_interventions(world, cfg, day=5, interventions=[inter], rng=rng)
    school_mask = world.group_type == GroupType.SCHOOL
    other_mask = ~school_mask
    # Schools have reduced attendance, others are unchanged
    assert (world.group_attendance_mult[school_mask] < 1.0).all()
    assert np.allclose(world.group_attendance_mult[other_mask], 1.0)


def test_contact_reduction_zero_people_no_effect():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    inter = contact_reduction(
        people=0, events_per_week=3, encounters_per_event=5,
        window=(0, 10),
    )
    apply_interventions(world, cfg, day=5, interventions=[inter], rng=rng)
    assert np.allclose(world.group_attendance_mult, 1.0)




def test_run_comparison_returns_result():
    cfg = default_config()
    # Make small for speed
    cfg = type(cfg)(**{**cfg.__dict__, "N": 500, "T": 30})
    interventions = [transmission_reduction(p_factor=0.5, window=(5, 25))]
    result = run_comparison(
        cfg, interventions, n_runs=2, base_seed=0,
        initial_infected=5, parallel=False,
    )
    assert len(result.baseline_results) == 1
    assert len(result.treatment_results) == 1
    assert result.baseline_results[0].I_history.shape == (2, cfg.T + 1)
    assert result.treatment_results[0].I_history.shape == (2, cfg.T + 1)


def test_run_comparison_no_interventions_baseline_equals_treatment():
    cfg = default_config()
    cfg = type(cfg)(**{**cfg.__dict__, "N": 500, "T": 20})
    result = run_comparison(
        cfg, interventions=[], n_runs=2, base_seed=0,
        initial_infected=5, parallel=False,
    )
    # With empty interventions list, treatment ≡ baseline
    np.testing.assert_array_equal(
        result.baseline_results[0].I_history,
        result.treatment_results[0].I_history,
    )


def test_run_comparison_with_healthcare():
    cfg = default_config()
    cfg = type(cfg)(**{**cfg.__dict__, "N": 500, "T": 60})
    hc = default_healthcare_config()
    result = run_comparison(
        cfg, interventions=[], n_runs=2, base_seed=0,
        initial_infected=5, parallel=False, healthcare=hc,
    )
    assert result.baseline_healthcare is not None
    assert result.treatment_healthcare is not None
    assert len(result.baseline_healthcare) == 1
    assert len(result.baseline_healthcare[0]) == 2




def test_contact_reduction_with_distribution_returns_uncertain():
    from sir.comparison import UncertainIntervention
    inter = contact_reduction(
        people=dist.uniform(800, 1200),
        events_per_week=3,
        encounters_per_event=5,
        window=(10, 30),
    )
    assert isinstance(inter, UncertainIntervention)


def test_run_comparison_with_uncertainty():
    cfg = default_config()
    cfg = type(cfg)(**{**cfg.__dict__, "N": 500, "T": 30})
    inter = transmission_reduction(
        p_factor=dist.uniform(0.3, 0.7), window=(5, 25),
    )
    result = run_comparison(
        cfg, interventions=[inter], n_runs=2, n_psa_samples=3,
        base_seed=0, initial_infected=5, parallel=False,
    )
    assert len(result.baseline_results) == 3
    assert len(result.treatment_results) == 3
    assert result.intervention_samples is not None
    assert len(result.intervention_samples) == 3


def test_uncertain_intervention_samples_are_reproducible():
    cfg = default_config()
    cfg = type(cfg)(**{**cfg.__dict__, "N": 500, "T": 20})
    inter = transmission_reduction(
        p_factor=dist.uniform(0.3, 0.7), window=(5, 15),
    )
    r1 = run_comparison(
        cfg, interventions=[inter], n_runs=2, n_psa_samples=3,
        base_seed=42, initial_infected=5, parallel=False,
    )
    r2 = run_comparison(
        cfg, interventions=[inter], n_runs=2, n_psa_samples=3,
        base_seed=42, initial_infected=5, parallel=False,
    )
    # Same seeds → same samples
    for s1, s2 in zip(r1.intervention_samples, r2.intervention_samples):
        for k in s1:
            assert s1[k] == s2[k]





def test_outcome_deltas_returns_dataframe():
    cfg = default_config()
    cfg = type(cfg)(**{**cfg.__dict__, "N": 500, "T": 30})
    hc = default_healthcare_config()
    inter = transmission_reduction(p_factor=0.5, window=(5, 25))
    result = run_comparison(
        cfg, interventions=[inter], n_runs=2, base_seed=0,
        initial_infected=5, parallel=False, healthcare=hc,
    )
    df = result.outcome_deltas()
    assert isinstance(df, pd.DataFrame)
    expected_cols = {"baseline", "treatment", "delta", "delta_pct", "ci_lo", "ci_hi"}
    assert expected_cols.issubset(df.columns)
    expected_outcomes = {
        "peak_I", "final_attack_rate", "welfare",
        "total_deaths", "total_yll", "peak_hosp_prev", "peak_icu_prev",
    }
    assert expected_outcomes.issubset(set(df.index))


def test_outcome_deltas_without_healthcare_omits_healthcare_outcomes():
    cfg = default_config()
    cfg = type(cfg)(**{**cfg.__dict__, "N": 500, "T": 30})
    result = run_comparison(
        cfg, interventions=[], n_runs=2, base_seed=0,
        initial_infected=5, parallel=False,
    )
    df = result.outcome_deltas()
    # Health outcomes should NOT be in the table when healthcare is None
    assert "total_deaths" not in df.index
    assert "peak_I" in df.index


# -------------------- Vaccination --------------------


def test_vaccination_returns_intervention():
    inter = vaccination(doses_per_day=50, window=(10, 30))
    assert isinstance(inter, Intervention)
    assert inter.start_day == 10
    assert inter.end_day == 30


def test_vaccination_default_uptake_is_full():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    inter = vaccination(doses_per_day=100, window=(0, 5))
    apply_interventions(world, cfg, day=2, interventions=[inter], rng=rng)
    assert (world.state == DiseaseState.V).sum() == 100


def test_vaccination_uptake_scales_effective_doses():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    inter = vaccination(doses_per_day=100, window=(0, 5), uptake_rate=0.6)
    apply_interventions(world, cfg, day=2, interventions=[inter], rng=rng)
    # 100 doses × 0.6 uptake = 60 actually vaccinated
    assert (world.state == DiseaseState.V).sum() == 60


def test_vaccination_uptake_zero_no_one_vaccinated():
    cfg = default_config()
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng)
    inter = vaccination(doses_per_day=100, window=(0, 5), uptake_rate=0.0)
    apply_interventions(world, cfg, day=2, interventions=[inter], rng=rng)
    assert (world.state == DiseaseState.V).sum() == 0


def test_vaccination_with_distribution_returns_uncertain():
    inter = vaccination(
        doses_per_day=dist.uniform(40, 60),
        window=(10, 30),
    )
    assert isinstance(inter, UncertainIntervention)


def test_vaccination_uncertain_on_uptake():
    inter = vaccination(
        doses_per_day=50,
        window=(10, 30),
        uptake_rate=dist.beta(a=8, b=2),  # mean 0.8
    )
    assert isinstance(inter, UncertainIntervention)


def test_run_comparison_with_vaccination():
    cfg = default_config()
    cfg = type(cfg)(**{**cfg.__dict__, "N": 500, "T": 60})
    hc = default_healthcare_config()
    inter = vaccination(doses_per_day=10, window=(5, 50), uptake_rate=0.8)
    result = run_comparison(
        cfg, interventions=[inter], n_runs=2, base_seed=0,
        initial_infected=5, parallel=False, healthcare=hc,
    )
    # Vaccination should reduce or at least not increase deaths
    df = result.outcome_deltas()
    assert "total_deaths" in df.index
