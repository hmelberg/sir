import numpy as np

from sir.comparison import ComparisonResult


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


from sir.comparison import transmission_reduction
from sir.constants import GroupType
from sir.config import default_config
from sir.interventions import Intervention, apply_interventions
from sir.world import build_world


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


from sir.comparison import contact_reduction


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
