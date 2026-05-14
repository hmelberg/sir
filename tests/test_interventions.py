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
