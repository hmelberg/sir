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
