import numpy as np

from sir.config import default_config
from sir.constants import GroupType
from sir.foc import solve_foc_by_age


# Small-alpha override used in tests that need interior theta solutions.
# Default alphas in `default_config` give utility ~0.73 at theta=1, which
# dominates the infection cost V*A for any realistic V/A — so theta corners
# at 1. To exercise the interior-θ branch of the FOC, we override α to
# values comparable to the infection-cost scale.
def _small_alpha(cfg):
    return type(cfg)(**{
        **cfg.__dict__,
        "group_alpha": {gt: 0.01 for gt in GroupType},
    })


def test_zero_prevalence_gives_theta_one_e_zero():
    cfg = default_config()
    prevalence_by_group_type = {gt: 0.0 for gt in GroupType}
    theta, e = solve_foc_by_age(cfg, prevalence_by_group_type)
    assert np.allclose(theta, 1.0, atol=1e-6)
    assert np.allclose(e, 0.0, atol=1e-6)


def test_zero_V_gives_theta_one_e_zero():
    cfg_orig = default_config()
    cfg = type(cfg_orig)(**{**cfg_orig.__dict__, "V_a": (0.0,) * 7})
    prevalence_by_group_type = {gt: 0.5 for gt in GroupType}
    theta, e = solve_foc_by_age(cfg, prevalence_by_group_type)
    assert np.allclose(theta, 1.0, atol=1e-6)
    assert np.allclose(e, 0.0, atol=1e-6)


def test_high_V_drives_theta_down_and_e_up():
    # Tuned for default-config p_g_baseline calibration; see Step 1 in commit
    # history. Under the stronger (R0 ≈ 2.5) calibration, A is ~5x larger, so
    # the same regime that V=100 reached previously is reached at V=20: e*
    # lands at ~0.68 (interior, not saturated) and theta corners at 0. At much
    # higher V (e.g. 30+), e* saturates to 1, which makes (1-e*)*A = 0 in the
    # theta-FOC RHS so theta returns to 1. V=20 is the regime where both
    # controls are active.
    cfg_orig = default_config()
    cfg = type(cfg_orig)(**{**cfg_orig.__dict__, "V_a": (20.0,) * 7})
    cfg = _small_alpha(cfg)
    prevalence_by_group_type = {gt: 0.1 for gt in GroupType}
    theta, e = solve_foc_by_age(cfg, prevalence_by_group_type)
    assert (theta < 0.5).all()
    assert (e > 0.5).all()


def test_e_capped_at_one():
    cfg_orig = default_config()
    cfg = type(cfg_orig)(**{**cfg_orig.__dict__, "V_a": (1e10,) * 7})
    prevalence_by_group_type = {gt: 0.5 for gt in GroupType}
    theta, e = solve_foc_by_age(cfg, prevalence_by_group_type)
    assert (e <= 1.0).all()
    assert (e >= 0.0).all()


def test_older_agents_react_more_at_same_prevalence():
    # Tuned for default-config p_g_baseline calibration; see Step 2 in commit
    # history. Under the stronger (R0 ≈ 2.5) calibration, A is ~5x larger, so
    # the differential-response regime moves from prev=0.1 down to prev=0.05.
    # At prev=0.05 with small alpha, age 6 (V_a=30) has V*A large enough to
    # corner theta at 0 while age 0 (V_a=0.5) stays at theta=1. e[6] ~ 0.51
    # vs e[0] ~ 0.009.
    cfg = _small_alpha(default_config())
    prevalence_by_group_type = {gt: 0.05 for gt in GroupType}
    theta, e = solve_foc_by_age(cfg, prevalence_by_group_type)
    # Age bin 6 (60+) has V_a = 30; age bin 0 (0-9) has V_a = 0.5
    assert theta[6] < theta[0]
    assert e[6] > e[0]
