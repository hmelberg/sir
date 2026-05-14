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
    # Tuned for default-config p_g_baseline (R0 ≈ 2.5 calibration). With small
    # alpha override + V=3 at prev=0.1, both controls are interior: theta
    # corners at 0 and e ≈ 0.51 (above the 0.5 threshold the test asserts).
    # Higher V saturates e to 1 → RHS of theta-FOC becomes 0 → theta returns
    # to 1. Lower V or lower prev keeps e below 0.5.
    cfg_orig = default_config()
    cfg = type(cfg_orig)(**{**cfg_orig.__dict__, "V_a": (3.0,) * 7})
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
    # Tuned for default-config p_g_baseline (R0 ≈ 2.5 calibration). With small
    # alpha + prev=0.005, age 6 (V_a=30) crosses into corner-theta + interior-e
    # while age 0 (V_a=0.5) stays at theta=1 with near-zero e. Higher prev
    # pushes e[6] to saturation which returns theta[6] to 1; lower prev mutes
    # the differential entirely.
    cfg = _small_alpha(default_config())
    prevalence_by_group_type = {gt: 0.005 for gt in GroupType}
    theta, e = solve_foc_by_age(cfg, prevalence_by_group_type)
    # Age bin 6 (60+) has V_a = 30; age bin 0 (0-9) has V_a = 0.5
    assert theta[6] < theta[0]
    assert e[6] > e[0]
