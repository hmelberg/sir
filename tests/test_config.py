import pytest

from sir.config import ScenarioConfig, default_config
from sir.constants import GroupType


def test_default_config_has_expected_population():
    cfg = default_config()
    assert cfg.N == 10_000
    assert cfg.T == 365


def test_default_config_has_all_seven_group_types():
    cfg = default_config()
    assert set(cfg.group_p_baseline.keys()) == set(GroupType)
    assert set(cfg.group_alpha.keys()) == set(GroupType)
    assert set(cfg.group_m_bar.keys()) == set(GroupType)


def test_default_config_voluntary_groups_listed():
    cfg = default_config()
    assert GroupType.RECURRING_LEISURE in cfg.voluntary_groups
    assert GroupType.COMMUNITY in cfg.voluntary_groups
    assert GroupType.KIN in cfg.voluntary_groups
    assert GroupType.HOUSEHOLD not in cfg.voluntary_groups
    assert GroupType.WORKPLACE not in cfg.voluntary_groups
    assert GroupType.SCHOOL not in cfg.voluntary_groups


def test_default_config_has_seven_v_a_values():
    cfg = default_config()
    assert len(cfg.V_a) == 7


def test_default_config_vaccine_efficacy_is_scalar():
    cfg = default_config()
    assert 0 <= cfg.v_eff <= 1


def test_config_rejects_negative_gamma():
    with pytest.raises(ValueError):
        ScenarioConfig(
            N=100, T=10, gamma=-0.1, kappa=1.0, v_eff=0.8, c_vax=0.01,
            sick_attendance_multiplier=0.3, V_a=(1.0,)*7,
            group_alpha={gt: 1.0 for gt in GroupType},
            group_p_baseline={gt: 0.01 for gt in GroupType},
            group_m_bar={gt: 1.0 for gt in GroupType},
            voluntary_groups=frozenset({GroupType.COMMUNITY}),
        )
