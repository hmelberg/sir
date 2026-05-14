from pathlib import Path

import numpy as np
import pytest

from sir.config import default_config
from sir.constants import AGE_BIN_LABELS, GroupType
from sir.psa import (
    ParameterDistribution,
    PSAResult,
    apply_sample,
    ci_to_distribution,
    draw_sample,
    load_psa_spec,
    run_psa,
)


def test_fixed_distribution_returns_value():
    d = ParameterDistribution("gamma", "fixed", {"value": 0.2})
    rng = np.random.default_rng(0)
    assert d.sample(rng) == 0.2


def test_normal_distribution_centered_correctly():
    d = ParameterDistribution("x", "normal", {"mean": 5.0, "sd": 0.0})
    rng = np.random.default_rng(0)
    assert d.sample(rng) == 5.0


def test_uniform_distribution_in_bounds():
    d = ParameterDistribution("x", "uniform", {"lo": 0.0, "hi": 1.0})
    rng = np.random.default_rng(0)
    for _ in range(100):
        assert 0.0 <= d.sample(rng) <= 1.0


def test_beta_distribution_in_unit_interval():
    d = ParameterDistribution("x", "beta", {"a": 2.0, "b": 5.0})
    rng = np.random.default_rng(0)
    for _ in range(100):
        v = d.sample(rng)
        assert 0.0 <= v <= 1.0


def test_lognormal_strictly_positive():
    d = ParameterDistribution("x", "lognormal", {"mu": 0.0, "sigma": 1.0})
    rng = np.random.default_rng(0)
    for _ in range(100):
        assert d.sample(rng) > 0.0


def test_triangular_within_bounds():
    d = ParameterDistribution("x", "triangular", {"lo": 1.0, "mode": 2.0, "hi": 3.0})
    rng = np.random.default_rng(0)
    for _ in range(100):
        v = d.sample(rng)
        assert 1.0 <= v <= 3.0


def test_unknown_distribution_raises():
    d = ParameterDistribution("x", "weibull", {"a": 1.0})
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        d.sample(rng)


def test_ci_to_normal_conversion():
    d = ci_to_distribution("x", [0.0, 10.0], family="normal")
    assert d.dist == "normal"
    assert np.isclose(d.params["mean"], 5.0)
    # 95% CI under normal: half-width = 1.96 * sd
    assert np.isclose(d.params["sd"], (10.0 - 0.0) / (2 * 1.96))


def test_ci_to_lognormal_conversion():
    d = ci_to_distribution("x", [1.0, 100.0], family="lognormal")
    assert d.dist == "lognormal"
    # mu = mean of log-bounds
    assert np.isclose(d.params["mu"], (np.log(1.0) + np.log(100.0)) / 2)


def test_load_psa_spec_parses_example(tmp_path: Path):
    spec_path = Path(__file__).parent.parent / "configs" / "psa_example.toml"
    spec = load_psa_spec(spec_path)
    names = {d.name for d in spec}
    assert "gamma" in names
    assert "kappa" in names
    assert "V_a.60+" in names
    assert "group_p_baseline.HOUSEHOLD" in names
    # CI shorthand was converted
    c_vax = [d for d in spec if d.name == "c_vax"][0]
    assert c_vax.dist in ("normal", "lognormal", "beta")


def test_draw_sample_reproducible_with_same_seed():
    spec = [
        ParameterDistribution("gamma", "lognormal", {"mu": -2.0, "sigma": 0.1}),
        ParameterDistribution("kappa", "uniform", {"lo": 0.5, "hi": 2.0}),
    ]
    s1 = draw_sample(spec, np.random.default_rng(42))
    s2 = draw_sample(spec, np.random.default_rng(42))
    assert s1 == s2


def test_apply_sample_overrides_scalar():
    base = default_config()
    sample = {"gamma": 0.25, "kappa": 1.5}
    new_cfg = apply_sample(base, sample)
    assert new_cfg.gamma == 0.25
    assert new_cfg.kappa == 1.5
    # Untouched parameters preserved
    assert new_cfg.v_eff == base.v_eff


def test_apply_sample_overrides_v_a_element():
    base = default_config()
    # 60+ is age bin 6, label "60+"
    sample = {"V_a.60+": 99.0}
    new_cfg = apply_sample(base, sample)
    assert new_cfg.V_a[6] == 99.0
    # Other bins unchanged
    for i in range(6):
        assert new_cfg.V_a[i] == base.V_a[i]


def test_apply_sample_overrides_group_mapping_element():
    base = default_config()
    sample = {"group_p_baseline.HOUSEHOLD": 0.123}
    new_cfg = apply_sample(base, sample)
    assert new_cfg.group_p_baseline[GroupType.HOUSEHOLD] == 0.123
    # Other group types unchanged
    assert new_cfg.group_p_baseline[GroupType.COMMUNITY] == base.group_p_baseline[GroupType.COMMUNITY]


def test_apply_sample_unknown_path_raises():
    base = default_config()
    with pytest.raises(KeyError):
        apply_sample(base, {"nonexistent": 1.0})


def test_run_psa_returns_correct_shape():
    base = default_config()
    # Make config small for speed
    base = type(base)(**{**base.__dict__, "N": 500, "T": 30})
    spec = [
        ParameterDistribution("gamma", "uniform", {"lo": 0.1, "hi": 0.2}),
    ]
    result = run_psa(
        base, spec, interventions=[],
        n_psa_samples=3, n_mc_per_sample=2,
        base_seed=0, initial_infected=5,
    )
    assert isinstance(result, PSAResult)
    assert result.welfare_per_sample.shape == (3,)
    assert len(result.samples) == 3


def test_psa_result_quantiles():
    res = PSAResult(
        samples=[{"x": 1.0}, {"x": 2.0}, {"x": 3.0}, {"x": 4.0}, {"x": 5.0}],
        welfare_per_sample=np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        peak_I_per_sample=np.array([100, 200, 300, 400, 500]),
        cumulative_infections_per_sample=np.array([1000, 2000, 3000, 4000, 5000]),
        cumulative_60plus_per_sample=np.array([100, 200, 300, 400, 500]),
        welfare_components_per_sample={"infection_cost": np.array([10.0]*5)},
    )
    lo, med, hi = res.welfare_ci(alpha=0.10)
    assert np.isclose(med, 30.0)
    assert np.isclose(lo, np.percentile(res.welfare_per_sample, 5))
    assert np.isclose(hi, np.percentile(res.welfare_per_sample, 95))
