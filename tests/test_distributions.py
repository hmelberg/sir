import numpy as np

from sir.distributions import dist
from sir.psa import ParameterDistribution


def test_fixed_returns_parameter_distribution():
    d = dist.fixed(1.5)
    assert isinstance(d, ParameterDistribution)
    assert d.dist == "fixed"
    rng = np.random.default_rng(0)
    assert d.sample(rng) == 1.5


def test_normal_constructor():
    d = dist.normal(mean=5.0, sd=2.0)
    assert d.dist == "normal"
    assert d.params["mean"] == 5.0
    assert d.params["sd"] == 2.0


def test_uniform_constructor():
    d = dist.uniform(0.0, 1.0)
    assert d.dist == "uniform"
    assert d.params["lo"] == 0.0
    assert d.params["hi"] == 1.0


def test_lognormal_constructor():
    d = dist.lognormal(mu=0.0, sigma=1.0)
    assert d.dist == "lognormal"
    assert d.params["mu"] == 0.0
    assert d.params["sigma"] == 1.0


def test_beta_constructor():
    d = dist.beta(a=2.0, b=5.0)
    assert d.dist == "beta"
    assert d.params["a"] == 2.0
    assert d.params["b"] == 5.0


def test_triangular_constructor():
    d = dist.triangular(lo=1.0, mode=2.0, hi=3.0)
    assert d.dist == "triangular"
    assert d.params == {"lo": 1.0, "mode": 2.0, "hi": 3.0}


def test_ci95_normal():
    d = dist.ci95(0.0, 10.0)
    assert d.dist == "normal"
    assert np.isclose(d.params["mean"], 5.0)


def test_ci95_lognormal():
    d = dist.ci95(1.0, 100.0, family="lognormal")
    assert d.dist == "lognormal"


def test_sampling_normal_is_reproducible():
    d = dist.normal(mean=10.0, sd=1.0)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    assert d.sample(rng1) == d.sample(rng2)
