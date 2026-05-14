"""Fluent constructors for ParameterDistribution.

Lets users write `dist.normal(mean=5, sd=1)` instead of constructing
`ParameterDistribution("x", "normal", {"mean": 5, "sd": 1})` directly.
Names are anonymous by default ("inline-N"); for PSA reporting and
tornado plots, prefer named distributions or the TOML-based PSA spec.
"""

from __future__ import annotations

from sir.psa import ParameterDistribution, ci_to_distribution

_anon_counter = 0


def _anon_name() -> str:
    global _anon_counter
    _anon_counter += 1
    return f"inline-{_anon_counter}"


class _DistNamespace:
    def fixed(self, value: float) -> ParameterDistribution:
        return ParameterDistribution(_anon_name(), "fixed", {"value": float(value)})

    def normal(self, mean: float, sd: float) -> ParameterDistribution:
        return ParameterDistribution(_anon_name(), "normal", {"mean": float(mean), "sd": float(sd)})

    def lognormal(self, mu: float, sigma: float) -> ParameterDistribution:
        return ParameterDistribution(_anon_name(), "lognormal", {"mu": float(mu), "sigma": float(sigma)})

    def uniform(self, lo: float, hi: float) -> ParameterDistribution:
        return ParameterDistribution(_anon_name(), "uniform", {"lo": float(lo), "hi": float(hi)})

    def beta(self, a: float, b: float) -> ParameterDistribution:
        return ParameterDistribution(_anon_name(), "beta", {"a": float(a), "b": float(b)})

    def triangular(self, lo: float, mode: float, hi: float) -> ParameterDistribution:
        return ParameterDistribution(
            _anon_name(), "triangular",
            {"lo": float(lo), "mode": float(mode), "hi": float(hi)},
        )

    def ci95(self, lo: float, hi: float, family: str = "normal") -> ParameterDistribution:
        d = ci_to_distribution(_anon_name(), [float(lo), float(hi)], family=family)
        return d


dist = _DistNamespace()
