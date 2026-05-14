"""Probabilistic sensitivity analysis (PSA).

Workflow:
  1. Load TOML spec -> list[ParameterDistribution]
  2. For each PSA sample: draw values, apply to base config -> ScenarioConfig
  3. Run n_mc_per_sample stochastic MC runs at that config
  4. Aggregate per-sample summaries -> PSAResult with quantile methods
"""

from __future__ import annotations

import tomllib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

import numpy as np

from sir.config import ScenarioConfig
from sir.constants import AGE_BIN_LABELS, GroupType
from sir.interventions import Intervention
from sir.simulation import simulate


# -------- Parameter distributions --------


@dataclass
class ParameterDistribution:
    name: str
    dist: str
    params: dict

    def sample(self, rng: np.random.Generator) -> float:
        if self.dist == "fixed":
            return float(self.params["value"])
        if self.dist == "normal":
            return float(rng.normal(self.params["mean"], self.params["sd"]))
        if self.dist == "lognormal":
            return float(rng.lognormal(self.params["mu"], self.params["sigma"]))
        if self.dist == "uniform":
            return float(rng.uniform(self.params["lo"], self.params["hi"]))
        if self.dist == "beta":
            return float(rng.beta(self.params["a"], self.params["b"]))
        if self.dist == "triangular":
            return float(rng.triangular(self.params["lo"], self.params["mode"], self.params["hi"]))
        raise ValueError(f"Unknown distribution: {self.dist}")


def ci_to_distribution(name: str, ci95: list[float], family: str = "normal") -> ParameterDistribution:
    """Convert a 95% CI [lo, hi] to a fitted distribution."""
    lo, hi = ci95
    if family == "normal":
        mean = (lo + hi) / 2
        sd = (hi - lo) / (2 * 1.96)
        return ParameterDistribution(name, "normal", {"mean": mean, "sd": sd})
    if family == "lognormal":
        # Fit so that exp(mu ± 1.96 sigma) = lo, hi
        mu = (np.log(lo) + np.log(hi)) / 2
        sigma = (np.log(hi) - np.log(lo)) / (2 * 1.96)
        return ParameterDistribution(name, "lognormal", {"mu": mu, "sigma": sigma})
    if family == "beta":
        # Method of moments: solve for (a, b) given mean and variance
        mean = (lo + hi) / 2
        sd = (hi - lo) / (2 * 1.96)
        var = sd ** 2
        if var <= 0 or mean <= 0 or mean >= 1:
            raise ValueError(f"Beta CI infeasible for {name}: lo={lo}, hi={hi}")
        common = mean * (1 - mean) / var - 1
        a = mean * common
        b = (1 - mean) * common
        return ParameterDistribution(name, "beta", {"a": a, "b": b})
    raise ValueError(f"Unknown family for CI shorthand: {family}")


# -------- Spec loading --------


def load_psa_spec(path: Path | str) -> list[ParameterDistribution]:
    path = Path(path)
    with open(path, "rb") as f:
        data = tomllib.load(f)
    out: list[ParameterDistribution] = []
    for name, body in data.items():
        if "ci95" in body:
            family = body.get("family", "normal")
            out.append(ci_to_distribution(name, body["ci95"], family))
        else:
            dist = body["dist"]
            params = {k: v for k, v in body.items() if k != "dist"}
            out.append(ParameterDistribution(name, dist, params))
    return out


# -------- Sampling and application --------


def draw_sample(spec: list[ParameterDistribution], rng: np.random.Generator) -> dict[str, float]:
    return {d.name: d.sample(rng) for d in spec}


_AGE_BIN_INDEX = {label: i for i, label in enumerate(AGE_BIN_LABELS)}
_GROUP_TYPE_INDEX = {gt.name: gt for gt in GroupType}


def apply_sample(base: ScenarioConfig, sample: dict[str, float]) -> ScenarioConfig:
    """Return a new ScenarioConfig with `sample` overrides applied.

    Path syntax:
      - "gamma"                          -> scalar attribute
      - "V_a.60+"                        -> tuple element by age-bin label
      - "group_p_baseline.HOUSEHOLD"     -> mapping element by GroupType name
    """
    scalar_overrides: dict[str, float] = {}
    v_a_list = list(base.V_a)
    group_alpha = dict(base.group_alpha)
    group_p = dict(base.group_p_baseline)
    group_m = dict(base.group_m_bar)
    v_a_touched = False
    group_alpha_touched = False
    group_p_touched = False
    group_m_touched = False

    for path, value in sample.items():
        if "." not in path:
            if not hasattr(base, path):
                raise KeyError(f"Unknown scalar parameter: {path}")
            scalar_overrides[path] = value
            continue
        parent, child = path.split(".", 1)
        if parent == "V_a":
            if child not in _AGE_BIN_INDEX:
                raise KeyError(f"Unknown age bin: {child}")
            v_a_list[_AGE_BIN_INDEX[child]] = value
            v_a_touched = True
        elif parent == "group_alpha":
            if child not in _GROUP_TYPE_INDEX:
                raise KeyError(f"Unknown group type: {child}")
            group_alpha[_GROUP_TYPE_INDEX[child]] = value
            group_alpha_touched = True
        elif parent == "group_p_baseline":
            if child not in _GROUP_TYPE_INDEX:
                raise KeyError(f"Unknown group type: {child}")
            group_p[_GROUP_TYPE_INDEX[child]] = value
            group_p_touched = True
        elif parent == "group_m_bar":
            if child not in _GROUP_TYPE_INDEX:
                raise KeyError(f"Unknown group type: {child}")
            group_m[_GROUP_TYPE_INDEX[child]] = value
            group_m_touched = True
        else:
            raise KeyError(f"Unknown parameter path: {path}")

    kwargs: dict = dict(scalar_overrides)
    if v_a_touched:
        kwargs["V_a"] = tuple(v_a_list)
    if group_alpha_touched:
        kwargs["group_alpha"] = group_alpha
    if group_p_touched:
        kwargs["group_p_baseline"] = group_p
    if group_m_touched:
        kwargs["group_m_bar"] = group_m
    return replace(base, **kwargs)


# -------- PSA runner --------


@dataclass
class PSAResult:
    samples: list[dict[str, float]]
    welfare_per_sample: np.ndarray
    peak_I_per_sample: np.ndarray
    cumulative_infections_per_sample: np.ndarray
    cumulative_60plus_per_sample: np.ndarray
    welfare_components_per_sample: dict[str, np.ndarray]

    def welfare_ci(self, alpha: float = 0.05) -> tuple[float, float, float]:
        lo = float(np.percentile(self.welfare_per_sample, 100 * alpha / 2))
        med = float(np.percentile(self.welfare_per_sample, 50))
        hi = float(np.percentile(self.welfare_per_sample, 100 * (1 - alpha / 2)))
        return lo, med, hi

    def parameter_correlations(self, outcome: np.ndarray) -> dict[str, float]:
        """Spearman-like rank correlation of each parameter with an outcome array."""
        if not self.samples:
            return {}
        param_names = list(self.samples[0].keys())
        from scipy.stats import spearmanr
        out: dict[str, float] = {}
        for p in param_names:
            vals = np.array([s[p] for s in self.samples])
            if vals.std() == 0:
                out[p] = 0.0
            else:
                rho, _ = spearmanr(vals, outcome)
                out[p] = float(rho)
        return out


def _summarize_run(
    cfg: ScenarioConfig,
    interventions: list[Intervention],
    n_mc: int,
    seed: int,
    initial_infected: int,
) -> dict:
    """Run n_mc stochastic simulations at cfg; return mean summary."""
    welfare_list = []
    peak_I_list = []
    cum_inf_list = []
    cum_60_list = []
    components_acc: dict[str, list[float]] = {}
    for k in range(n_mc):
        rng = np.random.default_rng(seed * 1000 + k)
        result = simulate(cfg, interventions, rng, initial_infected=initial_infected)
        welfare_list.append(result.welfare.total_welfare())
        peak_I_list.append(int(result.I_history.max()))
        cum_inf = int(result.R_history[-1] + result.V_history[-1] - result.V_history[0])
        cum_inf_list.append(cum_inf)
        # Cumulative 60+ infections: count agents in age_bin 6 who are R at end
        ages = result.final_world.age_bin
        is_60plus = ages == 6
        was_infected = (result.final_world.state[is_60plus] == 3) | (result.final_world.state[is_60plus] == 1)
        cum_60_list.append(int(was_infected.sum()))
        for ck, cv in result.welfare.totals.items():
            components_acc.setdefault(ck, []).append(cv)
    return {
        "welfare": float(np.mean(welfare_list)),
        "peak_I": float(np.mean(peak_I_list)),
        "cum_inf": float(np.mean(cum_inf_list)),
        "cum_60": float(np.mean(cum_60_list)),
        "components": {k: float(np.mean(v)) for k, v in components_acc.items()},
    }


def _outer_iteration(args: tuple) -> tuple[dict[str, float], dict]:
    base_cfg, spec, interventions, n_mc, seed, initial_infected = args
    rng = np.random.default_rng(seed)
    sample = draw_sample(spec, rng)
    cfg = apply_sample(base_cfg, sample)
    summary = _summarize_run(cfg, interventions, n_mc, seed, initial_infected)
    return sample, summary


def run_psa(
    base_cfg: ScenarioConfig,
    spec: list[ParameterDistribution],
    interventions: Iterable[Intervention],
    n_psa_samples: int,
    n_mc_per_sample: int,
    base_seed: int = 0,
    initial_infected: int = 10,
    parallel: bool = True,
) -> PSAResult:
    interventions_list = list(interventions)
    args_list = [
        (base_cfg, spec, interventions_list, n_mc_per_sample, base_seed + i, initial_infected)
        for i in range(n_psa_samples)
    ]
    if parallel and n_psa_samples > 1:
        with ProcessPoolExecutor() as pool:
            outputs = list(pool.map(_outer_iteration, args_list))
    else:
        outputs = [_outer_iteration(a) for a in args_list]

    samples = [o[0] for o in outputs]
    welfare = np.array([o[1]["welfare"] for o in outputs])
    peak_I = np.array([o[1]["peak_I"] for o in outputs])
    cum_inf = np.array([o[1]["cum_inf"] for o in outputs])
    cum_60 = np.array([o[1]["cum_60"] for o in outputs])
    component_keys = list(outputs[0][1]["components"].keys())
    components = {
        k: np.array([o[1]["components"][k] for o in outputs])
        for k in component_keys
    }
    return PSAResult(
        samples=samples,
        welfare_per_sample=welfare,
        peak_I_per_sample=peak_I,
        cumulative_infections_per_sample=cum_inf,
        cumulative_60plus_per_sample=cum_60,
        welfare_components_per_sample=components,
    )
