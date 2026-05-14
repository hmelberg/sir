"""High-level comparison wrapper: baseline (no interventions) vs treatment.

Wraps run_mc to produce a paired with-vs-without comparison. Supports
intervention-parameter uncertainty via PSA over distributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from sir.config import ScenarioConfig
from sir.constants import GroupType
from sir.healthcare import HealthcareConfig, HealthcareOutcomes
from sir.interventions import (
    Intervention,
    _apply_set_p_mult,
    _filter_group_type_in,
)
from sir.monte_carlo import MCResult, run_mc
from sir.psa import ParameterDistribution


@dataclass
class ComparisonResult:
    cfg: ScenarioConfig | None
    interventions_resolved: list[list[Intervention]]
    baseline_results: list[MCResult]
    treatment_results: list[MCResult]
    baseline_healthcare: list[list[HealthcareOutcomes]] | None
    treatment_healthcare: list[list[HealthcareOutcomes]] | None
    intervention_samples: list[dict[str, float]] | None

    def outcome_deltas(self) -> "pd.DataFrame":
        """Tidy DataFrame of with-vs-without deltas across PSA samples and MC runs.

        Rows: outcomes (peak_I, final_attack_rate, welfare, and healthcare outcomes
        if available). Columns: baseline (mean), treatment (mean), delta, delta_pct,
        ci_lo, ci_hi (95% interval).
        """
        rows = []

        def _add(name, base_vals, treat_vals):
            base_arr = np.asarray(base_vals, dtype=float)
            treat_arr = np.asarray(treat_vals, dtype=float)
            delta = treat_arr - base_arr
            base_mean = float(base_arr.mean())
            treat_mean = float(treat_arr.mean())
            delta_mean = float(delta.mean())
            delta_pct = (delta_mean / base_mean * 100) if base_mean != 0 else float("nan")
            ci_lo = float(np.percentile(delta, 2.5)) if delta.size > 1 else delta_mean
            ci_hi = float(np.percentile(delta, 97.5)) if delta.size > 1 else delta_mean
            rows.append({
                "outcome": name,
                "baseline": base_mean,
                "treatment": treat_mean,
                "delta": delta_mean,
                "delta_pct": delta_pct,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            })

        # Disease outcomes
        b_peak = np.concatenate([r.I_history.max(axis=1) for r in self.baseline_results])
        t_peak = np.concatenate([r.I_history.max(axis=1) for r in self.treatment_results])
        _add("peak_I", b_peak, t_peak)

        N = float(self.cfg.N) if self.cfg is not None else 1.0
        b_ar = np.concatenate([r.R_history[:, -1] / N for r in self.baseline_results])
        t_ar = np.concatenate([r.R_history[:, -1] / N for r in self.treatment_results])
        _add("final_attack_rate", b_ar, t_ar)

        b_w = np.concatenate([r.welfare_totals for r in self.baseline_results])
        t_w = np.concatenate([r.welfare_totals for r in self.treatment_results])
        _add("welfare", b_w, t_w)

        # Healthcare outcomes if available
        if self.baseline_healthcare is not None:
            b_d = np.array([
                hc.total_deaths
                for psa_runs in self.baseline_healthcare
                for hc in psa_runs
            ])
            t_d = np.array([
                hc.total_deaths
                for psa_runs in self.treatment_healthcare
                for hc in psa_runs
            ])
            _add("total_deaths", b_d, t_d)

            b_yll = np.array([
                hc.total_yll
                for psa_runs in self.baseline_healthcare
                for hc in psa_runs
            ])
            t_yll = np.array([
                hc.total_yll
                for psa_runs in self.treatment_healthcare
                for hc in psa_runs
            ])
            _add("total_yll", b_yll, t_yll)

            b_hp = np.array([
                hc.hosp_prev.max()
                for psa_runs in self.baseline_healthcare
                for hc in psa_runs
            ])
            t_hp = np.array([
                hc.hosp_prev.max()
                for psa_runs in self.treatment_healthcare
                for hc in psa_runs
            ])
            _add("peak_hosp_prev", b_hp, t_hp)

            b_ip = np.array([
                hc.icu_prev.max()
                for psa_runs in self.baseline_healthcare
                for hc in psa_runs
            ])
            t_ip = np.array([
                hc.icu_prev.max()
                for psa_runs in self.treatment_healthcare
                for hc in psa_runs
            ])
            _add("peak_icu_prev", b_ip, t_ip)

        df = pd.DataFrame(rows).set_index("outcome")
        return df


@dataclass
class UncertainIntervention:
    """A symbolic intervention spec with ParameterDistribution-valued parameters.

    Resolved into a concrete Intervention by `resolve(rng)`.
    """
    name: str
    builder: Callable[[dict], Intervention]
    params: dict[str, Any]  # each value is float or ParameterDistribution

    def resolve(self, rng: np.random.Generator) -> tuple[Intervention, dict[str, float]]:
        resolved = {}
        for k, v in self.params.items():
            if isinstance(v, ParameterDistribution):
                resolved[k] = v.sample(rng)
            else:
                resolved[k] = v
        sampled_only = {
            f"{self.name}.{k}": v
            for k, v in resolved.items()
            if isinstance(self.params[k], ParameterDistribution)
        }
        return self.builder(resolved), sampled_only


# ---------------- transmission_reduction ----------------

def _build_transmission_reduction_from_resolved(
    resolved: dict,
    window: tuple[int, int],
    targets: set[GroupType] | None,
    direct_cost_per_day: float,
) -> Intervention:
    return _build_transmission_reduction(
        resolved["p_factor"], window, targets, direct_cost_per_day,
    )


def transmission_reduction(
    p_factor,
    window: tuple[int, int],
    targets: set[GroupType] | None = None,
    direct_cost_per_day: float = 0.0,
):
    """Reduce per-meeting transmission by `p_factor` during the window.

    Returns Intervention if p_factor is a float, or UncertainIntervention if
    p_factor is a ParameterDistribution.
    """
    if isinstance(p_factor, ParameterDistribution):
        params = {"p_factor": p_factor}
        builder = partial(
            _build_transmission_reduction_from_resolved,
            window=window, targets=targets, direct_cost_per_day=direct_cost_per_day,
        )
        return UncertainIntervention(name="transmission_reduction", builder=builder, params=params)
    return _build_transmission_reduction(p_factor, window, targets, direct_cost_per_day)


def _build_transmission_reduction(
    p_factor: float, window: tuple[int, int],
    targets: set[GroupType] | None, direct_cost_per_day: float,
) -> Intervention:
    if targets is None:
        target_ints = np.array([int(gt) for gt in GroupType])
    else:
        target_ints = np.array([int(gt) for gt in targets])
    return Intervention(
        name=f"transmission_reduction_{p_factor:.2f}",
        start_day=window[0],
        end_day=window[1],
        target_filter=partial(_filter_group_type_in, target_ints=target_ints),
        apply_to_groups=partial(_apply_set_p_mult, factor=p_factor),
        direct_cost_per_day=direct_cost_per_day,
    )


# ---------------- contact_reduction ----------------

def _build_contact_reduction_from_resolved(
    resolved: dict,
    window: tuple[int, int],
    target: GroupType | None,
    spillover_cost_per_person_per_day: float,
) -> Intervention:
    return _build_contact_reduction(
        resolved["people"], resolved["events_per_week"],
        resolved["encounters_per_event"], window, target,
        spillover_cost_per_person_per_day,
    )


def contact_reduction(
    people,
    events_per_week,
    encounters_per_event,
    window: tuple[int, int],
    target: GroupType | None = None,
    spillover_cost_per_person_per_day: float = 0.0,
):
    """X·Y·Z contact-reduction intervention.

    Any of people/events_per_week/encounters_per_event may be a
    ParameterDistribution. Returns Intervention if all are floats,
    UncertainIntervention otherwise.
    """
    has_dist = any(
        isinstance(v, ParameterDistribution)
        for v in (people, events_per_week, encounters_per_event)
    )
    if has_dist:
        params = {
            "people": people,
            "events_per_week": events_per_week,
            "encounters_per_event": encounters_per_event,
        }
        builder = partial(
            _build_contact_reduction_from_resolved,
            window=window, target=target,
            spillover_cost_per_person_per_day=spillover_cost_per_person_per_day,
        )
        return UncertainIntervention(name="contact_reduction", builder=builder, params=params)
    return _build_contact_reduction(
        people, events_per_week, encounters_per_event,
        window, target, spillover_cost_per_person_per_day,
    )


def _apply_contact_reduction(
    world, mask: np.ndarray, meetings_per_day: float
) -> None:
    if not mask.any() or meetings_per_day == 0.0:
        return
    target_groups = np.where(mask)[0]
    in_target = np.isin(world.membership_group_id, target_groups)
    baseline_target_meetings = float(in_target.sum())
    if baseline_target_meetings == 0.0:
        return
    factor = max(0.0, 1.0 - meetings_per_day / baseline_target_meetings)
    world.group_attendance_mult[mask] *= factor


def _spillover_contact_reduction(
    world, cfg: ScenarioConfig, cost_per_person_per_day: float, people: float
) -> float:
    return cost_per_person_per_day * people


def _build_contact_reduction(
    people: float, events_per_week: float, encounters_per_event: float,
    window: tuple[int, int], target: GroupType | None,
    spillover_cost_per_person_per_day: float,
) -> Intervention:
    meetings_per_day = people * events_per_week * encounters_per_event / 7.0
    if target is None:
        target_ints = np.array([int(gt) for gt in GroupType])
    else:
        target_ints = np.array([int(target)])
    spillover_fn = None
    if spillover_cost_per_person_per_day > 0:
        spillover_fn = partial(
            _spillover_contact_reduction,
            cost_per_person_per_day=spillover_cost_per_person_per_day,
            people=people,
        )
    return Intervention(
        name=f"contact_reduction_{int(people)}x{events_per_week}x{encounters_per_event}",
        start_day=window[0],
        end_day=window[1],
        target_filter=partial(_filter_group_type_in, target_ints=target_ints),
        apply_to_groups=partial(_apply_contact_reduction, meetings_per_day=meetings_per_day),
        spillover_cost_fn=spillover_fn,
    )


# ---------------- run_comparison ----------------

def run_comparison(
    cfg: ScenarioConfig,
    interventions: Sequence,  # list of Intervention | UncertainIntervention
    n_runs: int,
    n_psa_samples: int = 1,
    base_seed: int = 0,
    initial_infected: int = 10,
    healthcare: HealthcareConfig | None = None,
    parallel: bool = True,
) -> ComparisonResult:
    """Run baseline vs treatment, with optional PSA over uncertain intervention parameters."""
    interventions_list = list(interventions)
    has_uncertainty = any(isinstance(i, UncertainIntervention) for i in interventions_list)

    if not has_uncertainty:
        n_psa_samples = 1
        resolved_list = [interventions_list]
        samples_list = None
    else:
        master_rng = np.random.default_rng(base_seed)
        resolved_list = []
        samples_list = []
        for _psa_idx in range(n_psa_samples):
            sub_seed = int(master_rng.integers(0, 2**31 - 1))
            sub_rng = np.random.default_rng(sub_seed)
            resolved_interventions = []
            sample_dict = {}
            for inter in interventions_list:
                if isinstance(inter, UncertainIntervention):
                    concrete, sampled = inter.resolve(sub_rng)
                    resolved_interventions.append(concrete)
                    sample_dict.update(sampled)
                else:
                    resolved_interventions.append(inter)
            resolved_list.append(resolved_interventions)
            samples_list.append(sample_dict)

    baseline_results = []
    treatment_results = []
    baseline_hc = [] if healthcare is not None else None
    treatment_hc = [] if healthcare is not None else None

    for psa_idx, resolved in enumerate(resolved_list):
        seed_offset = base_seed + psa_idx * n_runs * 2
        baseline_mc = run_mc(
            cfg, interventions=[], n_runs=n_runs, base_seed=seed_offset,
            initial_infected=initial_infected, parallel=parallel,
            healthcare=healthcare,
        )
        treatment_mc = run_mc(
            cfg, interventions=resolved, n_runs=n_runs, base_seed=seed_offset,
            initial_infected=initial_infected, parallel=parallel,
            healthcare=healthcare,
        )
        baseline_results.append(baseline_mc)
        treatment_results.append(treatment_mc)
        if healthcare is not None:
            baseline_hc.append(baseline_mc.healthcare_outcomes)
            treatment_hc.append(treatment_mc.healthcare_outcomes)

    return ComparisonResult(
        cfg=cfg,
        interventions_resolved=resolved_list,
        baseline_results=baseline_results,
        treatment_results=treatment_results,
        baseline_healthcare=baseline_hc,
        treatment_healthcare=treatment_hc,
        intervention_samples=samples_list,
    )
