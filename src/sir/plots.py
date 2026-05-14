"""Standard plots: epidemic curves, welfare decomposition."""

import matplotlib.pyplot as plt
import numpy as np

from sir.monte_carlo import MCResult


def plot_epidemic_curves(result: MCResult, title: str = "Epidemic curves") -> plt.Figure:
    """Plot mean and 10-90 percentile bands for S/I/R/V trajectories."""
    fig, ax = plt.subplots(figsize=(10, 6))
    days = np.arange(result.S_history.shape[1])
    for label, hist, color in [
        ("Susceptible", result.S_history, "tab:blue"),
        ("Infected", result.I_history, "tab:red"),
        ("Recovered", result.R_history, "tab:green"),
        ("Vaccinated", result.V_history, "tab:purple"),
    ]:
        mean = hist.mean(axis=0)
        q10 = np.percentile(hist, 10, axis=0)
        q90 = np.percentile(hist, 90, axis=0)
        ax.plot(days, mean, label=label, color=color)
        ax.fill_between(days, q10, q90, alpha=0.2, color=color)
    ax.set_xlabel("Days")
    ax.set_ylabel("Number of agents")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_welfare_decomposition(
    results_by_scenario: dict[str, MCResult],
    title: str = "Welfare decomposition by scenario",
) -> plt.Figure:
    """Stacked bar chart of welfare components per scenario."""
    component_keys = [
        "attendance_utility",
        "precaution_cost",
        "infection_cost",
        "vaccination_cost",
        "direct_policy_cost",
        "spillover_policy_cost",
    ]
    component_signs = {
        "attendance_utility": +1,
        "precaution_cost": -1,
        "infection_cost": -1,
        "vaccination_cost": -1,
        "direct_policy_cost": -1,
        "spillover_policy_cost": -1,
    }
    scenarios = list(results_by_scenario.keys())
    means = {
        k: [results_by_scenario[s].welfare_components[k].mean() for s in scenarios]
        for k in component_keys
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(scenarios))
    width = 0.8
    bottoms_pos = np.zeros(len(scenarios))
    bottoms_neg = np.zeros(len(scenarios))
    for k in component_keys:
        signed = np.array(means[k]) * component_signs[k]
        for i, v in enumerate(signed):
            if v >= 0:
                ax.bar(x[i], v, width, bottom=bottoms_pos[i], label=k if i == 0 else "")
                bottoms_pos[i] += v
            else:
                ax.bar(x[i], v, width, bottom=bottoms_neg[i], label=k if i == 0 else "")
                bottoms_neg[i] += v
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Welfare (utility units)")
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.3, axis="y")
    return fig


def plot_welfare_totals(
    results_by_scenario: dict[str, MCResult],
    title: str = "Total welfare by scenario",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    scenarios = list(results_by_scenario.keys())
    means = [results_by_scenario[s].welfare_totals.mean() for s in scenarios]
    stds = [results_by_scenario[s].welfare_totals.std() for s in scenarios]
    x = np.arange(len(scenarios))
    ax.bar(x, means, yerr=stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Total welfare")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    return fig


from sir.comparison import ComparisonResult
from sir.healthcare import HealthcareOutcomes


def plot_healthcare_curves(
    hc: HealthcareOutcomes, title: str = "Healthcare burden"
) -> plt.Figure:
    """Plot hospital and ICU admissions (incidence) + prevalence (occupancy)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    days = np.arange(hc.hosp_prev.size)
    ax.fill_between(days, 0, hc.hosp_prev, alpha=0.18, color="#0891b2",
                    label="Hospital occupancy")
    ax.fill_between(days, 0, hc.icu_prev, alpha=0.18, color="#be123c",
                    label="ICU occupancy")
    ax.plot(days, hc.hosp_admit, color="#0891b2", linestyle="--",
            label="Hospital admissions / day")
    ax.plot(days, hc.icu_admit, color="#be123c", linestyle="--",
            label="ICU admissions / day")
    ax.set_xlabel("Day")
    ax.set_ylabel("People")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    return fig


def plot_deaths_by_age(
    hc: HealthcareOutcomes, age_labels: tuple[str, ...] = (
        "0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60+"
    ),
    title: str = "Cumulative deaths by age bin",
) -> plt.Figure:
    """Stacked area of cumulative deaths over time, by age bin."""
    fig, ax = plt.subplots(figsize=(10, 6))
    days = np.arange(hc.deaths_by_age.shape[0])
    prev = np.zeros_like(days, dtype=np.float64)
    palette = ["#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa", "#3b82f6", "#1d4ed8", "#1e3a8a"]
    for a in range(7):
        band = hc.deaths_by_age[:, a]
        ax.fill_between(days, prev, prev + band, color=palette[a],
                        label=age_labels[a], alpha=0.9)
        prev = prev + band
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative deaths")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize="small")
    ax.grid(True, alpha=0.3)
    return fig


def plot_outcome_comparison(
    cr: ComparisonResult, outcome: str = "I", title: str | None = None,
) -> plt.Figure:
    """Plot a single outcome over time, with baseline (gray) and treatment (blue) lines.

    `outcome` is one of: 'I', 'S', 'R', 'new_inf', 'hosp_prev', 'hosp_admit',
    'icu_prev', 'icu_admit', 'cum_deaths', 'daily_deaths'.
    """
    def _stack(results, attr):
        return np.vstack([
            getattr(mc, attr) for mc in results
        ])

    def _stack_hc(hc_lists, attr):
        rows = []
        for psa_runs in hc_lists:
            for hc in psa_runs:
                rows.append(getattr(hc, attr))
        return np.vstack(rows)

    if outcome == "I":
        b = _stack(cr.baseline_results, "I_history")
        t = _stack(cr.treatment_results, "I_history")
        ylabel = "Infected"
    elif outcome == "S":
        b = _stack(cr.baseline_results, "S_history")
        t = _stack(cr.treatment_results, "S_history")
        ylabel = "Susceptible"
    elif outcome == "R":
        b = _stack(cr.baseline_results, "R_history")
        t = _stack(cr.treatment_results, "R_history")
        ylabel = "Cumulative recovered"
    elif outcome == "new_inf":
        b = _stack(cr.baseline_results, "new_infections_history")
        t = _stack(cr.treatment_results, "new_infections_history")
        ylabel = "New infections / day"
    elif outcome in ("hosp_prev", "hosp_admit", "icu_prev", "icu_admit",
                      "cum_deaths", "daily_deaths"):
        if cr.baseline_healthcare is None:
            raise ValueError(f"Outcome '{outcome}' requires healthcare; pass HealthcareConfig.")
        b = _stack_hc(cr.baseline_healthcare, outcome)
        t = _stack_hc(cr.treatment_healthcare, outcome)
        ylabel = outcome
    else:
        raise ValueError(f"Unknown outcome: {outcome}")

    days = np.arange(b.shape[1])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(days, np.percentile(b, 10, axis=0), np.percentile(b, 90, axis=0),
                    alpha=0.15, color="gray")
    ax.plot(days, b.mean(axis=0), color="#737373", linestyle=":",
            label="Baseline (no interventions)")
    ax.fill_between(days, np.percentile(t, 10, axis=0), np.percentile(t, 90, axis=0),
                    alpha=0.15, color="#1e40af")
    ax.plot(days, t.mean(axis=0), color="#1e40af",
            label="With interventions")
    ax.set_xlabel("Day")
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{outcome}: with vs without interventions")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig
