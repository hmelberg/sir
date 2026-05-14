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
