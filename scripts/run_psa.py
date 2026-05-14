"""Run probabilistic sensitivity analysis using configs/psa_example.toml.

Outputs a summary of welfare CIs and a tornado plot of parameter sensitivities.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sir.config import default_config
from sir.psa import load_psa_spec, run_psa


def plot_tornado(
    correlations: dict[str, float],
    outcome_name: str,
    out_path: Path,
) -> None:
    items = sorted(correlations.items(), key=lambda kv: abs(kv[1]), reverse=True)
    names = [k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(names) + 1)))
    colors = ["tab:red" if v < 0 else "tab:blue" for v in values]
    ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(f"Rank correlation with {outcome_name}")
    ax.set_title(f"Parameter sensitivity tornado: {outcome_name}")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    base_cfg = default_config()
    spec_path = Path(__file__).parent.parent / "configs" / "psa_example.toml"
    spec = load_psa_spec(spec_path)
    print(f"Loaded {len(spec)} parameter distributions from {spec_path.name}")

    n_psa = 30
    n_mc = 5
    print(f"Running PSA: {n_psa} samples x {n_mc} MC runs = {n_psa * n_mc} simulations")
    result = run_psa(
        base_cfg, spec,
        interventions=[],
        n_psa_samples=n_psa,
        n_mc_per_sample=n_mc,
        base_seed=0,
        initial_infected=10,
    )

    lo, med, hi = result.welfare_ci(alpha=0.05)
    print(f"Total welfare: median {med:.1f}, 95% CI [{lo:.1f}, {hi:.1f}]")
    peak_med = float(np.median(result.peak_I_per_sample))
    peak_lo = float(np.percentile(result.peak_I_per_sample, 2.5))
    peak_hi = float(np.percentile(result.peak_I_per_sample, 97.5))
    print(f"Peak infected: median {peak_med:.0f}, 95% CI [{peak_lo:.0f}, {peak_hi:.0f}]")
    cum_med = float(np.median(result.cumulative_infections_per_sample))
    print(f"Cumulative infections (median): {cum_med:.0f}")

    out_dir = Path(__file__).parent.parent / "output"
    out_dir.mkdir(exist_ok=True)
    welfare_corr = result.parameter_correlations(result.welfare_per_sample)
    plot_tornado(welfare_corr, "total welfare", out_dir / "psa_tornado_welfare.png")
    peak_corr = result.parameter_correlations(result.peak_I_per_sample)
    plot_tornado(peak_corr, "peak infected", out_dir / "psa_tornado_peak.png")
    print(f"Saved tornado plots to {out_dir}")


if __name__ == "__main__":
    main()
