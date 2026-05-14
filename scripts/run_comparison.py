"""Compare baseline, mask mandate, school closure, and vaccination scenarios."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sir.config import default_config
from sir.constants import GroupType
from sir.interventions import (
    close_schools,
    mask_mandate,
    vaccinate_eldest_first,
)
from sir.monte_carlo import run_mc
from sir.plots import (
    plot_epidemic_curves,
    plot_welfare_decomposition,
    plot_welfare_totals,
)


def main() -> None:
    cfg = default_config()
    n_runs = 10
    out_dir = Path(__file__).parent.parent / "output"
    out_dir.mkdir(exist_ok=True)

    scenarios = {
        "baseline": [],
        "mask_mandate": [
            mask_mandate(
                start_day=30, end_day=180,
                target_types={
                    GroupType.SCHOOL, GroupType.WORKPLACE, GroupType.COMMUNITY,
                },
                p_factor=0.5,
            ),
        ],
        "school_closure": [
            close_schools(start_day=30, end_day=120),
        ],
        "vaccination": [
            vaccinate_eldest_first(start_day=30, end_day=200, doses_per_day=50),
        ],
    }

    results = {}
    for name, interventions in scenarios.items():
        print(f"Running scenario: {name}")
        results[name] = run_mc(
            cfg, interventions, n_runs=n_runs, base_seed=42,
            initial_infected=10, parallel=False,
        )
        print(f"  Mean welfare: {results[name].welfare_totals.mean():.1f}")
        print(f"  Mean peak I: {results[name].I_history.max(axis=1).mean():.0f}")
        print(f"  Mean final R: {results[name].R_history[:, -1].mean():.0f}")

    # Per-scenario epidemic curves
    for name, r in results.items():
        fig = plot_epidemic_curves(r, title=f"Scenario: {name}")
        fig.savefig(out_dir / f"curves_{name}.png", dpi=120)
        plt.close(fig)

    # Welfare comparison
    fig = plot_welfare_decomposition(results, title="Welfare decomposition by scenario")
    fig.savefig(out_dir / "welfare_decomposition.png", dpi=120)
    plt.close(fig)
    fig = plot_welfare_totals(results, title="Total welfare by scenario")
    fig.savefig(out_dir / "welfare_totals.png", dpi=120)
    plt.close(fig)
    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
