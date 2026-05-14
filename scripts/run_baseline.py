"""Run a no-intervention baseline scenario and save outputs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from sir.config import default_config
from sir.monte_carlo import run_mc
from sir.plots import plot_epidemic_curves


def main() -> None:
    cfg = default_config()
    print(f"Running baseline: N={cfg.N}, T={cfg.T}")
    result = run_mc(cfg, interventions=[], n_runs=10, base_seed=0, initial_infected=10)
    print(f"Mean total welfare: {result.welfare_totals.mean():.1f}")
    print(f"Mean peak infected: {result.I_history.max(axis=1).mean():.0f}")
    print(f"Mean final recovered: {result.R_history[:, -1].mean():.0f}")

    out_dir = Path(__file__).parent.parent / "output"
    out_dir.mkdir(exist_ok=True)
    fig = plot_epidemic_curves(result, title="Baseline (no intervention)")
    fig.savefig(out_dir / "baseline_curves.png", dpi=120)
    plt.close(fig)
    print(f"Saved: {out_dir / 'baseline_curves.png'}")


if __name__ == "__main__":
    main()
