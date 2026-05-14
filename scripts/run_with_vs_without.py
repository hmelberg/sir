"""End-to-end demo: compare a baseline scenario vs an intervention scenario,
with optional uncertainty on intervention parameters. Mirrors the web demo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sir.comparison import contact_reduction, run_comparison, transmission_reduction
from sir.config import default_config
from sir.constants import GroupType
from sir.distributions import dist
from sir.healthcare import default_healthcare_config
from sir.plots import (
    plot_outcome_comparison,
    plot_healthcare_curves,
    plot_deaths_by_age,
)


def main() -> None:
    cfg = default_config()
    hc = default_healthcare_config()
    out_dir = Path(__file__).parent.parent / "output"
    out_dir.mkdir(exist_ok=True)

    # Scenario: combined mask mandate + school closure
    interventions = [
        transmission_reduction(
            p_factor=0.5, window=(20, 120),
            targets={GroupType.SCHOOL, GroupType.WORKPLACE, GroupType.COMMUNITY},
        ),
        contact_reduction(
            people=2500, events_per_week=5, encounters_per_event=10,
            window=(20, 120), target=GroupType.SCHOOL,
        ),
    ]

    print("Running deterministic comparison...")
    result = run_comparison(
        cfg, interventions=interventions, n_runs=10, base_seed=0,
        initial_infected=10, healthcare=hc, parallel=False,
    )
    df = result.outcome_deltas()
    print("\nOutcome deltas (baseline vs treatment):")
    print(df.round(2).to_string())

    fig = plot_outcome_comparison(result, outcome="cum_deaths",
                                  title="Cumulative deaths - with vs without combined intervention")
    fig.savefig(out_dir / "comparison_cum_deaths.png", dpi=120)
    plt.close(fig)

    fig = plot_outcome_comparison(result, outcome="hosp_prev",
                                  title="Hospital occupancy - with vs without combined intervention")
    fig.savefig(out_dir / "comparison_hosp_prev.png", dpi=120)
    plt.close(fig)

    # Single-run healthcare detail (first PSA sample, first MC run)
    hc_first = result.treatment_healthcare[0][0]
    fig = plot_healthcare_curves(hc_first, title="Healthcare burden (treatment, 1 run)")
    fig.savefig(out_dir / "comparison_healthcare_curves.png", dpi=120)
    plt.close(fig)

    fig = plot_deaths_by_age(hc_first, title="Cumulative deaths by age (treatment, 1 run)")
    fig.savefig(out_dir / "comparison_deaths_by_age.png", dpi=120)
    plt.close(fig)

    print(f"\nPlots saved to {out_dir}")

    # Now with uncertainty on intervention strength
    print("\nRunning PSA over intervention p_factor uncertainty...")
    interventions_psa = [
        transmission_reduction(
            p_factor=dist.beta(a=8, b=8),  # mean 0.5, sd ~0.12
            window=(20, 120),
            targets={GroupType.SCHOOL, GroupType.WORKPLACE, GroupType.COMMUNITY},
        ),
    ]
    result_psa = run_comparison(
        cfg, interventions=interventions_psa,
        n_runs=5, n_psa_samples=10, base_seed=0,
        initial_infected=10, healthcare=hc, parallel=False,
    )
    df_psa = result_psa.outcome_deltas()
    print("\nOutcome deltas with PSA on intervention strength:")
    print(df_psa.round(2).to_string())


if __name__ == "__main__":
    main()
