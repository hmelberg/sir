# sir — Microfounded epidemic model

A discrete-time, agent-based SIR model where the transmission rate `β` is decomposed into observable, policy-addressable components, and where rational agents endogenously adjust their behavior in response to perceived infection risk. Designed for cost-benefit analysis of non-pharmaceutical interventions.

> ⚠ **AI-assisted work in progress. May contain mistakes. Do not use for policy purposes.**

**Live demo:** https://hmelberg.github.io/sir/
**Full documentation:** https://hmelberg.github.io/sir/docs.html

## Highlights

- **Decomposed transmission**: `β = m · p` where `m` is meetings per day and `p` is per-meeting transmission probability — both directly affected by interventions and behavior.
- **Microfounded behavior**: agents solve a per-period FOC for attendance `θ` and precaution effort `e`, trading off social-contact utility against expected infection cost.
- **Bipartite group structure**: seven group types (household, kin, school, workplace, recurring leisure, one-off events, community) with age-stratified mixing.
- **Interventions**: composable building blocks (mask mandates, school closures, WFH, gathering limits, vaccination, plus a generic X·Y·Z contact-reduction constructor).
- **Healthcare outcomes**: hospital admissions and occupancy, ICU admissions and occupancy, deaths, life-years lost — age-stratified.
- **Comparison wrapper**: `run_comparison()` runs baseline vs treatment with paired seeds and produces a tidy delta table with 95% CIs.
- **Uncertainty (PSA)**: any intervention parameter can be a probability distribution; the wrapper samples and reports outcome ranges.
- **Welfare accounting**: 6-component decomposed welfare ledger (attendance utility, precaution cost, infection cost, vaccination cost, direct + spillover policy costs).

## Installation

Install from GitHub (Python 3.11+):

```bash
pip install git+https://github.com/hmelberg/sir.git
```

For development (clone + editable install with test dependencies):

```bash
git clone https://github.com/hmelberg/sir.git
cd sir
pip install -e ".[dev]"
pytest
```

## Quick start

```python
from sir.comparison import run_comparison, transmission_reduction, contact_reduction
from sir.config import default_config
from sir.constants import GroupType
from sir.healthcare import default_healthcare_config

cfg = default_config()
hc = default_healthcare_config()

# Combine a mask mandate and a school closure for days 20–120
interventions = [
    transmission_reduction(
        p_factor=0.5,
        window=(20, 120),
        targets={GroupType.SCHOOL, GroupType.WORKPLACE, GroupType.COMMUNITY},
    ),
    contact_reduction(
        people=2500, events_per_week=5, encounters_per_event=10,
        window=(20, 120), target=GroupType.SCHOOL,
    ),
]

result = run_comparison(
    cfg, interventions=interventions, n_runs=10,
    initial_infected=10, healthcare=hc,
)

print(result.outcome_deltas().round(2))
```

Produces something like:

```
                      baseline    treatment     delta  delta_pct
outcome
peak_I                 146.3         38.2     -108.1    -73.9
final_attack_rate        0.27         0.02     -0.25    -92.1
total_deaths            18.9          2.3     -16.5    -87.6
total_yll              270.9         28.2    -242.7    -89.6
peak_hosp_prev           6.0          2.7      -3.4    -56.1
peak_icu_prev            2.3          1.0      -1.3    -55.5
welfare           10968402.5   10846299.3 -122103.2     -1.1
```

See the [full documentation](https://hmelberg.github.io/sir/docs.html) for:

- The mathematical model (FOCs, network structure, welfare equations)
- API reference for every module
- How to add probabilistic sensitivity analysis on intervention parameters
- How to customize healthcare assumptions (age-stratified rates, lags, LOS)
- Examples and demo scripts

## Demo scripts

After installation, four ready-to-run scripts in `scripts/`:

```bash
python scripts/run_baseline.py            # Baseline epidemic, no interventions
python scripts/run_comparison.py          # Four scenarios side by side
python scripts/run_psa.py                 # Probabilistic sensitivity over disease params
python scripts/run_with_vs_without.py     # Combined intervention + PSA on strength
```

Output PNGs land in `output/`.

## Browser demo

The interactive web demo runs the model live in your browser via Pyodide and is the easiest way to build intuition:

**https://hmelberg.github.io/sir/**

## Disclaimer

This is an experimental, AI-assisted model. The math may contain mistakes; the calibration is illustrative, not derived from a specific pathogen; the welfare formulation reflects one set of modeling choices among many. **Do not use for policy decisions.**

## License

MIT. See [LICENSE](LICENSE).

## Author

Hans Olav Melberg, UiT The Arctic University of Norway.
