# Bangladesh PV Module Degradation, Lifetime and Waste Forecasting

A reproducible research framework for estimating photovoltaic (PV) module retirement and end-of-life waste by installation cohort. The project combines survival analysis, cohort forecasting, module-mass accounting, degradation modelling and material recovery scenarios.

> **Evidence status:** the bundled CSVs are synthetic demonstration data. The outputs are not an estimate of Bangladesh's current or future PV waste, and they must not be cited as empirical results. The project is a modelling framework that needs verified installation, retirement and module-mass data before a national forecast can be produced.

## Start here

New to the project? Read [the beginner guide](docs/BEGINNER_GUIDE.md) first. It explains the project in plain language, describes the files and walks through the full run. The [data dictionary](docs/DATA_DICTIONARY.md) defines each input field, and the [data-source register](docs/DATA_SOURCES.md) explains what Bangladesh-specific evidence is still needed.

### Quick start

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest -q
python scripts/run_forecast.py
python scripts/run_degradation_models.py
```

The forecast script writes demonstration outputs under `outputs/`. These output files are excluded from version control so that generated results do not get confused with source data.

## What question does this project answer?

PV modules do not all retire at the same age. Some are replaced early; others remain in service longer than the nominal warranty period. A useful waste forecast therefore needs to account for:

1. **How many modules were installed, and when?** This defines the fleet cohorts.
2. **How long do modules remain in service?** A lifetime distribution describes retirement probability.
3. **How much material is associated with each cohort?** Capacity must be translated into module mass using a documented mass-per-kW factor.
4. **What happens after retirement?** Reuse, refurbishment, recycling, storage, export and disposal are different material pathways.

The framework keeps degradation and retirement separate. Degradation reduces energy output; it does not automatically mean a module has become waste. The new ML workflow splits by physical module ID, so repeated observations from one module cannot appear in both train and test sets. To include XGBoost, install `pip install -r requirements-xgboost.txt` and call `compare_degradation_models(..., include_xgboost=True)`.

## Modelling workflow

| Stage | What the code does | Main output |
|---|---|---|
| 1. Data audit | Checks observed lifetimes, censoring flags and cohort fields | Validated input tables |
| 2. Lifetime estimation | Fits a two-parameter Weibull distribution with right-censored observations | Shape and scale estimates |
| 3. Survival analysis | Calculates survival and conditional retirement probabilities | Survival curve and annual failure probabilities |
| 4. Cohort projection | Applies retirement probabilities to capacity cohorts | Annual retired capacity |
| 5. Mass conversion | Converts retired capacity to tonnes using module mass intensity | Annual module waste |
| 6. Material pathways | Applies explicit reuse/recycling/recovery assumptions | Reuse, recovery and residual streams |
| 7. Degradation demonstration | Compares ML regressors on a synthetic module-level dataset | Illustrative MAE, RMSE and R² |
| 8. Sensitivity | Varies lifetime and material-flow assumptions | Scenario comparison |

## Methods in plain language

### Weibull lifetime model

The Weibull distribution describes how the chance of retirement changes as a module gets older. Its shape parameter controls the pattern of the hazard rate; its scale parameter controls the age range over which retirement becomes likely.

`S(t) = exp[-(t / lambda)^k]`

where `S(t)` is the probability of surviving beyond age `t`, `k` is the shape, and `lambda` is the scale.

The lifetime fit supports right-censored observations. A censored module is one that was still operating when observation ended. It must not be treated as if it failed on the last observation date.

### Cohort waste forecast

The model estimates the expected share of each installation cohort that retires during each year. It then converts that capacity into module mass. The unit conversion is:

`waste tonnes = retired capacity (MW) × module mass (kg/kW)`

This works numerically because 1 MW = 1,000 kW and 1 tonne = 1,000 kg.

### Degradation model

The dedicated degradation workflow compares Random Forest and Extra Trees, with optional XGBoost. It uses a grouped split by module ID so repeated observations from the same physical panel do not leak between train and test. The bundled example panel is synthetic, so model scores and permutation importance demonstrate the workflow only; they are not evidence of Bangladesh degradation rates.

## Repository structure

```text
config/       editable forecast settings
data/demo/    synthetic examples only
docs/         beginner guide, methods, sources and provenance
notebooks/    interactive end-to-end walkthrough
scripts/      command-line forecast and sensitivity runs
src/          reusable Python modelling modules
tests/        unit tests
outputs/      generated tables and figures (not committed)
```

See the [data dictionary](docs/DATA_DICTIONARY.md), [methodology](docs/METHODOLOGY.md), [data-source register](docs/DATA_SOURCES.md) and [provenance template](docs/DATA_PROVENANCE.md) before replacing demo inputs.

## Bangladesh data: where to begin

- **SREDA / National Database of Renewable Energy:** potential source for Bangladesh project and capacity records. Verify historical coverage and definitions before treating it as a commissioning-cohort dataset: https://ndre.sreda.gov.bd/
- **Power Division, National Renewable Energy Strategy 2026–2030:** current policy context and related programme documents: https://powerdivision.gov.bd/pages/static-pages/694032be35ce18e1c0561a07
- **Dhaka PV module dataset (2026):** a CC0 dataset reporting outdoor-exposed four-year-old modules in Mirpur, Dhaka. It may support a small degradation-method demonstration, but its 40-module sample is not representative of the national fleet: https://doi.org/10.7910/DVN/Q56G63
- **Bangladesh PV waste literature:** useful background and research gap identification, not a substitute for observed national retirement data: https://doi.org/10.1016/j.heliyon.2022. e08970

Record each data release, download date, licence, coverage, units, transformations and known gaps in `docs/DATA_PROVENANCE.md`. A recent installed-capacity snapshot is not a complete historical time series.

## What is not yet modelled

The current framework is intentionally transparent rather than pretending to solve every research problem. A full empirical study should consider technology-specific lifetime distributions, competing risks, early repowering, repair and refurbishment, parameter uncertainty, bootstrap or Bayesian intervals, independent validation and uncertainty in the installed base and module mass.

## Research and engineering cautions

- Do not mix MW, kW, kg and tonnes.
- Do not treat censored assets as observed failures.
- Do not equate technical retirement with collected waste.
- Do not infer degradation rates from synthetic training data.
- Keep module waste separate from inverter, cable, mounting and battery waste unless the study explicitly includes those components.
- Report scenarios and uncertainty, not only a single deterministic line.

## Reproducibility

The included tests can be run with `pytest -q`. The notebook is a walkthrough, while the scripts provide repeatable command-line runs. No GitHub Actions or CI workflow is configured; testing is run manually.

## Licence

MIT for the code. Dataset licences remain separate; check them before redistribution.
