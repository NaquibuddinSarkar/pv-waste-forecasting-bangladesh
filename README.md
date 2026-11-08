# PV Waste Forecasting — Bangladesh

A research-oriented framework for modelling photovoltaic (PV) module lifetime, degradation, cohort retirement and end-of-life waste flows.

## Project contents

- `src/pv_waste_forecasting/model.py`: two-parameter Weibull maximum-likelihood fit with right-censored lifetime observations, survival functions, conditional retirement probabilities and cohort waste forecasting.
- `scripts/run_forecast.py`: fits the synthetic lifetime sample and generates an annual cohort forecast.
- `config/base_case.json`: forecast horizon and illustrative settings.
- `data/demo/`: synthetic lifetime and installation-cohort records.
- `notebooks/01_pv_waste_forecasting_walkthrough.ipynb`: interactive walkthrough.
- `tests/`: manual unit tests for survival, fitting and cohort logic.
- `docs/`: methodology and data-provenance guidance.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
python scripts/run_forecast.py
```

Open the notebook in JupyterLab, VS Code or Google Colab.

## Modelling scope

The Weibull model estimates lifetime distribution parameters while using right-censored observations. Cohort projections convert commissioning-year capacity into expected annual retirements and module mass. The package also includes an optional Random Forest degradation demonstration.

Technical retirement is not the same as collected waste. Refurbishment, second-life reuse, exports, storage, recycling and residual disposal require explicit flow assumptions.

## Data and limitations

**All bundled CSVs and default values are synthetic or illustrative.** They are not Bangladesh installation records, measured degradation or a national PV waste estimate. Replace them with documented installation cohorts, observed failures and censoring information, module mass/composition data and validated degradation measurements before empirical use.

A defensible research study should report uncertainty in Weibull parameters, test alternative lifetime assumptions, document data provenance and validate against independent observations.

No GitHub Actions or CI workflow is configured. Tests are available for manual local execution.

## Licence

MIT. See [LICENSE](LICENSE).
