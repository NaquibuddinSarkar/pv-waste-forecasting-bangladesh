# PV Waste Forecasting — Bangladesh

A research-oriented framework for photovoltaic (PV) module end-of-life capacity and waste-flow forecasting. The complete project source package is provided in [source-code.zip](./source-code.zip).

## What is included

The source archive contains the Python package, a walkthrough notebook, Weibull survival and right-censored lifetime estimation, cohort-based retirement forecasts, degradation model demonstrations, scenario/sensitivity scripts, sample CSV datasets, tests, methodology notes and data-source registers.

## Quick start

Download and extract [source-code.zip](./source-code.zip), then run from the extracted project directory:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
python scripts/run_forecast.py
```

## Data status

Bundled CSVs and base-case values are synthetic or illustrative. They are not Bangladesh installation records, measured degradation or a national PV waste estimate. The source package includes a data-provenance register and candidate public sources. Replace the demo data with documented data before empirical research.

## Scope and limitations

The modelling framework separates technical retirement from actual waste generation and includes explicit reuse, recycling and residual-disposal assumptions. Any empirical or policy conclusion requires validated cohort data, censoring definitions, material composition data and uncertainty analysis.

**No GitHub Actions or CI workflow is included, as requested.** Local tests are retained so the package can be checked manually.

[Download complete source package](./source-code.zip)
