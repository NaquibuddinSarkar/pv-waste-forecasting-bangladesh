# Beginner Guide: PV Module Lifetime and Waste Forecasting

This guide explains the project from the beginning. You do not need to know survival analysis before reading it.

## 1. What problem are we trying to solve?

A solar PV module produces electricity for many years. Eventually it may be replaced, removed, damaged or retired. When this happens, it becomes part of a material-management problem: modules may be repaired, reused, recycled, stored or disposed of.

A waste forecast asks: **given the modules installed in different years and an assumed lifetime pattern, how much module capacity and mass might retire in each future year?**

The answer depends on data and assumptions. This repository demonstrates the method; its sample data are synthetic and do not represent Bangladesh's actual solar fleet.

## 2. Five important concepts

### Installation cohort
A cohort is a group of modules installed in the same year (or another defined period). For example, all modules commissioned during 2018 form one cohort. Cohort data should include capacity and module mass.

### Lifetime and retirement
A module lifetime is the time from commissioning to the defined retirement event. Define the event carefully: failure, removal, replacement, repowering or permanent decommissioning may not mean the same thing.

### Right-censoring
Some modules are still operating when data collection ends. Their exact retirement date is unknown; we only know that they survived at least until the observation date. This is called right-censoring. Do not mark such a module as a failure.

### Survival probability
The survival function S(t) is the probability that a module remains in service beyond age t. A value of 0.80 at age 20 would mean that the fitted model estimates 80% of comparable modules survive past 20 years.

### Waste mass
Capacity is not mass. To convert retired capacity to module mass, multiply capacity by a documented module mass per kW. Ensure the units are consistent.

## 3. Why Weibull?

A single average lifetime cannot describe how retirement probability changes with age. The two-parameter Weibull model is flexible:
- shape below 1: the modelled hazard tends to decline with age;
- shape equal to 1: constant hazard;
- shape above 1: hazard tends to rise with age.

This is a statistical description, not an engineering explanation. The model does not prove why modules retire. Other distributions, technology-specific groups and non-parametric survival estimates should be considered in a research study.

## 4. Run the project

From the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
python scripts/run_forecast.py
```

The script reads two synthetic CSV files, fits the Weibull distribution, estimates annual cohort retirements, converts retired capacity into tonnes and writes results to `outputs/`.

## 5. Understand the outputs

- `weibull_fit_demo.json`: fitted shape, scale and sample-size information.
- `pv_waste_forecast_demo.csv`: annual retired capacity and module waste.

The waste number is an expected value under the model, not a prediction of an exact number of panels. It also represents technical module retirements, not necessarily collected waste.

## 6. What must be replaced for a real Bangladesh analysis?

At minimum:
1. A historical installation inventory by commissioning year, technology and project type.
2. Actual module removals/failures and records of modules still operating at the observation date.
3. A consistent definition of retirement.
4. Module mass by product/technology generation.
5. Information on repair, reuse, export, storage, collection and recycling pathways.
6. A documented source, licence, extraction date and unit for every dataset.

A recent national installed-capacity snapshot cannot by itself establish how much capacity was installed in each historical year.

## 7. Common mistakes to avoid

- Treating censored modules as failed modules.
- Using a single lifetime assumption without sensitivity analysis.
- Mixing module capacity with total PV system capacity that includes inverters or batteries.
- Treating degradation as equivalent to retirement.
- Reporting synthetic results as Bangladesh estimates.
- Applying recycling rates to total capacity rather than to a defined mass stream.

## 8. Suggested research extensions

Compare Weibull, lognormal and Kaplan-Meier survival; stratify by technology or climate where sample sizes allow; bootstrap parameter uncertainty; add early-replacement and repowering hazards; and use Monte Carlo simulation to propagate uncertainty through the annual waste forecast.
