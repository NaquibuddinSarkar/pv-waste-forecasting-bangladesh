# Data Dictionary

The demonstration files are synthetic. Column definitions below describe the intended input contract for empirical replacements.

## Installation cohort data

File: `data/demo/pv_installation_cohorts_synthetic.csv`

| Column | Type | Unit | Meaning | Validation |
|---|---|---|---|---|
| commissioning_year | integer | calendar year | Year the cohort entered service | Must be a plausible year; no future cohort in historical base year |
| installed_capacity_kw | numeric | kW DC | Initially installed module capacity for the cohort | Non-negative |
| module_mass_kg_per_kw | numeric | kg/kW | Module mass per unit of rated DC capacity | Positive |
| annual_degradation_rate (if present) | numeric | fraction/year | Assumed annual output degradation | 0 to below 1 |

A cohort-level capacity should be defined consistently. Do not mix module DC nameplate, inverter AC rating and project capacity.

## Lifetime observations

File: `data/demo/module_lifetime_observations_synthetic.csv`

| Column | Type | Unit | Meaning | Validation |
|---|---|---|---|---|
| observed_lifetime_years | numeric | years | Observed time to retirement or last follow-up | Positive and finite |
| event_observed | integer/boolean | 0/1 | 1 if retirement event observed; 0 if still operating at last follow-up | Only 0 or 1 |

If the original data contains commissioning and removal dates, calculate duration from those dates and document how partial years and missing dates are handled.

## Recommended additional fields for empirical research

- module technology, manufacturer, model and wattage;
- project type and installation location;
- commissioning month/year;
- failure/removal date and reason;
- observation end date for modules still operating;
- module mass and material composition;
- repair, reuse, resale, export and recycling outcome.

## Provenance record

For every empirical file, record publisher, exact dataset title, URL or persistent identifier, download date, release date, licence, geographic coverage, time coverage, unit, data transformations and known limitations in `docs/DATA_PROVENANCE.md`.
