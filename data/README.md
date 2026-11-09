# PV forecasting data

The files under `demo/` are synthetic teaching datasets. They are not measurements from Bangladesh PV installations and must not be cited as empirical estimates.

## Demo files

- `module_lifetime_observations_synthetic.csv`: sample durations and event/censoring flags for Weibull method demonstration.
- `pv_installation_cohorts_synthetic.csv`: illustrative installation capacity by commissioning cohort and module mass intensity.
- `module_degradation_panel_synthetic.csv`: repeated synthetic observations by physical module ID, designed to demonstrate grouped train/test validation for Random Forest and Extra Trees models.

The degradation panel has one row per module-year. The same module ID appears more than once; therefore, do not randomly split rows across training and testing. The dedicated degradation workflow uses a grouped holdout by module ID.

## Replacing demo data

Keep source files unchanged in a separate raw-data location. Add cleaned empirical files using clear names, document the access date and licence, and complete the template in `docs/DATA_PROVENANCE.md`. Preserve a script that reproduces cleaning and unit conversion steps.

See the [data dictionary](../docs/DATA_DICTIONARY.md) for variable definitions and validation rules.
