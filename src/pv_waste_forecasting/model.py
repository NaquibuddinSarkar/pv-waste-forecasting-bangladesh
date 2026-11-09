"""PV module survival, cohort retirement and waste-flow modelling.

All data inputs must be documented. The bundled demo records are synthetic and
must not be used as evidence about Bangladesh's national PV fleet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _finite_vector(values, name):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a one-dimensional finite array")
    return array


def fit_weibull_lifetime(lifetimes_years, event_observed):
    """Estimate a two-parameter Weibull lifetime distribution by MLE.

    Parameters
    ----------
    lifetimes_years:
        Positive follow-up durations. A value is either a retirement age or
        the age at which an otherwise-operating module was last observed.
    event_observed:
        1/True if the defined retirement event was observed, 0/False if the
        observation is right-censored.

    Statistical assumptions
    -----------------------
    The fit assumes independent observations and non-informative right
    censoring. It also assumes one homogeneous Weibull population. If
    technologies, climates or replacement practices differ, fit justified
    strata or a more flexible survival model rather than forcing one curve.

    Returns shape, scale_years, event/censor counts and optimiser diagnostics.
    """
    t = _finite_vector(lifetimes_years, "lifetimes_years")
    raw_event = np.asarray(event_observed)
    if raw_event.ndim != 1 or len(raw_event) != len(t):
        raise ValueError("event_observed must be one-dimensional and match lifetimes")
    if not np.isin(raw_event, [0, 1, False, True]).all():
        raise ValueError("event_observed must contain only 0/1 or bool values")
    event = raw_event.astype(bool)

    if len(t) < 5:
        raise ValueError("at least five lifetime observations are required")
    if np.any(t <= 0):
        raise ValueError("lifetimes must be strictly positive")
    if event.sum() < 2:
        raise ValueError("at least two observed retirement events are required")

    def negative_log_likelihood(log_parameters):
        shape, scale = np.exp(log_parameters)
        z = t / scale
        log_survival = -(z ** shape)
        log_density = (
            np.log(shape) - np.log(scale)
            + (shape - 1.0) * np.log(z) + log_survival
        )
        log_likelihood = np.where(event, log_density, log_survival)
        if not np.isfinite(log_likelihood).all():
            return 1e100
        return float(-log_likelihood.sum())

    initial = np.log([2.5, max(float(np.median(t)), 1.0)])
    result = minimize(
        negative_log_likelihood,
        initial,
        method="Nelder-Mead",
        options={"maxiter": 30000, "xatol": 1e-9, "fatol": 1e-8},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Weibull maximum-likelihood fit failed: {result.message}")

    shape, scale = np.exp(result.x)
    return {
        "shape": float(shape),
        "scale_years": float(scale),
        "success": bool(result.success),
        "n_total": int(len(t)),
        "n_observed": int(event.sum()),
        "n_censored": int((~event).sum()),
        "negative_log_likelihood": float(result.fun),
        "assumption": "two-parameter Weibull; independent non-informative right censoring",
    }


def survival_probability(age_years, shape, scale_years):
    """Probability that a module survives beyond age t under a Weibull model."""
    if not np.isfinite(shape) or not np.isfinite(scale_years) or shape <= 0 or scale_years <= 0:
        raise ValueError("Weibull shape and scale must be positive finite values")
    age = np.asarray(age_years, dtype=float)
    if not np.isfinite(age).all() or np.any(age < 0):
        raise ValueError("age_years must be finite and non-negative")
    return np.exp(-np.power(age / scale_years, shape))


def weibull_hazard(age_years, shape, scale_years):
    """Instantaneous Weibull hazard per year at the supplied module age."""
    if not np.isfinite(shape) or not np.isfinite(scale_years) or shape <= 0 or scale_years <= 0:
        raise ValueError("Weibull shape and scale must be positive finite values")
    age = np.asarray(age_years, dtype=float)
    if not np.isfinite(age).all() or np.any(age < 0):
        raise ValueError("age_years must be finite and non-negative")
    # At age zero, the hazard is 0, shape/scale, or +infinity depending on shape.
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (shape / scale_years) * np.power(age / scale_years, shape - 1.0)
    if shape < 1:
        result = np.where(age == 0, np.inf, result)
    elif shape == 1:
        result = np.where(age == 0, 1.0 / scale_years, result)
    else:
        result = np.where(age == 0, 0.0, result)
    return result


def conditional_retirement_probability(age_years, shape, scale_years, interval_years=1.0):
    """P(retirement within next interval | the module survived to current age)."""
    age = np.asarray(age_years, dtype=float)
    if not np.isfinite(interval_years) or interval_years <= 0:
        raise ValueError("interval_years must be a positive finite number")
    if not np.isfinite(age).all() or np.any(age < 0):
        raise ValueError("age_years must be finite and non-negative")
    start = survival_probability(age, shape, scale_years)
    end = survival_probability(age + interval_years, shape, scale_years)
    probability = 1.0 - end / np.maximum(start, np.finfo(float).tiny)
    return np.clip(probability, 0.0, 1.0)


def _validate_cohorts(cohorts: pd.DataFrame) -> pd.DataFrame:
    required = {"commissioning_year", "installed_capacity_kw", "module_mass_kg_per_kw"}
    missing = required.difference(cohorts.columns)
    if missing:
        raise ValueError(f"cohort table is missing columns: {sorted(missing)}")
    if cohorts.empty:
        raise ValueError("cohort table cannot be empty")

    result = cohorts.copy()
    result["commissioning_year"] = pd.to_numeric(
        result["commissioning_year"], errors="raise"
    )
    if not np.isfinite(result["commissioning_year"]).all():
        raise ValueError("commissioning_year must be finite")
    if not np.equal(result["commissioning_year"], np.floor(result["commissioning_year"])).all():
        raise ValueError("commissioning_year must contain whole calendar years")
    result["commissioning_year"] = result["commissioning_year"].astype(int)

    for column in ["installed_capacity_kw", "module_mass_kg_per_kw"]:
        result[column] = pd.to_numeric(result[column], errors="raise")
        if not np.isfinite(result[column]).all() or (result[column] < 0).any():
            raise ValueError(f"{column} must be finite and non-negative")
    if (result["module_mass_kg_per_kw"] <= 0).any():
        raise ValueError("module_mass_kg_per_kw must be strictly positive")
    if result["commissioning_year"].duplicated().any():
        raise ValueError("each commissioning year must appear once; aggregate duplicate cohorts first")
    return result.sort_values("commissioning_year").reset_index(drop=True)


def forecast_cohort_waste(
    cohorts: pd.DataFrame,
    base_year: int,
    forecast_years,
    shape: float,
    scale_years: float,
):
    """Forecast expected annual retirements and module mass by calendar year.

    Required cohort columns:
      commissioning_year: integer year, assumed commissioned at the start
        of that year;
      installed_capacity_kw: initial module DC capacity in kW;
      module_mass_kg_per_kw: module-only mass intensity in kg per kW DC.

    A cohort's annual retired share is S(age at start) - S(age at end).
    This is an unconditional share of initial cohort capacity, so cohorts
    already in operation at the base year are not treated as newly installed.

    Returns one row per forecast year. The output is an expectation under
    the lifetime model, not an exact physical count of modules.
    """
    if not isinstance(base_year, (int, np.integer)):
        raise TypeError("base_year must be an integer calendar year")
    years = np.asarray(forecast_years, dtype=int)
    if years.ndim != 1 or len(years) == 0 or not np.all(np.isfinite(years)):
        raise ValueError("forecast_years must be a non-empty one-dimensional year sequence")
    if len(np.unique(years)) != len(years) or np.any(np.diff(years) <= 0):
        raise ValueError("forecast_years must be unique and strictly increasing")
    if np.any(years <= base_year):
        raise ValueError("all forecast years must be later than base_year")

    data = _validate_cohorts(cohorts)
    rows = []
    for year in years:
        retired_kw = 0.0
        waste_kg = 0.0
        survivors_end_kw = 0.0
        for row in data.itertuples(index=False):
            if row.commissioning_year > year:
                continue
            # Commissioning is defined at the start of the calendar year.
            # At the start of forecast year Y, cohort age is Y - commissioning_year.
            age_start = max(int(year) - row.commissioning_year, 0)
            age_end = age_start + 1
            s_start = float(survival_probability(age_start, shape, scale_years))
            s_end = float(survival_probability(age_end, shape, scale_years))
            retired_fraction = max(s_start - s_end, 0.0)
            cohort_retired_kw = row.installed_capacity_kw * retired_fraction
            cohort_survivors_kw = row.installed_capacity_kw * s_end
            retired_kw += cohort_retired_kw
            survivors_end_kw += cohort_survivors_kw
            waste_kg += cohort_retired_kw * row.module_mass_kg_per_kw

        rows.append({
            "year": int(year),
            "expected_retired_capacity_kw": float(retired_kw),
            "expected_module_waste_tonnes": float(waste_kg / 1000.0),
            "expected_surviving_capacity_kw": float(survivors_end_kw),
        })
    return pd.DataFrame(rows)


def apply_material_pathways(
    forecast: pd.DataFrame,
    reuse_share: float = 0.0,
    recycling_share_of_nonreuse: float = 0.0,
    recycling_yield: float = 0.85,
):
    """Split projected module waste into reuse, recovered material and residual.

    All three arguments are mass fractions from 0 to 1. Recycling yield is
    the recovered fraction of material sent to recycling, not the share sent.
    A zero mass-balance residual confirms arithmetic closure only; it does not
    prove that any assumed reuse/recycling pathway is commercially available.
    """
    for name, value in [
        ("reuse_share", reuse_share),
        ("recycling_share_of_nonreuse", recycling_share_of_nonreuse),
        ("recycling_yield", recycling_yield),
    ]:
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")

    if "expected_module_waste_tonnes" not in forecast.columns:
        raise ValueError("forecast must contain expected_module_waste_tonnes")
    out = forecast.copy()
    waste = pd.to_numeric(out["expected_module_waste_tonnes"], errors="raise")
    if not np.isfinite(waste).all() or (waste < 0).any():
        raise ValueError("module waste tonnes must be finite and non-negative")

    out["reused_tonnes"] = waste * reuse_share
    non_reused = waste - out["reused_tonnes"]
    out["recycling_input_tonnes"] = non_reused * recycling_share_of_nonreuse
    out["recovered_material_tonnes"] = out["recycling_input_tonnes"] * recycling_yield
    out["recycling_residue_tonnes"] = (
        out["recycling_input_tonnes"] - out["recovered_material_tonnes"]
    )
    out["residual_disposal_tonnes"] = (
        non_reused - out["recycling_input_tonnes"] + out["recycling_residue_tonnes"]
    )
    out["mass_balance_residual_tonnes"] = (
        waste - out["reused_tonnes"] - out["recovered_material_tonnes"]
        - out["residual_disposal_tonnes"]
    )
    return out


def degradation_prediction_frame(df, target_col="annual_degradation"):
    """Fit a clearly labelled Random Forest degradation demonstration.

    This helper is intended for learning and should not be used to claim
    Bangladesh-specific accuracy without a representative, independent data
    set. If multiple rows belong to the same physical module, use a grouped
    split by module ID rather than a random row split.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    features = ["module_age_years", "mean_temperature_c", "irradiance_kwh_m2_day"]
    required = set(features + [target_col])
    if not required.issubset(df.columns):
        raise ValueError(f"data must contain {sorted(required)}")
    sample = df[list(required)].apply(pd.to_numeric, errors="raise").dropna()
    if len(sample) < 20:
        raise ValueError("at least 20 complete rows are required for the demonstration")
    if not np.isfinite(sample.to_numpy()).all():
        raise ValueError("degradation data must be finite")
    if (sample[target_col] < 0).any():
        raise ValueError("degradation target must be non-negative")

    x_train, x_test, y_train, y_test = train_test_split(
        sample[features], sample[target_col], test_size=0.25, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, max_features=1.0,
        random_state=42, n_jobs=-1
    )
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, prediction))),
        "R2": float(r2_score(y_test, prediction)),
        "test_rows": int(len(y_test)),
        "note": "Synthetic/illustrative benchmark only; validate on independent module-level observations.",
    }
    return model, metrics
