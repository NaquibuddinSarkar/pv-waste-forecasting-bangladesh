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


def bootstrap_weibull_lifetime(
    lifetimes_years,
    event_observed,
    n_bootstrap=200,
    confidence_level=0.95,
    random_state=42,
):
    """Estimate percentile intervals for Weibull parameters by paired bootstrap.

    The lifetime and event/censoring flag are resampled together. This
    quantifies sampling uncertainty under the fitted homogeneous Weibull
    model; it does not account for selection bias, site clustering or
    uncertainty in the installed fleet and module mass.
    """
    t = _finite_vector(lifetimes_years, "lifetimes_years")
    event = np.asarray(event_observed)
    if event.ndim != 1 or len(event) != len(t):
        raise ValueError("event_observed must match lifetimes_years")
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 20:
        raise ValueError("n_bootstrap must be an integer of at least 20")
    if not np.isfinite(confidence_level) or not 0.5 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0.5 and 1")

    rng = np.random.default_rng(random_state)
    shape_values = []
    scale_values = []
    n = len(t)
    for _ in range(int(n_bootstrap)):
        indices = rng.integers(0, n, size=n)
        try:
            fit = fit_weibull_lifetime(t[indices], event[indices])
        except (ValueError, RuntimeError):
            continue
        shape_values.append(fit["shape"])
        scale_values.append(fit["scale_years"])

    minimum_valid = max(20, int(0.7 * n_bootstrap))
    if len(shape_values) < minimum_valid:
        raise RuntimeError(
            f"only {len(shape_values)} of {n_bootstrap} bootstrap fits succeeded; "
            "check event counts and censoring structure"
        )

    alpha = (1.0 - confidence_level) / 2.0
    return {
        "confidence_level": float(confidence_level),
        "n_bootstrap_requested": int(n_bootstrap),
        "n_bootstrap_valid": int(len(shape_values)),
        "shape_median": float(np.median(shape_values)),
        "shape_lower": float(np.quantile(shape_values, alpha)),
        "shape_upper": float(np.quantile(shape_values, 1.0 - alpha)),
        "scale_years_median": float(np.median(scale_values)),
        "scale_years_lower": float(np.quantile(scale_values, alpha)),
        "scale_years_upper": float(np.quantile(scale_values, 1.0 - alpha)),
        "note": "Percentile paired-bootstrap intervals conditional on model and sampling assumptions.",
    }


def degradation_prediction_frame(df, target_col="annual_degradation_rate"):
    """Backward-compatible wrapper using module-grouped validation.

    The current implementation uses the dedicated degradation module so
    repeated observations from one module cannot leak across train/test.
    """
    from .degradation import compare_degradation_models

    result = compare_degradation_models(df, target_col=target_col)
    best_name = result["best_model_name"]
    best = result["comparison"].set_index("model").loc[best_name]
    metrics = {
        "MAE": float(best["MAE"]),
        "RMSE": float(best["RMSE"]),
        "R2": float(best["R2"]),
        "model": best_name,
        "validation_method": result["validation_method"],
    }
    return result["models"][best_name], metrics
