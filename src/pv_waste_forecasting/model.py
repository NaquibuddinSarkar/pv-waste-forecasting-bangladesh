"""PV module survival and cohort waste forecasting.

All calculations accept explicit inputs. Demo data shipped with the project are
synthetic and must not be interpreted as Bangladesh national PV waste estimates.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import weibull_min


def fit_weibull_lifetime(lifetimes_years, event_observed):
    """Fit a two-parameter Weibull by MLE, supporting right-censored observations.

    Parameters
    ----------
    lifetimes_years : array-like
        Observed lifetime or censoring time in years; all values must be > 0.
    event_observed : array-like of bool or 0/1
        True for observed retirement/failure; False for right-censored assets.

    Returns
    -------
    dict with shape, scale, success, n_observed, n_censored.
    """
    t = np.asarray(lifetimes_years, dtype=float)
    event = np.asarray(event_observed, dtype=bool)
    if t.ndim != 1 or event.ndim != 1 or len(t) != len(event) or len(t) < 3:
        raise ValueError("lifetimes and event flags must be 1-D arrays of equal length >= 3")
    if not np.all(np.isfinite(t)) or np.any(t <= 0):
        raise ValueError("lifetimes must be finite and strictly positive")
    if not np.any(event):
        raise ValueError("at least one observed retirement is required to estimate the scale")

    def nll(log_params):
        shape, scale = np.exp(log_params)
        z = t / scale
        log_pdf = np.log(shape) - np.log(scale) + (shape - 1) * np.log(z) - z**shape
        log_survival = -(z**shape)
        ll = np.where(event, log_pdf, log_survival)
        return float(-np.sum(ll))

    initial = np.log([2.5, max(float(np.median(t)), 1.0)])
    result = minimize(nll, initial, method="Nelder-Mead", options={"maxiter": 20000})
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"Weibull maximum-likelihood fit failed: {result.message}")
    shape, scale = np.exp(result.x)
    return {
        "shape": float(shape), "scale_years": float(scale),
        "success": bool(result.success), "n_observed": int(event.sum()),
        "n_censored": int((~event).sum()), "negative_log_likelihood": float(result.fun)
    }


def survival_probability(age_years, shape, scale_years):
    """Weibull probability a module survives beyond a specified age."""
    if shape <= 0 or scale_years <= 0:
        raise ValueError("shape and scale must be positive")
    age = np.asarray(age_years, dtype=float)
    if np.any(age < 0):
        raise ValueError("age cannot be negative")
    return np.exp(-np.power(age / scale_years, shape))


def conditional_retirement_probability(age_years, shape, scale_years, interval_years=1.0):
    """Probability of retirement in the next interval conditional on survival to age."""
    age = np.asarray(age_years, dtype=float)
    if interval_years <= 0:
        raise ValueError("interval_years must be positive")
    s0 = survival_probability(age, shape, scale_years)
    s1 = survival_probability(age + interval_years, shape, scale_years)
    return np.clip(1.0 - s1 / np.maximum(s0, 1e-15), 0.0, 1.0)


def forecast_cohort_waste(cohorts: pd.DataFrame, base_year: int, forecast_years,
                          shape: float, scale_years: float):
    """Forecast expected annual retirements and module mass from commissioning cohorts.

    Required columns: commissioning_year, installed_capacity_kw, module_mass_kg_per_kw.
    The model assumes a homogeneous lifetime distribution within each cohort and
    does not include refurbishment, reuse or export unless added as explicit flows.
    """
    required = {"commissioning_year", "installed_capacity_kw", "module_mass_kg_per_kw"}
    missing = required.difference(cohorts.columns)
    if missing:
        raise ValueError(f"cohorts missing columns: {sorted(missing)}")
    if (cohorts["installed_capacity_kw"] < 0).any() or (cohorts["module_mass_kg_per_kw"] <= 0).any():
        raise ValueError("capacity must be non-negative and mass intensity positive")
    rows = []
    for year in forecast_years:
        failed_kw = 0.0
        waste_kg = 0.0
        for row in cohorts.itertuples(index=False):
            if year <= base_year:
                continue
            # Annual retirement flow: survival from the start of this calendar year
            # to the end of this year, rather than cumulative retirements since base year.
            age_start = max(int(year) - 1 - int(row.commissioning_year), 0)
            age_end = max(int(year) - int(row.commissioning_year), 0)
            survival_start = survival_probability(age_start, shape, scale_years)
            survival_end = survival_probability(age_end, shape, scale_years)
            failed_fraction = max(survival_start - survival_end, 0.0)
            expected_failed = float(row.installed_capacity_kw) * failed_fraction
            failed_kw += expected_failed
            waste_kg += expected_failed * float(row.module_mass_kg_per_kw)
        rows.append({
            "year": int(year),
            "expected_retired_capacity_kw": failed_kw,
            "expected_module_waste_tonnes": waste_kg / 1000.0
        })
    return pd.DataFrame(rows)


def degradation_prediction_frame(df, target_col="annual_degradation"):
    """Fit a transparent Random Forest degradation demo and return fitted model + metrics."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    features = ["module_age_years", "mean_temperature_c", "irradiance_kwh_m2_day"]
    needed = set(features + [target_col])
    if not needed.issubset(df.columns):
        raise ValueError(f"data must contain {sorted(needed)}")
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target_col], test_size=0.25, random_state=42
    )
    model = RandomForestRegressor(n_estimators=250, min_samples_leaf=3, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": float(r2_score(y_test, pred))
    }
    return model, metrics
