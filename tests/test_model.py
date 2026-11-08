import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pv_waste_forecasting.model import (
    fit_weibull_lifetime, survival_probability,
    conditional_retirement_probability, forecast_cohort_waste
)


def test_survival_decreases_with_age():
    s = survival_probability(np.array([0, 10, 20]), 3.0, 25.0)
    assert s[0] == 1.0
    assert s[0] > s[1] > s[2]


def test_conditional_retirement_probability_is_bounded():
    p = conditional_retirement_probability(np.array([0, 10, 20]), 3.0, 25.0)
    assert np.all((p >= 0) & (p <= 1))


def test_weibull_fit_with_right_censoring():
    rng = np.random.default_rng(12)
    true = rng.weibull(3.0, 500) * 25
    censored_at = 18.0
    observed = np.minimum(true, censored_at)
    event = true <= censored_at
    result = fit_weibull_lifetime(observed, event)
    assert result["success"]
    assert result["shape"] > 0
    assert result["scale_years"] > 0


def test_cohort_forecast_nonnegative():
    cohorts = pd.DataFrame({
        "commissioning_year": [2015, 2020],
        "installed_capacity_kw": [1000, 2000],
        "module_mass_kg_per_kw": [70, 68]
    })
    result = forecast_cohort_waste(cohorts, 2025, [2026, 2027], 3.0, 28.0)
    assert len(result) == 2
    assert (result["expected_retired_capacity_kw"] >= 0).all()
    assert (result["expected_module_waste_tonnes"] >= 0).all()
