import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pv_waste_forecasting.model import (
    fit_weibull_lifetime,
    survival_probability,
    weibull_hazard,
    conditional_retirement_probability,
    forecast_cohort_waste,
    apply_material_pathways,
    bootstrap_weibull_lifetime,
)


def sample_cohorts():
    return pd.DataFrame({
        "commissioning_year": [2015, 2020],
        "installed_capacity_kw": [1000, 2000],
        "module_mass_kg_per_kw": [70, 68],
    })


def test_survival_decreases_with_age():
    s = survival_probability(np.array([0, 10, 20]), 3.0, 25.0)
    assert s[0] == 1.0
    assert s[0] > s[1] > s[2]


def test_weibull_hazard_behaves_for_shape_greater_than_one():
    h = weibull_hazard(np.array([0, 10, 20]), 3.0, 25.0)
    assert h[0] == 0
    assert h[1] > 0
    assert h[2] > h[1]


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
    assert result["n_observed"] + result["n_censored"] == 500


def test_weibull_fit_rejects_invalid_event_flags():
    with pytest.raises(ValueError, match="0/1"):
        fit_weibull_lifetime([1, 2, 3, 4, 5], [1, 0, 0, 2, 1])


def test_cohort_forecast_nonnegative_and_unit_conversion():
    result = forecast_cohort_waste(sample_cohorts(), 2025, [2026, 2027], 3.0, 28.0)
    assert len(result) == 2
    assert (result["expected_retired_capacity_kw"] >= 0).all()
    assert (result["expected_module_waste_tonnes"] >= 0).all()
    single = pd.DataFrame({
        "commissioning_year": [2015],
        "installed_capacity_kw": [1000],
        "module_mass_kg_per_kw": [70],
    })
    one = forecast_cohort_waste(single, 2025, [2026], 3.0, 28.0)
    assert one["expected_module_waste_tonnes"].iloc[0] == pytest.approx(
        one["expected_retired_capacity_kw"].iloc[0] * 70 / 1000
    )


def test_cohort_forecast_rejects_duplicate_years():
    cohorts = pd.concat([sample_cohorts(), sample_cohorts().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="once"):
        forecast_cohort_waste(cohorts, 2025, [2026], 3.0, 28.0)


def test_material_pathway_mass_balance_closes():
    forecast = forecast_cohort_waste(sample_cohorts(), 2025, [2026, 2027], 3.0, 28.0)
    flows = apply_material_pathways(
        forecast, reuse_share=0.10,
        recycling_share_of_nonreuse=0.70,
        recycling_yield=0.82,
    )
    assert np.allclose(flows["mass_balance_residual_tonnes"], 0.0, atol=1e-9)
    assert (flows["reused_tonnes"] >= 0).all()
    assert (flows["recovered_material_tonnes"] <= flows["recycling_input_tonnes"]).all()


def test_material_pathway_rejects_invalid_fraction():
    forecast = forecast_cohort_waste(sample_cohorts(), 2025, [2026], 3.0, 28.0)
    with pytest.raises(ValueError):
        apply_material_pathways(forecast, reuse_share=1.2)


def test_weibull_bootstrap_returns_ordered_intervals():
    rng = np.random.default_rng(2026)
    true_lifetime = rng.weibull(3.0, 180) * 25.0
    censor_time = 18.0
    observed = np.minimum(true_lifetime, censor_time)
    event = true_lifetime <= censor_time
    result = bootstrap_weibull_lifetime(
        observed, event, n_bootstrap=30, confidence_level=0.90, random_state=7
    )
    assert result["n_bootstrap_valid"] >= 20
    assert result["shape_lower"] <= result["shape_median"] <= result["shape_upper"]
    assert result["scale_years_lower"] <= result["scale_years_median"] <= result["scale_years_upper"]
