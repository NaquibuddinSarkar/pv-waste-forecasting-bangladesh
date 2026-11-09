import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pv_waste_forecasting.degradation import (
    validate_degradation_data,
    compare_degradation_models,
)


def test_synthetic_degradation_panel_has_no_module_leakage():
    data = pd.read_csv(ROOT / "data" / "demo" / "module_degradation_panel_synthetic.csv")
    result = compare_degradation_models(data)
    assert result["comparison"]["MAE"].notna().all()
    assert result["comparison"]["RMSE"].notna().all()
    assert set(result["train_module_ids"]).isdisjoint(set(result["test_module_ids"]))
    assert len(result["models"]) == 2


def test_degradation_validation_rejects_negative_target():
    data = pd.read_csv(ROOT / "data" / "demo" / "module_degradation_panel_synthetic.csv")
    data.loc[0, "annual_degradation_rate"] = -0.01
    with pytest.raises(ValueError, match="non-negative"):
        validate_degradation_data(data)


def test_degradation_validation_requires_module_id():
    data = pd.read_csv(ROOT / "data" / "demo" / "module_degradation_panel_synthetic.csv")
    with pytest.raises(ValueError, match="module_id"):
        validate_degradation_data(data.drop(columns=["module_id"]))
