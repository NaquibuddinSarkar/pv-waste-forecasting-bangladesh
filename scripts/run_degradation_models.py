"""Run the grouped PV module degradation-model comparison on synthetic data."""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pv_waste_forecasting.degradation import compare_degradation_models


def main():
    data = pd.read_csv(ROOT / "data" / "demo" / "module_degradation_panel_synthetic.csv")
    result = compare_degradation_models(data, target_col="annual_degradation_rate")

    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    result["comparison"].to_csv(output / "degradation_model_comparison_demo.csv", index=False)
    result["feature_importance"].to_csv(output / "degradation_feature_importance_demo.csv", index=False)

    print("DEGRADATION MODEL COMPARISON — SYNTHETIC MODULE DATA")
    print(result["comparison"].to_string(index=False))
    print("\nPERMUTATION IMPORTANCE")
    print(result["feature_importance"].to_string(index=False))
    print("\nValidation:", result["validation_method"])
    print("Train modules:", len(result["train_module_ids"]))
    print("Test modules:", len(result["test_module_ids"]))
    print("\nWARNING:", result["caution"])


if __name__ == "__main__":
    main()
