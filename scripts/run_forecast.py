"""Run a synthetic PV retirement and waste forecast from the project root."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pv_waste_forecasting.model import fit_weibull_lifetime, forecast_cohort_waste


def main():
    data_dir = ROOT / "data" / "demo"
    lifetime = pd.read_csv(data_dir / "module_lifetime_observations_synthetic.csv")
    cohorts = pd.read_csv(data_dir / "pv_installation_cohorts_synthetic.csv")
    params = fit_weibull_lifetime(lifetime["observed_lifetime_years"], lifetime["event_observed"])
    settings = json.loads((ROOT / "config" / "base_case.json").read_text())
    years = np.arange(settings["forecast_start_year"], settings["forecast_end_year"] + 1)
    forecast = forecast_cohort_waste(
        cohorts, settings["base_year"], years, params["shape"], params["scale_years"]
    )
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    forecast.to_csv(output / "pv_waste_forecast_demo.csv", index=False)
    (output / "weibull_fit_demo.json").write_text(json.dumps(params, indent=2))
    print("Weibull fit:", params)
    print(forecast.to_string(index=False))
    print("\nWARNING: Demo inputs are synthetic; results are not a Bangladesh national estimate.")


if __name__ == "__main__":
    main()
