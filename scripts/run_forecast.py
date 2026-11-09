"""Run the synthetic PV retirement and material-flow forecast from the project root."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pv_waste_forecasting.model import (
    fit_weibull_lifetime,
    bootstrap_weibull_lifetime,
    forecast_cohort_waste,
    apply_material_pathways,
)


def main():
    data_dir = ROOT / "data" / "demo"
    lifetime = pd.read_csv(data_dir / "module_lifetime_observations_synthetic.csv")
    cohorts = pd.read_csv(data_dir / "pv_installation_cohorts_synthetic.csv")
    settings = json.loads((ROOT / "config" / "base_case.json").read_text())

    fit = fit_weibull_lifetime(
        lifetime["observed_lifetime_years"],
        lifetime["event_observed"],
    )
    bootstrap_ci = bootstrap_weibull_lifetime(
        lifetime["observed_lifetime_years"],
        lifetime["event_observed"],
        n_bootstrap=100,
        confidence_level=0.95,
        random_state=42,
    )
    years = np.arange(
        settings["forecast_start_year"],
        settings["forecast_end_year"] + 1,
    )
    forecast = forecast_cohort_waste(
        cohorts,
        settings["base_year"],
        years,
        fit["shape"],
        fit["scale_years"],
    )
    pathways = apply_material_pathways(
        forecast,
        reuse_share=settings.get("reuse_share_demo", 0.0),
        recycling_share_of_nonreuse=settings.get("recycling_share_of_nonreuse_demo", 0.0),
        recycling_yield=settings.get("recycling_yield_demo", 0.0),
    )

    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    forecast.to_csv(output / "pv_waste_forecast_demo.csv", index=False)
    pathways.to_csv(output / "pv_material_pathways_demo.csv", index=False)
    (output / "weibull_fit_demo.json").write_text(json.dumps(fit, indent=2))
    (output / "weibull_bootstrap_intervals_demo.json").write_text(
        json.dumps(bootstrap_ci, indent=2)
    )

    print("WEIBULL LIFETIME FIT — SYNTHETIC DATA")
    print(json.dumps(fit, indent=2))
    print("\n95% PARAMETER INTERVALS — PAIRED BOOTSTRAP")
    print(json.dumps(bootstrap_ci, indent=2))
    print("\nANNUAL PV RETIREMENT FORECAST — DEMONSTRATION ONLY")
    print(forecast.head(10).to_string(index=False))
    print("\nMATERIAL PATHWAYS — ILLUSTRATIVE SHARES")
    print(pathways.head(10).to_string(index=False))
    print("\nWARNING: Synthetic inputs and illustrative pathway assumptions. "
          "This is not a Bangladesh national PV waste estimate.")


if __name__ == "__main__":
    main()
