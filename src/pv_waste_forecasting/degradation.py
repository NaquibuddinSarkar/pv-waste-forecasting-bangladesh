"""Leakage-aware machine-learning workflow for PV module degradation data.

Rows from the same physical module must not be split across training and test
sets. The workflow therefore uses module-level GroupShuffleSplit validation.
Use real, documented module measurements before interpreting performance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


NUMERIC_FEATURES = [
    "module_age_years",
    "mean_temperature_c",
    "irradiance_kwh_m2_day",
    "relative_humidity_pct",
]
CATEGORICAL_FEATURES = ["technology"]


def validate_degradation_data(df: pd.DataFrame, target_col="annual_degradation_rate"):
    """Validate a module-year panel and return a clean modelling frame."""
    required = set(
        ["module_id", target_col] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    )
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"degradation data are missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("degradation dataset cannot be empty")

    result = df[list(required)].copy()
    result["module_id"] = result["module_id"].astype(str)
    if result["module_id"].str.strip().eq("").any():
        raise ValueError("module_id cannot be blank")

    for column in NUMERIC_FEATURES + [target_col]:
        result[column] = pd.to_numeric(result[column], errors="raise")
        if not np.isfinite(result[column].dropna().to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain only finite values or missing values")
    if (result["module_age_years"].dropna() < 0).any():
        raise ValueError("module_age_years cannot be negative")
    if ((result["relative_humidity_pct"].dropna() < 0)
        | (result["relative_humidity_pct"].dropna() > 100)).any():
        raise ValueError("relative_humidity_pct must be in [0,100]")
    if result[target_col].isna().any():
        raise ValueError(f"target {target_col} cannot contain missing values")
    if (result[target_col] < 0).any():
        raise ValueError("degradation rate cannot be negative")
    result["technology"] = result["technology"].astype("string")
    if result["technology"].isna().any():
        raise ValueError("technology category cannot be missing")
    if result["module_id"].nunique() < 8:
        raise ValueError("at least eight distinct modules are needed for group validation")
    return result.reset_index(drop=True)


def _build_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def compare_degradation_models(
    df: pd.DataFrame,
    target_col="annual_degradation_rate",
    test_size=0.25,
    random_state=42,
    include_xgboost=False,
):
    """Compare Random Forest and Extra Trees on unseen physical modules.

    Returns a dictionary with fitted pipelines, comparison metrics,
    permutation importance for the leading model and split diagnostics.
    XGBoost is optional and must be installed separately when requested.
    """
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.1 and 0.5")
    data = validate_degradation_data(df, target_col=target_col)
    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data[target_col]
    groups = data["module_id"]

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    if train_groups.intersection(test_groups):
        raise RuntimeError("group split leakage detected: module IDs overlap")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    estimator_factories = {
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=350,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Extra Trees": lambda: ExtraTreesRegressor(
            n_estimators=350,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    if include_xgboost:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "XGBoost is optional. Install requirements-xgboost.txt to enable it."
            ) from exc
        estimator_factories["XGBoost"] = lambda: XGBRegressor(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )

    fitted = {}
    rows = []
    for name, factory in estimator_factories.items():
        pipeline = Pipeline([
            ("preprocess", _build_preprocessor()),
            ("model", factory()),
        ])
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)
        fitted[name] = pipeline
        rows.append({
            "model": name,
            "n_train_rows": int(len(train_idx)),
            "n_test_rows": int(len(test_idx)),
            "n_train_modules": int(len(train_groups)),
            "n_test_modules": int(len(test_groups)),
            "MAE": float(mean_absolute_error(y_test, prediction)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, prediction))),
            "R2": float(r2_score(y_test, prediction)),
        })

    comparison = pd.DataFrame(rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True)
    best_name = comparison.iloc[0]["model"]
    best_model = fitted[best_name]
    importance = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=random_state,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    feature_importance = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return {
        "models": fitted,
        "comparison": comparison,
        "best_model_name": best_name,
        "feature_importance": feature_importance,
        "train_module_ids": sorted(train_groups),
        "test_module_ids": sorted(test_groups),
        "target": target_col,
        "validation_method": "single GroupShuffleSplit by physical module ID",
        "caution": (
            "Synthetic benchmark or observational association only. "
            "Do not interpret feature importance as causal effect. "
            "Consider site/time holdouts and external validation for empirical studies."
        ),
    }
