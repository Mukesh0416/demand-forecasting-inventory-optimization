"""Machine Learning Forecasting for M5 Demand Dataset.

Trains Random Forest and XGBoost models to forecast next-day demand
using leakage-safe features. Models are compared against the MA-28 baseline.

IMPORTANT (evaluation audit, Milestone 11):
  - MA-28 is recomputed leakage-safely on the same validation/test populations
    as the ML models, so the benchmark is directly comparable.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features_dev.parquet"
RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "ml_results.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "ml_predictions.parquet"
IMPORTANCE_PATH = PROJECT_ROOT / "data" / "processed" / "feature_importance.csv"

RANDOM_STATE = 42
TRAIN_PCT = 0.80
VAL_PCT = 0.10

# Feature groups
DEMAND_FEATURES = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
    "rolling_std_7", "rolling_std_28",
    "zero_demand_rate_28", "mean_demand_28", "std_demand_28",
]

CALENDAR_FEATURES = [
    "day_of_week", "day_of_month", "month", "quarter",
    "year", "week_of_year", "is_weekend",
]

EVENT_FEATURES = ["has_event"]

PRICE_FEATURES = ["sell_price", "price_change", "price_change_pct", "price_missing"]

CATEGORICAL_FEATURES = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]

ALL_FEATURES = DEMAND_FEATURES + CALENDAR_FEATURES + EVENT_FEATURES + PRICE_FEATURES + CATEGORICAL_FEATURES


def load_data() -> pd.DataFrame:
    """Load feature dataset and sort by product and date."""
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values(["id", "date"]).reset_index(drop=True)
    return df


def calculate_mae(y_true, y_pred) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def calculate_rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def calculate_wape(y_true, y_pred) -> float:
    """Weighted Absolute Percentage Error."""
    numerator = np.sum(np.abs(np.asarray(y_true) - np.asarray(y_pred)))
    denominator = np.sum(np.abs(np.asarray(y_true)))
    if denominator == 0:
        return np.nan
    return float(numerator / denominator)
def time_based_split(df: pd.DataFrame) -> tuple:
    """Split data chronologically into train, validation, and test sets."""
    unique_dates = df["date"].unique()
    unique_dates = np.sort(unique_dates)
    n_dates = len(unique_dates)
    n_train = int(n_dates * TRAIN_PCT)
    n_val = int(n_dates * VAL_PCT)
    train_dates = unique_dates[:n_train]
    val_dates = unique_dates[n_train: n_train + n_val]
    test_dates = unique_dates[n_train + n_val:]
    train_df = df[df["date"].isin(train_dates)].copy()
    val_df = df[df["date"].isin(val_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()
    return train_df, val_df, test_df


def encode_categoricals(df: pd.DataFrame, encoders=None, fit: bool = True) -> tuple:
    """Label-encode categorical features.

    When `fit=True`, encoders are created from the given DataFrame (training
    data only). When `fit=False`, the supplied encoders are reused and unseen
    categories are mapped to -1 to avoid crashes.
    """
    df = df.copy()
    if encoders is None:
        encoders = {}

    def resolve(le, value):
        try:
            return int(le.transform([str(value)])[0])
        except ValueError:
            return -1

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = df[col].map(lambda v: resolve(le, v))
    return df, encoders


def evaluate_model(model, X, y, name, split) -> dict:
    """Compute and return evaluation metrics for a fitted model."""
    preds = np.maximum(model.predict(X), 0)
    return {
        "model": name,
        "split": split,
        "MAE": calculate_mae(y, preds),
        "RMSE": calculate_rmse(y, preds),
        "WAPE": calculate_wape(y, preds),
    }


def evaluate_ma28(df, val_df, test_df) -> list:
    """Grade the MA-28 baseline leakage-safely on the ML populations.

    MA-28 for date t = mean of the previous 28 actual demands (shift 1).
    Uses the *full* history up to t-1 to be consistent with the audit.
    """
    hist = (
        df.sort_values(["id", "date"])
        .groupby("id")["demand"]
        .transform(lambda s: s.shift(1).rolling(window=28, min_periods=28).mean())
    )
    results = []
    for split_df, split_name in [(val_df, "validation"), (test_df, "test")]:
        idx = df.index.isin(split_df.index)
        y = df["demand"].values[idx]
        p = hist.values[idx]
        mask = ~np.isnan(p)
        denom = np.sum(np.abs(y[mask])) if mask.sum() else 0
        results.append({
            "model": "MA-28",
            "split": split_name,
            "MAE": calculate_mae(y[mask], p[mask]),
            "RMSE": calculate_rmse(y[mask], p[mask]),
            "WAPE": calculate_wape(y[mask], p[mask]),
        })
    return results
def main() -> None:
    """Run the ML forecasting pipeline and compare against MA-28."""
    print("=" * 72)
    print("ML FORECASTING (Random Forest + XGBoost)")
    print("=" * 72)

    print("\nLoading data...")
    df = load_data()
    print(f"  Rows: {len(df):,}   Unique products: {df['id'].nunique()}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    print("\nChronological split...")
    train_df, val_df, test_df = time_based_split(df)
    print(f"  Train: {len(train_df):,} rows  ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"  Val:   {len(val_df):,} rows  ({val_df['date'].min()} to {val_df['date'].max()})")
    print(f"  Test:  {len(test_df):,} rows  ({test_df['date'].min()} to {test_df['date'].max()})")
    assert train_df["date"].max() < val_df["date"].min(), "Train/validation overlap!"
    assert val_df["date"].max() < test_df["date"].min(), "Validation/test overlap!"
    print("  Chronological split verified (no leakage).")

    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    X_train = train_df[feature_cols].copy()
    y_train = train_df["demand"].copy()
    X_val = val_df[feature_cols].copy()
    y_val = val_df["demand"].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df["demand"].copy()

    print("\nEncoding categorical features...")
    encoders = {}
    X_train, encoders = encode_categoricals(X_train, encoders=encoders, fit=True)
    X_val, _ = encode_categoricals(X_val, encoders=encoders, fit=False)
    X_test, _ = encode_categoricals(X_test, encoders=encoders, fit=False)

    print("Filling missing price features...")
    for col in PRICE_FEATURES:
        if col in X_train.columns:
            X_train[col] = X_train[col].fillna(0)
            X_val[col] = X_val[col].fillna(0)
            X_test[col] = X_test[col].fillna(0)

    print("\nTraining Random Forest...")
    start = time.time()
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    rf_time = time.time() - start
    print(f"  RF training time: {rf_time:.2f}s")

    print("\nTraining XGBoost...")
    start = time.time()
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    xgb_time = time.time() - start
    print(f"  XGBoost training time: {xgb_time:.2f}s")

    print("\nEvaluating models...")
    results = []
    results.append(evaluate_model(rf_model, X_val, y_val, "Random Forest", "validation"))
    results.append(evaluate_model(rf_model, X_test, y_test, "Random Forest", "test"))
    results.append(evaluate_model(xgb_model, X_val, y_val, "XGBoost", "validation"))
    results.append(evaluate_model(xgb_model, X_test, y_test, "XGBoost", "test"))

    print("\nEvaluating MA-28 baseline (leakage-safe)...")
    results.extend(evaluate_ma28(df, val_df, test_df))

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 72)
    print("MODEL RESULTS")
    print("=" * 72)
    print(results_df.round(4).to_string(index=False))

    print("\nSaving results...")
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"  Results saved to: {RESULTS_PATH}")

    # Predictions for test set only (validation predictions not persisted)
    rf_pred = np.maximum(rf_model.predict(X_test), 0)
    xgb_pred = np.maximum(xgb_model.predict(X_test), 0)
    predictions = pd.DataFrame({
        "id": test_df["id"].values,
        "item_id": test_df["item_id"].values,
        "store_id": test_df["store_id"].values,
        "date": test_df["date"].values,
        "actual_demand": y_test.values,
        "random_forest_prediction": rf_pred,
        "xgboost_prediction": xgb_pred,
    })
    predictions.to_parquet(PREDICTIONS_PATH, index=False)
    print(f"  Predictions saved to: {PREDICTIONS_PATH}")

    # Feature importance
    importance_rows = []
    for model, name in [(rf_model, "Random Forest"), (xgb_model, "XGBoost")]:
        imp = model.feature_importances_
        df_imp = pd.DataFrame({"feature": feature_cols, "importance": imp})
        df_imp["model"] = name
        importance_rows.append(df_imp)
    importance = pd.concat(importance_rows, ignore_index=True)
    importance.to_csv(IMPORTANCE_PATH, index=False)
    print(f"  Feature importance saved to: {IMPORTANCE_PATH}")

    print("\n" + "=" * 72)
    print("ML FORECASTING COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()