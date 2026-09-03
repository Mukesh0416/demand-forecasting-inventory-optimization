"""Baseline Forecasting Models for M5 Demand Dataset.

Implements simple forecasting baselines:
- Naive (random walk)
- Moving Average (7-day and 28-day)
- Seasonal Naive (weekly, 7-day season)

All models use only historical information to avoid leakage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "demand_dev.parquet"
RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "baseline_results.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "baseline_predictions.parquet"

TRAIN_PCT = 0.80
VAL_PCT = 0.10
TEST_PCT = 0.10
RANDOM_STATE = 42


def load_data():
    """Load processed dataset and sort by product and date."""
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values(["id", "date"]).reset_index(drop=True)
    return df


def time_based_split(df, train_pct=TRAIN_PCT, val_pct=VAL_PCT):
    """Split data chronologically into train, validation, and test sets."""
    unique_dates = df["date"].unique()
    unique_dates = np.sort(unique_dates)
    n_dates = len(unique_dates)
    n_train = int(n_dates * train_pct)
    n_val = int(n_dates * val_pct)
    train_dates = unique_dates[:n_train]
    val_dates = unique_dates[n_train : n_train + n_val]
    test_dates = unique_dates[n_train + n_val :]
    train_df = df[df["date"].isin(train_dates)].copy()
    val_df = df[df["date"].isin(val_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()
    return train_df, val_df, test_df


def naive_forecast(series):
    """Naive forecast: forecast(t+1) = demand(t)."""
    return series.shift(1)


def moving_average_forecast(series, window):
    """Moving average forecast."""
    return series.shift(1).rolling(window=window, min_periods=1).mean()


def seasonal_naive_forecast(series, season_length=7):
    """Seasonal naive forecast."""
    return series.shift(season_length)

def calculate_mae(actual, predicted):
    """Mean Absolute Error."""
    mask = predicted.notna()
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(actual[mask] - predicted[mask])))


def calculate_rmse(actual, predicted):
    """Root Mean Squared Error."""
    mask = predicted.notna()
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2)))


def calculate_wape(actual, predicted):
    """Weighted Absolute Percentage Error (WAPE)."""
    mask = predicted.notna()
    if mask.sum() == 0:
        return np.nan
    numerator = np.sum(np.abs(actual[mask] - predicted[mask]))
    denominator = np.sum(np.abs(actual[mask]))
    if denominator == 0:
        return np.nan
    return float(numerator / denominator)


def calculate_mase(actual, predicted, train_series):
    """Mean Absolute Scaled Error (MASE)."""
    mask = predicted.notna()
    if mask.sum() == 0:
        return np.nan
    mae_model = np.mean(np.abs(actual[mask] - predicted[mask]))
    naive_errors = np.abs(train_series.diff().dropna())
    if len(naive_errors) == 0 or naive_errors.mean() == 0:
        return np.nan
    mae_naive = naive_errors.mean()
    return float(mae_model / mae_naive)


def evaluate_single_series(train_series, eval_series, item_id, store_id, model_name):
    """Evaluate a single model on a single product/store series."""
    full_series = pd.concat([train_series, eval_series])
    if model_name == "Naive":
        predictions = naive_forecast(full_series).loc[eval_series.index]
    elif model_name == "MA_7":
        predictions = moving_average_forecast(full_series, window=7).loc[eval_series.index]
    elif model_name == "MA_28":
        predictions = moving_average_forecast(full_series, window=28).loc[eval_series.index]
    elif model_name == "Seasonal_Naive_7":
        predictions = seasonal_naive_forecast(full_series, season_length=7).loc[eval_series.index]
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return {
        "item_id": item_id,
        "store_id": store_id,
        "model": model_name,
        "mae": calculate_mae(eval_series, predictions),
        "rmse": calculate_rmse(eval_series, predictions),
        "wape": calculate_wape(eval_series, predictions),
        "mase": calculate_mase(eval_series, predictions, train_series),
    }


def evaluate_all_models(train_df, eval_df, split_name="validation"):
    """Evaluate all baseline models on all product/store series."""
    models = ["Naive", "MA_7", "MA_28", "Seasonal_Naive_7"]
    results = []
    train_groups = train_df.groupby("id")
    eval_groups = eval_df.groupby("id")
    for product_id in eval_groups.groups.keys():
        if product_id not in train_groups.groups:
            continue
        train_series = train_groups.get_group(product_id).set_index("date")["demand"]
        eval_series = eval_groups.get_group(product_id).set_index("date")["demand"]
        item_id = train_groups.get_group(product_id)["item_id"].iloc[0]
        store_id = train_groups.get_group(product_id)["store_id"].iloc[0]
        for model_name in models:
            result = evaluate_single_series(train_series, eval_series, item_id, store_id, model_name)
            result["split"] = split_name
            results.append(result)
    return pd.DataFrame(results)

def segment_series(df):
    """Segment product/store series by demand characteristics."""
    stats = df.groupby("id")["demand"].agg(["mean", "std"]).reset_index()
    stats.columns = ["id", "mean_demand", "std_demand"]
    zero_pct = df.groupby("id")["demand"].apply(lambda x: (x == 0).mean()).reset_index()
    zero_pct.columns = ["id", "zero_pct"]
    stats = stats.merge(zero_pct, on="id")
    stats["cv"] = np.where(stats["mean_demand"] > 0, stats["std_demand"] / stats["mean_demand"], np.nan)
    median_demand = stats["mean_demand"].median()
    def classify(row):
        if row["zero_pct"] > 0.5:
            return "Intermittent"
        elif row["mean_demand"] > median_demand:
            return "High-Volume"
        else:
            return "Low-Volume"
    stats["segment"] = stats.apply(classify, axis=1)
    return stats[["id", "segment", "mean_demand", "zero_pct", "cv"]]


def generate_predictions(train_df, eval_df):
    """Generate predictions for all models."""
    eval_groups = eval_df.groupby("id")
    train_groups = train_df.groupby("id")
    predictions = []
    for product_id in eval_groups.groups.keys():
        if product_id not in train_groups.groups:
            continue
        train_series = train_groups.get_group(product_id).set_index("date")["demand"]
        eval_series = eval_groups.get_group(product_id).set_index("date")["demand"]
        full_series = pd.concat([train_series, eval_series])
        naive_pred = naive_forecast(full_series).loc[eval_series.index]
        ma7_pred = moving_average_forecast(full_series, 7).loc[eval_series.index]
        ma28_pred = moving_average_forecast(full_series, 28).loc[eval_series.index]
        seasonal_pred = seasonal_naive_forecast(full_series, 7).loc[eval_series.index]
        pred_df = pd.DataFrame({
            "id": product_id,
            "item_id": train_groups.get_group(product_id)["item_id"].iloc[0],
            "store_id": train_groups.get_group(product_id)["store_id"].iloc[0],
            "date": eval_series.index,
            "actual_demand": eval_series.values,
            "naive_prediction": naive_pred.values,
            "ma_7_prediction": ma7_pred.values,
            "ma_28_prediction": ma28_pred.values,
            "seasonal_naive_prediction": seasonal_pred.values,
        })
        predictions.append(pred_df)
    return pd.concat(predictions, ignore_index=True)

def main():
    """Run baseline forecasting pipeline."""
    print("=" * 60)
    print("BASELINE FORECASTING")
    print("=" * 60)
    print()
    print("Loading data...")
    df = load_data()
    print(f"  Rows: {len(df):,}")
    print(f"  Unique products: {df['id'].nunique()}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print()
    print("Splitting data chronologically...")
    train_df, val_df, test_df = time_based_split(df)
    print(f"  Training:   {train_df['date'].min()} to {train_df['date'].max()} ({len(train_df):,} rows)")
    print(f"  Validation: {val_df['date'].min()} to {val_df['date'].max()} ({len(val_df):,} rows)")
    print(f"  Test:       {test_df['date'].min()} to {test_df['date'].max()} ({len(test_df):,} rows)")
    assert train_df["date"].max() < val_df["date"].min(), "Train/Val overlap!"
    assert val_df["date"].max() < test_df["date"].min(), "Val/Test overlap!"
    print()
    print("  Chronological split verified (no leakage).")
    print()
    print("Segmenting series...")
    segments = segment_series(df)
    print(f"  {segments['segment'].value_counts().to_dict()}")
    print()
    print("Evaluating on validation set...")
    val_results = evaluate_all_models(train_df, val_df, "validation")
    print("Evaluating on test set...")
    train_val_df = pd.concat([train_df, val_df])
    test_results = evaluate_all_models(train_val_df, test_df, "test")
    all_results = pd.concat([val_results, test_results], ignore_index=True)
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for split in ["validation", "test"]:
        print()
        print(f"{split.upper()}:")
        split_results = all_results[all_results["split"] == split]
        summary = split_results.groupby("model")[["mae", "rmse", "wape"]].mean()
        print(summary.round(4))
    print()
    print("Generating predictions...")
    predictions = generate_predictions(train_val_df, test_df)
    print()
    print("Saving results...")
    Path(PROJECT_ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    all_results.to_csv(RESULTS_PATH, index=False)
    predictions.to_parquet(PREDICTIONS_PATH, index=False)
    print(f"  Results saved to: {RESULTS_PATH}")
    print(f"  Predictions saved to: {PREDICTIONS_PATH}")
    print()
    print("=" * 60)
    print("BASELINE FORECASTING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
