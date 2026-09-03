"""Feature Engineering for M5 Demand Forecasting."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "demand_dev.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "features_dev.parquet"
WARMUP_DAYS = 28
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    """Load processed dataset and sort by product and date."""
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values(["id", "date"]).reset_index(drop=True)
    return df


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag features: demand from previous days."""
    lags = [1, 7, 14, 28]
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("id")["demand"].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling demand features using only historical observations."""
    shifted = df.groupby("id")["demand"].shift(1)
    windows = [7, 14, 28]
    for window in windows:
        df[f"rolling_mean_{window}"] = shifted.groupby(df["id"]).transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    for window in [7, 28]:
        df[f"rolling_std_{window}"] = shifted.groupby(df["id"]).transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
    return df


def create_demand_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create demand behavior features."""
    shifted = df.groupby("id")["demand"].shift(1)
    df["zero_demand_rate_28"] = shifted.groupby(df["id"]).transform(
        lambda x: (x == 0).rolling(window=28, min_periods=1).mean()
    )
    df["mean_demand_28"] = shifted.groupby(df["id"]).transform(
        lambda x: x.rolling(window=28, min_periods=1).mean()
    )
    df["std_demand_28"] = shifted.groupby(df["id"]).transform(
        lambda x: x.rolling(window=28, min_periods=1).std()
    )
    return df

def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar features from date column."""
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    return df


def create_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create event features."""
    df["has_event"] = (
        df["event_name_1"].notna() | df["event_name_2"].notna()
    ).astype(int)
    return df


def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe price features."""
    df["price_missing"] = df["sell_price"].isna().astype(int)
    df["price_change"] = df.groupby("id")["sell_price"].diff()
    prev_price = df.groupby("id")["sell_price"].shift(1)
    df["price_change_pct"] = np.where(
        prev_price > 0,
        (df["sell_price"] - prev_price) / prev_price,
        np.nan
    )
    return df

def remove_warmup_rows(df: pd.DataFrame, warmup_days: int = WARMUP_DAYS) -> pd.DataFrame:
    """Remove rows that cannot support the required forecasting features."""
    min_date_by_id = df.groupby("id")["date"].transform("min")
    cutoff_date = min_date_by_id + pd.Timedelta(days=warmup_days)
    df = df[df["date"] >= cutoff_date].copy()
    return df


def validate_features(df: pd.DataFrame) -> dict:
    """Run validation checks on feature dataset."""
    results = {}
    duplicates = df.duplicated(subset=["id", "date"]).sum()
    results["duplicate_id_date"] = int(duplicates)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    results["infinite_values"] = int(inf_count)
    missing = df.isna().sum()
    results["missing_values"] = missing[missing > 0].to_dict()
    results["total_missing"] = int(missing.sum())
    results["dtypes"] = df.dtypes.astype(str).to_dict()
    return results


def save_features(df: pd.DataFrame) -> None:
    """Save feature dataset to parquet."""
    Path(PROJECT_ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

def main():
    """Run feature engineering pipeline."""
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    print()
    print("Loading data...")
    df = load_data()
    input_rows = len(df)
    print(f"  Input rows: {input_rows:,}")
    print(f"  Unique products: {df['id'].nunique()}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print()
    print("Creating lag features...")
    df = create_lag_features(df)
    print("Creating rolling features...")
    df = create_rolling_features(df)
    print("Creating demand behavior features...")
    df = create_demand_behavior_features(df)
    print("Creating calendar features...")
    df = create_calendar_features(df)
    print("Creating event features...")
    df = create_event_features(df)
    print("Creating price features...")
    df = create_price_features(df)
    print(f"Removing warmup rows ({WARMUP_DAYS} days per series)...")
    df = remove_warmup_rows(df, WARMUP_DAYS)
    output_rows = len(df)
    rows_removed = input_rows - output_rows
    print(f"  Rows removed: {rows_removed:,}")
    print(f"  Output rows: {output_rows:,}")
    print()
    print("Validating features...")
    validation = validate_features(df)
    print(f"  Duplicate (id, date) rows: {validation['duplicate_id_date']}")
    print(f"  Infinite values: {validation['infinite_values']}")
    print(f"  Total missing values: {validation['total_missing']:,}")
    print()
    feature_cols = [c for c in df.columns if c not in ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "date", "d", "demand"]]
    lag_features = [c for c in feature_cols if c.startswith("lag_")]
    rolling_features = [c for c in feature_cols if c.startswith("rolling_")]
    calendar_features = ["day_of_week", "day_of_month", "month", "quarter", "year", "week_of_year", "is_weekend"]
    price_features = ["price_change", "price_change_pct", "price_missing"]
    print("=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"  Input rows: {input_rows:,}")
    print(f"  Output rows: {output_rows:,}")
    print(f"  Rows removed: {rows_removed:,}")
    print(f"  Number of features: {len(feature_cols)}")
    print(f"  Lag features: {len(lag_features)}")
    print(f"  Rolling features: {len(rolling_features)}")
    print(f"  Calendar features: {len(calendar_features)}")
    print(f"  Price features: {len(price_features)}")
    print(f"  Missing values: {validation['total_missing']:,}")
    print(f"  Infinite values: {validation['infinite_values']}")
    print(f"  Duplicate (id, date) rows: {validation['duplicate_id_date']}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Output: {OUTPUT_PATH}")
    print("=" * 60)
    
    # Save features
    save_features(df)
    print("Features saved successfully.")


if __name__ == "__main__":
    main()
