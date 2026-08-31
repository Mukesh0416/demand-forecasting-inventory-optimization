"""M5 Dataset Validation and Profiling Script.

Loads the M5 Forecasting dataset files, validates their structure,
and prints a detailed profile report to the console.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REQUIRED_FILES = [
    "calendar.csv",
    "sales_train_validation.csv",
    "sell_prices.csv",
]

SALES_ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CALENDAR_DATE_COLUMN = "date"
CALENDAR_WEEKDAY_COLUMN = "weekday"
CALENDAR_EVENT_COLUMNS = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
SELL_PRICES_COLUMNS = ["store_id", "item_id", "wm_yr_wk", "sell_price"]


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def get_raw_data_dir(project_root: Path) -> Path:
    return project_root / "data" / "raw"


def validate_files_exist(raw_dir: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (raw_dir / name).is_file()]
    if missing:
        missing_list = "\n  - ".join(missing)
        raise FileNotFoundError(
            f"Missing required data file(s) in '{raw_dir}':\n  - {missing_list}\n\n"
            "Download the M5 dataset from "
            "https://www.kaggle.com/competitions/m5-forecasting-accuracy/data "
            "and place the files in the 'data/raw/' directory."
        )


def load_csv(raw_dir: Path, filename: str) -> pd.DataFrame:
    return pd.read_csv(raw_dir / filename)


def basic_profile(df: pd.DataFrame, name: str) -> dict:
    n_rows, n_cols = df.shape
    missing_count = int(df.isna().sum().sum())
    missing_pct = (missing_count / (n_rows * n_cols) * 100) if n_rows and n_cols else 0.0
    duplicate_rows = int(df.duplicated().sum())
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    return {
        "file_name": name,
        "rows": n_rows,
        "columns": n_cols,
        "column_names": list(df.columns),
        "missing_count": missing_count,
        "missing_pct": round(missing_pct, 2),
        "duplicate_rows": duplicate_rows,
        "memory_mb": round(memory_mb, 2),
    }


def get_day_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("d_")]

def profile_sales(df: pd.DataFrame) -> dict:
    """Compute M5-specific business metrics for sales_train_validation.csv."""
    day_cols = get_day_columns(df)
    n_days = len(day_cols)
    unique_items = int(df["item_id"].nunique()) if "item_id" in df.columns else 0
    unique_depts = int(df["dept_id"].nunique()) if "dept_id" in df.columns else 0
    unique_cats = int(df["cat_id"].nunique()) if "cat_id" in df.columns else 0
    unique_stores = int(df["store_id"].nunique()) if "store_id" in df.columns else 0
    unique_states = int(df["state_id"].nunique()) if "state_id" in df.columns else 0
    if day_cols:
        day_data = df[day_cols].values.flatten()
        total_units = int(day_data.sum())
        min_daily = int(day_data.min())
        max_daily = int(day_data.max())
    else:
        total_units = 0
        min_daily = 0
        max_daily = 0
    return {
        "unique_items": unique_items,
        "unique_departments": unique_depts,
        "unique_categories": unique_cats,
        "unique_stores": unique_stores,
        "unique_states": unique_states,
        "historical_days": n_days,
        "total_units_sold": total_units,
        "min_daily_sales": min_daily,
        "max_daily_sales": max_daily,
    }


def profile_calendar(df: pd.DataFrame) -> dict:
    """Compute M5-specific business metrics for calendar.csv."""
    if CALENDAR_DATE_COLUMN in df.columns:
        dates = pd.to_datetime(df[CALENDAR_DATE_COLUMN])
        min_date = str(dates.min().date())
        max_date = str(dates.max().date())
        n_unique_dates = int(dates.nunique())
    else:
        min_date = "N/A"
        max_date = "N/A"
        n_unique_dates = 0
    weekday_dist: dict = {}
    if CALENDAR_WEEKDAY_COLUMN in df.columns:
        weekday_dist = df[CALENDAR_WEEKDAY_COLUMN].value_counts().sort_index().to_dict()
    available_event_cols = [c for c in CALENDAR_EVENT_COLUMNS if c in df.columns]
    return {
        "min_date": min_date,
        "max_date": max_date,
        "unique_dates": n_unique_dates,
        "weekday_distribution": weekday_dist,
        "event_columns": available_event_cols,
    }


def profile_sell_prices(df: pd.DataFrame) -> dict:
    """Compute M5-specific business metrics for sell_prices.csv."""
    unique_stores = int(df["store_id"].nunique()) if "store_id" in df.columns else 0
    unique_items = int(df["item_id"].nunique()) if "item_id" in df.columns else 0
    unique_weeks = int(df["wm_yr_wk"].nunique()) if "wm_yr_wk" in df.columns else 0
    if "sell_price" in df.columns:
        min_price = float(df["sell_price"].min())
        max_price = float(df["sell_price"].max())
        avg_price = float(df["sell_price"].mean())
    else:
        min_price = 0.0
        max_price = 0.0
        avg_price = 0.0
    return {
        "unique_stores": unique_stores,
        "unique_items": unique_items,
        "unique_weeks": unique_weeks,
        "min_price": round(min_price, 2),
        "max_price": round(max_price, 2),
        "avg_price": round(avg_price, 2),
    }

def validate_sales(df: pd.DataFrame) -> None:
    """Validate the sales dataset structure."""
    missing_cols = [c for c in SALES_ID_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"sales_train_validation.csv is missing required columns: {missing_cols}")
    day_cols = get_day_columns(df)
    if not day_cols:
        raise ValueError("sales_train_validation.csv contains no day columns (expected d_1, d_2, etc.).")


def validate_calendar(df: pd.DataFrame) -> None:
    """Validate the calendar dataset structure."""
    if CALENDAR_DATE_COLUMN not in df.columns:
        raise ValueError(f"calendar.csv is missing required column: '{CALENDAR_DATE_COLUMN}'")


def validate_sell_prices(df: pd.DataFrame) -> None:
    """Validate the sell-prices dataset structure."""
    missing_cols = [c for c in SELL_PRICES_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"sell_prices.csv is missing required columns: {missing_cols}")


def validate_not_empty(df: pd.DataFrame, name: str) -> None:
    """Ensure a DataFrame is not completely empty."""
    if df.empty:
        raise ValueError(f"'{name}' is completely empty (0 rows).")


def print_divider(title=None) -> None:
    """Print a horizontal divider, optionally with a title."""
    width = 50
    if title:
        print(f"\n{'=' * width}\n{title}\n{'=' * width}")
    else:
        print(f"\n{'=' * width}")


def print_basic_profile(profile: dict) -> None:
    """Print the basic profiling section."""
    print(f"  File:              {profile['file_name']}")
    print(f"  Rows:              {profile['rows']:,}")
    print(f"  Columns:           {profile['columns']}")
    print(f"  Missing Values:    {profile['missing_count']:,} ({profile['missing_pct']}%)")
    print(f"  Duplicate Rows:    {profile['duplicate_rows']:,}")
    print(f"  Memory Usage:      {profile['memory_mb']} MB")
    col_names = profile["column_names"]
    display = ", ".join(col_names[:10])
    if len(col_names) > 10:
        display += f" ... ({len(col_names)} total)"
    print(f"  Column Names:      {display}")


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'-' * 50}")
    print(f"[{title}]")
    print(f"{'-' * 50}")

def main() -> None:
    """Run the full M5 data-profiling pipeline."""
    project_root = get_project_root()
    raw_dir = get_raw_data_dir(project_root)
    print_divider("M5 DATA PROFILE")

    # Step 1: Validate file existence
    print_section("File Validation")
    try:
        validate_files_exist(raw_dir)
        for name in REQUIRED_FILES:
            print(f"  OK {name}")
    except FileNotFoundError as exc:
        print(f"\n  ERROR: {exc}")
        sys.exit(1)

    # Step 2: Load datasets
    print_section("Loading Datasets")
    calendar_df = load_csv(raw_dir, "calendar.csv")
    sales_df = load_csv(raw_dir, "sales_train_validation.csv")
    prices_df = load_csv(raw_dir, "sell_prices.csv")
    print("  OK All three datasets loaded successfully.")

    # Step 3: Basic validation
    print_section("Basic Validation")
    datasets = {
        "calendar.csv": calendar_df,
        "sales_train_validation.csv": sales_df,
        "sell_prices.csv": prices_df,
    }
    try:
        for name, df in datasets.items():
            validate_not_empty(df, name)
            print(f"  OK {name} - not empty")
        validate_sales(sales_df)
        print("  OK sales_train_validation.csv - structure OK")
        validate_calendar(calendar_df)
        print("  OK calendar.csv - structure OK")
        validate_sell_prices(prices_df)
        print("  OK sell_prices.csv - structure OK")
    except ValueError as exc:
        print(f"\n  VALIDATION ERROR: {exc}")
        sys.exit(1)

    # Step 4: Generate basic profiles
    print_section("Basic Profiling")
    for name, df in datasets.items():
        profile = basic_profile(df, name)
        print_basic_profile(profile)

    # Step 5: M5-specific business profiling
    print_section("M5 Business Metrics - sales_train_validation.csv")
    sales_metrics = profile_sales(sales_df)
    print(f"  Unique Items:          {sales_metrics['unique_items']:,}")
    print(f"  Unique Departments:    {sales_metrics['unique_departments']:,}")
    print(f"  Unique Categories:     {sales_metrics['unique_categories']:,}")
    print(f"  Unique Stores:         {sales_metrics['unique_stores']:,}")
    print(f"  Unique States:         {sales_metrics['unique_states']:,}")
    print(f"  Historical Day Cols:   {sales_metrics['historical_days']:,}")
    print(f"  Total Units Sold:      {sales_metrics['total_units_sold']:,}")
    print(f"  Min Daily Sales:       {sales_metrics['min_daily_sales']:,}")
    print(f"  Max Daily Sales:       {sales_metrics['max_daily_sales']:,}")

    print_section("M5 Business Metrics - calendar.csv")
    cal_metrics = profile_calendar(calendar_df)
    print(f"  Min Date:              {cal_metrics['min_date']}")
    print(f"  Max Date:              {cal_metrics['max_date']}")
    print(f"  Unique Dates:          {cal_metrics['unique_dates']:,}")
    event_cols = ", ".join(cal_metrics["event_columns"]) or "None"
    print(f"  Event Columns:         {event_cols}")
    print(f"  Weekday Distribution:")
    for day, count in cal_metrics["weekday_distribution"].items():
        print(f"    - {day}: {count}")

    print_section("M5 Business Metrics - sell_prices.csv")
    price_metrics = profile_sell_prices(prices_df)
    print(f"  Unique Stores:         {price_metrics['unique_stores']:,}")
    print(f"  Unique Items:          {price_metrics['unique_items']:,}")
    print(f"  Unique Weeks:          {price_metrics['unique_weeks']:,}")
    print(f"  Min Price:             ${price_metrics['min_price']:.2f}")
    print(f"  Max Price:             ${price_metrics['max_price']:.2f}")
    print(f"  Avg Price:             ${price_metrics['avg_price']:.2f}")

    # Step 6: Final summary
    print_divider("DATA VALIDATION COMPLETE")


if __name__ == "__main__":
    main()
