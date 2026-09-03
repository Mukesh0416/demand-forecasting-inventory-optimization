"""M5 Data Transformation Pipeline.

Converts the M5 sales data from wide format to long format and enriches it
with calendar and price information for analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEV_STORE_LIMIT = 3
DEV_ITEM_LIMIT = 100

SALES_ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CALENDAR_USEFUL_COLUMNS = [
    "d",
    "date",
    "wm_yr_wk",
    "weekday",
    "wday",
    "month",
    "year",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
]
PRICES_USEFUL_COLUMNS = ["store_id", "item_id", "wm_yr_wk", "sell_price"]

FINAL_REQUIRED_COLUMNS = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "d",
    "demand",
    "date",
    "wm_yr_wk",
    "weekday",
    "month",
    "year",
    "sell_price",
]


def get_project_root() -> Path:
    """Return the project root directory (two levels up from this file)."""
    return Path(__file__).resolve().parent.parent.parent


def get_raw_data_dir(project_root: Path) -> Path:
    """Return the path to the raw data directory."""
    return project_root / "data" / "raw"


def get_processed_data_dir(project_root: Path) -> Path:
    """Return the path to the processed data directory."""
    return project_root / "data" / "processed"


def get_day_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of historical day columns (d_1, d_2, etc.)."""
    return [c for c in df.columns if c.startswith("d_")]

def load_sales_data(raw_dir: Path) -> pd.DataFrame:
    """Load the sales_train_validation.csv file."""
    filepath = raw_dir / "sales_train_validation.csv"
    return pd.read_csv(filepath)


def load_calendar_data(raw_dir: Path) -> pd.DataFrame:
    """Load calendar.csv with only useful columns and parse dates."""
    filepath = raw_dir / "calendar.csv"
    df = pd.read_csv(filepath, usecols=CALENDAR_USEFUL_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_prices_data(raw_dir: Path) -> pd.DataFrame:
    """Load sell_prices.csv with only useful columns."""
    filepath = raw_dir / "sell_prices.csv"
    return pd.read_csv(filepath, usecols=PRICES_USEFUL_COLUMNS)


def create_development_subset(
    df: pd.DataFrame,
    store_limit: int = DEV_STORE_LIMIT,
    item_limit: int = DEV_ITEM_LIMIT,
) -> pd.DataFrame:
    """Create a development subset by selecting limited stores and items."""
    unique_stores = df["store_id"].unique()[:store_limit]
    df_filtered = df[df["store_id"].isin(unique_stores)]

    unique_items = df_filtered["item_id"].unique()[:item_limit]
    df_subset = df_filtered[df_filtered["item_id"].isin(unique_items)]

    return df_subset.copy()


def convert_wide_to_long(
    df: pd.DataFrame,
    day_columns: list[str],
) -> pd.DataFrame:
    """Convert wide-format sales data to long format using melt."""
    melted = pd.melt(
        df,
        id_vars=SALES_ID_COLUMNS,
        value_vars=day_columns,
        var_name="d",
        value_name="demand",
    )
    return melted


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage with categorical dtypes."""
    categorical_columns = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "weekday"]
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df

def validate_final_dataset(df: pd.DataFrame) -> dict:
    """Run validation checks on the final dataset."""
    results = {
        "empty_dataset": len(df) == 0,
        "duplicate_id_date": int(df.duplicated(subset=["id", "d"]).sum()),
        "negative_demand": int((df["demand"] < 0).sum()) if "demand" in df.columns else 0,
        "negative_prices": int((df["sell_price"] < 0).sum()) if "sell_price" in df.columns else 0,
        "missing_demand": int(df["demand"].isna().sum()) if "demand" in df.columns else 0,
        "missing_prices": int(df["sell_price"].isna().sum()) if "sell_price" in df.columns else 0,
        "missing_dates": int(df["date"].isna().sum()) if "date" in df.columns else 0,
        "total_rows": len(df),
        "total_columns": len(df.columns),
    }
    return results


def print_divider(title: str | None = None) -> None:
    """Print a horizontal divider, optionally with a title."""
    width = 58
    if title:
        print(f"\n{'=' * width}\n{title}\n{'=' * width}")
    else:
        print(f"\n{'=' * width}")


def print_section(title: str) -> None:
    """Print a section header."""
    width = 58
    print(f"\n{'-' * width}")
    print(f"[{title}]")
    print(f"{'-' * width}")



def validate_final_dataset(df: pd.DataFrame) -> dict:
    """Run validation checks on the final dataset."""
    results = {
        "empty_dataset": len(df) == 0,
        "duplicate_id_date": int(df.duplicated(subset=["id", "d"]).sum()),
        "negative_demand": int((df["demand"] < 0).sum()) if "demand" in df.columns else 0,
        "negative_prices": int((df["sell_price"] < 0).sum()) if "sell_price" in df.columns else 0,
        "missing_demand": int(df["demand"].isna().sum()) if "demand" in df.columns else 0,
        "missing_prices": int(df["sell_price"].isna().sum()) if "sell_price" in df.columns else 0,
        "missing_dates": int(df["date"].isna().sum()) if "date" in df.columns else 0,
        "total_rows": len(df),
        "total_columns": len(df.columns),
    }
    return results


def print_divider(title: str | None = None) -> None:
    """Print a horizontal divider, optionally with a title."""
    width = 58
    if title:
        print(f"\n{'=' * width}\n{title}\n{'=' * width}")
    else:
        print(f"\n{'=' * width}")


def print_section(title: str) -> None:
    """Print a section header."""
    width = 58
    print(f"\n{'-' * width}")
    print(f"[{title}]")
    print(f"{'-' * width}")


def main() -> None:
    """Run the full M5 data transformation pipeline."""
    project_root = get_project_root()
    raw_dir = get_raw_data_dir(project_root)
    processed_dir = get_processed_data_dir(project_root)

    print_divider("M5 DATA TRANSFORMATION")

    # Step 1: Load sales data
    print_section("Loading Sales Data")
    sales_df = load_sales_data(raw_dir)
    day_columns = get_day_columns(sales_df)
    print(f"  Raw sales rows:         {len(sales_df):,}")
    print(f"  Historical day columns: {len(day_columns):,}")

    # Step 2: Create development subset
    print_section("Creating Development Subset")
    sales_subset = create_development_subset(sales_df)
    selected_stores = sales_subset["store_id"].nunique()
    selected_items = sales_subset["item_id"].nunique()
    print(f"  Selected stores:        {selected_stores}")
    print(f"  Selected items:         {selected_items}")

    # Free memory from full dataset
    del sales_df

    # Step 3: Convert wide to long format
    print_section("Converting Wide to Long Format")
    long_df = convert_wide_to_long(sales_subset, day_columns)
    print(f"  Long-format rows:       {len(long_df):,}")

    # Free memory from subset
    del sales_subset

    # Step 4: Load and join calendar data
    print_section("Joining Calendar Data")
    calendar_df = load_calendar_data(raw_dir)
    print(f"  Calendar rows:          {len(calendar_df):,}")

    long_df = long_df.merge(calendar_df, on="d", how="left")
    print(f"  Rows after calendar join: {len(long_df):,}")

    # Free memory from calendar
    del calendar_df

    # Step 5: Load and join price data
    print_section("Joining Price Data")
    prices_df = load_prices_data(raw_dir)
    print(f"  Price rows:             {len(prices_df):,}")

    long_df = long_df.merge(
        prices_df,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )
    print(f"  Rows after price join:  {len(long_df):,}")

    # Free memory from prices
    del prices_df

    # Step 6: Optimize dtypes
    print_section("Optimizing Data Types")
    long_df = optimize_dtypes(long_df)
    print("  Categorical dtypes applied to ID and label columns.")

    # Step 7: Validate final dataset
    print_section("Validating Final Dataset")
    validation = validate_final_dataset(long_df)

    # Check required columns
    missing_columns = [c for c in FINAL_REQUIRED_COLUMNS if c not in long_df.columns]
    if missing_columns:
        print(f"  WARNING: Missing columns: {missing_columns}")
    else:
        print("  All required columns present.")

    print(f"  Duplicate (id, d) combinations: {validation['duplicate_id_date']:,}")
    print(f"  Negative demand values:         {validation['negative_demand']:,}")
    print(f"  Negative price values:          {validation['negative_prices']:,}")
    print(f"  Missing demand values:          {validation['missing_demand']:,}")
    print(f"  Missing price values:           {validation['missing_prices']:,}")
    print(f"  Missing date values:            {validation['missing_dates']:,}")

    # Step 8: Print final summary
    print_section("FINAL DATASET")
    print(f"  Rows:               {len(long_df):,}")
    print(f"  Columns:            {len(long_df.columns)}")
    print(f"  Date range:         {long_df['date'].min()} to {long_df['date'].max()}")
    print(f"  Unique products:    {long_df['id'].nunique():,}")
    print(f"  Unique stores:      {long_df['store_id'].nunique():,}")
    print(f"  Missing demand:     {validation['missing_demand']:,}")
    print(f"  Missing prices:     {validation['missing_prices']:,}")
    print(f"  Duplicate item-date: {validation['duplicate_id_date']:,}")

    # Step 9: Save to parquet
    print_section("Saving Processed Data")
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "demand_dev.parquet"
    long_df.to_parquet(output_path, index=False)
    print(f"  Output: {output_path}")

    # Get file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")

    print_divider("TRANSFORMATION COMPLETE")


if __name__ == "__main__":
    main()

