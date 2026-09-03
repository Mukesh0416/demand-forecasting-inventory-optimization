"""Data-loading and helper utilities for the Milestone-13 Streamlit dashboard.

All data comes from processed artifacts produced by Milestones 11-12.
Nothing is recalculated here: loaders only read, rename and filter.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# Milestone-12 recommendation (validation-based). Used for highlighting and
# for cross-checking that the dashboard shows the already-selected policy.
EXPECTED_RECOMMENDED_SL = 0.99
EXPECTED_RECOMMENDED_LT = 3


@st.cache_data(show_spinner="Loading model evaluation results ...")
def load_audited_comparison() -> pd.DataFrame:
    """Audited Milestone-11 model comparison (MA-28 vs RF vs XGB)."""
    return pd.read_csv(DATA_DIR / "audited_model_comparison.csv")


@st.cache_data(show_spinner="Loading scenario results ...")
def load_scenarios() -> pd.DataFrame:
    """3x3 service-level x lead-time scenario grid (validation + test)."""
    return pd.read_csv(DATA_DIR / "inventory_scenarios.csv")


@st.cache_data(show_spinner="Loading ABC classification ...")
def load_abc_series() -> pd.DataFrame:
    """ABC classification per series (window suffix stripped from id)."""
    abc = pd.read_csv(DATA_DIR / "inventory_abc_series.csv")
    abc["base_id"] = abc["id"].str.replace(r"_(validation|test)$", "", regex=True)
    parts = abc["base_id"].str.split("_")
    abc["item_id"] = parts.str[:3].str.join("_")
    abc["store_id"] = parts.str[3]
    return abc


@st.cache_data(show_spinner="Loading ABC performance ...")
def load_abc_performance() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "inventory_abc_performance.csv")


@st.cache_data(show_spinner="Loading business cases ...")
def load_business_cases() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "inventory_business_cases.csv")


@st.cache_data(show_spinner="Loading recommendations ...")
def load_recommendations() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "inventory_recommendations.csv")


@st.cache_data(show_spinner="Loading evaluation population ...")
def load_eval_population() -> pd.DataFrame:
    """Common evaluation population with actual demand + all model forecasts."""
    return pd.read_parquet(
        DATA_DIR / "evaluation_population.parquet",
        columns=[
            "id", "item_id", "store_id", "date", "demand", "split",
            "ma_28_prediction", "random_forest_prediction", "xgboost_prediction",
        ],
    )


@st.cache_data(show_spinner="Loading inventory policy ...")
def load_policy() -> pd.DataFrame:
    """Full per-day, per-scenario policy table (1.0M rows, cached once)."""
    return pd.read_parquet(
        DATA_DIR / "inventory_policy.parquet",
        columns=[
            "id", "store_id", "item_id", "date", "forecast", "lead_time",
            "service_level", "lead_time_demand", "safety_stock", "reorder_point",
        ],
    )


def get_recommended_policy() -> dict:
    """The Milestone-12 recommended scenario, parsed from the processed
    recommendations artifact (selected on VALIDATION performance only)."""
    rec = load_recommendations()
    row = rec[rec["scope"] == "recommended_scenario"].iloc[0]
    seg = row["segment"]  # e.g. "SL=0.99, LT=3d"
    sl = float(seg.split("SL=")[1].split(",")[0])
    lt = int(seg.split("LT=")[1].rstrip("d"))
    return {
        "service_level": sl,
        "lead_time": lt,
        "segment": seg,
        "stockout_rate": float(row["stockout_rate"]),
        "average_inventory": float(row["average_inventory"]),
        "inventory_turnover": float(row["inventory_turnover"]),
        "recommendation": str(row["recommendation"]),
    }


def get_scenario_row(window: str, service_level: float, lead_time: int) -> pd.Series:
    """One row of the scenario grid."""
    sc = load_scenarios()
    row = sc[
        (sc["window"] == window)
        & (sc["service_level"] == service_level)
        & (sc["lead_time"] == lead_time)
    ]
    return row.iloc[0]
