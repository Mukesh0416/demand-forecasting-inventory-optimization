"""Milestone 13 dashboard tests.

Verifies that the dashboard imports, that every processed artifact it relies
on exists with the required columns, and that the Milestone-12
recommendation/policy data feeding the dashboard is consistent.

These tests READ processed artifacts only - no forecasting or inventory
results are modified or recalculated.
"""

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

REQUIRED_FILES = [
    "features_dev.parquet",
    "baseline_results.csv",
    "ml_results.csv",
    "ml_predictions.parquet",
    "audited_model_comparison.csv",
    "evaluation_population.parquet",
    "inventory_policy.parquet",
    "inventory_scenarios.csv",
    "inventory_abc_series.csv",
    "inventory_abc_performance.csv",
    "inventory_business_cases.csv",
    "inventory_recommendations.csv",
]


@pytest.fixture(scope="module")
def scenarios() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "inventory_scenarios.csv")


@pytest.fixture(scope="module")
def abc_series() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "inventory_abc_series.csv")


@pytest.fixture(scope="module")
def recommendations() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "inventory_recommendations.csv")


@pytest.fixture(scope="module")
def policy() -> pd.DataFrame:
    return pd.read_parquet(
        DATA_DIR / "inventory_policy.parquet",
        columns=["id", "lead_time", "service_level", "safety_stock",
                 "reorder_point"],
    )

# ---------------------------------------------------------------------------
# 1. Dashboard imports successfully
# ---------------------------------------------------------------------------
def test_dashboard_imports():
    sys.path.insert(0, str(DASHBOARD_DIR))
    try:
        mod = importlib.import_module("app")
        assert hasattr(mod, "main"), "app.py must expose main()"
        for page in [
            "page_executive_overview", "page_demand_forecasting",
            "page_inventory_optimization", "page_scenario_analysis",
            "page_abc_analysis", "page_business_cases",
            "page_recommendations",
        ]:
            assert hasattr(mod, page), f"app.py must expose {page}()"
        assert len(mod.PAGES) == 7, "dashboard must have exactly 7 pages"
    finally:
        sys.path.remove(str(DASHBOARD_DIR))
        sys.modules.pop("app", None)
        sys.modules.pop("utils", None)


# ---------------------------------------------------------------------------
# 2. Required processed files exist
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("filename", REQUIRED_FILES)
def test_required_files_exist(filename):
    assert (DATA_DIR / filename).is_file(), f"missing artifact: {filename}"


# ---------------------------------------------------------------------------
# 3. Required columns exist
# ---------------------------------------------------------------------------
def test_audited_comparison_columns():
    aud = pd.read_csv(DATA_DIR / "audited_model_comparison.csv")
    for col in ["model", "split", "evaluation_rows", "MAE", "RMSE", "WAPE"]:
        assert col in aud.columns
    assert {"MA-28", "Random Forest", "XGBoost"} <= set(aud["model"])
    assert {"validation", "test"} <= set(aud["split"])


def test_scenarios_columns(scenarios):
    for col in [
        "service_level", "lead_time", "window", "total_demand",
        "stockout_units", "stockout_days", "service_level_actual_weighted",
        "average_inventory", "maximum_inventory", "inventory_turnover",
        "stockout_rate",
    ]:
        assert col in scenarios.columns


def test_policy_columns(policy):
    for col in ["id", "lead_time", "service_level", "safety_stock",
                "reorder_point"]:
        assert col in policy.columns

# ---------------------------------------------------------------------------
# 4. Recommendation exists and matches Milestone 12
# ---------------------------------------------------------------------------
def test_recommendation_exists(recommendations):
    assert len(recommendations) > 0
    rec = recommendations[recommendations["scope"] == "recommended_scenario"]
    assert len(rec) == 1, "exactly one recommended scenario expected"


def test_recommendation_matches_milestone_12(recommendations, scenarios):
    rec = recommendations[
        recommendations["scope"] == "recommended_scenario"].iloc[0]
    seg = rec["segment"]  # e.g. "SL=0.99, LT=3d"
    sl = float(seg.split("SL=")[1].split(",")[0])
    lt = int(seg.split("LT=")[1].rstrip("d"))
    assert (sl, lt) == (0.99, 3), "M12 recommendation is 99% SL / 3-day LT"
    # the recommended scenario must exist in the VALIDATION grid
    row = scenarios[
        (scenarios["window"] == "validation")
        & (scenarios["service_level"] == sl)
        & (scenarios["lead_time"] == lt)
    ]
    assert len(row) == 1
    # weighted fill >= 98% target used by the M12 selection rule
    assert row["service_level_actual_weighted"].iloc[0] >= 0.98


# ---------------------------------------------------------------------------
# 5-6. Three service levels and three lead times
# ---------------------------------------------------------------------------
def test_three_service_levels(scenarios):
    assert sorted(scenarios["service_level"].unique().tolist()) == [
        0.90, 0.95, 0.99]


def test_three_lead_times(scenarios):
    assert sorted(scenarios["lead_time"].unique().tolist()) == [3, 7, 14]


def test_full_grid_present(scenarios):
    for window in ["validation", "test"]:
        sub = scenarios[scenarios["window"] == window]
        assert len(sub) == 9, f"expected 3x3 grid for {window}"


# ---------------------------------------------------------------------------
# 7. ABC classes A/B/C exist
# ---------------------------------------------------------------------------
def test_abc_classes(abc_series):
    assert {"A", "B", "C"} <= set(abc_series["abc_class"])
    counts = abc_series["abc_class"].value_counts()
    assert counts.get("A") == 96 and counts.get("B") == 90 \
        and counts.get("C") == 114, "M12 ABC counts: A=96, B=90, C=114"


# ---------------------------------------------------------------------------
# 8-9. No negative safety stock / reorder point
# ---------------------------------------------------------------------------
def test_no_negative_safety_stock(policy):
    assert (policy["safety_stock"] >= 0).all()


def test_no_negative_reorder_point(policy):
    assert (policy["reorder_point"] >= 0).all()


def test_policy_grid_matches_scenarios(policy):
    assert sorted(policy["service_level"].unique().tolist()) == [
        0.90, 0.95, 0.99]
    assert sorted(policy["lead_time"].unique().tolist()) == [3, 7, 14]


