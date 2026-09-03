"""Regression tests for leakage-safe forecasting and feature engineering.

These tests use tiny deterministic series (no raw M5 data, no large artifacts).
They guard against:
  - the invalid frozen MA-28 (a look-ahead constant = mean of the last 28 obs
    of the whole dataset, applied to every row)
  - lag/rolling features that leak the current observation or cross series

Scientific context: MA-28 is the selected primary forecasting method
(forecast(t) = mean(demand[t-28 : t-1])); Random Forest / XGBoost are
challengers and must remain un-tuned.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from models.evaluation_audit import recompute_ma28, buggy_ma28_constant  # noqa: E402
from features.build_features import (  # noqa: E402
    create_lag_features,
    create_rolling_features,
    create_demand_behavior_features,
)


def _series(n=60, start=0.0, sid="s1", seed_date="2024-01-01"):
    dates = pd.date_range(seed_date, periods=n, freq="D")
    demand = np.arange(start, n + start, dtype=float)
    df = pd.DataFrame({
        "id": [sid] * n,
        "date": dates,
        "demand": demand,
        "sell_price": 1.0,
        "event_name_1": [None] * n,
        "event_name_2": [None] * n,
    })
    return df.sort_values(["id", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# MA-28 (leakage-safe rolling mean of the previous 28 actual demands)
# ---------------------------------------------------------------------------
def test_ma28_first_28_days_have_no_prediction():
    df = _series(60)
    pred = recompute_ma28(df).sort_values(["id", "date"]).reset_index(drop=True)
    # rolling(28, min_periods=28) applied to shift(1): first 28 rows per series NaN
    assert pred["ma_28_prediction"].iloc[:28].isna().all()
    assert pred["ma_28_prediction"].iloc[28:].notna().all()


def test_ma28_equals_mean_of_previous_28_actuals():
    df = _series(60)
    pred = recompute_ma28(df).sort_values(["id", "date"]).reset_index(drop=True)[
        "ma_28_prediction"
    ].values
    demand = np.arange(0, 60, dtype=float)
    for t in (28, 40, 59):
        # forecast(t) = mean of demand[t-28 : t-1]
        assert pred[t] == pytest.approx(demand[t - 28:t].mean())


def test_ma28_is_not_frozen_and_ignores_future():
    df = _series(60)
    pred = recompute_ma28(df).sort_values(["id", "date"]).reset_index(drop=True)[
        "ma_28_prediction"
    ].values
    # demand is strictly increasing -> a real rolling forecast must vary, not freeze
    assert not np.allclose(pred[28:], pred[28])
    # a future observation (day 55) must not change the forecast at day 30
    future = df.copy()
    future["demand"] = np.where(future["date"].eq(future["date"].iloc[55]), 9999.0,
                                future["demand"])
    pred_f = recompute_ma28(future).sort_values(["id", "date"]).reset_index(drop=True)[
        "ma_28_prediction"
    ].values
    assert pred_f[30] == pytest.approx(pred[30])


def test_ma28_frozen_tail_bug_is_caught():
    df = _series(60)
    pred = recompute_ma28(df).sort_values(["id", "date"]).reset_index(drop=True)[
        "ma_28_prediction"
    ].values
    frozen = buggy_ma28_constant(df)["buggy_ma28"].iloc[0]
    # the correct per-date forecast differs from the frozen constant
    assert pred[50] != pytest.approx(frozen)
    # and differs across dates (not a single constant)
    assert pred[30] != pytest.approx(pred[50])


# ---------------------------------------------------------------------------
# Feature leakage (lag / rolling must exclude the current observation)
# ---------------------------------------------------------------------------
def test_lag1_excludes_current_observation():
    d = create_lag_features(_series(40))
    assert pd.isna(d.loc[0, "lag_1"])
    assert d.loc[1, "lag_1"] == pytest.approx(d.loc[0, "demand"])
    assert d.loc[5, "lag_1"] == pytest.approx(d.loc[4, "demand"])


def test_rolling_excludes_current_observation():
    d = create_rolling_features(_series(40))
    last = len(d) - 1
    shifted = d["demand"].shift(1)
    expected = shifted.iloc[last - 6:last + 1].mean()  # last 7 shifted obs (<= t-1)
    assert d.loc[last, "rolling_mean_7"] == pytest.approx(expected)
    # the window must exclude the current-day demand (index `last`)
    assert not np.isclose(d.loc[last, "rolling_mean_7"],
                          d["demand"].iloc[last - 6:last + 1].mean())


def test_lag_features_are_grouped_per_id():
    a = _series(10, sid="s1")
    a["demand"] = np.arange(1, 11, dtype=float)
    b = _series(10, sid="s2", seed_date="2024-02-01")
    b["demand"] = np.arange(101, 111, dtype=float)
    df = pd.concat([a, b], ignore_index=True).sort_values(["id", "date"]).reset_index(drop=True)
    d = create_lag_features(df)
    # each series starts with NaN lag_1 (no leakage across the series boundary)
    assert pd.isna(d.loc[0, "lag_1"]) and pd.isna(d.loc[10, "lag_1"])
    assert d.loc[1, "lag_1"] == pytest.approx(d.loc[0, "demand"])   # s1 internal
    assert d.loc[11, "lag_1"] == pytest.approx(d.loc[10, "demand"])  # s2 internal


def test_demand_behavior_features_exclude_current():
    d = create_demand_behavior_features(_series(40))
    last = len(d) - 1
    shifted = d["demand"].shift(1)
    expected = shifted.iloc[max(0, last - 27):last + 1].mean()
    assert d.loc[last, "mean_demand_28"] == pytest.approx(expected)
