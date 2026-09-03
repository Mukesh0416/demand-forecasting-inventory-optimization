"""Unit tests for the Milestone-12 inventory optimization engine.

Covers the seven required behaviours:
1. safety stock increases with service level
2. safety stock increases with variability
3. reorder point increases with lead time
4. zero demand produces sensible inventory requirements
5. stockout calculation works correctly
6. inventory never becomes negative after stockout logic
7. ABC classification covers all series
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inventory.inventory_optimizer import (  # noqa: E402
    Z_SCORES,
    calculate_forecast_error,
    calculate_inventory_metrics,
    calculate_reorder_point,
    calculate_safety_stock,
    classify_abc,
    simulate_inventory,
)


# ---------------------------------------------------------------------------
# 1-2. Safety stock behaviour
# ---------------------------------------------------------------------------
def test_safety_stock_increases_with_service_level():
    sigma, lt = 3.0, 7
    values = [calculate_safety_stock(Z_SCORES[sl], sigma, lt)
              for sl in (0.90, 0.95, 0.99)]
    assert values[0] < values[1] < values[2]
    for sl, z in ((0.90, 1.282), (0.95, 1.645), (0.99, 2.326)):
        assert calculate_safety_stock(z, sigma, lt) == pytest.approx(
            z * sigma * np.sqrt(7)
        )


def test_safety_stock_increases_with_variability():
    z, lt = 1.645, 7
    assert (
        calculate_safety_stock(z, 1.0, lt)
        < calculate_safety_stock(z, 5.0, lt)
        < calculate_safety_stock(z, 10.0, lt)
    )
    # zero variability -> zero safety stock; never negative
    assert calculate_safety_stock(z, 0.0, lt) == 0.0


# ---------------------------------------------------------------------------
# 3. Reorder point behaviour
# ---------------------------------------------------------------------------
def test_reorder_point_increases_with_lead_time():
    forecast, ss = 4.0, 5.0
    rops = [calculate_reorder_point(forecast, lt, ss) for lt in (3, 7, 14)]
    assert rops[0] < rops[1] < rops[2]
    # ROP = ceil(f*LT) + ceil(SS)
    assert calculate_reorder_point(4.0, 7, 5.2) == int(np.ceil(28) + np.ceil(5.2))


# ---------------------------------------------------------------------------
# 4. Zero demand produces sensible requirements
# ---------------------------------------------------------------------------
def test_zero_demand_sensible_requirements():
    assert calculate_safety_stock(1.645, 0.0, 7) == 0.0
    assert calculate_reorder_point(0.0, 7, 0.0) == 0
    # zero forecast but positive sigma still reserves safety stock
    assert calculate_reorder_point(0.0, 7, 3.0) == 3
    # all-zero-demand series: no stockouts, no orders needed
    zeros = np.zeros(60)
    sim = simulate_inventory(zeros, zeros, reorder_point=0, lead_time_days=7)
    m = calculate_inventory_metrics(sim)
    assert m["stockout_units"] == 0
    assert m["stockout_days"] == 0
    assert m["service_level_actual"] == 1.0
    assert (sim["ending_inventory"] == 0).all()

# ---------------------------------------------------------------------------
# 5. Stockout calculation works correctly
# ---------------------------------------------------------------------------
def test_stockout_calculation():
    # constant demand 2/day, no replenishment (order_quantity=0),
    # start with 5 units
    demand = np.full(10, 2.0)
    forecast = np.full(10, 2.0)
    sim = simulate_inventory(demand, forecast, reorder_point=999,
                             lead_time_days=3, starting_inventory=5,
                             order_quantity=np.zeros(10))
    # 5 units available -> day 0, 1 fully served, day 2 partially (1 unit)
    assert sim.loc[0:1, "stockout_units"].sum() == 0
    assert sim.loc[2, "stockout_units"] == 1
    assert sim.loc[3:, "stockout_units"].sum() == 14
    assert sim["stockout_day"].sum() == 8  # days 2..9
    m = calculate_inventory_metrics(sim)
    assert m["total_demand"] == 20
    assert m["stockout_units"] == 15
    assert m["stockout_days"] == 8
    # a zero-demand day with empty shelves is NOT a stockout day
    demand2 = np.array([5.0, 0.0, 3.0])
    sim2 = simulate_inventory(demand2, demand2.astype(float),
                              reorder_point=999, lead_time_days=3,
                              starting_inventory=4,
                              order_quantity=np.zeros(3))
    assert not sim2.loc[1, "stockout_day"]
    assert sim2.loc[1, "ending_inventory"] == 0.0
    assert sim2["stockout_day"].sum() == 2


# ---------------------------------------------------------------------------
# 6. Inventory never becomes negative
# ---------------------------------------------------------------------------
def test_inventory_never_negative():
    rng = np.random.default_rng(42)
    for _ in range(20):
        demand = rng.integers(0, 12, size=120).astype(float)
        forecast = np.clip(demand + rng.normal(0, 2, size=120), 0, None)
        rop = rng.integers(0, 40, size=120)
        sim = simulate_inventory(demand, forecast, reorder_point=rop,
                                 lead_time_days=7, starting_inventory=10)
        assert (sim["ending_inventory"] >= 0).all()
        assert (sim["fulfilled_units"] >= 0).all()
        assert (sim["fulfilled_units"] <= sim["demand"] + 1e-9).all()
        # conservation: beginning - fulfilled = ending
        assert np.allclose(
            sim["beginning_inventory"] - sim["fulfilled_units"],
            sim["ending_inventory"],
        )

# ---------------------------------------------------------------------------
# Simulation respects lead time
# ---------------------------------------------------------------------------
def test_simulation_respects_lead_time():
    # order placed on day 0 with LT=5 must arrive on day 5, not earlier
    demand = np.full(12, 3.0)
    forecast = np.full(12, 3.0)
    sim = simulate_inventory(demand, forecast, reorder_point=30,
                             lead_time_days=5, starting_inventory=9)
    placed = sim.index[sim["order_placed_units"] > 0].tolist()
    assert placed, "expected at least one order to be placed"
    first = placed[0]
    # stockout must occur before the replenishment arrives
    # (start 9 units, demand 3/day -> empty at start of day 3)
    assert sim.loc[3:4, "stockout_units"].sum() > 0
    assert sim.loc[4, "ending_inventory"] == 0.0
    # inventory strictly increases on the arrival day
    assert (
        sim.loc[first + 5, "beginning_inventory"]
        > sim.loc[first + 4, "ending_inventory"]
    )
    # nothing arrives between the order and its arrival day
    assert (
        sim.loc[first + 1: first + 4, "beginning_inventory"].diff().iloc[1:] <= 0
    ).all()


# ---------------------------------------------------------------------------
# 7. ABC classification covers all series
# ---------------------------------------------------------------------------
def test_abc_covers_all_series():
    rng = np.random.default_rng(42)
    ids = [f"s{i}" for i in range(50)]
    totals = pd.Series(rng.gamma(0.8, 10, size=50), index=ids)
    abc = classify_abc(totals)
    assert len(abc) == 50
    assert set(abc.index) == set(ids)
    assert set(abc["abc_class"]) <= {"A", "B", "C"}
    assert abc["abc_class"].notna().all()
    assert abc["cumulative_demand_percentage"].max() == pytest.approx(1.0)
    assert abc["demand_percentage"].sum() == pytest.approx(1.0)
    # the top-demand series must be an A item
    assert abc.loc[totals.idxmax(), "abc_class"] == "A"
    # all-zero demand edge case: everything is C, still fully covered
    abc0 = classify_abc(pd.Series(0.0, index=ids))
    assert (abc0["abc_class"] == "C").all()
    assert len(abc0) == 50


# ---------------------------------------------------------------------------
# Forecast-error statistics sanity
# ---------------------------------------------------------------------------
def test_forecast_error_stats():
    actual = np.array([2.0, 4.0, 6.0, 8.0])
    forecast = np.array([1.0, 4.0, 8.0, 8.0])
    stats = calculate_forecast_error(actual, forecast)
    err = actual - forecast
    assert stats["MAE"] == pytest.approx(np.mean(np.abs(err)))
    assert stats["RMSE"] == pytest.approx(np.sqrt(np.mean(err ** 2)))
    assert stats["mean_error"] == pytest.approx(np.mean(err))
    assert stats["std_error"] == pytest.approx(np.std(err, ddof=1))
    # rows with NaN in either input are ignored
    actual2 = np.array([2.0, np.nan, 6.0, 8.0])
    assert calculate_forecast_error(actual2, forecast)["n"] == 3



from inventory.inventory_optimizer import portfolio_turnover  # noqa: E402
from pathlib import Path as _Path  # noqa: E402


# ---------------------------------------------------------------------------
# Portfolio-level inventory turnover (Milestone 15B regression)
# ---------------------------------------------------------------------------
def test_portfolio_turnover_uses_portfolio_sums_not_mean_of_per_series():
    # High-volume fast-turning series vs low-volume slow-turning series.
    per = pd.DataFrame({
        "total_demand": [1000.0, 100.0],
        "average_inventory": [50.0, 100.0],
    })
    per_series_turn = per["total_demand"] / per["average_inventory"]
    assert per_series_turn.tolist() == [20.0, 1.0]
    portfolio = portfolio_turnover(
        per["total_demand"].sum(), per["average_inventory"].sum()
    )
    # portfolio = 1100 / 150 = 7.333...
    assert portfolio == pytest.approx(1100.0 / 150.0)
    # must NOT equal the mean of the per-series turnovers (10.5)
    assert portfolio != pytest.approx(per_series_turn.mean())


def test_commited_scenarios_turnover_is_portfolio_ratio():
    p = _Path(__file__).resolve().parents[1] / "data" / "processed" / "inventory_scenarios.csv"
    if not p.is_file():
        pytest.skip("inventory_scenarios.csv not present")
    sc = pd.read_csv(p)
    assert (sc["total_demand"] > 0).all()
    assert (sc["average_inventory"] > 0).all()
    expected = sc["total_demand"] / sc["average_inventory"]
    pd.testing.assert_series_equal(
        sc["inventory_turnover"], expected,
        check_names=False, check_dtype=False, check_exact=False, atol=1e-9,
    )
    rec = sc[(sc["window"] == "validation") & (sc["service_level"] == 0.99)
             & (sc["lead_time"] == 3)].iloc[0]
    assert rec["inventory_turnover"] == pytest.approx(67120.0 / 2826.574468)
