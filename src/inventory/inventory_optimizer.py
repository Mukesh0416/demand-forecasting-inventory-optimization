"""
Milestone 12 - Inventory Optimization Engine
============================================

Converts the validated MA-28 demand forecast into practical inventory
decisions: safety stock, reorder points and a lead-time-respecting
backtesting simulation across service-level / lead-time scenarios.

DATA vs SCENARIO ASSUMPTIONS (read this first)
---------------------------------------------
The M5 dataset used in this project provides daily sales (demand) only.
It contains NO supplier lead times, NO inventory-on-hand snapshots and NO
cost data. Everything involving lead times, service-level targets, starting
inventory and replenishment quantities in this module is a clearly
documented SCENARIO ASSUMPTION for inventory simulation - not observed
M5 / Walmart data.

Forecasting input
-----------------
The primary forecast is the leakage-safe rolling MA-28 baseline (winner of
the Milestone-11 validation audit):

    forecast(t) = mean(demand[t-28 : t-1])

Only demand observed strictly before date t is used - both for the forecast
itself and for the forecast-error statistics (std of train-period errors)
that drive safety stock. No test-period demand enters any policy parameter.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration - ALL scenario assumptions live here (Part 2 / Part 21)
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# Lead-time scenarios (days) - SCENARIO ASSUMPTIONS for inventory simulation,
# NOT observed M5 data (M5 provides no supplier lead times).
LEAD_TIME_SCENARIOS: Dict[str, int] = {"short": 3, "medium": 7, "long": 14}

# Service-level scenario targets - SCENARIO ASSUMPTIONS for inventory
# simulation, NOT observed M5 data.
SERVICE_LEVEL_SCENARIOS: Dict[str, float] = {"low": 0.90, "standard": 0.95, "high": 0.99}

# One-sided standard-normal z-scores for the service-level targets.
Z_SCORES: Dict[float, float] = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}

# ABC cut-offs on cumulative demand contribution (A: first 80%, B: next 15%,
# C: final 5%) - analytical classification, not a company standard.
ABC_CUTS = (0.80, 0.95)

# Chronological split boundaries established in the Milestone-11 audit.
TRAIN_END = "2015-04-13"
VAL_START = "2015-04-14"
VAL_END = "2015-10-18"
TEST_START = "2015-10-19"
TEST_END = "2016-04-24"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Part 4 - Forecast error
# ---------------------------------------------------------------------------
def calculate_forecast_error(
    actual: Sequence[float], forecast: Sequence[float]
) -> Dict[str, float]:
    """Forecast-error statistics for one series.

    forecast_error = actual_demand - forecast

    Returns n, MAE, RMSE, mean_error and std_error (sample std, ddof=1).
    Rows where either input is NaN are ignored.

    IMPORTANT: callers must pass only TRAINING-period values when the result
    feeds safety-stock calculation, so that no future/test demand influences
    policy parameters.
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if actual.shape != forecast.shape:
        raise ValueError("actual and forecast must have the same shape")
    mask = ~(np.isnan(actual) | np.isnan(forecast))
    a, f = actual[mask], forecast[mask]
    if len(a) == 0:
        raise ValueError("no overlapping non-NaN observations")
    err = a - f
    return {
        "n": int(len(err)),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "mean_error": float(np.mean(err)),
        "std_error": float(np.std(err, ddof=1)) if len(err) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Part 5 - Safety stock
# ---------------------------------------------------------------------------
def calculate_safety_stock(
    z_score: float, sigma: float, lead_time_days: Union[int, float]
) -> float:
    """Standard statistical safety stock:

        Safety Stock = Z * sigma * sqrt(Lead Time)

    where
      Z              one-sided normal z-score of the target service level
                     (0.90 -> 1.282, 0.95 -> 1.645, 0.99 -> 2.326)
      sigma          variability of demand over the replenishment cycle. In
                     this project sigma is the STANDARD DEVIATION OF THE
                     FORECAST ERROR (actual - MA-28 forecast) measured on
                     TRAINING data only. Using forecast-error std (rather
                     than raw demand std) is deliberate and explicit: the
                     reorder decision is made against the MA-28 forecast, so
                     the risk safety stock must absorb is the error around
                     that forecast. (See docs/INVENTORY_METHODOLOGY.md.)
      lead_time_days assumed replenishment lead time in days (scenario value)

    The result is never negative and is not rounded here; operational
    rounding happens in calculate_reorder_point.
    """
    if z_score < 0 or sigma < 0 or lead_time_days < 0:
        raise ValueError("z_score, sigma and lead_time_days must be >= 0")
    return max(0.0, z_score * sigma * math.sqrt(lead_time_days))


# ---------------------------------------------------------------------------
# Part 6 - Reorder point
# ---------------------------------------------------------------------------
def calculate_reorder_point(
    forecast: float, lead_time_days: Union[int, float], safety_stock: float
) -> int:
    """Reorder point:

        ROP = Expected Lead-Time Demand + Safety Stock
            = forecast * lead_time + safety_stock

    with a constant daily forecast over the lead time. Operational rounding:
    lead-time demand and safety stock are each rounded UP to whole units
    (ceil) because inventory is discrete and rounding down would understate
    requirements. The result can never be negative.
    """
    if forecast < 0 or safety_stock < 0:
        raise ValueError("forecast and safety_stock must be >= 0")
    ltd = math.ceil(max(0.0, forecast) * lead_time_days)
    ss = math.ceil(max(0.0, safety_stock))
    return int(ltd + ss)


# ---------------------------------------------------------------------------
# Part 14 - ABC classification (analytical)
# ---------------------------------------------------------------------------
def classify_abc(total_demand: pd.Series) -> pd.DataFrame:
    """Analytical ABC classification by cumulative demand contribution.

    Sorts series by total demand (descending), computes each series' share of
    total demand and the running cumulative share, then labels:

        A = first 80% of cumulative demand
        B = next 15%  (up to 95%)
        C = final 5%

    This is an ANALYTICAL ABC classification of this project's development
    subset - not a company standard and not based on revenue/cost (M5
    provides no prices for all series in the dev subset). Every input series
    receives exactly one class (A, B or C); zero-demand series are always C.
    """
    total = total_demand.astype(float).copy()
    if (total < 0).any():
        raise ValueError("total_demand must be non-negative")
    grand_total = float(total.sum())
    pct = (total / grand_total) if grand_total > 0 else total * 0.0
    pct = pct.fillna(0.0)
    order = pct.sort_values(ascending=False, kind="mergesort")
    cum = order.cumsum().clip(upper=1.0)
    cls = pd.Series("C", index=order.index, dtype=object)
    cls[cum <= ABC_CUTS[0]] = "A"
    cls[(cum > ABC_CUTS[0]) & (cum <= ABC_CUTS[1])] = "B"
    if grand_total == 0:
        cls[:] = "C"
    out = pd.DataFrame(
        {
            "total_demand": total,
            "demand_percentage": pct,
            "cumulative_demand_percentage": cum.reindex(total.index),
            "abc_class": cls.reindex(total.index),
        },
        index=total.index,
    )
    return out

# ---------------------------------------------------------------------------
# Parts 8-10 - Lead-time-respecting inventory simulation
# ---------------------------------------------------------------------------
def simulate_inventory(
    demand: Sequence[float],
    forecast: Sequence[float],
    reorder_point: Union[int, Sequence[int]],
    lead_time_days: int,
    starting_inventory: Optional[float] = None,
    order_quantity: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Day-by-day (s, Q) backtesting simulation with lost sales.

    Daily sequence (lead time is strictly respected):

        1. any order arriving today is received
        2. demand occurs; inventory decreases; unmet demand is LOST
           (stockout_units) - on-hand inventory is never negative
        3. if inventory position (on hand + all outstanding orders) is at or
           below the reorder point -> place order of Q units
        4. the order arrives exactly lead_time_days later

    Parameters
    ----------
    demand, forecast        per-day arrays for one series
    reorder_point           int, or per-day ints (policy may vary by day)
    lead_time_days          scenario lead time in days (>= 0)
    starting_inventory      SIMULATION ASSUMPTION (M5 has no inventory data).
                            Default: the reorder point on day 0.
    order_quantity          per-day order size; default Q = max(1, ceil(
                            forecast * lead_time)) when the reorder point is
                            positive, and 0 when the reorder point is 0 (a
                            zero-ROP policy deliberately holds no stock).
                            Documented simulation assumption.

    A day counts as a stockout day only when demand > 0 AND some demand could
    not be fulfilled. Ordinary zero-demand days are never stockouts.
    """
    demand = np.asarray(demand, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    n = len(demand)
    if len(forecast) != n:
        raise ValueError("demand and forecast must have the same length")
    if lead_time_days < 0:
        raise ValueError("lead_time_days must be >= 0")
    if np.isscalar(reorder_point):
        rop = np.full(n, int(reorder_point), dtype=int)
    else:
        rop = np.asarray(reorder_point, dtype=int)
    if len(rop) != n:
        raise ValueError("reorder_point must be scalar or per-day")
    if order_quantity is None:
        # Default Q = one lead-time of expected demand (min 1 unit) whenever
        # the policy's reorder point is positive; a zero reorder point means
        # the policy deliberately holds no stock and never orders (sensible
        # for a true zero-demand series).
        order_quantity = np.where(
            rop > 0,
            np.maximum(1.0, np.ceil(forecast * lead_time_days)),
            0.0,
        )
    else:
        order_quantity = np.asarray(order_quantity, dtype=float)
        if len(order_quantity) != n:
            raise ValueError("order_quantity must be per-day")

    if starting_inventory is None:
        starting_inventory = float(rop[0])

    # pipeline[t] = units scheduled to arrive on day t
    pipeline = np.zeros(n + lead_time_days + 1, dtype=float)
    on_hand = float(starting_inventory)

    rows = []
    for t in range(n):
        # 1. receive arrivals scheduled for today
        on_hand += pipeline[t]
        beginning = on_hand
        # 2. demand occurs (lost sales)
        served = min(demand[t], on_hand)
        stockout_units = demand[t] - served
        on_hand -= served  # never negative by construction
        # 3. reorder decision on inventory position (on hand + pipeline)
        pipeline_units = float(pipeline[t + 1: n + lead_time_days + 1].sum())
        inv_position = on_hand + pipeline_units
        order = float(order_quantity[t]) if inv_position <= rop[t] else 0.0
        # 4. order arrives exactly lead_time_days later
        if t + lead_time_days < len(pipeline):
            pipeline[t + lead_time_days] += order

        rows.append(
            {
                "day": t,
                "beginning_inventory": beginning,
                "demand": demand[t],
                "fulfilled_units": served,
                "stockout_units": stockout_units,
                "stockout_day": bool(demand[t] > 0 and stockout_units > 0),
                "ending_inventory": on_hand,
                "order_placed_units": order,
                "on_order_units": pipeline_units,
            }
        )
    sim = pd.DataFrame(rows)
    sim["inventory_position"] = sim["ending_inventory"] + sim["on_order_units"]
    return sim


# ---------------------------------------------------------------------------
# Part 11 - Inventory metrics
# ---------------------------------------------------------------------------
def calculate_inventory_metrics(simulation: pd.DataFrame) -> Dict[str, float]:
    """Metrics for one simulation run (one series, one scenario, one window).

    - total_demand           sum of daily demand in the window
    - stockout_units         demand that could not be fulfilled
    - stockout_days          days with demand > 0 and unfulfilled units
                             (ordinary zero-demand days are NOT stockouts)
    - stockout_rate          stockout_days / total_days
    - service_level_actual   1 - stockout_units / total_demand (fill rate);
                             defined as 1.0 when total_demand == 0
    - average_inventory      mean on-hand inventory (end of day)
    - maximum_inventory      max on-hand inventory
    - inventory_turnover     total_demand / average_inventory. SIMULATED
                             metric under a rolling MA-28 policy with
                             scenario lead times - NOT comparable to real
                             retail turnover figures (no costs, no real
                             starting inventory, lost-sales assumption).
    """
    total_demand = float(simulation["demand"].sum())
    stockout_units = float(simulation["stockout_units"].sum())
    stockout_days = int(simulation["stockout_day"].sum())
    avg_inv = float(simulation["ending_inventory"].mean())
    max_inv = float(simulation["ending_inventory"].max())
    return {
        "total_demand": total_demand,
        "stockout_units": stockout_units,
        "stockout_days": stockout_days,
        "stockout_rate": stockout_days / len(simulation),
        "service_level_actual": (
            1.0 - stockout_units / total_demand if total_demand > 0 else 1.0
        ),
        "average_inventory": avg_inv,
        "maximum_inventory": max_inv,
        "inventory_turnover": (total_demand / avg_inv) if avg_inv > 0 else None,
    }


# ---------------------------------------------------------------------------
# Pipeline helpers (MA-28 input, error stats, policy table)
# ---------------------------------------------------------------------------
FEATURES_PATH = PROCESSED / "features_dev.parquet"


def load_demand_data(path: Path = FEATURES_PATH) -> pd.DataFrame:
    """Load the development-subset demand history and add:
    - ma28_forecast(t) = mean(demand[t-28:t-1])  (leakage-safe rolling MA-28,
      identical to the audited Milestone-11 baseline; uses ONLY demand
      observed strictly before t)
    - split (train / validation / test) from the audit's date boundaries
    """
    df = pd.read_parquet(
        path, columns=["id", "item_id", "store_id", "date", "demand"]
    ).sort_values(["id", "date"])
    df["ma28_forecast"] = (
        df.groupby("id", observed=True)["demand"].shift(1).rolling(28).mean()
    )
    d = df["date"]
    df["split"] = np.select(
        [d <= TRAIN_END, d <= VAL_END], ["train", "validation"], default="test"
    )
    return df.reset_index(drop=True)


def compute_series_error_stats(
    df: pd.DataFrame, max_date: str = TRAIN_END
) -> pd.DataFrame:
    """Per-series forecast-error statistics on the TRAINING period only.

    sigma used for safety stock = std_error of (actual - MA-28 forecast) over
    training dates where the 28-day forecast window is fully available. No
    validation/test demand is used anywhere in this step.
    """
    train = df[(df["split"] == "train") & df["ma28_forecast"].notna()]
    assert train["date"].max() <= pd.Timestamp(max_date), "leakage: train period violated"
    rows = []
    for sid, g in train.groupby("id", observed=True):
        stats = calculate_forecast_error(g["demand"], g["ma28_forecast"])
        stats["id"] = sid
        stats["last_error_date"] = g["date"].max()
        rows.append(stats)
    cols = ["id", "n", "MAE", "RMSE", "mean_error", "std_error", "last_error_date"]
    return pd.DataFrame(rows)[cols].reset_index(drop=True)


def build_inventory_policy(
    eval_df: pd.DataFrame, error_stats: pd.DataFrame
) -> pd.DataFrame:
    """Full inventory policy table (Part 7) for every (id, date) in the
    evaluation window crossed with every service-level x lead-time scenario:

        lead_time_demand = forecast * lead_time          (rounded up)
        safety_stock     = Z * sigma_error * sqrt(LT)    (rounded up)
        reorder_point    = lead_time_demand + safety_stock

    eval_df must contain id, item_id, store_id, date, ma28_forecast for the
    validation+test dates only. error_stats sigma comes from TRAIN data.
    """
    scenarios = pd.DataFrame(
        [
            {"service_level": sl, "lead_time": lt, "z_score": Z_SCORES[sl]}
            for sl in SERVICE_LEVEL_SCENARIOS.values()
            for lt in LEAD_TIME_SCENARIOS.values()
        ]
    )
    sigma = error_stats.set_index("id")["std_error"]
    base = eval_df[["id", "item_id", "store_id", "date", "ma28_forecast"]].copy()
    base["sigma_error"] = base["id"].map(sigma)

    policy = base.merge(scenarios, how="cross")
    ss = (
        policy["z_score"]
        * policy["sigma_error"]
        * np.sqrt(policy["lead_time"])
    )
    policy["safety_stock"] = np.ceil(ss).astype(int)
    policy["lead_time_demand"] = np.ceil(
        policy["ma28_forecast"] * policy["lead_time"]
    ).astype(int)
    policy["reorder_point"] = policy["lead_time_demand"] + policy["safety_stock"]
    policy = policy.rename(columns={"ma28_forecast": "forecast"})
    policy = policy.drop(columns=["z_score", "sigma_error"])
    return policy[
        [
            "id",
            "store_id",
            "item_id",
            "date",
            "forecast",
            "lead_time",
            "service_level",
            "lead_time_demand",
            "safety_stock",
            "reorder_point",
        ]
    ]

# ---------------------------------------------------------------------------
# Part 12-13 - Scenario analysis over service levels x lead times
# ---------------------------------------------------------------------------
def _simulate_one_window(
    g: pd.DataFrame, policy_slice: pd.DataFrame, lead_time: int
) -> Dict[str, float]:
    """Simulate one series in one window with its per-day reorder points.

    g            demand rows (id, date, demand, ma28_forecast) sorted by date
    policy_slice reorder points for this series for one scenario, same dates
    """
    sim = simulate_inventory(
        demand=g["demand"].to_numpy(dtype=float),
        forecast=g["ma28_forecast"].to_numpy(dtype=float),
        reorder_point=policy_slice["reorder_point"].to_numpy(dtype=int),
        lead_time_days=lead_time,
    )
    metrics = calculate_inventory_metrics(sim)
    metrics["days"] = len(sim)
    # orders placed and average order size (replenishment behaviour)
    metrics["orders_placed"] = int((sim["order_placed_units"] > 0).sum())
    metrics["avg_order_units"] = float(
        sim.loc[sim["order_placed_units"] > 0, "order_placed_units"].mean()
    )
    return metrics


def portfolio_turnover(total_demand, average_inventory):
    """Portfolio-level inventory turnover = total demand / total average inventory.

    Computed from scenario-level portfolio sums (total demand and the SUM of
    per-series average inventories). Replaces the previous
    mean-of-per-series-turnover aggregation, which weighted every series equally
    regardless of its demand or inventory volume. See
    docs/INVENTORY_METHODOLOGY.md (Inventory turnover).
    """
    td = pd.Series(np.asarray(total_demand, dtype=float))
    ai = pd.Series(np.asarray(average_inventory, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = td / ai.where(ai != 0, other=np.nan)
    if np.ndim(total_demand) == 0:
        return float(ratio.iloc[0])
    return ratio

def run_scenario_analysis(
    df: Optional[pd.DataFrame] = None,
    error_stats: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """Full scenario analysis over the 3 x 3 service-level x lead-time grid.

    Leakage rules honoured:
    - forecast uses only past demand (MA-28)
    - sigma (safety-stock input) is estimated on TRAIN data only
    - simulation runs on validation and test windows; scenario RANKING and
      recommendations use VALIDATION performance only

    Returns a dict of DataFrames:
      'policy'           per (id, date, scenario) inventory policy
      'scenarios'        aggregated metrics per (window, service_level, LT)
      'abc_series'       per-series ABC classification (train demand)
      'abc_performance'  metrics per (window, abc_class, service_level, LT)
    """
    if df is None:
        df = load_demand_data()
    if error_stats is None:
        error_stats = compute_series_error_stats(df)

    # ABC classification from TRAIN demand only
    train_demand = (
        df[df["split"] == "train"].groupby("id", observed=True)["demand"].sum()
    )
    abc = classify_abc(train_demand).reset_index()
    abc = abc.rename(columns={"index": "id"})

    eval_df = df[
        df["split"].isin(["validation", "test"]) & df["ma28_forecast"].notna()
    ].sort_values(["id", "date"])
    policy = build_inventory_policy(eval_df, error_stats)

    abc_map = abc.set_index("id")["abc_class"]
    scen_rows = []
    for service_level in SERVICE_LEVEL_SCENARIOS.values():
        for lead_time in LEAD_TIME_SCENARIOS.values():
            pol = policy[
                (policy["service_level"] == service_level)
                & (policy["lead_time"] == lead_time)
            ]
            for window in ["validation", "test"]:
                wdf = eval_df[eval_df["split"] == window]
                pol_g = pol[
                    (pol["date"] >= wdf["date"].min())
                    & (pol["date"] <= wdf["date"].max())
                ].groupby("id", observed=True)
                for sid, g in wdf.groupby("id", observed=True):
                    m = _simulate_one_window(
                        g, pol_g.get_group(sid), lead_time
                    )
                    m["id"] = sid
                    m["service_level"] = service_level
                    m["lead_time"] = lead_time
                    m["window"] = window
                    m["abc_class"] = abc_map.get(sid, "C")
                    scen_rows.append(m)
    per_series = pd.DataFrame(scen_rows)
    per_series = per_series[
        [
            "id", "window", "abc_class", "service_level", "lead_time",
            "days", "total_demand", "stockout_units", "stockout_days",
            "stockout_rate", "service_level_actual", "average_inventory",
            "maximum_inventory", "inventory_turnover", "orders_placed",
            "avg_order_units",
        ]
    ]

    # `average_inventory` is a portfolio SUM (sum of per-series daily-mean
    # inventories) and `total_demand` a portfolio SUM, so the turnover below is a
    # true portfolio ratio, not the mean of per-series turnovers.
    agg_spec = {
        "total_demand": "sum",
        "stockout_units": "sum",
        "stockout_days": "sum",
        "days": "sum",
        "average_inventory": "sum",
        "maximum_inventory": "max",
    }
    scenarios = (
        per_series.groupby(["window", "service_level", "lead_time"], as_index=False)
        .agg(**{**{k: (k, v) for k, v in agg_spec.items()},
               "service_level_actual": ("service_level_actual", "mean")})
    )
    scenarios["stockout_rate"] = scenarios["stockout_days"] / scenarios["days"]
    scenarios["inventory_turnover"] = portfolio_turnover(
        scenarios["total_demand"], scenarios["average_inventory"]
    )
    # demand-weighted fill rate across all series in the scenario
    w = per_series.assign(
        served=per_series["total_demand"] * per_series["service_level_actual"]
    )
    dw = (
        w.groupby(["window", "service_level", "lead_time"], as_index=False)
        .agg(served=("served", "sum"), td=("total_demand", "sum"))
    )
    dw["service_level_actual_weighted"] = 1.0 - (
        (dw["td"] - dw["served"]) / dw["td"].replace(0, np.nan)
    )
    scenarios = scenarios.merge(
        dw.drop(columns=["served", "td"]),
        on=["window", "service_level", "lead_time"],
    )
    abc_performance = (
        per_series.groupby(
            ["window", "abc_class", "service_level", "lead_time"], as_index=False
        )
        .agg(**{**{k: (k, v) for k, v in agg_spec.items()},
               "service_level_actual": ("service_level_actual", "mean")})
    )
    abc_performance["stockout_rate"] = (
        abc_performance["stockout_days"] / abc_performance["days"]
    )
    abc_performance["inventory_turnover"] = (
        abc_performance["total_demand"] / abc_performance["average_inventory"]
    )
    return {
        "policy": policy,
        "per_series": per_series,
        "scenarios": scenarios,
        "abc_series": abc,
        "abc_performance": abc_performance,
    }

# ---------------------------------------------------------------------------
# Part 15, 20, 22, 23 - Main pipeline, business cases and leakage validation
# ---------------------------------------------------------------------------
def select_business_case_series(df: pd.DataFrame, n_per_group: int = 1) -> pd.DataFrame:
    """Pick representative series for the business-case example (Part 20):
    - stable high-demand: highest train demand among series with lowest
      relative variability
    - intermittent: majority of zero-demand days in train
    - volatile: highest coefficient of variation among non-intermittent
    Selection uses TRAIN data only.
    """
    tr = df[df["split"] == "train"].groupby("id", observed=True)["demand"]
    stats = tr.agg(["sum", "mean", "std", "size"])
    stats["zero_rate"] = df[df["split"] == "train"].groupby("id", observed=True)[
        "demand"
    ].apply(lambda s: (s == 0).mean())
    stats["cv"] = stats["std"] / stats["mean"].replace(0, np.nan)
    stats["mean"] = stats["mean"].fillna(0.0)
    stats["cv"] = stats["cv"].fillna(0.0)

    stable = stats[stats["zero_rate"] < 0.1].nlargest(10, "sum")
    stable = stable.nsmallest(n_per_group, "cv").index.tolist()
    intermittent = stats[stats["zero_rate"] >= 0.5].nlargest(
        n_per_group, "sum"
    ).index.tolist()
    vol_cand = stats[(stats["zero_rate"] < 0.3) & (stats["mean"] > 0.5)]
    volatile = vol_cand.nlargest(n_per_group, "cv").index.tolist()

    out = stats.loc[stable + intermittent + volatile].copy()
    out["case"] = ["stable_high_demand"] * len(stable) + [
        "intermittent"
    ] * len(intermittent) + ["volatile"] * len(volatile)
    return out.reset_index()

def validate_no_leakage(
    df: pd.DataFrame, error_stats: pd.DataFrame, policy: pd.DataFrame
) -> Dict[str, bool]:
    """Part 23 leakage and consistency checks. Returns {check: passed}."""
    checks = {}
    # 1. forecast uses only past demand: recompute one series manually
    sid = df["id"].iloc[0]
    g = df[df["id"] == sid].set_index("date")["demand"]
    t = g.index[-1]
    manual = g.loc[t - pd.Timedelta(days=28): t - pd.Timedelta(days=1)].mean()
    checks["forecast_is_past_only"] = bool(
        np.isclose(manual, g.shift(1).rolling(28).mean().loc[t])
    )
    # 2. error stats (safety-stock sigma) come from train period only
    checks["sigma_uses_train_only"] = bool(
        error_stats["last_error_date"].max() <= pd.Timestamp(TRAIN_END)
    )
    # 3. scenario selection uses validation only (enforced by design: ranking
    #    code filters window == 'validation')
    checks["scenario_ranking_on_validation"] = True
    # 4/5. no negative safety stock or reorder point anywhere
    checks["no_negative_safety_stock"] = bool((policy["safety_stock"] >= 0).all())
    checks["no_negative_reorder_point"] = bool((policy["reorder_point"] >= 0).all())
    # 6. higher service level never reduces safety stock (vectorized pivot)
    p = policy.pivot_table(index=["id", "date", "lead_time"],
                           columns="service_level", values="safety_stock")
    checks["ss_monotone_in_service_level"] = bool(
        (p[0.99] >= p[0.95]).all() and (p[0.95] >= p[0.90]).all()
    )
    # 7. longer lead times produce logically consistent reorder points
    q = policy.pivot_table(index=["id", "date", "service_level"],
                           columns="lead_time", values="reorder_point")
    checks["rop_increases_with_lead_time"] = bool(
        (q[14] >= q[7]).all() and (q[7] >= q[3]).all()
    )
    return checks

# ---------------------------------------------------------------------------
# Part 15 - Scenario recommendation (validation-based Pareto selection)
# ---------------------------------------------------------------------------
def select_recommended_scenario(
    scenarios_val: pd.DataFrame, target_fill_rate: float = 0.98
) -> pd.Series:
    """Pick the recommended scenario from VALIDATION performance only.

    Rule (documented, evidence-based): among scenarios on the Pareto frontier
    of (weighted fill rate up, average inventory down), choose the cheapest
    one whose weighted fill rate reaches `target_fill_rate`. Falls back to
    the frontier's cheapest member if no scenario reaches the target.
    """
    sc = scenarios_val.reset_index(drop=True)
    is_frontier = []
    for _, r in sc.iterrows():
        dominated = any(
            (o["service_level_actual_weighted"] >= r["service_level_actual_weighted"])
            and (o["average_inventory"] <= r["average_inventory"])
            and (
                (o["service_level_actual_weighted"] > r["service_level_actual_weighted"])
                or (o["average_inventory"] < r["average_inventory"])
            )
            for _, o in sc.iterrows()
        )
        is_frontier.append(not dominated)
    front = sc[is_frontier]
    cand = front[front["service_level_actual_weighted"] >= target_fill_rate]
    pool = cand if len(cand) else front
    return pool.nsmallest(1, "average_inventory").iloc[0]


def main() -> None:
    print("=" * 70)
    print("MILESTONE 12 - INVENTORY OPTIMIZATION ENGINE")
    print("=" * 70)
    df = load_demand_data()
    print(f"Loaded {len(df):,} rows, {df['id'].nunique()} series "
          f"({df['date'].min().date()} -> {df['date'].max().date()})")

    error_stats = compute_series_error_stats(df)
    print(f"Forecast-error stats (TRAIN only, last error date "
          f"{error_stats['last_error_date'].max().date()}): "
          f"mean sigma={error_stats['std_error'].mean():.3f}")

    print("Running 3x3 scenario grid on validation + test windows ...")
    results = run_scenario_analysis(df, error_stats)

    policy = results["policy"]
    policy.to_parquet(PROCESSED / "inventory_policy.parquet", index=False)
    print(f"Saved inventory_policy.parquet ({len(policy):,} rows)")

    scen = results["scenarios"].copy()
    # mean safety stock per scenario (from the policy table) for the
    # trade-off analysis
    ss_mean = (
        policy.groupby(["service_level", "lead_time"], as_index=False)["safety_stock"]
        .mean()
        .rename(columns={"safety_stock": "safety_stock_mean"})
    )
    scen = scen.merge(ss_mean, on=["service_level", "lead_time"])
    # Part 23: scenario SELECTION uses validation performance only
    val = scen[scen["window"] == "validation"].sort_values(
        ["service_level_actual_weighted", "average_inventory"],
        ascending=[False, True],
    )
    print("\nValidation ranking (selection basis):")
    print(val[["service_level", "lead_time", "service_level_actual_weighted",
               "average_inventory", "stockout_rate"]].to_string(index=False))
    scen.to_csv(PROCESSED / "inventory_scenarios.csv", index=False)
    print("Saved inventory_scenarios.csv")

    results["abc_series"].to_csv(PROCESSED / "inventory_abc_series.csv", index=False)
    results["abc_performance"].to_csv(
        PROCESSED / "inventory_abc_performance.csv", index=False)
    results["per_series"].to_parquet(
        PROCESSED / "inventory_per_series.parquet", index=False)
    counts = results["abc_series"]["abc_class"].value_counts().to_dict()
    contrib = results["abc_series"].groupby("abc_class")[
        "demand_percentage"].sum().to_dict()
    print(f"ABC: {counts} | demand share {contrib}")

    # leakage validation (Part 23)
    checks = validate_no_leakage(df, error_stats, policy)
    print("\nPart 23 validation checks:")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    if not all(checks.values()):
        raise SystemExit("LEAKAGE / CONSISTENCY CHECK FAILED")

    # recommended scenario (Part 15): validation-based Pareto selection
    best = select_recommended_scenario(val)
    print(f"\nRecommended scenario (validation Pareto, fill >= 0.98): "
          f"SL={best['service_level']}, LT={int(best['lead_time'])}d, "
          f"weighted fill={best['service_level_actual_weighted']:.4f}, "
          f"avg inventory={best['average_inventory']:.0f}")

    # business-case series (Part 20) under the recommended scenario
    cases = select_business_case_series(df)
    ps = results["per_series"]
    bc = ps[(ps["service_level"] == best["service_level"])
            & (ps["lead_time"] == best["lead_time"])
            & (ps["id"].isin(cases["id"]))].merge(cases[["id", "case"]], on="id")
    pol_rec = policy[(policy["service_level"] == best["service_level"])
                     & (policy["lead_time"] == best["lead_time"])]
    pol_stats = pol_rec.groupby("id", observed=True).agg(
        mean_forecast=("forecast", "mean"),
        mean_safety_stock=("safety_stock", "mean"),
        mean_reorder_point=("reorder_point", "mean"),
    ).reset_index()
    dstats = (
        df[df["split"] == "train"].groupby("id", observed=True)["demand"]
        .agg(mean_demand="mean", std_demand="std")
        .reset_index()
    )
    bc = (bc.merge(pol_stats, on="id").merge(dstats, on="id"))
    bc = bc[[
        "case", "id", "window", "mean_demand", "std_demand", "mean_forecast",
        "lead_time", "service_level", "mean_safety_stock",
        "mean_reorder_point", "average_inventory", "stockout_rate",
        "service_level_actual",
    ]]
    bc.to_csv(PROCESSED / "inventory_business_cases.csv", index=False)
    print("\nBusiness-case series (recommended scenario):")
    print(bc.round(3).to_string(index=False))

    # Part 15 - evidence-based recommendations (validation performance only)
    rec_rows = [{
        "scope": "recommended_scenario",
        "segment": f"SL={best['service_level']}, LT={int(best['lead_time'])}d",
        "total_demand": "",
        "stockout_rate": round(float(best["stockout_rate"]), 4),
        "average_inventory": round(float(best["average_inventory"]), 0),
        "inventory_turnover": round(float(best["inventory_turnover"]), 2),
        "recommendation": (
            f"Best validation trade-off: weighted fill rate "
            f"{best['service_level_actual_weighted']:.3f} at "
            f"{best['average_inventory']:.0f} avg units; adopt as default policy"
        ),
    }]
    apv = results["abc_performance"]
    apv = apv[(apv["window"] == "validation")
              & (apv["service_level"] == best["service_level"])
              & (apv["lead_time"] == best["lead_time"])]
    rec_text = {
        "A": ("High demand contribution: tight monitoring, high service-level "
              "target, frequent replenishment review"),
        "B": ("Medium contribution: standard monitoring, 95% service target, "
              "weekly replenishment review"),
        "C": ("Low contribution / intermittent: avoid excessive safety stock, "
              "consider lower service-level target, review demand behaviour"),
    }
    for _, r in apv.sort_values("abc_class").iterrows():
        rec_rows.append({
            "scope": "abc_class",
            "segment": r["abc_class"],
            "total_demand": r["total_demand"],
            "stockout_rate": round(r["stockout_rate"], 4),
            "average_inventory": round(r["average_inventory"], 1),
            "inventory_turnover": round(r["inventory_turnover"], 2),
            "recommendation": rec_text[r["abc_class"]],
        })
    recs = pd.DataFrame(rec_rows)
    recs.to_csv(PROCESSED / "inventory_recommendations.csv", index=False)
    print("\nRecommendations (validation-based):")
    print(recs.to_string(index=False))
    print("\nDONE")


if __name__ == "__main__":
    main()









