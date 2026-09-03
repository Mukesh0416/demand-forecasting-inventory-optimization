# Inventory Optimization Methodology

**Milestone 12 — Demand Forecasting & Inventory Optimization**

This document explains, in simple business language, the concepts behind the
inventory optimization engine (`src/inventory/inventory_optimizer.py`) and the
assumptions we make. Read it before interpreting any numbers in
`docs/INVENTORY_RESULTS.md`.

> **Observed data vs scenario assumptions.** The M5 dataset used in this
> project provides **daily sales (demand) only**. It contains no supplier lead
> times, no inventory-on-hand history and no cost data. Every lead time,
> service-level target, starting inventory and order quantity in this
> milestone is a **scenario assumption for inventory simulation** — not
> observed M5/Walmart data.

---

## 1. Demand forecast

A demand forecast is our best estimate of how many units a product will sell
per day at a given store. The company cannot stock exactly what it sells
because sales happen continuously while replenishment happens in batches, so
every inventory decision starts from a forecast.

**Formula (MA-28, the selected method from the Milestone-11 audit):**

```
forecast(t) = mean( demand[t-28] ... demand[t-1] )
```

Only demand observed strictly before day `t` is used. MA-28 was selected
because it won on the validation set (MAE 1.0820 vs Random Forest 1.0860 and
XGBoost 1.0897); the ML models are treated as challengers.

## 2. Forecast error

The forecast is never exactly right. The forecast error is:

```
forecast_error(t) = actual_demand(t) - forecast(t)
```

We summarize it per series with MAE, RMSE, mean error and the **standard
deviation of the error (σ)**. The mean error tells us whether we
systematically over- or under-forecast; σ tells us how much surprise we must
protect against. σ is estimated **on the training period only**
(≤ 2015-04-13) — no validation or test demand enters any safety-stock number.

## 3. Lead time

Lead time is the number of days between placing a replenishment order and the
goods arriving on the shelf. If a supplier needs 7 days to deliver, an order
placed today protects demand 7 days from now.

**Scenario assumptions (not M5 data):**

| Scenario | Lead time |
|---|---|
| Short  | 3 days |
| Medium | 7 days |
| Long   | 14 days |

## 4. Lead-time demand

Lead-time demand is the total demand expected **while an order is on its
way**. With a constant daily forecast:

```
Expected lead-time demand = forecast × lead_time
```

This is the demand that must be covered by stock on hand at the moment of
reordering.

## 5. Service level

The service level is the probability (or, in this project's simulation, the
realized *fill rate* = fraction of demand served) of **not** running out.
Higher service levels require more stock. **Scenario targets:**

| Target | z-score (one-sided normal) |
|---|---|
| 90% | 1.282 |
| 95% | 1.645 |
| 99% | 2.326 |

## 6. Safety stock

Safety stock is the extra buffer held above expected demand to absorb
forecast error and demand spikes. Standard statistical formulation:

```
Safety Stock = Z × σ × sqrt(lead_time)
```

* `Z` — service-level z-score from the table above
* `σ` — demand variability over the replenishment cycle. **In this project σ
  is the standard deviation of the forecast error** (`actual - MA-28
  forecast`, training period only). This is an explicit modelling choice: the
  reorder decision is made against the MA-28 forecast, so the risk that
  safety stock must absorb is the error *around that forecast*, not the raw
  day-to-day demand spread. Using raw demand std would double-count the
  variability already captured by the forecast.
* `sqrt(lead_time)` — variability accumulates sub-linearly over the lead time
  (standard square-root rule for independent daily errors).

Assumptions behind the formula: daily forecast errors are approximately
independent and identically distributed; demand during lead time is
approximately normal; no demand correlation across days. These are standard
textbook simplifications, not properties we verified in M5 data.

## 7. Reorder point

The reorder point (ROP) is the inventory position (on hand + on order) at
which a new order must be placed:

```
ROP = expected lead-time demand + safety stock
    = forecast × lead_time + Z × σ × sqrt(lead_time)
```

Operational rounding: both lead-time demand and safety stock are rounded
**up** to whole units (inventory is discrete; rounding down understates
requirements). ROP can never be negative.

## 8. Stockout

A stockout is demand that cannot be served because inventory is insufficient:

```
stockout_units(t) = demand(t) - min(demand(t), inventory(t))
```

An **ordinary zero-demand day is NOT a stockout** — a day counts as a
stockout day only when demand > 0 **and** some demand could not be fulfilled.
Unmet demand is treated as **lost sales** (the customer leaves), not
backordered.

## 9. Holding inventory

Holding inventory ties up money and shelf space and exposes the company to
spoilage and obsolescence. We measure it with average on-hand inventory,
maximum inventory and a simulated inventory turnover:

```
inventory_turnover = total_demand / average_inventory
```

⚠ This turnover is a **simulated metric** under scenario lead times and an
assumed starting position — it is not comparable to real retail turnover
figures and involves no observed costs.

## 10. Inventory policy

An inventory policy is the rule that says **when to buy** and **how much**.
This project uses a continuous-review **(s, Q)** policy:

* **s (reorder point)** — computed per series/day from the MA-28 forecast,
  training-period σ, and the scenario service level & lead time
* **Q (order quantity)** — default one lead-time of expected demand:
  `Q = max(1, ceil(forecast × lead_time))`; when the reorder point is 0 the
  policy holds no stock and never orders. Documented simulation assumption.

Daily simulation loop (lead time strictly respected):

```
1. receive any order arriving today
2. demand occurs; inventory decreases; unmet demand is lost (stockout)
3. if inventory position (on hand + on order) ≤ ROP → place order of Q
4. the order arrives exactly lead_time days later
```

**Starting inventory (simulation assumption):** the reorder point on day 0.
M5 has no inventory data; this only seeds the simulation and is not claimed
to be historical inventory.

---

## Where the numbers come from

| Quantity | Source |
|---|---|
| Demand, dates, stores, items | **Observed M5 data** (development subset: 300 HOBBIES_1 series — 100 items × CA_1/CA_2/CA_3 stores) |
| MA-28 forecast | Leakage-safe rolling mean, validated in Milestone 11 |
| σ (forecast-error std) | Estimated on training period only (≤ 2015-04-13) |
| Lead times (3/7/14 days) | **Scenario assumption** |
| Service levels (90/95/99%) | **Scenario assumption** |
| Starting inventory = ROP | **Simulation assumption** |
| Order quantity Q | **Simulation assumption** |
| Costs (holding, stockout, ordering) | Not observed; not used (no cost optimization) |

