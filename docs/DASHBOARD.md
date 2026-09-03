# Dashboard Documentation

**Milestone 13 — Supply Chain Analytics Dashboard**

## Purpose

The Streamlit dashboard presents the completed project results — demand
forecasting, the audited model evaluation, scenario-based inventory
optimization, ABC analysis and business recommendations — as a
supply-chain analytics application. It is **presentation/analytics only**:
it loads processed artifacts and does not retrain, retune or recompute any
model, and it does not modify any processed file.

> **Scenario assumptions.** M5 provides daily demand only. Lead times,
> service-level targets, starting inventory and replenishment quantities are
> **scenario assumptions for inventory simulation**, and the dashboard labels
> them as such everywhere they appear.

## How to Run Locally

```bash
streamlit run dashboard/app.py
```

Requirements: Python 3.12+, `streamlit`, `pandas`, `numpy`,
`matplotlib` (see `requirements.txt`). The dashboard reads only
`data/processed/` artifacts — no raw M5 files are needed at runtime.

## Pages

| # | Page | Content |
|---|---|---|
| 1 | **Executive Overview** | KPI cards (Total Demand, Series, Recommended SL/Lead Time, Average Inventory, Weighted Fill Rate, Stockout Rate, Inventory Turnover), the RECOMMENDED POLICY banner (99% SL / 3-day LT with validation metrics), executive summary |
| 2 | **Demand Forecasting** | Store/item/date-range filters; actual vs MA-28 forecast chart; audited MA-28 / RF / XGBoost metrics; demand-behavior indicators (zero-demand rate, mean, std); "MA-28 is the selected primary forecasting model" |
| 3 | **Inventory Optimization** | SL (90/95/99) and lead-time (3/7/14) controls; policy metrics (lead-time demand, safety stock, ROP, avg/max inventory, stockouts, fill rate, turnover) from `inventory_policy.parquet` + `inventory_scenarios.csv`; ROP methodology explainer |
| 4 | **Scenario Analysis** | Full 3×3 grid (fill rate, avg inventory, stockout rate); three trade-off charts; recommended scenario highlighted; **VALIDATION = selection basis / TEST = final evaluation only** distinction |
| 5 | **ABC Analysis** | A/B/C counts (96/90/114) and demand contribution (79.7/15.2/5.1%); Pareto chart; per-class inventory performance; ABC-class filter |
| 6 | **Business Cases** | Three representative profiles (stable/high, intermittent, volatile); per-case policy metrics and comparison chart |
| 7 | **Recommendations** | The five evidence-backed recommendations, each labeled as an analytical conclusion from this development subset/scenario simulation |

## Data Sources (read-only)

- `features_dev.parquet` — demand history, MA-28 forecast, store/item info
- `audited_model_comparison.csv` — audited Milestone-11 metrics
- `baseline_results.csv`, `ml_results.csv`, `ml_predictions.parquet` — original model outputs
- `inventory_policy.parquet` — per (id, date, SL, LT) policy parameters
- `inventory_scenarios.csv` — 3×3 scenario grid (validation + test)
- `inventory_abc_series.csv`, `inventory_abc_performance.csv` — ABC classes and per-class performance
- `inventory_business_cases.csv` — three representative demand profiles
- `inventory_recommendations.csv` — recommendation logic backing

All loading goes through cached helpers in `dashboard/utils.py`
(`@st.cache_data`), loading only the columns each page needs; large parquet
files are not re-read on every widget interaction.

## Filters

- Store (`CA_1/CA_2/CA_3`), item, and date range — Demand Forecasting page
- Service level and lead time — Inventory Optimization page
- ABC class — ABC Analysis page
- Sidebar navigation across the 7 pages

## KPIs

- **Total demand** — sum of daily demand, evaluation window
- **Weighted fill rate** — `1 − stockout_units / total_demand` (Milestone-12
  definition), demand-weighted across series
- **Stockout rate** — days with unmet demand / all days (zero-demand days
  are never stockouts)
- **Average / maximum inventory** — mean/max on-hand units in simulation
- **Inventory turnover** — `total_demand / average_inventory` (simulated
  metric; documented limitation)

## Recommendation Logic

The recommended policy (**99% service level / 3-day lead time**) is loaded
directly from the Milestone-12 recommendation artifact
(`inventory_recommendations.csv`) — it was selected **on validation
performance only** as the lowest-inventory scenario meeting the ≥98%
weighted fill target on the Pareto frontier. The dashboard never selects a
scenario using test results, and test rows are always labeled
"final evaluation only".

## Limitations

- Development subset only: 300 HOBBIES_1 series across CA_1/CA_2/CA_3.
- All inventory figures are scenario simulations, not observed operations.
- No costs are modeled; the turnover KPI is simulated and not comparable to
  real retail turnover.
- Forecast metrics shown are the audited Milestone-11 results; no model is
  retrained or re-evaluated in the dashboard.
