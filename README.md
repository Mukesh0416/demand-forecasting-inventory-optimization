# AI-Powered Demand Forecasting & Inventory Optimization

An end-to-end supply chain analytics project: demand forecasting on the M5
retail dataset, an audited model comparison, scenario-based inventory
optimization, ABC analysis, and an interactive Streamlit dashboard.

## Project Overview

The project answers two business questions:

1. **How much demand should we expect?** — forecasting with a leakage-safe
   evaluation methodology (Milestones 1-11).
2. **How much inventory should we hold and when should we reorder?** —
   converting the selected forecast into safety stock, reorder points and a
   lead-time-respecting backtesting simulation (Milestone 12).

All results are presented in a supply-chain analytics dashboard
(Milestone 13).

## Business Problem

Retail businesses must forecast product demand accurately and maintain
sufficient inventory while minimizing stockouts and excess stock. Forecast
error propagates directly into inventory decisions: too little stock means
lost sales, too much means tied-up working capital. This project quantifies
that link - from forecast quality, through forecast uncertainty, to safety
stock and reorder policy.

## Dataset

The **M5 Forecasting dataset** (Walmart retail sales, [Kaggle - M5 Forecasting
Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data))
contains daily product-level sales, calendar information, store metadata and
selling prices.

Required raw files (not stored in this repository - download and place in
data/raw/):

- calendar.csv
- sales_train_validation.csv
- sell_prices.csv

**Scope of this project:** a development subset of **300 HOBBIES_1 item-store
series** across stores CA_1, CA_2, CA_3 (100 items x 3 stores), chosen
to keep the pipeline tractable while covering stable, intermittent and
volatile demand patterns.

> **Important:** M5 provides *demand only*. It contains **no inventory
> on-hand, no supplier lead times, no ordering/holding costs**. All inventory
> results in this project are **scenario-based analytical simulations**, not
> observed Walmart operations.

## Architecture

`	ext
data/raw                  M5 source files (downloaded, not in repo)
src/data                  acquisition, validation, transformation
src/features              leakage-safe feature engineering
src/models                baseline (MA-28), Random Forest, XGBoost,
                          evaluation audit
src/inventory             inventory optimization engine + scenario runs
notebooks/                EDA, error analysis, inventory optimization
data/processed            all intermediate/final artifacts (parquet/csv)
dashboard/                Streamlit analytics application
tests/                    unit tests (inventory engine + dashboard)
docs/                     methodology and results reports
`

Pipeline: raw M5 -> validation -> transformation -> EDA -> features ->
baseline/ML models -> **evaluation audit** -> inventory scenarios -> ABC ->
dashboard.

## Forecasting Approach

- **Primary model - MA-28**: orecast(t) = mean(demand[t-28 : t-1]), a
  leakage-safe rolling mean using only demand observed strictly before 	.
- **Challengers - Random Forest & XGBoost** on lag/rolling/calendar/price
  features (leakage-safe construction).
- Chronological split: train <= 2015-04-13 . validation 2015-04-14 ->
  2015-10-18 . test 2015-10-19 -> 2016-04-24. Model selection uses
  **validation only**; the test set is used once for final reporting.

## Model Evaluation

An evaluation audit (Milestone 11) found that an earlier ML-vs-baseline
comparison used an invalid frozen MA-28 calculation (a look-ahead constant,
not a per-date forecast). After correcting it, the audited apples-to-apples
result is:

| Model | Validation MAE | Test MAE |
|---|---|---|
| **MA-28 (selected)** | **1.0820** | 1.1138 |
| Random Forest | 1.0860 | 1.1120 |
| XGBoost | 1.0897 | 1.1136 |

**MA-28 is the selected primary forecasting method** because it won on
validation. Random Forest's nominal test edge (+0.17%) is not a basis for
selection: it did not beat MA-28 on validation and wins on only 44% of
individual series. ML models are treated as challengers.

## Inventory Optimization

- **Safety stock**: Z x sigma x sqrt(lead_time), where sigma is the standard
  deviation of the MA-28 forecast error estimated on **training data only**.
- **Reorder point**: orecast x lead_time + safety_stock (rounded up).
- **Backtesting simulation**: day-by-day (s, Q) policy with orders arriving
  exactly lead_time days later; unmet demand treated as lost sales;
  inventory never negative. Starting inventory = reorder point on day 0
  (**simulation assumption**).
- **Scenario grid**: service levels {90%, 95%, 99%} x lead times
  {3, 7, 14 days} - all **scenario assumptions**, since M5 provides no
  operational data.

**Recommended policy (selected on validation): 99% service level / 3-day
lead time** - weighted fill rate 0.9884 with the lowest average inventory
among scenarios meeting the >=98% fill target. Within the modeled scenario
space, this scenario provided the best validation-based trade-off,
achieving approximately 98.8% weighted fill with substantially lower
simulated inventory than longer lead-time alternatives.

## ABC Analysis

Analytical classification of the 300 series by cumulative demand
contribution (train period): **A = first 80%** (96 series, 79.7% of demand),
**B = next 15%** (90 series, 15.2%), **C = last 5%** (114 series, 5.1%).
A-class series justify tighter monitoring and higher service targets;
C-class (often intermittent) should not carry heavy buffers. This is an
analytical classification of this project's development subset - not an
official Walmart policy.

## Dashboard

An interactive Streamlit application presenting all results:

`ash
streamlit run dashboard/app.py
`

Pages: Executive Overview . Demand Forecasting . Inventory Optimization .
Scenario Analysis . ABC Analysis . Business Cases . Recommendations.
See docs/DASHBOARD.md for details.

## Key Findings

1. **MA-28 is a strong, robust benchmark** - the ML challengers did not
   provide a robust validation improvement after the evaluation audit.
2. **Trade-offs confirmed by simulation**: higher service level -> higher
   safety stock -> higher inventory -> lower stockout rate (monotone).
3. **Lead-time reduction dominates z-score increases**: at 99% SL, moving
   14d -> 3d cuts average inventory ~60% (7,120 -> 2,827 units) while
   *improving* fill rate.
4. **Demand concentration**: ~80% of demand comes from 96 A-class series;
   C-class series are mostly intermittent and behave differently.
5. **Intermittent demand is the hard case** - all models overpredict on
   zero-demand days; conventional (s, Q) policies are least reliable there.

## Limitations

- Development subset only (300 HOBBIES_1 series, 3 stores) - not the full M5
  dataset.
- Initial, lightly tuned models; the project prioritizes evaluation
  integrity over leaderboard-style tuning.
- M5 contains **no inventory on-hand, lead-time, ordering-cost or
  holding-cost data**. All inventory metrics come from **scenario
  simulations** with documented assumptions (lead times, service levels,
  starting inventory, order quantity, lost sales) - they do not represent
  historical company performance.
- Costs are not modeled; no cost optimization was performed.

## Reproducing the Project

A technically competent person can reproduce all results in order. M5 provides
daily sales (demand) only - it contains no inventory-on-hand, no supplier
lead times, no ordering/holding costs. All inventory metrics are scenario
simulations under documented assumptions; they are not observed Walmart
operations.

### Step 1 - Environment

`ash
python -m venv .venv
.venv\Scripts\activate        # Windows (PowerShell)
# or: source .venv/bin/activate   # macOS/Linux
python -m pip install --upgrade pip
pip install -r requirements.txt
`

### Step 2 - Dataset

Download the **M5 Forecasting dataset** (Walmart retail sales) from
<https://www.kaggle.com/competitions/m5-forecasting-accuracy/data>.
Place these three files in data/raw/ (intentionally excluded from Git due to
size, ~500+ MB):

- calendar.csv
- sales_train_validation.csv
- sell_prices.csv

### Step 3 - Data profiling

`ash
python src/data/profile_data.py
`

Validates file structure and prints a profile of the raw M5 files. It writes no
outputs - it confirms the data is ready for transformation.

### Step 4 - Data transformation

`ash
python src/data/transform_data.py
`

Wide->long melt, joins calendar and sell prices, selects the development
subset (300 HOBBIES_1 item-store series across CA_1, CA_2, CA_3) ->
data/processed/demand_dev.parquet.

### Step 5 - EDA

Open 
otebooks/01_exploratory_data_analysis.ipynb for read-only exploratory
analysis of the transformed demand data.

### Step 6 - Baseline forecasting

`ash
python src/models/baseline_forecasting.py
`

Computes Naive, Seasonal-Naive-7, MA-7 and MA-28 baselines with a strict
chronological split; writes aseline_results.csv and
aseline_predictions.parquet.

otebooks/02_baseline_forecasting.ipynb reviews these results.

### Step 7 - Feature engineering

`ash
python src/features/build_features.py
`

Builds leakage-safe lag / rolling / calendar / event / price features and
removes the 28-day warmup -> data/processed/features_dev.parquet.

### Step 8 - ML forecasting (challengers only)

`ash
python src/models/ml_forecasting.py
`

Trains Random Forest and XGBoost challengers on the leakage-safe features;
writes ml_results.csv, ml_predictions.parquet, eature_importance.csv.

**IMPORTANT:** per the Milestone-11 audit, **MA-28 remains the primary
forecasting method**. Random Forest and XGBoost did not robustly outperform
it (RF's nominal test edge of +0.17% is not a basis for selection). They are
treated as challenger models throughout.

### Step 9 - Evaluation audit & error analysis

`ash
python src/models/evaluation_audit.py
`

Recomputes the common evaluation population and an apples-to-apples comparison
(leakage-safe rolling MA-28 vs RF vs XGB) -> udited_model_comparison.csv,
evaluation_population.parquet, and the udit_*.csv error-analysis
artifacts. 
otebooks/04_error_analysis.ipynb performs the error analysis.
The historical invalid frozen MA-28 comparison is excluded from all active code
and artifacts.

### Step 10 - Inventory optimization

`ash
python src/inventory/inventory_optimizer.py
`

Uses the selected MA-28 forecast and evaluates lead times {3, 7, 14} days x
service levels {90%, 95%, 99%}. It computes safety stock
(Z x sigma x sqrt(lead_time), with sigma = standard deviation of the MA-28
forecast error on **training data only**), reorder points, a continuous-review
(s, Q) backtesting simulation (orders arrive exactly lead_time days later,
unmet demand = lost sales, inventory never negative), stockout rate, fill rate
and ABC classification -> the inventory_*.csv / inventory_*.parquet
artifacts.

All lead times, service-level targets, starting inventory (= reorder point on
day 0) and order quantity Q = max(1, ceil(forecast x lead_time)) are
**scenario assumptions** (M5 provides no operational data). Results are policy
simulations, not observed operations.


otebooks/05_inventory_optimization.ipynb reviews the results.

**Recommended policy (selected on validation only):** 99% service level /
3-day lead time. Within the modeled scenario space, this scenario provided the
best validation-based trade-off, achieving approximately 98.8% weighted fill
with substantially lower simulated inventory than longer lead-time
alternatives (the lowest-inventory scenario meeting the >=98% fill target).

### Step 11 - Dashboard

`ash
streamlit run dashboard/app.py
`

The dashboard reads the processed artifacts only (no retraining / no
recomputation). See docs/DASHBOARD.md.

### Step 12 - Tests

`ash
python -m pytest tests/ -q
`

- 	ests/test_inventory_optimizer.py - 9 unit tests of the inventory engine
  (safety-stock/reorder-point math, simulation lead-time behaviour, ABC
  classification). Self-contained; need no artifacts.
- 	ests/test_dashboard.py - 25 checks that the dashboard imports, that the
  committed processed artifacts exist with the required schema, and that the
  recommendation is consistent (99% SL / 3-day LT, validation fill >= 0.98).
  These checks read the **committed dashboard artifacts**. A few
  file-existence checks cover pipeline-only intermediates
  (eatures_dev.parquet, aseline_results.csv, ml_results.csv,
  ml_predictions.parquet) that are regenerated by Steps 4-8 and are
  intentionally not committed (see the artifact table below); those specific
  checks will report missing artifacts until the pipeline above has been run.

## Processed artifacts in this repository

data/raw/ is **ignored** (large M5 source files). data/processed/ uses a
mixed strategy:

**Committed (small set the dashboard reads - ~1.7 MB total, works on a fresh
clone):**

| File | Size | Used by |
|---|---|---|
| audited_model_comparison.csv | <1 KB | dashboard, docs |
| inventory_scenarios.csv | ~3 KB | dashboard |
| inventory_abc_series.csv | ~24 KB | dashboard |
| inventory_abc_performance.csv | ~7 KB | dashboard |
| inventory_business_cases.csv | ~1 KB | dashboard |
| inventory_recommendations.csv | <1 KB | dashboard |
| evaluation_population.parquet | ~0.8 MB | dashboard |
| inventory_policy.parquet | ~0.9 MB | dashboard |

**Generated by the pipeline (git-ignored; large or regenerated):**
demand_dev.parquet, eatures_dev.parquet, aseline_results.csv,
aseline_predictions.parquet, ml_results.csv, ml_predictions.parquet,
eature_importance.csv, inventory_per_series.parquet, and the
udit_*.csv / udit_product_details.csv audit tables. These are produced
by Steps 3-9 and are not needed to launch the dashboard.

## Project Structure

`	ext
dashboard/app.py                     Streamlit dashboard (7 pages)
dashboard/utils.py                   cached data-loading helpers
src/data/                            acquisition / validation / transformation
src/features/                        leakage-safe feature engineering
src/models/                          MA-28, RF, XGBoost, evaluation audit
src/inventory/inventory_optimizer.py inventory engine + scenario runner
tests/                               unit tests (inventory + dashboard)
notebooks/                           EDA, error analysis, inventory
data/processed/                      artifacts (forecasts, policies, ABC)
docs/                                methodology & results reports
`

## Technology

Python . pandas . NumPy . scikit-learn . XGBoost . matplotlib . seaborn . Plotly . Streamlit . pytest . Parquet / pyarrow
