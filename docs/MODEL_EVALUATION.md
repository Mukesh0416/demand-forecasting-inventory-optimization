# MODEL EVALUATION — Audited Report (Milestone 11)

**Scope:** MA-28 vs Random Forest vs XGBoost on the M5 HOBBIES development subset
(300 series / 3 CA stores / 1 department).

## 1. Evaluation methodology

**Chronological splitting.** The features dataset (`features_dev.parquet`, 565,500
rows, 300 products, 2011-02-26 → 2016-04-24) is split strictly by date, never
randomly:

| split | dates | rows |
|---|---|---|
| train | 2011-02-26 → 2015-04-13 | 452,400 |
| validation | 2015-04-14 → 2015-10-18 | 56,400 |
| test | 2015-10-19 → 2016-04-24 | 56,700 |

**Common evaluation population** (`data/processed/evaluation_population.parquet`):
every model is scored on exactly the same `(id, date, demand)` rows —
113,100 rows (56,400 validation + 56,700 test) over 300 series, verified to have
one row per `(id, date)` and all three predictions present on every test row.
Metrics are computed identically for every model:

- MAE = `mean |y − ŷ|`
- RMSE = `sqrt(mean (y − ŷ)²)`
- WAPE = `Σ |y − ŷ| / Σ y`

**Model selection uses validation only.** The test set is touched once, for final
reporting.

## 2. Baseline — MA-28

MA-28 forecasts demand at date *t* as the mean of the 28 actual demands
immediately preceding *t* (`demand.shift(1).rolling(28).mean()` per series) — a
pure rolling-mean forecast that uses no information from *t* or later.

**Audit finding (the 1.1157 → 1.1501 discrepancy).** The original baseline
experiment (`baseline_forecasting.py`) used the correct rolling formula and
reported test MAE = **1.1157**. The latest ML experiment stored its MA-28
comparison as `groupby('id')['demand'].shift(1).tail(28).groupby('id').mean()` —
i.e. **one frozen constant per series**, taken from the last 28 days of the whole
features dataset (a window ending in April 2016, *inside* the test period). That
value (a) leaks future test-period demand and (b) is not a per-date forecast at
all. It reproduces the stored `MA-28 Test MAE = 1.1501` exactly. **The two
numbers are not comparable.** Recomputing the leakage-free rolling MA-28 on the
common population gives **1.1138 (test) / 1.0820 (validation)** — better than
both previously reported figures. The previously claimable ~3.3% ML improvement
over MA-28 is therefore **invalid**.

## 3. ML models

- **Random Forest** — `RandomForestRegressor` on engineered features (lags 1/7/14/28,
  rolling means/stds over 7/14/28 days, calendar features, price features,
  categorical encodings), trained on the chronological train split with a
  time-ordered internal validation set.
- **XGBoost** — `XGBRegressor` on the identical feature matrix; gradient-boosted
  trees capture different interaction structure than the bagged RF.

Both models predict the raw demand level; predictions are clipped at 0 (RF's
minimum stored prediction is 0.025, so clipping was inert for it).

## 4. Final metrics (audited, common population)

### Validation (2015-04-14 → 2015-10-18, 56,400 rows)

| model | MAE | RMSE | WAPE |
|---|---|---|---|
| **MA-28** | **1.0820** | 2.5006 | **0.9091** |
| Random Forest | 1.0860 | **2.4787** | 0.9125 |
| XGBoost | 1.0897 | 2.5073 | 0.9157 |

### Test (2015-10-19 → 2016-04-24, 56,700 rows)

| model | MAE | RMSE | WAPE |
|---|---|---|---|
| **Random Forest** | **1.1120** | **2.5131** | **0.9404** |
| XGBoost | 1.1136 | 2.5453 | 0.9418 |
| MA-28 | 1.1138 | 2.5508 | 0.9419 |

Source: `data/processed/audited_model_comparison.csv` (recomputed independently
in `src/models/evaluation_audit.py`; previously stored `ml_results.csv` values
for RF/XGB are reproduced, MA-28 rows in that file are superseded).

## 5. Improvement vs MA-28 (positive = model better)

| model | split | MAE | RMSE | WAPE |
|---|---|---|---|---|
| Random Forest | validation | **−0.37% (worse)** | **+0.87%** | −0.37% (worse) |
| XGBoost | validation | **−0.72% (worse)** | −0.27% (worse) | −0.72% (worse) |
| Random Forest | test | +0.17% | +1.48% | +0.17% |
| XGBoost | test | +0.02% | +0.22% | +0.02% |

Per the scientific rule, no improvement is claimed where the model is actually
worse: **on validation, neither ML model beats MA-28 on MAE or WAPE.**

## 6. Error analysis

**Zero demand (test):** 34,401 of 56,700 rows (60.7%) have actual demand 0.

| model | MAE | mean pred | median pred | % pred > 0 | avg overpred |
|---|---|---|---|---|---|
| MA-28 | 0.6650 | 0.6650 | 0.357 | 92.9% | 0.716 |
| Random Forest | 0.6811 | 0.6811 | 0.371 | 100.0% | 0.681 |
| XGBoost | **0.6641** | 0.6641 | 0.364 | 99.9% | 0.665 |

ML does **not** systematically overpredict inactive products beyond the baseline:
every model (including MA-28) predicts >0 on most zero-demand days, because the
28-day average of a noisy count is almost never exactly 0. XGBoost is marginally
the best model on these rows; RF is marginally the worst. This is a structural
property of intermittent count data, not an ML defect.

**Non-zero demand (22,299 rows):** all models underforecast by ~1 unit on average
(MA-28 −1.022, RF −1.003, XGB −0.991) — averages regress toward zero.

**Demand segments** (project's existing intermittent/stable/volatile classification, test):

| segment | rows | MA-28 MAE | RF MAE | XGB MAE | best |
|---|---|---|---|---|---|
| Intermittent | 43,281 | **0.6689** | 0.6832 | 0.6756 | MA-28 |
| Stable | 6,615 | 1.7976 | **1.7665** | 1.7784 | Random Forest |
| Volatile | 6,804 | 3.2794 | **3.2033** | 3.2536 | Random Forest |

MA-28 wins on intermittent series (60%+ of rows, where errors are tiny either
way); RF wins on stable and volatile series — the commercially important
high-volume segment. The two effects nearly cancel globally.

## 7. Feature importance

| feature | Random Forest | XGBoost |
|---|---|---|
| rolling_mean_28 | 0.454 | 0.188 |
| mean_demand_28 | 0.330 | 0.300 |
| rolling_mean_14 | 0.044 | 0.039 |
| is_weekend | 0.003 | 0.036 |
| store_id | 0.003 | 0.025 |

Business interpretation: the dominant features (rolling_mean_28, mean_demand_28)
are precisely the signal MA-28 uses — recent 28-day demand level is the single
best predictor of next-day demand. That ML models lean on the same signal
explains why they improve on MA-28 only marginally. rolling_mean_14 adds
shorter-horizon trend detection (useful on volatile series), is_weekend /
day_of_week capture the weekend peak in HOBBIES demand, and store_id / item_id
encode scale differences (CA_3 volume > CA_1 > CA_2). Feature importance is
associative, **not causal**.

## 8. Limitations

- **Development subset.** All results cover 300 series in HOBBIES_1 across CA_1–CA_3
  only; the full M5 dataset (3,049 series × 10 stores, FOODS/HOUSEHOLD) is not yet
  modelled. Category-level comparisons are not possible in this subset.
- **Initial models.** RF/XGB use their first sensible hyperparameters; no tuning
  has been performed (deliberately deferred until this audit).
- **Limited tuning.** Differences between models (~0.2% MAE) are far smaller than
  what tuning typically achieves.
- **Intermittent demand.** 60.7% of test rows are zero-demand; MAE on such data is
  dominated by near-dead SKUs where any averaging forecaster is near-optimal.
- **Missing prices.** `price_missing` exists in the feature set and sell-price
  coverage is incomplete in the dev subset, so price effects are under-used
  (price features rank low in importance).
- **Relatively small performance difference.** RF beats MA-28 on only 44.0% of
  the 300 series on test (XGB: 49.7%); global averages mask this unevenness.
- **Reproducibility caveat.** Validation predictions for RF/XGB were not persisted
  by the original run, so their validation metrics are taken from `ml_results.csv`
  (whose test values were verified against recomputation); re-training in this
  environment does not reproduce saved predictions bit-for-bit across library
  versions, so saved predictions are always treated as authoritative.

## 9. Final conclusion

**ML does not genuinely beat MA-28 after the audit.**

- The reported `MA-28 = 1.1501` was invalid (look-ahead leak + frozen constant
  forecast). The audited leakage-free MA-28 is **1.1138 (test)**.
- On the honest selection set (**validation**), **MA-28 is the best model**
  (MAE 1.0820 vs RF 1.0860, XGB 1.0897).
- On test, RF is nominally best (1.1120) but by only +0.17% over MA-28, and it
  wins on fewer than half of the individual series — not a defensible
  superiority claim.
- The real, actionable signal is *where* RF wins: stable and volatile
  high-volume series (and RMSE on validation, +0.87%). This is the target for
  the next milestone — after this audit, tuning is now permitted.

