# Feature Documentation

## Overview

This document describes all engineered features for the M5 demand forecasting project.

## Feature Table

| Feature | Meaning | Lookback | Leakage-safe? |
|---------|---------|----------|---------------|
| **Lag Features** | | | |
| `lag_1` | Demand from previous day | 1 day | Yes |
| `lag_7` | Demand from 7 days ago | 7 days | Yes |
| `lag_14` | Demand from 14 days ago | 14 days | Yes |
| `lag_28` | Demand from 28 days ago | 28 days | Yes |
| **Rolling Features** | | | |
| `rolling_mean_7` | Mean demand over last 7 days | 7 days | Yes |
| `rolling_mean_14` | Mean demand over last 14 days | 14 days | Yes |
| `rolling_mean_28` | Mean demand over last 28 days | 28 days | Yes |
| `rolling_std_7` | Std dev of demand over last 7 days | 7 days | Yes |
| `rolling_std_28` | Std dev of demand over last 28 days | 28 days | Yes |
| **Demand Behavior Features** | | | |
| `zero_demand_rate_28` | % of zero-demand days in last 28 days | 28 days | Yes |
| `mean_demand_28` | Mean demand over last 28 days | 28 days | Yes |
| `std_demand_28` | Std dev of demand over last 28 days | 28 days | Yes |
| **Calendar Features** | | | |
| `day_of_week` | Day of week (0=Monday, 6=Sunday) | N/A | Yes |
| `day_of_month` | Day of month (1-31) | N/A | Yes |
| `month` | Month (1-12) | N/A | Yes |
| `quarter` | Quarter (1-4) | N/A | Yes |
| `year` | Year | N/A | Yes |
| `week_of_year` | Week of year (1-53) | N/A | Yes |
| `is_weekend` | Weekend indicator (1=weekend) | N/A | Yes |
| **Event Features** | | | |
| `has_event` | Event indicator (1=has event) | N/A | Yes |
| **Price Features** | | | |
| `price_change` | Change from previous price | 1 period | Yes |
| `price_change_pct` | Percentage change from previous price | 1 period | Yes |
| `price_missing` | Missing price indicator (1=missing) | N/A | Yes |

## Feature Design Rationale

### Why Lag Features?

Lag features capture recent demand patterns:
- `lag_1`: Captures short-term demand persistence
- `lag_7`: Captures weekly patterns (same day last week)
- `lag_14`: Captures bi-weekly patterns
- `lag_28`: Captures monthly patterns

These features are essential for tree-based models to learn temporal dependencies.

### Why Rolling Features?

Rolling features capture demand trends and variability:
- `rolling_mean_*`: Smoothed demand level (trend)
- `rolling_std_*`: Demand volatility (uncertainty)

These help models distinguish between stable and volatile products.

### Why Demand Behavior Features?

- `zero_demand_rate_28`: Identifies intermittent demand patterns
- `mean_demand_28`: Long-term demand level
- `std_demand_28`: Long-term demand variability

These features help models handle the high percentage (66.6%) of zero-demand observations.

### Why Calendar Features?

Calendar features capture seasonal patterns:
- `day_of_week`: Weekly shopping patterns
- `month`/`quarter`: Annual seasonality
- `is_weekend`: Weekend vs. weekday effects

### Why Price Features?

Price features capture price-demand relationships:
- `price_change`: Recent price movements
- `price_change_pct`: Relative price changes
- `price_missing`: Identifies products without price data

## Handling Intermittent Demand

The dataset contains approximately 66.6% zero-demand observations. This is addressed by:
1. `zero_demand_rate_28`: Explicitly models the zero-demand proportion
2. Rolling means smooth over zero-demand periods
3. Lag features capture the timing of zero-demand days

## Handling Missing Prices

Missing prices are handled carefully:
1. `price_missing` flag identifies missing values
2. `price_change` and `price_change_pct` are computed only when prices are available
3. No blind forward-filling that could introduce leakage

## Leakage Prevention

All features are designed to prevent data leakage:
1. **Lag features**: Use `shift(lag)` to access only past values
2. **Rolling features**: Use `shift(1)` before rolling calculations to exclude current day
3. **Price features**: Use `shift(1)` for previous price comparisons
4. **No future information**: All features use only information available at prediction time

## Warm-up Period

The first 28 days of each product/store series are removed because:
- `lag_28` requires 28 days of history
- Rolling features need sufficient history
- This ensures all features have valid values

## Target Variable

The prediction target is:
- `demand`: The demand value to predict for the current day

All features represent information available at time `t` to predict demand at time `t`.
