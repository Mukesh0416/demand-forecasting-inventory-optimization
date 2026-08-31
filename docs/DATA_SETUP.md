# Data Setup

## Dataset

This project uses the **M5 Forecasting dataset**, a retail sales forecasting dataset containing historical product-level sales, calendar information, store information, and selling prices.

## Source

https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

## Required Files

Download the following files from the dataset source:

* `calendar.csv`
* `sales_train_validation.csv`
* `sell_prices.csv`

## Local Directory

Place the downloaded files inside:

```text
data/raw/
├── calendar.csv
├── sales_train_validation.csv
└── sell_prices.csv
```

## Important

Raw data files are intentionally excluded from GitHub because of their size.

The project code operates on the files stored locally under:

`data/raw/`