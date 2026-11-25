# Validation Leaderboard (2023)

Primary metric: `val_mae` (MWh, lower is better). Generated from `results/val_2023/*.json`.

## Hourly (H = 24h)

| Rank | Model | Family | MAE (MWh) | RMSE (MWh) | MAPE (%) | sMAPE (%) | MASE | Weekday MAE | Weekend MAE | Holiday MAE | Source | Commit |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | `seasonal_naive_week_before` | baseline | 2,315.3 | 3,741.5 | 4.59 | 4.53 | 0.928 | 2,155.1 | 1,783.3 | 10,969.0 | `baselines_hourly.json` | `9732e70` |
| 2 | `naive_day_before` | baseline | 3,916.1 | 5,913.9 | 7.65 | 7.72 | 1.570 | 3,307.4 | 5,322.0 | 4,281.5 | `baselines_hourly.json` | `9732e70` |

## Daily (H = 1d)

| Rank | Model | Family | MAE (MWh) | RMSE (MWh) | MAPE (%) | sMAPE (%) | MASE | Weekday MAE | Weekend MAE | Holiday MAE | Source | Commit |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | `seasonal_naive_week_before` | baseline | 49,920.8 | 81,771.9 | 4.12 | 4.07 | 0.912 | 45,272.9 | 38,529.8 | 261,228.9 | `baselines_daily.json` | `9732e70` |
| 2 | `naive_day_before` | baseline | 88,204.7 | 125,199.7 | 7.30 | 7.37 | 1.611 | 72,583.7 | 124,308.9 | 97,861.7 | `baselines_daily.json` | `9732e70` |
