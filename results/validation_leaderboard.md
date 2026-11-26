# Validation Leaderboard (2023)

Primary metric: `val_mae` (MWh, lower is better). Generated from `results/val_2023/*.json`.

## Hourly (H = 24h)

| Rank | Model | Family | MAE (MWh) | RMSE (MWh) | MAPE (%) | sMAPE (%) | MASE | Weekday MAE | Weekend MAE | Holiday MAE | Source | Commit |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | `lightgbm_linear` | tree | 1,055.9 | 1,369.5 | 2.08 | 2.06 | 0.423 | 1,024.4 | 1,071.1 | 1,727.9 | `trees_hourly.json` | `5665f31` |
| 2 | `lightgbm` | tree | 1,075.9 | 1,390.8 | 2.13 | 2.11 | 0.431 | 1,032.2 | 1,107.1 | 1,806.7 | `trees_hourly.json` | `5665f31` |
| 3 | `xgboost` | tree | 1,184.2 | 1,539.1 | 2.33 | 2.30 | 0.475 | 1,137.2 | 1,214.8 | 2,039.2 | `trees_hourly.json` | `5665f31` |
| 4 | `random_forest` | tree | 1,186.0 | 1,539.0 | 2.36 | 2.33 | 0.475 | 1,114.3 | 1,244.4 | 2,356.2 | `trees_hourly.json` | `5665f31` |
| 5 | `catboost` | tree | 1,249.3 | 1,605.3 | 2.49 | 2.45 | 0.501 | 1,213.3 | 1,268.4 | 2,178.0 | `trees_hourly.json` | `5665f31` |
| 6 | `seasonal_naive_week_before` | baseline | 2,315.3 | 3,741.5 | 4.59 | 4.53 | 0.928 | 2,155.1 | 1,783.3 | 10,969.0 | `baselines_hourly.json` | `9732e70` |
| 7 | `naive_day_before` | baseline | 3,916.1 | 5,913.9 | 7.65 | 7.72 | 1.570 | 3,307.4 | 5,322.0 | 4,281.5 | `baselines_hourly.json` | `9732e70` |

## Daily (H = 1d)

| Rank | Model | Family | MAE (MWh) | RMSE (MWh) | MAPE (%) | sMAPE (%) | MASE | Weekday MAE | Weekend MAE | Holiday MAE | Source | Commit |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | `lightgbm_linear` | tree | 24,604.9 | 32,701.7 | 2.01 | 1.98 | 0.450 | 22,890.5 | 24,576.1 | 64,655.7 | `trees_daily.json` | `5665f31` |
| 2 | `xgboost` | tree | 26,566.3 | 33,950.5 | 2.21 | 2.17 | 0.485 | 25,108.2 | 27,979.8 | 48,342.6 | `trees_daily.json` | `5665f31` |
| 3 | `catboost` | tree | 27,679.6 | 35,214.7 | 2.28 | 2.25 | 0.506 | 26,918.8 | 28,643.0 | 44,148.4 | `trees_daily.json` | `5665f31` |
| 4 | `random_forest` | tree | 30,893.6 | 38,484.0 | 2.56 | 2.51 | 0.564 | 30,195.9 | 29,985.0 | 58,833.3 | `trees_daily.json` | `5665f31` |
| 5 | `lightgbm` | tree | 32,344.2 | 41,283.5 | 2.66 | 2.61 | 0.591 | 31,068.5 | 31,840.6 | 67,261.8 | `trees_daily.json` | `5665f31` |
| 6 | `seasonal_naive_week_before` | baseline | 49,920.8 | 81,771.9 | 4.12 | 4.07 | 0.912 | 45,272.9 | 38,529.8 | 261,228.9 | `baselines_daily.json` | `9732e70` |
| 7 | `naive_day_before` | baseline | 88,204.7 | 125,199.7 | 7.30 | 7.37 | 1.611 | 72,583.7 | 124,308.9 | 97,861.7 | `baselines_daily.json` | `9732e70` |
