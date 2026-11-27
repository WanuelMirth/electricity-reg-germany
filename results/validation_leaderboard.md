# Validation Leaderboard (2023)

Primary metric: `val_mae` (MWh, lower is better). Generated from `results/val_2023/*.json`.

## Hourly (H = 24h)

| Rank | Model | Family | MAE (MWh) | RMSE (MWh) | MAPE (%) | sMAPE (%) | MASE | Weekday MAE | Weekend MAE | Holiday MAE | Source | Commit |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | `timestral_209m` | icl | 824.2 | 1,145.6 | 1.58 | 1.58 | 0.330 | 806.1 | 791.6 | 1,540.6 | `icl_hourly.json` | `a2343b9` |
| 2 | `timeh2o_v1` | icl | 903.8 | 1,415.6 | 1.77 | 1.75 | 0.362 | 824.9 | 811.6 | 3,659.0 | `icl_hourly.json` | `a2343b9` |
| 3 | `spline_arx` | autoregressive | 1,039.6 | 1,413.8 | 1.99 | 1.97 | 0.417 | 1,034.5 | 1,011.2 | 1,465.3 | `autoregressive_hourly.json` | `5665f31` |
| 4 | `ridge_arx` | autoregressive | 1,055.6 | 1,440.9 | 2.01 | 1.99 | 0.423 | 1,067.7 | 981.5 | 1,505.3 | `autoregressive_hourly.json` | `5665f31` |
| 5 | `lightgbm_linear` | tree | 1,055.9 | 1,369.5 | 2.08 | 2.06 | 0.423 | 1,024.4 | 1,071.1 | 1,727.9 | `trees_hourly.json` | `5665f31` |
| 6 | `lightgbm` | tree | 1,075.9 | 1,390.8 | 2.13 | 2.11 | 0.431 | 1,032.2 | 1,107.1 | 1,806.7 | `trees_hourly.json` | `5665f31` |
| 7 | `xgboost` | tree | 1,184.2 | 1,539.1 | 2.33 | 2.30 | 0.475 | 1,137.2 | 1,214.8 | 2,039.2 | `trees_hourly.json` | `5665f31` |
| 8 | `random_forest` | tree | 1,186.0 | 1,539.0 | 2.36 | 2.33 | 0.475 | 1,114.3 | 1,244.4 | 2,356.2 | `trees_hourly.json` | `5665f31` |
| 9 | `catboost` | tree | 1,249.3 | 1,605.3 | 2.49 | 2.45 | 0.501 | 1,213.3 | 1,268.4 | 2,178.0 | `trees_hourly.json` | `5665f31` |
| 10 | `seasonal_naive_week_before` | baseline | 2,315.3 | 3,741.5 | 4.59 | 4.53 | 0.928 | 2,155.1 | 1,783.3 | 10,969.0 | `baselines_hourly.json` | `9732e70` |
| 11 | `naive_day_before` | baseline | 3,916.1 | 5,913.9 | 7.65 | 7.72 | 1.570 | 3,307.4 | 5,322.0 | 4,281.5 | `baselines_hourly.json` | `9732e70` |

## Daily (H = 1d)

| Rank | Model | Family | MAE (MWh) | RMSE (MWh) | MAPE (%) | sMAPE (%) | MASE | Weekday MAE | Weekend MAE | Holiday MAE | Source | Commit |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | `timestral_209m` | icl | 18,491.4 | 24,919.4 | 1.51 | 1.51 | 0.338 | 16,779.4 | 21,430.5 | 28,757.6 | `icl_daily.json` | `a2343b9` |
| 2 | `timeh2o_v1` | icl | 18,655.5 | 27,502.0 | 1.54 | 1.53 | 0.341 | 15,060.8 | 19,650.7 | 90,191.2 | `icl_daily.json` | `a2343b9` |
| 3 | `sarimax` | autoregressive | 19,644.5 | 26,113.8 | 1.60 | 1.58 | 0.359 | 19,418.4 | 18,414.8 | 35,197.1 | `autoregressive_daily.json` | `d46112a` |
| 4 | `lightgbm_linear` | tree | 24,604.9 | 32,701.7 | 2.01 | 1.98 | 0.450 | 22,890.5 | 24,576.1 | 64,655.7 | `trees_daily.json` | `5665f31` |
| 5 | `spline_arx` | autoregressive | 25,700.6 | 32,079.6 | 2.07 | 2.05 | 0.470 | 26,554.7 | 22,603.2 | 34,450.3 | `autoregressive_daily.json` | `d46112a` |
| 6 | `ridge_arx` | autoregressive | 25,775.9 | 32,936.3 | 2.09 | 2.06 | 0.471 | 26,730.1 | 22,066.1 | 38,247.8 | `autoregressive_daily.json` | `d46112a` |
| 7 | `xgboost` | tree | 26,566.3 | 33,950.5 | 2.21 | 2.17 | 0.485 | 25,108.2 | 27,979.8 | 48,342.6 | `trees_daily.json` | `5665f31` |
| 8 | `catboost` | tree | 27,679.6 | 35,214.7 | 2.28 | 2.25 | 0.506 | 26,918.8 | 28,643.0 | 44,148.4 | `trees_daily.json` | `5665f31` |
| 9 | `random_forest` | tree | 30,893.6 | 38,484.0 | 2.56 | 2.51 | 0.564 | 30,195.9 | 29,985.0 | 58,833.3 | `trees_daily.json` | `5665f31` |
| 10 | `lightgbm` | tree | 32,344.2 | 41,283.5 | 2.66 | 2.61 | 0.591 | 31,068.5 | 31,840.6 | 67,261.8 | `trees_daily.json` | `5665f31` |
| 11 | `seasonal_naive_week_before` | baseline | 49,920.8 | 81,771.9 | 4.12 | 4.07 | 0.912 | 45,272.9 | 38,529.8 | 261,228.9 | `baselines_daily.json` | `9732e70` |
| 12 | `naive_day_before` | baseline | 88,204.7 | 125,199.7 | 7.30 | 7.37 | 1.611 | 72,583.7 | 124,308.9 | 97,861.7 | `baselines_daily.json` | `9732e70` |
