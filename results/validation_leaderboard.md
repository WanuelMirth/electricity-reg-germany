# Validation Leaderboard (2023)

Primary metric: `val_mae` (MWh, lower is better). Generated from `results/val_2023/*.json`.

## Hourly (H = 24h)

| Rank | Model | Family | MAE (MWh) | RMSE (MWh) | MAPE (%) | sMAPE (%) | MASE | Weekday MAE | Weekend MAE | Holiday MAE | Source | Commit |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | `spline_arx` | autoregressive | 1,039.6 | 1,413.8 | 1.99 | 1.97 | 0.417 | 1,034.5 | 1,011.2 | 1,465.3 | `autoregressive_hourly.json` | `5665f31` |
| 2 | `ridge_arx` | autoregressive | 1,055.6 | 1,440.9 | 2.01 | 1.99 | 0.423 | 1,067.7 | 981.5 | 1,505.3 | `autoregressive_hourly.json` | `5665f31` |
| 3 | `seasonal_naive_week_before` | baseline | 2,315.3 | 3,741.5 | 4.59 | 4.53 | 0.928 | 2,155.1 | 1,783.3 | 10,969.0 | `baselines_hourly.json` | `9732e70` |
| 4 | `naive_day_before` | baseline | 3,916.1 | 5,913.9 | 7.65 | 7.72 | 1.570 | 3,307.4 | 5,322.0 | 4,281.5 | `baselines_hourly.json` | `9732e70` |

## Daily (H = 1d)

| Rank | Model | Family | MAE (MWh) | RMSE (MWh) | MAPE (%) | sMAPE (%) | MASE | Weekday MAE | Weekend MAE | Holiday MAE | Source | Commit |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | `sarimax` | autoregressive | 19,483.1 | 25,980.4 | 1.58 | 1.57 | 0.356 | 19,267.4 | 18,257.5 | 34,716.1 | `autoregressive_daily.json` | `5665f31` |
| 2 | `spline_arx` | autoregressive | 25,700.6 | 32,079.6 | 2.07 | 2.05 | 0.470 | 26,554.7 | 22,603.2 | 34,450.3 | `autoregressive_daily.json` | `5665f31` |
| 3 | `ridge_arx` | autoregressive | 25,775.9 | 32,936.3 | 2.09 | 2.06 | 0.471 | 26,730.1 | 22,066.1 | 38,247.8 | `autoregressive_daily.json` | `5665f31` |
| 4 | `seasonal_naive_week_before` | baseline | 49,920.8 | 81,771.9 | 4.12 | 4.07 | 0.912 | 45,272.9 | 38,529.8 | 261,228.9 | `baselines_daily.json` | `9732e70` |
| 5 | `naive_day_before` | baseline | 88,204.7 | 125,199.7 | 7.30 | 7.37 | 1.611 | 72,583.7 | 124,308.9 | 97,861.7 | `baselines_daily.json` | `9732e70` |
