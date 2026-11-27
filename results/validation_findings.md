# 2023 Validation Findings (`val_2023`)

Key takeaways across all four model families (`train: 2019–2022`, `val: 2023`, `weather_mode=observed_historical`). Full table: [validation_leaderboard.md](validation_leaderboard.md).

## 1. Baselines

- **Weekly > daily lag:** `seasonal_naive_week_before` (`lag_168h` / `lag_7d`) cuts MAE by ~41–43% vs. `naive_day_before` (`2,315` vs `3,916` MWh hourly; `49,921` vs `88,205` MWh daily).
- **Holiday breakdown:** Weekly persistence fails sharply on German holidays (`10,969` MWh hourly; `261,229` MWh daily) because the prior week's same weekday is usually a normal workday.

## 2. Tree Models

- **Linear-leaf LightGBM wins both tasks:** `lightgbm_linear` ranked #1 among trees on hourly (`1,056` MWh) and daily (`24,605` MWh), beating standard `lightgbm` by **-23.9%** on daily (`32,344` MWh) because piecewise-linear leaves extrapolate smooth load/temperature responses better on small daily data (`1,454` rows).
- **Task-dependent depth:** Daily boosted trees preferred shallow depth to avoid overfitting (`XGBoost depth=3`, `CatBoost depth=4`), whereas hourly favored deeper trees (`XGBoost/CatBoost depth=8`, `RF depth=17`).
- **Speed & holiday weakness:** Very fast (`~6–77s` hourly, `~2–10s` daily), but all tree models struggled on sparse holiday dates (`1,728–2,356` MWh hourly; `44.1k–67.3k` MWh daily).

## 3. Autoregressive Models

- **Best classical family on both tasks:** 24h-stratified `spline_arx` (`1,040` MWh, `n_knots=5`, `alpha=50`) beat all 5 tree models hourly and posted the **lowest hourly holiday MAE (`1,465` MWh)** of all models; daily `SARIMAX(2,0,1)(1,0,1,7)` (`19,645` MWh) beat all trees by **-20%+** and achieved the **lowest daily weekend MAE (`18,415` MWh)**.
- **Surprise — no linear trend:** Inner CV chose `include_trend=False` across every ARX model; autoregressive load lags already absorb structural demand shifts.

## 4. In-Context Learning Models (`timestral_209m`, `timeh2o_v1`)

Zero-shot inference on a local **GTX 1080 Ti (11 GB VRAM, `torch 2.14.1+cu126`)** using raw load + known-future covariates (no manual target lags).

| Task | Model | `3m` MAE (Time / VRAM) | `1y` MAE (Time / VRAM) | `max` MAE (Time / VRAM) | `max` Steps |
| --- | --- | --- | --- | --- | ---: |
| **Hourly** | `timestral_209m` | 997.1 (112.5s / 1,321 MB) | 944.7 (397.5s / 1,568 MB) | **824.2** (1,900.1s / 2,503 MB) | 32,768 |
| **Hourly** | `timeh2o_v1` | 1,002.8 (328.8s / 913 MB) | 951.1 (165.2s / 904 MB) | **903.8** (385.0s / 1,324 MB) | 16,384 |
| **Daily** | `timestral_209m` | 24,989.3 (48.0s / 1,243 MB) | 19,996.4 (49.2s / 1,253 MB) | **18,491.4** (82.2s / 1,304 MB) | 1,825 |
| **Daily** | `timeh2o_v1` | 24,897.4 (161.5s / 1,279 MB) | 21,383.5 (192.9s / 1,383 MB) | **18,655.5** (483.7s / 1,961 MB) | 1,825 |

- **Zero-shot #1 on both tasks:** `timestral_209m (max)` beat the best classical models by **-20.7%** hourly (`824` vs `1,040` MWh) and **-5.9%** daily (`18,491` vs `19,645` MWh); even `3m` hourly beat every tree and AR model.
- **More context always helped:** Strictly monotonic `3m → 1y → max` gains across all 4 sweeps (`-10%` to `-26%` MAE).
- **Holidays separate the two ICL models:** `timeh2o_v1` actually edges out `timestral_209m` on daily weekdays (`15,061` vs `16,779`) and weekends (`19,651` vs `21,431`), but `timestral_209m` dominates German holidays (`28,758` vs `90,191` daily; `1,541` vs `3,659` hourly).
- **Low VRAM & speed crossover:** Peak VRAM stayed **0.9–2.5 GB** (out of 11 GB). `timestral_209m` was **5.9× faster** daily (`82s` vs `484s`, since `timeh2o_v1` uses multi-pass `patch_ensemble`), while `timeh2o_v1` (`16k` cap) was **4.9× faster** at hourly `max` (`6.4 min` vs `31.7 min` at `32k`).

## 5. Selection for Locked `test_2024` (Databricks)

- **Keep all four model families (`baselines`, `trees`, `autoregressive`, `icl`):** Classical models run in seconds–minutes, giving a complete 2024 benchmark across every family.
- **Only prune `3m` and `1y` ICL context presets:** Freeze both `timestral_209m` and `timeh2o_v1` at `context_preset="max"` (`32,768`h / `16,384`h hourly; full history daily), cutting ICL compute roughly in half without sacrificing accuracy.
