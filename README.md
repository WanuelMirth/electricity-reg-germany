# electricity-reg-germany

Forecasting German electricity grid load (`Netzlast [MWh]`) using SMARD consumption data, Open-Meteo weather covariates, and German calendar/holiday features.

## Forecasting Tasks & Rolling-Origin Setup

We solve two tasks under one consistent, leak-free **1-day-ahead rolling-origin setup**:

- **Hourly ($H = 24\text{h}$):** Predict the next 24 hourly grid load values, one calendar day at a time.
- **Daily ($H = 1\text{d}$):** Predict the next daily total grid load, one calendar day at a time.

Using a shared 1-day-ahead horizon for both tasks keeps the evaluation pipeline unified across model families:

- **Train & local validation (2019–2023):** Models are trained on `2019–2022` and tuned/compared locally across all 365 days of `2023` (`val_mae` in MWh as primary metric, alongside `RMSE`, `MAPE`, `sMAPE`, `MASE`, and weekday/weekend/holiday MAE slices).
- **Final test (2024, locked locally):** Models are trained/tuned strictly on `2019–2023`, frozen on December 31, 2023, and stepped forward one day at a time across all 366 days of `2024` on Databricks without peeking at future load or re-tuning on 2024.

## Setup & Commands

Requires Python 3.13 and [`uv`](https://docs.astral.sh/uv/).

```powershell
# Install dependencies
uv sync

# Run 2023 validation baselines and log runs/models/plots to local MLflow
uv run python scripts/run_baselines.py

# Run 2023 autoregressive models (Ridge-ARX, Spline-ARX, daily SARIMAX) with train-only CV
uv run python scripts/run_autoregressive.py

# Optional strict approval mode (assigns @challenger only; promote manually)
uv run python scripts/run_baselines.py --no-auto-promote
uv run python scripts/run_autoregressive.py --no-auto-promote
uv run python scripts/promote_model.py --task all

# Build Markdown leaderboard from stored JSON summaries (or --split test_2024 later)
uv run python scripts/build_leaderboard.py

# Open local MLflow UI (Models -> @champion/@challenger aliases; Experiments -> Artifacts/plots)
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

# Lint and run tests
uv run ruff check .
uv run pytest
```

## Project Structure

```text
data/                     Raw SMARD CSVs, Open-Meteo cache, and processed feature CSVs
notebooks/                EDA and raw data audit notebooks (2019–2023)
src/
  data_loader.py          SMARD CSV loader with Europe/Berlin DST handling
  weather.py              Open-Meteo population-weighted German weather loader
  features.py             Calendar/holiday features and DST-safe day-ahead lag builders
  outliers.py             Retrospective EDA comparison helpers
  evaluation/             Train/val/test splits, rolling-origin iterator, metrics, plots
  models/                 BaseForecaster, backtest runner, baselines, autoregressive
                          (upcoming: trees.py, icl.py)
  tracking/               Portable MLflow logging, quality gate, aliases, leaderboard renderer
scripts/                  CLI runners: run_baselines.py, run_autoregressive.py,
                          build_leaderboard.py, promote_model.py (upcoming: run_trees.py, run_icl.py)
results/
  validation_leaderboard.md   Combined 2023 validation table (generated via build_leaderboard.py)
  val_2023/*.json             Git-tracked 2023 per-family summaries (future test: test_2024/)
tests/                    Pytest unit tests
mlruns/                   Local gitignored MLflow SQLite DB + logged plots/artifacts
```

## Model Families & Team Workflow

`data/processed/*.csv` stores only raw load and known-future covariates (no target lags):
- **Baselines (`src/models/baselines.py`):** `naive_day_before` (`lag_24h` / `lag_1d`) and `seasonal_naive_week_before` (`lag_168h` / `lag_7d`).
- **Autoregressive (`src/models/autoregressive.py`):** `ridge_arx` (24h-stratified / daily Ridge-ARX), `spline_arx` (GAM/LEAR-style B-spline ARX with full yesterday 24h profile on hourly), and `sarimax` (daily `SARIMAX(p,d,q)(P,D,Q,7)`), tuned strictly on `train_df` via inner-year CV.
- **Tree models (upcoming `src/models/trees.py`):** `LightGBM` / `CatBoost` + Optuna, using `build_tree_features(df, task=...)` in `src/features.py` for DST-safe day-ahead lags and cyclical calendar encodings.
- **ICL models (upcoming `src/models/icl.py`):** `TimeH2O-v1` / `Timestral-209M`, using raw past load + `HOURLY_ICL_COVARIATE_COLS` / `DAILY_ICL_COVARIATE_COLS` without manual target lags.

**Working across branches:**
1. Pull `main` and run `uv run python scripts/run_baselines.py` once locally so your local `mlruns/mlflow.db` has the baseline `@champion` to compare against.
2. Implement your model family in `src/models/` + CLI runner in `scripts/`, saving JSON summaries to `results/val_2023/<family>_<task>.json`.
3. Run `uv run python scripts/build_leaderboard.py` to refresh `results/validation_leaderboard.md`.
4. Commit code + `results/` (never `mlruns/`). For Databricks/Unity Catalog later, override `MLFLOW_TRACKING_URI`, `MLFLOW_REGISTRY_URI`, and `MLFLOW_UC_PREFIX`.

## Data Attribution

- Load data: Bundesnetzagentur | SMARD.de (see `data/smard-user-guide-09-2021-data.pdf`).
- Weather data: Open-Meteo historical archive.
