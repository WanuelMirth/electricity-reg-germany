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

# Tune and evaluate tree models on both tasks (10 trials per configuration/task)
uv run python scripts/run_trees.py

# Optional strict approval mode (assigns @challenger only; promote manually)
uv run python scripts/run_baselines.py --no-auto-promote
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
  models/                 BaseForecaster, backtest runner, baselines, trees (upcoming: icl.py)
  tracking/               Portable MLflow logging, quality gate, aliases, leaderboard renderer
scripts/                  CLI runners: run_baselines.py, run_trees.py, build_leaderboard.py,
                          promote_model.py (upcoming: run_icl.py)
results/
  validation_leaderboard.md   Combined 2023 validation table (generated via build_leaderboard.py)
  val_2023/*.json             Git-tracked 2023 per-family summaries (future test: test_2024/)
tests/                    Pytest unit tests
mlruns/                   Local gitignored MLflow SQLite DB + logged plots/artifacts
```

## Model Families & Team Workflow

`data/processed/*.csv` stores only raw load and known-future covariates (no target lags):
- **Baselines (`src/models/baselines.py`):** `naive_day_before` (`lag_24h` / `lag_1d`) and `seasonal_naive_week_before` (`lag_168h` / `lag_7d`).
- **Tree models (`src/models/trees.py`):** Random Forest, XGBoost, LightGBM, LightGBM linear trees, and CatBoost, using the shared tree feature builder and Optuna tuning within the training years.
- **ICL models (upcoming `src/models/icl.py`):** `TimeH2O-v1` / `Timestral-209M`, using raw past load + `HOURLY_ICL_COVARIATE_COLS` / `DAILY_ICL_COVARIATE_COLS` without manual target lags.

**Working across branches:**
1. Pull `main` and run `uv run python scripts/run_baselines.py` once locally so your local `mlruns/mlflow.db` has the baseline `@champion` to compare against.
2. Implement your model family in `src/models/` + CLI runner in `scripts/`, saving JSON summaries to `results/val_2023/<family>_<task>.json`.
3. Run `uv run python scripts/build_leaderboard.py` to refresh `results/validation_leaderboard.md`.
4. Commit code + `results/` (never `mlruns/`). For Databricks/Unity Catalog later, override `MLFLOW_TRACKING_URI`, `MLFLOW_REGISTRY_URI`, and `MLFLOW_UC_PREFIX`.

## Data Attribution

- Load data: Bundesnetzagentur | SMARD.de (see `data/smard-user-guide-09-2021-data.pdf`).
- Weather data: Open-Meteo historical archive.

## Tree models

Run from the repository root after `uv sync`. On macOS, LightGBM and XGBoost
also need the native OpenMP runtime: `brew install libomp`.

```bash
# First complete run: LightGBM, hourly and daily
uv run python scripts/run_trees.py --models lightgbm --n-trials 10

# All five configurations, separately for hourly and daily data
uv run python scripts/run_trees.py --models all --n-trials 10

# Individual models/tasks or an untuned first check
uv run python scripts/run_trees.py --models random_forest xgboost catboost --task hourly
uv run python scripts/run_trees.py --models lightgbm --no-tune

# Keep the shared champion unchanged when a better candidate is found
uv run python scripts/run_trees.py --no-auto-promote

# Inspect the same tracking database used by baselines
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5002
```

**Training protocol:** fit trial models on 2019–2021, select hyperparameters by
2022 MAE, then refit on 2019–2022. The shared rolling backtest evaluates all days
of 2023 with fixed fitted models and updated past-load features. Each local
forecast starts at midnight and predicts the following calendar day: 23, 24 or
25 hourly rows, or one daily total. It assumes the entire previous day's load
is available then. The runner does not offer access to the 2024 test split.

All models use the existing 23 hourly / 19 daily features. Only the
first seven training days are removed for missing lag history; any remaining
non-finite inputs fail explicitly. Linear-tree LightGBM uses a scaler fitted
on each training partition, including a fresh scaler for the final refit.

**Weather qualification:** models use the processed historical observed
weather, for consistency with the team's existing weather-assisted models.
These are retrospective experiments, not evaluations using weather forecasts
available on the forecast date. Runs carry `weather_mode=observed_historical`.

Each model logs the shared metrics and plots, plus R², runtime, the Optuna trial
history, exact features, feature importances and the fitted model. Tree feature
importances are model-specific diagnostics, not causal effects or directly
comparable numbers across different libraries. No outliers are automatically
removed. `--n-trials` controls a bounded search; ten trials are a starting budget,
not an exhaustive optimization. Seeds are fixed and Optuna trials run serially;
`--n-jobs` controls estimator threads (default 2).

The tree runner disables MLflow usage telemetry by default. Experiment tracking
still writes normally to the configured local database (or an explicitly
configured remote tracking service).

Results are checkpointed after each model in `results/val_2023/trees_hourly.json`
and `trees_daily.json`. Selecting
individual models updates their entries while retaining other model entries.
The shared leaderboard is rebuilt at the end. Commit code and JSON/Markdown
results, never the local MLflow database or artifacts.
