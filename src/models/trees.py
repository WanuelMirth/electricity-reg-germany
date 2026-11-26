"""Day-ahead tree forecasters using the shared calendar, weather and lag features."""

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING, Any

import mlflow.pyfunc
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.splits import TARGET_COL, TaskType
from src.features import (
    DAILY_TREE_FEATURE_COLS,
    DAILY_TREE_LAG_COLS,
    HOURLY_TREE_FEATURE_COLS,
    HOURLY_TREE_LAG_COLS,
    add_cyclical_calendar_features,
    build_tree_features,
    compute_single_day_lags,
)
from src.models.base import BaseForecaster

if TYPE_CHECKING:
    import optuna

TREE_MODELS = ("random_forest", "xgboost", "lightgbm", "lightgbm_linear", "catboost")


def _model_input(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing tree inputs: {sorted(missing)}")
    inputs = frame.loc[:, columns].astype(float)
    if not np.isfinite(inputs.to_numpy()).all():
        raise ValueError("Tree inputs contain missing or infinite values.")
    return inputs


class TreePyFunc(mlflow.pyfunc.PythonModel):
    """Serialized model accepts the same pre-built features as the backtest."""

    def __init__(self, estimator: Pipeline, feature_cols: list[str]) -> None:
        self.estimator = estimator
        self.feature_cols = list(feature_cols)

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None
    ) -> np.ndarray:
        return self.estimator.predict(_model_input(model_input, self.feature_cols))


class TreeForecaster(BaseForecaster):
    """Fit once on training years; update historical inputs at each daily origin.

    Hyperparameters are selected on the final year inside the training frame,
    never on the outer 2023 validation year. Only linear-tree LightGBM scales
    inputs; its scaler is fitted separately inside each training partition.
    """

    family = "tree"

    def __init__(
        self,
        task: TaskType,
        model_name: str = "lightgbm",
        *,
        tune: bool = True,
        n_trials: int = 10,
        seed: int = 42,
        n_jobs: int = 2,
        model_params: dict[str, Any] | None = None,
    ) -> None:
        if task not in {"hourly", "daily"}:
            raise ValueError(f"Unsupported task: {task}")
        if model_name not in TREE_MODELS:
            raise ValueError(f"Unsupported tree model: {model_name}")
        if n_trials < 1 or n_jobs < 1:
            raise ValueError("n_trials and n_jobs must be positive.")
        self.task = task
        self.model_name = model_name
        self.name = model_name
        self.tune = tune
        self.n_trials = n_trials
        self.seed = seed
        self.n_jobs = n_jobs
        self.model_params = dict(model_params or {})
        columns = HOURLY_TREE_FEATURE_COLS if task == "hourly" else DAILY_TREE_FEATURE_COLS
        self.feature_cols = list(columns)
        self.best_params_: dict[str, Any] = {}
        self.cv_best_mae_: float | None = None
        self.trials_: list[dict[str, Any]] = []
        self.estimator_: Pipeline | None = None

    def _make_estimator(self, params: dict[str, Any]) -> Pipeline:
        if self.model_name == "random_forest":
            estimator = RandomForestRegressor(
                **{
                    "n_estimators": 200,
                    "min_samples_leaf": 3,
                    "random_state": self.seed,
                    "n_jobs": self.n_jobs,
                    **params,
                }
            )
        elif self.model_name == "xgboost":
            from xgboost import XGBRegressor

            estimator = XGBRegressor(
                **{
                    "n_estimators": 300,
                    "max_depth": 5,
                    "learning_rate": 0.05,
                    "objective": "reg:squarederror",
                    "tree_method": "hist",
                    "random_state": self.seed,
                    "n_jobs": self.n_jobs,
                    **params,
                }
            )
        elif self.model_name.startswith("lightgbm"):
            from lightgbm import LGBMRegressor

            estimator = LGBMRegressor(
                **{
                    "n_estimators": 300,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "objective": "regression",
                    "linear_tree": self.model_name == "lightgbm_linear",
                    "random_state": self.seed,
                    "n_jobs": self.n_jobs,
                    "verbosity": -1,
                    "deterministic": True,
                    "force_col_wise": True,
                    **params,
                }
            )
        else:
            from catboost import CatBoostRegressor

            estimator = CatBoostRegressor(
                **{
                    "iterations": 300,
                    "depth": 6,
                    "learning_rate": 0.05,
                    "loss_function": "RMSE",
                    "random_seed": self.seed,
                    "thread_count": self.n_jobs,
                    "verbose": False,
                    "allow_writing_files": False,
                    **params,
                }
            )
        scaler = StandardScaler() if self.model_name == "lightgbm_linear" else "passthrough"
        return Pipeline([("scale", scaler), ("model", estimator)])

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        if self.model_name == "random_forest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=100),
                "max_depth": trial.suggest_int("max_depth", 5, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
                "max_features": trial.suggest_float("max_features", 0.5, 1.0),
            }
        params: dict[str, Any] = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        }
        if self.model_name == "catboost":
            params.update(
                iterations=trial.suggest_int("iterations", 200, 600, step=100),
                depth=trial.suggest_int("depth", 4, 8),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            )
        else:
            params["n_estimators"] = trial.suggest_int("n_estimators", 200, 600, step=100)
            params["reg_lambda"] = trial.suggest_float("reg_lambda", 0.01, 10.0, log=True)
            if self.model_name == "xgboost":
                params.update(
                    max_depth=trial.suggest_int("max_depth", 3, 8),
                    min_child_weight=trial.suggest_float("min_child_weight", 1.0, 20.0),
                    subsample=trial.suggest_float("subsample", 0.7, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
                )
            else:
                params.update(
                    num_leaves=trial.suggest_int("num_leaves", 15, 63),
                    min_child_samples=trial.suggest_int("min_child_samples", 20, 100),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
                )
                if self.model_name == "lightgbm_linear":
                    params["linear_lambda"] = trial.suggest_float(
                        "linear_lambda", 0.01, 10.0, log=True
                    )
        return params

    def fit(self, train_df: pd.DataFrame) -> TreeForecaster:
        if train_df.empty:
            raise ValueError("Training data cannot be empty.")
        dates = pd.to_datetime(train_df["date_local"]).dt.normalize()
        if not dates.is_monotonic_increasing:
            raise ValueError("Training data must be sorted by date_local.")
        features = build_tree_features(train_df, task=self.task)
        # Only the initial seven calendar days may lack consumption history.
        usable = dates >= dates.min() + pd.Timedelta(days=7)
        if not usable.any():
            raise ValueError("Need more than seven training days to build lag features.")
        X = _model_input(features.loc[usable], self.feature_cols)
        y = train_df.loc[usable, TARGET_COL].astype(float)
        if not np.isfinite(y.to_numpy()).all():
            raise ValueError("Training targets contain missing or infinite values.")
        self.train_rows_ = len(X)
        self.warmup_rows_ = int((~usable).sum())
        self.best_params_ = dict(self.model_params)
        self.cv_best_mae_ = None
        self.trials_ = []
        if self.tune:
            import optuna

            usable_dates = dates.loc[usable]
            final_year = int(usable_dates.dt.year.max())
            inner_val = usable_dates.dt.year == final_year
            if inner_val.all() or not inner_val.any():
                raise ValueError("Tuning needs at least two calendar years of training data.")
            self.cv_train_end_ = str(usable_dates.loc[~inner_val].max().date())
            self.cv_val_start_ = str(usable_dates.loc[inner_val].min().date())
            self.cv_val_end_ = str(usable_dates.loc[inner_val].max().date())

            def objective(trial: optuna.Trial) -> float:
                params = {**self._suggest_params(trial), **self.model_params}
                estimator = self._make_estimator(params)
                estimator.fit(X.loc[~inner_val], y.loc[~inner_val])
                prediction = estimator.predict(X.loc[inner_val])
                return float(mean_absolute_error(y.loc[inner_val], prediction))

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.seed, n_startup_trials=5),
            )
            study.optimize(objective, n_trials=self.n_trials, n_jobs=1)
            self.best_params_ = {**study.best_params, **self.model_params}
            self.cv_best_mae_ = float(study.best_value)
            self.trials_ = [
                {
                    "number": t.number,
                    "inner_mae": t.value,
                    "params": t.params,
                    "state": t.state.name,
                }
                for t in study.trials
            ]
        self.estimator_ = self._make_estimator(self.best_params_)
        self.estimator_.fit(X, y)
        return self

    def build_day_features(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        features = add_cyclical_calendar_features(future_df, task=self.task)
        lags = compute_single_day_lags(history_df, future_df, task=self.task)
        lag_cols = HOURLY_TREE_LAG_COLS if self.task == "hourly" else DAILY_TREE_LAG_COLS
        for column in lag_cols:
            features[column] = lags[column].to_numpy()
        return _model_input(features, self.feature_cols)

    def predict_features(self, features_df: pd.DataFrame) -> np.ndarray:
        if self.estimator_ is None:
            raise ValueError("Fit the tree model before predicting.")
        return self.estimator_.predict(_model_input(features_df, self.feature_cols))

    def to_pyfunc_model(self) -> TreePyFunc:
        if self.estimator_ is None:
            raise ValueError("Fit the tree model before exporting.")
        return TreePyFunc(self.estimator_, self.feature_cols)

    def pip_requirements(self) -> list[str]:
        packages = ["mlflow", "numpy", "pandas", "scikit-learn", "holidays"]
        if self.model_name != "random_forest":
            packages.append(
                "lightgbm" if self.model_name.startswith("lightgbm") else self.model_name
            )
        return [f"{package}=={version(package)}" for package in packages]

    def get_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "task": self.task,
            "model": self.model_name,
            "tune": self.tune,
            "n_trials": self.n_trials,
            "seed": self.seed,
            "n_jobs": self.n_jobs,
            "weather_mode": "observed_historical",
            "feature_cols": self.feature_cols,
            **self.best_params_,
        }
        if self.estimator_ is not None:
            params["train_rows"] = self.train_rows_
            params["warmup_rows_dropped"] = self.warmup_rows_
            # XGBoost's default missing-value marker is NaN; keep summaries valid JSON.
            params["estimator_params"] = {
                key: str(value)
                if isinstance(value, (float, np.floating)) and not np.isfinite(value)
                else value
                for key, value in self.estimator_.named_steps["model"].get_params().items()
            }
        if self.cv_best_mae_ is not None:
            params.update(
                cv_best_mae=self.cv_best_mae_,
                cv_train_end=self.cv_train_end_,
                cv_val_start=self.cv_val_start_,
                cv_val_end=self.cv_val_end_,
            )
        return params

    def feature_importance(self) -> pd.DataFrame:
        if self.estimator_ is None:
            raise ValueError("Fit the tree model before inspecting importance.")
        return pd.DataFrame(
            {
                "feature": self.feature_cols,
                "importance": self.estimator_.named_steps["model"].feature_importances_,
            }
        ).sort_values("importance", ascending=False)
