"""Autoregressive electricity load forecasters (`Ridge-ARX`, `Spline-ARX`, daily `SARIMAX`).

All models implement `BaseForecaster`, perform strict train-only hyperparameter or order
selection (holding out the final calendar year of `train_df`), freeze fitted parameters
before rolling evaluation, and export compact stateless MLflow PyFunc models.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.evaluation.splits import TARGET_COL, TaskType
from src.features import (
    AR_DOW_DUMMY_COLS,
    DAILY_ARX_FEATURE_COLS,
    DAILY_SARIMAX_EXOG_BASE_COLS,
    DAILY_SPLINE_ARX_FEATURE_COLS,
    DAILY_SPLINE_WEATHER_COLS,
    HOURLY_ARX_FEATURE_COLS,
    HOURLY_SPLINE_ARX_FEATURE_COLS,
    HOURLY_SPLINE_WEATHER_COLS,
    add_ar_exogenous_features,
    build_arx_features,
    compute_single_day_arx_features,
)
from src.models.base import BaseForecaster


def _split_inner_temporal_cv(dates: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Split training rows into `(inner_train_mask, inner_val_mask)` strictly by calendar date.

    Uses the final calendar year of `train_df` (e.g., 2022 when `train_df` spans 2019-2022)
    when at least two calendar years are present; falls back to the final 20% of unique dates
    for shorter synthetic unit-test frames.
    """
    norm_dates = pd.to_datetime(dates).dt.normalize()
    unique_dates = pd.DatetimeIndex(norm_dates.unique()).sort_values()
    if len(unique_dates) < 10:
        raise ValueError(
            "Need at least 10 unique training dates for inner temporal CV, "
            f"got {len(unique_dates)}."
        )

    years = unique_dates.year.unique()
    if len(years) >= 2:
        val_year = int(years.max())
        val_mask = (norm_dates.dt.year == val_year).to_numpy()
        train_mask = ~val_mask
        if np.any(train_mask) and np.any(val_mask):
            return train_mask, val_mask

    cutoff_idx = max(1, int(len(unique_dates) * 0.8))
    cutoff_date = unique_dates[cutoff_idx]
    val_mask = (norm_dates >= cutoff_date).to_numpy()
    train_mask = ~val_mask
    return train_mask, val_mask


def _normalize_model_input(model_input: Any) -> pd.DataFrame:
    if isinstance(model_input, pd.DataFrame):
        return model_input
    return pd.DataFrame(model_input)


def _predict_stratified_pipelines(
    *,
    task: TaskType,
    feature_cols: list[str],
    models: Pipeline | dict[int, Pipeline],
    model_input: pd.DataFrame,
) -> np.ndarray:
    """Run stateless inference for a daily Pipeline or 24-hour-stratified Pipeline dict."""
    df = _normalize_model_input(model_input)
    required = set(feature_cols) | ({"hour"} if task == "hourly" else set())
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Model input missing required columns for task='{task}': {sorted(missing)}."
        )

    if task == "daily":
        assert isinstance(models, Pipeline)
        return np.asarray(models.predict(df[feature_cols]), dtype=float)

    assert isinstance(models, dict)
    hours = df["hour"].astype(int).to_numpy()
    preds = np.empty(len(df), dtype=float)
    for hour in np.unique(hours):
        h_int = int(hour)
        if h_int not in models:
            raise ValueError(f"Unexpected local hour '{h_int}' not found in fitted hourly models.")
        mask = hours == h_int
        preds[mask] = np.asarray(models[h_int].predict(df.loc[mask, feature_cols]), dtype=float)
    return preds


class RidgeARXPyFunc(mlflow.pyfunc.PythonModel):
    """Stateless MLflow PyFunc wrapper for daily or 24h-stratified Ridge-ARX."""

    def __init__(
        self,
        *,
        task: TaskType,
        feature_cols: list[str],
        models: Pipeline | dict[int, Pipeline],
    ) -> None:
        self.task = task
        self.feature_cols = list(feature_cols)
        self.models = models

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Predict load directly from pre-built ARX features."""
        return _predict_stratified_pipelines(
            task=self.task,
            feature_cols=self.feature_cols,
            models=self.models,
            model_input=model_input,
        )


class RidgeARXForecaster(BaseForecaster):
    """Regularized linear ARX forecaster (24-hour-stratified hourly / single daily)."""

    family = "autoregressive"
    name = "ridge_arx"

    def __init__(
        self,
        task: TaskType,
        *,
        alpha: float = 10.0,
        include_trend: bool = False,
        tune: bool = True,
        alpha_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 50.0, 100.0, 300.0, 1000.0),
        trend_grid: tuple[bool, ...] = (False, True),
    ) -> None:
        if task not in {"hourly", "daily"}:
            raise ValueError(f"Unsupported task '{task}'. Expected 'hourly' or 'daily'.")
        self.task: TaskType = task
        self.alpha = float(alpha)
        self.include_trend = bool(include_trend)
        self.tune = bool(tune)
        self.alpha_grid = tuple(float(a) for a in alpha_grid)
        self.trend_grid = tuple(bool(t) for t in trend_grid)

        self.selected_alpha_: float = self.alpha
        self.selected_include_trend_: bool = self.include_trend
        self.cv_best_mae_: float | None = None
        self.feature_cols_: list[str] = []
        self.models_: Pipeline | dict[int, Pipeline] | None = None

    def _feature_cols_for_trend(self, include_trend: bool) -> list[str]:
        base = list(HOURLY_ARX_FEATURE_COLS if self.task == "hourly" else DAILY_ARX_FEATURE_COLS)
        if include_trend:
            return ["time_years", *base]
        return base

    def _build_pipeline(self, alpha: float) -> Pipeline:
        return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])

    def _fit_pipelines(
        self,
        feat_df: pd.DataFrame,
        feature_cols: list[str],
        alpha: float,
    ) -> Pipeline | dict[int, Pipeline]:
        if self.task == "daily":
            pipe = self._build_pipeline(alpha)
            pipe.fit(feat_df[feature_cols], feat_df[TARGET_COL].to_numpy(dtype=float))
            return pipe

        hourly_models: dict[int, Pipeline] = {}
        hours = feat_df["hour"].astype(int).to_numpy()
        for hour in range(24):
            mask = hours == hour
            if not np.any(mask):
                raise ValueError(f"Training data has no valid rows for local hour {hour}.")
            pipe = self._build_pipeline(alpha)
            pipe.fit(
                feat_df.loc[mask, feature_cols],
                feat_df.loc[mask, TARGET_COL].to_numpy(dtype=float),
            )
            hourly_models[hour] = pipe
        return hourly_models

    def get_params(self) -> dict[str, Any]:
        """Return model configuration and train-only CV selection metadata for MLflow."""
        params: dict[str, Any] = {
            "task": self.task,
            "tune": self.tune,
            "selected_alpha": self.selected_alpha_,
            "selected_include_trend": self.selected_include_trend_,
            "n_features": len(self.feature_cols_) or len(
                self._feature_cols_for_trend(self.selected_include_trend_)
            ),
            "stratified_by_hour": self.task == "hourly",
        }
        if self.cv_best_mae_ is not None:
            params["cv_best_mae"] = round(self.cv_best_mae_, 4)
        return params

    def fit(self, train_df: pd.DataFrame) -> RidgeARXForecaster:
        """Fit Ridge-ARX strictly on `train_df` with optional train-only temporal CV."""
        full_feat = build_arx_features(
            train_df,
            task=self.task,
            target_col=TARGET_COL,
            include_prev_day_profile=False,
        )
        all_needed = list(
            dict.fromkeys(
                [
                    *self._feature_cols_for_trend(True),
                    TARGET_COL,
                    "date_local",
                    *(["hour"] if self.task == "hourly" else []),
                ]
            )
        )
        valid_feat = full_feat.dropna(subset=all_needed).reset_index(drop=True)
        if valid_feat.empty:
            raise ValueError("No valid training rows remain after constructing ARX lags.")

        best_alpha = self.alpha
        best_trend = self.include_trend
        best_mae: float | None = None

        if self.tune:
            inner_tr_mask, inner_val_mask = _split_inner_temporal_cv(valid_feat["date_local"])
            tr_df = valid_feat.loc[inner_tr_mask].reset_index(drop=True)
            va_df = valid_feat.loc[inner_val_mask].reset_index(drop=True)
            y_va = va_df[TARGET_COL].to_numpy(dtype=float)

            for include_trend in self.trend_grid:
                fcols = self._feature_cols_for_trend(include_trend)
                for alpha in self.alpha_grid:
                    candidate = self._fit_pipelines(tr_df, fcols, alpha)
                    preds = _predict_stratified_pipelines(
                        task=self.task,
                        feature_cols=fcols,
                        models=candidate,
                        model_input=va_df,
                    )
                    mae = float(np.mean(np.abs(y_va - preds)))
                    if best_mae is None or mae < best_mae:
                        best_mae = mae
                        best_alpha = alpha
                        best_trend = include_trend

        self.selected_alpha_ = best_alpha
        self.selected_include_trend_ = best_trend
        self.cv_best_mae_ = best_mae
        self.feature_cols_ = self._feature_cols_for_trend(best_trend)
        self.models_ = self._fit_pipelines(valid_feat, self.feature_cols_, best_alpha)
        return self

    def build_day_features(
        self,
        history_df: pd.DataFrame,
        future_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build leak-free day-ahead ARX features strictly from `history_df` and `future_df`."""
        if not self.feature_cols_:
            raise RuntimeError("RidgeARXForecaster must be fitted before building features.")
        day_df = compute_single_day_arx_features(
            history_df,
            future_df,
            task=self.task,
            target_col=TARGET_COL,
            include_prev_day_profile=False,
        )
        cols = (["hour"] if self.task == "hourly" else []) + self.feature_cols_
        out = day_df[cols].copy()
        for col in self.feature_cols_:
            out[col] = out[col].astype(float)
        return out

    def predict_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """Generate point forecasts from pre-built ARX features."""
        if self.models_ is None:
            raise RuntimeError("RidgeARXForecaster must be fitted before prediction.")
        return _predict_stratified_pipelines(
            task=self.task,
            feature_cols=self.feature_cols_,
            models=self.models_,
            model_input=features_df,
        )

    def to_pyfunc_model(self) -> mlflow.pyfunc.PythonModel:
        """Return a stateless MLflow PyFunc wrapper for Model Registry logging."""
        if self.models_ is None:
            raise RuntimeError("RidgeARXForecaster must be fitted before exporting PyFunc.")
        return RidgeARXPyFunc(
            task=self.task,
            feature_cols=self.feature_cols_,
            models=self.models_,
        )

    def pip_requirements(self) -> list[str]:
        """Return minimal pip dependencies required by `RidgeARXPyFunc`."""
        return ["mlflow", "numpy", "pandas", "scikit-learn"]


class SplineARXPyFunc(mlflow.pyfunc.PythonModel):
    """Stateless MLflow PyFunc wrapper for daily or 24h-stratified Spline-ARX."""

    def __init__(
        self,
        *,
        task: TaskType,
        feature_cols: list[str],
        models: Pipeline | dict[int, Pipeline],
    ) -> None:
        self.task = task
        self.feature_cols = list(feature_cols)
        self.models = models

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Predict load directly from pre-built Spline-ARX features."""
        return _predict_stratified_pipelines(
            task=self.task,
            feature_cols=self.feature_cols,
            models=self.models,
            model_input=model_input,
        )


class SplineARXForecaster(BaseForecaster):
    """GAM/LEAR-inspired penalized B-spline ARX forecaster (hourly 24h profile + daily)."""

    family = "autoregressive"
    name = "spline_arx"

    def __init__(
        self,
        task: TaskType,
        *,
        alpha: float = 50.0,
        n_knots: int = 6,
        include_trend: bool = False,
        tune: bool = True,
        alpha_grid: tuple[float, ...] = (1.0, 10.0, 50.0, 150.0, 500.0, 1500.0),
        knots_grid: tuple[int, ...] = (5, 7),
        trend_grid: tuple[bool, ...] = (False, True),
    ) -> None:
        if task not in {"hourly", "daily"}:
            raise ValueError(f"Unsupported task '{task}'. Expected 'hourly' or 'daily'.")
        self.task: TaskType = task
        self.alpha = float(alpha)
        self.n_knots = int(n_knots)
        self.include_trend = bool(include_trend)
        self.tune = bool(tune)
        self.alpha_grid = tuple(float(a) for a in alpha_grid)
        self.knots_grid = tuple(int(k) for k in knots_grid)
        self.trend_grid = tuple(bool(t) for t in trend_grid)

        self.selected_alpha_: float = self.alpha
        self.selected_n_knots_: int = self.n_knots
        self.selected_include_trend_: bool = self.include_trend
        self.cv_best_mae_: float | None = None
        self.feature_cols_: list[str] = []
        self.models_: Pipeline | dict[int, Pipeline] | None = None

    def _feature_cols_for_trend(self, include_trend: bool) -> list[str]:
        base = list(
            HOURLY_SPLINE_ARX_FEATURE_COLS
            if self.task == "hourly"
            else DAILY_SPLINE_ARX_FEATURE_COLS
        )
        if include_trend:
            return ["time_years", *base]
        return base

    def _build_pipeline(self, feature_cols: list[str], alpha: float, n_knots: int) -> Pipeline:
        weather_cols = (
            HOURLY_SPLINE_WEATHER_COLS if self.task == "hourly" else DAILY_SPLINE_WEATHER_COLS
        )
        doy_col = ["day_of_year"]
        linear_cols = [c for c in feature_cols if c not in set(weather_cols) and c != "day_of_year"]

        # Periodic annual spline spans [1, 367] so leap-year day 366 wraps cleanly.
        doy_knots = np.linspace(1.0, 367.0, 8).reshape(-1, 1)
        weather_transformer = Pipeline(
            [
                (
                    "spline",
                    SplineTransformer(
                        n_knots=n_knots,
                        degree=3,
                        knots="quantile",
                        extrapolation="linear",
                        include_bias=False,
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )
        doy_transformer = Pipeline(
            [
                (
                    "periodic_spline",
                    SplineTransformer(
                        degree=3,
                        knots=doy_knots,
                        extrapolation="periodic",
                        include_bias=False,
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("weather_spline", weather_transformer, weather_cols),
                ("doy_spline", doy_transformer, doy_col),
                ("linear", StandardScaler(), linear_cols),
            ],
            remainder="drop",
        )
        return Pipeline([("prep", preprocessor), ("ridge", Ridge(alpha=alpha))])

    def _fit_pipelines(
        self,
        feat_df: pd.DataFrame,
        feature_cols: list[str],
        alpha: float,
        n_knots: int,
    ) -> Pipeline | dict[int, Pipeline]:
        if self.task == "daily":
            pipe = self._build_pipeline(feature_cols, alpha, n_knots)
            pipe.fit(feat_df[feature_cols], feat_df[TARGET_COL].to_numpy(dtype=float))
            return pipe

        hourly_models: dict[int, Pipeline] = {}
        hours = feat_df["hour"].astype(int).to_numpy()
        for hour in range(24):
            mask = hours == hour
            if not np.any(mask):
                raise ValueError(f"Training data has no valid rows for local hour {hour}.")
            pipe = self._build_pipeline(feature_cols, alpha, n_knots)
            pipe.fit(
                feat_df.loc[mask, feature_cols],
                feat_df.loc[mask, TARGET_COL].to_numpy(dtype=float),
            )
            hourly_models[hour] = pipe
        return hourly_models

    def get_params(self) -> dict[str, Any]:
        """Return configuration and train-only CV selection metadata for MLflow."""
        params: dict[str, Any] = {
            "task": self.task,
            "tune": self.tune,
            "selected_alpha": self.selected_alpha_,
            "selected_n_knots": self.selected_n_knots_,
            "selected_include_trend": self.selected_include_trend_,
            "uses_prev_day_24h_profile": self.task == "hourly",
            "n_input_features": len(self.feature_cols_) or len(
                self._feature_cols_for_trend(self.selected_include_trend_)
            ),
            "stratified_by_hour": self.task == "hourly",
        }
        if self.cv_best_mae_ is not None:
            params["cv_best_mae"] = round(self.cv_best_mae_, 4)
        return params

    def fit(self, train_df: pd.DataFrame) -> SplineARXForecaster:
        """Fit Spline-ARX strictly on `train_df` with optional train-only temporal CV."""
        full_feat = build_arx_features(
            train_df,
            task=self.task,
            target_col=TARGET_COL,
            include_prev_day_profile=(self.task == "hourly"),
        )
        all_needed = list(
            dict.fromkeys(
                [
                    *self._feature_cols_for_trend(True),
                    TARGET_COL,
                    "date_local",
                    *(["hour"] if self.task == "hourly" else []),
                ]
            )
        )
        valid_feat = full_feat.dropna(subset=all_needed).reset_index(drop=True)
        if valid_feat.empty:
            raise ValueError("No valid training rows remain after constructing Spline-ARX lags.")

        best_alpha = self.alpha
        best_knots = self.n_knots
        best_trend = self.include_trend
        best_mae: float | None = None

        if self.tune:
            inner_tr_mask, inner_val_mask = _split_inner_temporal_cv(valid_feat["date_local"])
            tr_df = valid_feat.loc[inner_tr_mask].reset_index(drop=True)
            va_df = valid_feat.loc[inner_val_mask].reset_index(drop=True)
            y_va = va_df[TARGET_COL].to_numpy(dtype=float)

            for include_trend in self.trend_grid:
                fcols = self._feature_cols_for_trend(include_trend)
                for n_knots in self.knots_grid:
                    for alpha in self.alpha_grid:
                        candidate = self._fit_pipelines(tr_df, fcols, alpha, n_knots)
                        preds = _predict_stratified_pipelines(
                            task=self.task,
                            feature_cols=fcols,
                            models=candidate,
                            model_input=va_df,
                        )
                        mae = float(np.mean(np.abs(y_va - preds)))
                        if best_mae is None or mae < best_mae:
                            best_mae = mae
                            best_alpha = alpha
                            best_knots = n_knots
                            best_trend = include_trend

        self.selected_alpha_ = best_alpha
        self.selected_n_knots_ = best_knots
        self.selected_include_trend_ = best_trend
        self.cv_best_mae_ = best_mae
        self.feature_cols_ = self._feature_cols_for_trend(best_trend)
        self.models_ = self._fit_pipelines(
            valid_feat, self.feature_cols_, best_alpha, best_knots
        )
        return self

    def build_day_features(
        self,
        history_df: pd.DataFrame,
        future_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build leak-free day-ahead Spline-ARX features from `history_df` and `future_df`."""
        if not self.feature_cols_:
            raise RuntimeError("SplineARXForecaster must be fitted before building features.")
        day_df = compute_single_day_arx_features(
            history_df,
            future_df,
            task=self.task,
            target_col=TARGET_COL,
            include_prev_day_profile=(self.task == "hourly"),
        )
        cols = (["hour"] if self.task == "hourly" else []) + self.feature_cols_
        out = day_df[cols].copy()
        for col in self.feature_cols_:
            out[col] = out[col].astype(float)
        return out

    def predict_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """Generate point forecasts from pre-built Spline-ARX features."""
        if self.models_ is None:
            raise RuntimeError("SplineARXForecaster must be fitted before prediction.")
        return _predict_stratified_pipelines(
            task=self.task,
            feature_cols=self.feature_cols_,
            models=self.models_,
            model_input=features_df,
        )

    def to_pyfunc_model(self) -> mlflow.pyfunc.PythonModel:
        """Return a stateless MLflow PyFunc wrapper for Model Registry logging."""
        if self.models_ is None:
            raise RuntimeError("SplineARXForecaster must be fitted before exporting PyFunc.")
        return SplineARXPyFunc(
            task=self.task,
            feature_cols=self.feature_cols_,
            models=self.models_,
        )

    def pip_requirements(self) -> list[str]:
        """Return minimal pip dependencies required by `SplineARXPyFunc`."""
        return ["mlflow", "numpy", "pandas", "scikit-learn"]


@dataclass(frozen=True)
class FittedSARIMAXState:
    """Compact serializable parameters extracted from a fitted `statsmodels` SARIMAX model."""

    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    exog_cols: tuple[str, ...]
    has_const: bool
    y_mean: float
    y_std: float
    exog_mean: np.ndarray
    exog_scale: np.ndarray
    beta: np.ndarray
    const_beta: float
    arma_params: np.ndarray  # frozen AR/MA/seasonal AR/MA + sigma2 params for Kalman filter
    converged: bool
    aic: float


def _sarimax_exog_cols(seasonal_order: tuple[int, int, int, int]) -> list[str]:
    """Return collinearity-safe exogenous columns for daily SARIMAX.

    Under weekly seasonal differencing (`D >= 1`, `s = 7`), constant day-of-week dummies
    differentiate identically to zero (`Delta_7(dow_k) == 0`) and become rank-deficient.
    """
    _, seasonal_d, _, _ = seasonal_order
    if seasonal_d >= 1:
        return list(DAILY_SARIMAX_EXOG_BASE_COLS)
    return [*DAILY_SARIMAX_EXOG_BASE_COLS, *AR_DOW_DUMMY_COLS]


def _predict_sarima_residual_one_step(
    u_window: np.ndarray,
    *,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    arma_params: np.ndarray,
) -> float:
    """Forecast `u_t` 1-step ahead via exact frozen-parameter Kalman filtering."""
    u = np.asarray(u_window, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mod = SARIMAX(
            endog=u,
            order=order,
            seasonal_order=seasonal_order,
            trend="n",
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        filtered = mod.filter(arma_params)
        return float(filtered.forecast(steps=1)[0])


def _predict_from_sarimax_state(
    state: FittedSARIMAXState,
    residual_window: int,
    model_input: pd.DataFrame,
) -> np.ndarray:
    """Stateless batch prediction for a fitted daily `SARIMAX` model."""
    df = _normalize_model_input(model_input)
    resid_cols = [f"resid_lag_{k}" for k in range(1, residual_window + 1)]
    required = set(state.exog_cols) | set(resid_cols)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"SARIMAX input missing required columns: {sorted(missing)}.")

    exog_raw = df[list(state.exog_cols)].to_numpy(dtype=float)
    exog_scaled = (exog_raw - state.exog_mean) / state.exog_scale
    exog_hat = exog_scaled @ state.beta + state.const_beta

    # Columns are [resid_lag_1, ..., resid_lag_W]; reverse to chronological [t-W, ..., t-1].
    resid_matrix = df[resid_cols].to_numpy(dtype=float)[:, ::-1]

    u_hat = np.empty(len(df), dtype=float)
    for row_idx in range(len(df)):
        u_hat[row_idx] = _predict_sarima_residual_one_step(
            resid_matrix[row_idx],
            order=state.order,
            seasonal_order=state.seasonal_order,
            arma_params=state.arma_params,
        )

    y_scaled_hat = exog_hat + u_hat
    return state.y_mean + state.y_std * y_scaled_hat


class SARIMAXPyFunc(mlflow.pyfunc.PythonModel):
    """Compact stateless MLflow PyFunc wrapper for daily `SARIMAX`."""

    def __init__(self, *, state: FittedSARIMAXState, residual_window: int) -> None:
        self.state = state
        self.residual_window = int(residual_window)

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Predict daily load from day-d exogenous features and trailing residual window."""
        return _predict_from_sarimax_state(self.state, self.residual_window, model_input)


class SARIMAXForecaster(BaseForecaster):
    """Daily Box-Jenkins `SARIMAX(p,d,q)(P,D,Q,7)` forecaster with exogenous covariates."""

    family = "autoregressive"
    name = "sarimax"

    DEFAULT_CANDIDATE_ORDERS: tuple[tuple[tuple[int, int, int], tuple[int, int, int, int]], ...] = (
        ((1, 0, 1), (1, 0, 1, 7)),
        ((2, 0, 1), (1, 0, 1, 7)),
        ((1, 0, 1), (1, 1, 1, 7)),
        ((2, 0, 1), (1, 1, 1, 7)),
        ((1, 1, 1), (1, 1, 1, 7)),
        ((2, 1, 1), (0, 1, 1, 7)),
    )

    def __init__(
        self,
        task: TaskType = "daily",
        *,
        order: tuple[int, int, int] = (1, 0, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 7),
        residual_window: int = 42,
        tune: bool = True,
        candidate_orders: tuple[
            tuple[tuple[int, int, int], tuple[int, int, int, int]], ...
        ]
        | None = None,
        maxiter: int = 150,
    ) -> None:
        if task != "daily":
            raise ValueError(
                "SARIMAXForecaster supports task='daily' only. For hourly forecasting, use "
                "RidgeARXForecaster or SplineARXForecaster."
            )
        if residual_window < 21:
            raise ValueError(f"residual_window must be >= 21 days, got {residual_window}.")

        self.task: TaskType = "daily"
        self.order = tuple(int(x) for x in order)
        self.seasonal_order = tuple(int(x) for x in seasonal_order)
        self.residual_window = int(residual_window)
        self.tune = bool(tune)
        self.candidate_orders = (
            tuple(candidate_orders)
            if candidate_orders is not None
            else self.DEFAULT_CANDIDATE_ORDERS
        )
        self.maxiter = int(maxiter)

        self.selected_order_: tuple[int, int, int] = self.order  # type: ignore[assignment]
        self.selected_seasonal_order_: tuple[int, int, int, int] = self.seasonal_order  # type: ignore[assignment]
        self.cv_best_mae_: float | None = None
        self.state_: FittedSARIMAXState | None = None

    def _fit_single_order(
        self,
        df: pd.DataFrame,
        order: tuple[int, int, int],
        seasonal_order: tuple[int, int, int, int],
    ) -> FittedSARIMAXState:
        exog_cols = _sarimax_exog_cols(seasonal_order)
        _, d, _ = order
        _, seasonal_d, _, _ = seasonal_order
        has_const = d == 0 and seasonal_d == 0

        y_raw = df[TARGET_COL].to_numpy(dtype=float)
        y_mean = float(np.mean(y_raw))
        y_std = float(np.std(y_raw))
        if not np.isfinite(y_std) or y_std <= 0.0:
            raise ValueError("Daily target series has zero or non-finite standard deviation.")
        y_scaled = (y_raw - y_mean) / y_std

        exog_raw = df[exog_cols].to_numpy(dtype=float)
        exog_mean = np.mean(exog_raw, axis=0)
        exog_scale = np.std(exog_raw, axis=0)
        exog_scale = np.where(exog_scale < 1e-8, 1.0, exog_scale)
        exog_scaled = (exog_raw - exog_mean) / exog_scale

        if has_const:
            exog_design = np.column_stack([np.ones(len(exog_scaled), dtype=float), exog_scaled])
            exog_design_cols = ["_const", *exog_cols]
        else:
            exog_design = exog_scaled
            exog_design_cols = list(exog_cols)

        exog_frame = pd.DataFrame(exog_design, columns=exog_design_cols)
        model = SARIMAX(
            endog=y_scaled,
            exog=exog_frame,
            order=order,
            seasonal_order=seasonal_order,
            trend="n",
            enforce_stationarity=True,
            enforce_invertibility=True,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            res = model.fit(method="lbfgs", maxiter=self.maxiter, disp=False)
            converged = bool(res.mle_retvals.get("converged", False))
            if not converged:
                try:
                    res_alt = model.fit(
                        start_params=res.params,
                        method="bfgs",
                        maxiter=min(self.maxiter, 80),
                        disp=False,
                    )
                    if np.isfinite(res_alt.aic) and res_alt.aic <= res.aic:
                        res = res_alt
                        converged = bool(res.mle_retvals.get("converged", False))
                except Exception:
                    pass

        params_series = pd.Series(res.params, index=res.param_names)
        const_beta = float(params_series["_const"]) if has_const else 0.0
        beta = np.array([float(params_series[col]) for col in exog_cols], dtype=float)
        exog_name_set = set(exog_design_cols)
        arma_param_names = [name for name in res.param_names if name not in exog_name_set]
        arma_params = np.array([float(params_series[name]) for name in arma_param_names])

        return FittedSARIMAXState(
            order=order,
            seasonal_order=seasonal_order,
            exog_cols=tuple(exog_cols),
            has_const=has_const,
            y_mean=y_mean,
            y_std=y_std,
            exog_mean=exog_mean,
            exog_scale=exog_scale,
            beta=beta,
            const_beta=const_beta,
            arma_params=arma_params,
            converged=converged,
            aic=float(res.aic),
        )

    def _compute_scaled_residuals(
        self,
        df: pd.DataFrame,
        state: FittedSARIMAXState,
    ) -> np.ndarray:
        y_scaled = (df[TARGET_COL].to_numpy(dtype=float) - state.y_mean) / state.y_std
        exog_raw = df[list(state.exog_cols)].to_numpy(dtype=float)
        exog_scaled = (exog_raw - state.exog_mean) / state.exog_scale
        return y_scaled - (exog_scaled @ state.beta + state.const_beta)

    def _evaluate_candidate_on_inner_val(
        self,
        full_exog_df: pd.DataFrame,
        inner_tr_mask: np.ndarray,
        inner_val_mask: np.ndarray,
        state: FittedSARIMAXState,
    ) -> float:
        """Roll 1-day-ahead over inner validation rows with frozen parameters."""
        residuals = self._compute_scaled_residuals(full_exog_df, state)
        val_indices = np.flatnonzero(inner_val_mask)

        exog_raw = full_exog_df.iloc[val_indices][list(state.exog_cols)].to_numpy(dtype=float)
        exog_scaled = (exog_raw - state.exog_mean) / state.exog_scale
        exog_hat = exog_scaled @ state.beta + state.const_beta

        preds = np.empty(len(val_indices), dtype=float)
        for out_i, row_idx in enumerate(val_indices):
            start_idx = max(0, row_idx - self.residual_window)
            u_win = residuals[start_idx:row_idx]
            u_hat = _predict_sarima_residual_one_step(
                u_win,
                order=state.order,
                seasonal_order=state.seasonal_order,
                arma_params=state.arma_params,
            )
            preds[out_i] = state.y_mean + state.y_std * (exog_hat[out_i] + u_hat)

        y_true = full_exog_df.iloc[val_indices][TARGET_COL].to_numpy(dtype=float)
        return float(np.mean(np.abs(y_true - preds)))

    def get_params(self) -> dict[str, Any]:
        """Return SARIMAX order, configuration, and fit diagnostics for MLflow."""
        params: dict[str, Any] = {
            "task": self.task,
            "tune": self.tune,
            "selected_order": str(self.selected_order_),
            "selected_seasonal_order": str(self.selected_seasonal_order_),
            "residual_window": self.residual_window,
            "maxiter": self.maxiter,
        }
        if self.state_ is not None:
            params["n_exog_features"] = len(self.state_.exog_cols)
            params["converged"] = self.state_.converged
            params["train_aic"] = round(self.state_.aic, 2)
        if self.cv_best_mae_ is not None:
            params["cv_best_mae"] = round(self.cv_best_mae_, 4)
        return params

    def fit(self, train_df: pd.DataFrame) -> SARIMAXForecaster:
        """Fit daily SARIMAX strictly on `train_df` with optional train-only order CV."""
        sorted_df = train_df.sort_values("date_local").reset_index(drop=True)
        exog_df = add_ar_exogenous_features(sorted_df, task="daily")

        best_order = self.order
        best_seasonal = self.seasonal_order
        best_mae: float | None = None

        if self.tune:
            inner_tr_mask, inner_val_mask = _split_inner_temporal_cv(exog_df["date_local"])
            inner_tr_df = exog_df.loc[inner_tr_mask].reset_index(drop=True)
            if len(inner_tr_df) >= self.residual_window + 14:
                for cand_order, cand_seasonal in self.candidate_orders:
                    try:
                        cand_state = self._fit_single_order(inner_tr_df, cand_order, cand_seasonal)
                        cand_mae = self._evaluate_candidate_on_inner_val(
                            exog_df, inner_tr_mask, inner_val_mask, cand_state
                        )
                    except Exception:
                        continue
                    if np.isfinite(cand_mae) and (best_mae is None or cand_mae < best_mae):
                        best_mae = cand_mae
                        best_order = cand_order
                        best_seasonal = cand_seasonal

        self.selected_order_ = best_order  # type: ignore[assignment]
        self.selected_seasonal_order_ = best_seasonal  # type: ignore[assignment]
        self.cv_best_mae_ = best_mae
        self.state_ = self._fit_single_order(
            exog_df,
            self.selected_order_,
            self.selected_seasonal_order_,
        )
        return self

    def build_day_features(
        self,
        history_df: pd.DataFrame,
        future_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build day-d exogenous features plus trailing scaled residual lags from `history_df`."""
        if self.state_ is None:
            raise RuntimeError("SARIMAXForecaster must be fitted before building features.")
        if future_df.empty:
            raise ValueError("future_df cannot be empty.")

        origin_date = pd.Timestamp(future_df["date_local"].iloc[0]).normalize()
        hist_dates = pd.to_datetime(history_df["date_local"]).dt.normalize()
        if not history_df.empty and (hist_dates >= origin_date).any():
            raise ValueError(
                f"Leakage detected: history_df contains rows on/after {origin_date.date()}."
            )

        hist_tail = history_df.tail(self.residual_window).copy()
        if len(hist_tail) < self.residual_window:
            raise ValueError(
                f"Need at least {self.residual_window} historical daily rows for SARIMAX "
                f"state filtering, got {len(hist_tail)}."
            )

        hist_exog = add_ar_exogenous_features(hist_tail, task="daily")
        hist_resids = self._compute_scaled_residuals(hist_exog, self.state_)
        # Store most recent residual first: resid_lag_1 = u_{t-1}, ..., resid_lag_W = u_{t-W}.
        rev_resids = hist_resids[::-1]

        future_exog = add_ar_exogenous_features(future_df, task="daily")
        out = future_exog[list(self.state_.exog_cols)].astype(float).copy()
        for lag_k in range(1, self.residual_window + 1):
            out[f"resid_lag_{lag_k}"] = float(rev_resids[lag_k - 1])
        return out

    def predict_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict daily load from exogenous covariates and residual lag columns."""
        if self.state_ is None:
            raise RuntimeError("SARIMAXForecaster must be fitted before prediction.")
        return _predict_from_sarimax_state(self.state_, self.residual_window, features_df)

    def to_pyfunc_model(self) -> mlflow.pyfunc.PythonModel:
        """Return a compact stateless MLflow PyFunc wrapper for Registry logging."""
        if self.state_ is None:
            raise RuntimeError("SARIMAXForecaster must be fitted before exporting PyFunc.")
        return SARIMAXPyFunc(state=self.state_, residual_window=self.residual_window)

    def pip_requirements(self) -> list[str]:
        """Return minimal pip dependencies required by `SARIMAXPyFunc`."""
        return ["mlflow", "numpy", "pandas", "scikit-learn", "statsmodels"]
