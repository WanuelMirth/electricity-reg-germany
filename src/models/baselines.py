"""Day-ahead naive baseline forecasters (`lag_24h`/`lag_1d` and `lag_168h`/`lag_7d`)."""

from __future__ import annotations

from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd

from src.evaluation.splits import TARGET_COL, TaskType
from src.features import lookup_day_lag
from src.models.base import BaseForecaster


class LagBaselinePyFunc(mlflow.pyfunc.PythonModel):
    """Portable MLflow PyFunc wrapper for single-lag baseline models."""

    def __init__(self, feature_col: str, task: TaskType, lag_days: int) -> None:
        self.feature_col = feature_col
        self.task = task
        self.lag_days = lag_days

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Predict load directly from the pre-aligned lag feature column."""
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        if self.feature_col not in model_input.columns:
            raise ValueError(
                f"Expected lag column '{self.feature_col}' in model input, "
                f"got {list(model_input.columns)}."
            )
        return model_input[self.feature_col].to_numpy(dtype=float)


class LagBaselineForecaster(BaseForecaster):
    """Calendar-day-aligned lag baseline with DST-safe hourly lookup."""

    family = "baseline"

    def __init__(
        self,
        task: TaskType,
        *,
        lag_days: int,
        name: str,
        feature_col: str,
    ) -> None:
        if task not in {"hourly", "daily"}:
            raise ValueError(f"Unsupported task '{task}'. Expected 'hourly' or 'daily'.")
        if lag_days < 1:
            raise ValueError(f"lag_days must be >= 1, got {lag_days}.")
        self.task: TaskType = task
        self.lag_days = lag_days
        self.name = name
        self.feature_col = feature_col

    def get_params(self) -> dict[str, Any]:
        """Return baseline configuration parameters."""
        return {
            "task": self.task,
            "lag_days": self.lag_days,
            "feature_col": self.feature_col,
        }

    def fit(self, train_df: pd.DataFrame) -> LagBaselineForecaster:
        """Lag baselines require no parameter fitting."""
        return self

    def build_day_features(
        self,
        history_df: pd.DataFrame,
        future_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Look up `lag_days` prior load strictly inside `history_df`."""
        preds = lookup_day_lag(
            history_df,
            future_df,
            task=self.task,
            lag_days=self.lag_days,
            target_col=TARGET_COL,
        )
        return pd.DataFrame({self.feature_col: preds}, index=future_df.index)

    def predict_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """Extract the aligned lag column as float predictions."""
        return features_df[self.feature_col].to_numpy(dtype=float)

    def to_pyfunc_model(self) -> mlflow.pyfunc.PythonModel:
        """Create an MLflow PyFunc wrapper for Registry logging."""
        return LagBaselinePyFunc(
            feature_col=self.feature_col,
            task=self.task,
            lag_days=self.lag_days,
        )


class DayBeforeBaseline(LagBaselineForecaster):
    """Day-before naive baseline (`lag_24h` hourly / `lag_1d` daily)."""

    def __init__(self, task: TaskType) -> None:
        feature_col = "lag_24h_mwh" if task == "hourly" else "lag_1d_mwh"
        super().__init__(
            task,
            lag_days=1,
            name="naive_day_before",
            feature_col=feature_col,
        )


class SeasonalNaiveBaseline(LagBaselineForecaster):
    """Week-before seasonal naive baseline (`lag_168h` hourly / `lag_7d` daily)."""

    def __init__(self, task: TaskType) -> None:
        feature_col = "lag_168h_mwh" if task == "hourly" else "lag_7d_mwh"
        super().__init__(
            task,
            lag_days=7,
            name="seasonal_naive_week_before",
            feature_col=feature_col,
        )
