"""Base forecaster abstraction and leak-free rolling-origin backtest runner."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd

from src.evaluation.splits import (
    LEAKY_TARGET_COLS,
    TARGET_COL,
    TaskType,
    iter_daily_origins,
)


@dataclass(frozen=True)
class BacktestResult:
    """Outputs produced by a day-ahead rolling-origin backtest."""

    predictions_df: pd.DataFrame
    input_example: pd.DataFrame
    sample_output: np.ndarray


class BaseForecaster(ABC):
    """Common interface for baseline, tree-based, and ICL load forecasters."""

    name: str
    family: str
    task: TaskType

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return hyperparameters/configuration logged to MLflow."""

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> BaseForecaster:
        """Fit model state strictly on historical training data."""

    @abstractmethod
    def build_day_features(
        self,
        history_df: pd.DataFrame,
        future_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Construct model-ready features for one forecast origin day."""

    @abstractmethod
    def predict_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """Produce point forecasts from pre-built features."""

    @abstractmethod
    def to_pyfunc_model(self) -> mlflow.pyfunc.PythonModel:
        """Return an MLflow PythonModel wrapper for Registry logging and loading."""

    def predict_day(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> np.ndarray:
        """Build features and generate point forecasts for one forecast origin day."""
        features_df = self.build_day_features(history_df, future_df)
        return self.predict_features(features_df)

    def pip_requirements(self) -> list[str]:
        """Return minimal pip dependencies required by the serialized pyfunc model."""
        return ["mlflow", "numpy", "pandas"]

    def make_input_example(
        self,
        features_df: pd.DataFrame,
        sample_output: np.ndarray,
    ) -> tuple[pd.DataFrame, np.ndarray] | None:
        """Optionally return a compact `(input_example, sample_output)` for MLflow logging."""
        return None


def run_rolling_backtest(
    model: BaseForecaster,
    train_df: pd.DataFrame,
    eval_full_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    target_col: str = TARGET_COL,
) -> BacktestResult:
    """Fit `model` on `train_df` and run a leak-free day-ahead backtest."""
    model.fit(train_df)

    example_target_rows = 24 if model.task == "hourly" else 7
    collected_example_rows = 0
    feature_examples: list[pd.DataFrame] = []
    pred_examples: list[np.ndarray] = []
    custom_example: tuple[pd.DataFrame, np.ndarray] | None = None
    prediction_frames: list[pd.DataFrame] = []

    keep_meta_cols = [
        col
        for col in (
            "date_local",
            "timestamp_utc",
            "timestamp_local",
            "hour",
            "day_of_week",
            "is_weekend",
            "holiday_weight",
        )
        if col in eval_full_df.columns
    ]

    for _, history_df, target_df in iter_daily_origins(eval_full_df, start_date, end_date):
        drop_cols = [col for col in LEAKY_TARGET_COLS if col in target_df.columns]
        future_df = target_df.drop(columns=drop_cols)

        features_df = model.build_day_features(history_df, future_df)
        y_pred = np.asarray(model.predict_features(features_df), dtype=float)

        if len(y_pred) != len(target_df):
            raise ValueError(
                f"{model.name} returned {len(y_pred)} predictions for {len(target_df)} rows."
            )
        if not np.isfinite(y_pred).all():
            raise ValueError(f"{model.name} returned non-finite predictions.")

        day_out = target_df[keep_meta_cols].copy()
        day_out["y_true"] = target_df[target_col].to_numpy(dtype=float)
        day_out["y_pred"] = y_pred
        prediction_frames.append(day_out)

        if custom_example is None:
            candidate = model.make_input_example(features_df, y_pred)
            if candidate is not None:
                custom_example = candidate
            elif collected_example_rows < example_target_rows:
                feature_examples.append(features_df)
                pred_examples.append(y_pred)
                collected_example_rows += len(features_df)

    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    if custom_example is not None:
        input_example, sample_output = custom_example
    else:
        input_example = (
            pd.concat(feature_examples, ignore_index=True)
            .iloc[:example_target_rows]
            .reset_index(drop=True)
        )
        sample_output = np.concatenate(pred_examples)[:example_target_rows]

    return BacktestResult(
        predictions_df=predictions_df,
        input_example=input_example,
        sample_output=sample_output,
    )
