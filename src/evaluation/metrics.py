"""Forecast evaluation metrics with weekday, weekend, and holiday breakdowns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.splits import TARGET_COL, TaskType


def compute_seasonal_naive_scale(
    train_df: pd.DataFrame,
    task: TaskType,
    target_col: str = TARGET_COL,
) -> float:
    """Compute in-sample seasonal naive MAE on `train_df` for MASE scaling.

    Uses `m=168` (1 week of hours) for hourly and `m=7` (1 week of days) for daily.
    """
    if task not in {"hourly", "daily"}:
        raise ValueError(f"Unsupported task '{task}'. Expected 'hourly' or 'daily'.")

    season_period = 168 if task == "hourly" else 7
    y = train_df[target_col].to_numpy(dtype=float)
    if len(y) <= season_period:
        raise ValueError(
            f"Need more than {season_period} training rows to compute MASE scale, got {len(y)}."
        )

    scale = float(np.mean(np.abs(y[season_period:] - y[:-season_period])))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Invalid MASE scale computed from training data: {scale}.")
    return scale


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_forecast_metrics(
    predictions_df: pd.DataFrame,
    mase_scale: float,
    prefix: str = "val_",
) -> dict[str, float]:
    """Compute standard regression accuracy metrics plus calendar regime MAE slices."""
    if mase_scale <= 0.0 or not np.isfinite(mase_scale):
        raise ValueError(f"mase_scale must be finite and positive, got {mase_scale}.")

    required = {"y_true", "y_pred", "is_weekend", "holiday_weight"}
    missing = required - set(predictions_df.columns)
    if missing:
        raise ValueError(f"predictions_df missing required columns: {sorted(missing)}")

    y_true = predictions_df["y_true"].to_numpy(dtype=float)
    y_pred = predictions_df["y_pred"].to_numpy(dtype=float)
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on an empty predictions_df.")

    abs_err = np.abs(y_true - y_pred)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    nonzero = np.abs(y_true) > 0.0
    if not np.any(nonzero):
        raise ValueError("Cannot compute MAPE when all actual values are zero.")
    mape = float(np.mean(abs_err[nonzero] / np.abs(y_true[nonzero])) * 100.0)

    smape_denom = np.abs(y_true) + np.abs(y_pred)
    smape_terms = np.where(smape_denom == 0.0, 0.0, 2.0 * abs_err / smape_denom)
    smape = float(np.mean(smape_terms) * 100.0)
    mase = float(mae / mase_scale)

    metrics: dict[str, float] = {
        f"{prefix}mae": mae,
        f"{prefix}rmse": rmse,
        f"{prefix}mape": mape,
        f"{prefix}smape": smape,
        f"{prefix}mase": mase,
    }

    is_weekend = predictions_df["is_weekend"].to_numpy(dtype=int) == 1
    is_holiday = predictions_df["holiday_weight"].to_numpy(dtype=float) >= 0.5
    is_regular_weekday = (~is_weekend) & (~is_holiday)

    if np.any(is_regular_weekday):
        metrics[f"{prefix}mae_weekday"] = _mae(
            y_true[is_regular_weekday], y_pred[is_regular_weekday]
        )
    if np.any(is_weekend):
        metrics[f"{prefix}mae_weekend"] = _mae(y_true[is_weekend], y_pred[is_weekend])
    if np.any(is_holiday):
        metrics[f"{prefix}mae_holiday"] = _mae(y_true[is_holiday], y_pred[is_holiday])

    return metrics
