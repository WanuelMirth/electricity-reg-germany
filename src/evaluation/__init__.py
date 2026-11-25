"""Leak-free rolling-origin evaluation utilities, metrics, and diagnostic plots."""

from src.evaluation.metrics import compute_forecast_metrics, compute_seasonal_naive_scale
from src.evaluation.plots import create_validation_plots
from src.evaluation.splits import (
    LEAKY_TARGET_COLS,
    TARGET_COL,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
    get_split_frames,
    iter_daily_origins,
    load_feature_dataset,
)

__all__ = [
    "LEAKY_TARGET_COLS",
    "TARGET_COL",
    "TEST_END",
    "TEST_START",
    "TRAIN_END",
    "TRAIN_START",
    "VAL_END",
    "VAL_START",
    "compute_forecast_metrics",
    "compute_seasonal_naive_scale",
    "create_validation_plots",
    "get_split_frames",
    "iter_daily_origins",
    "load_feature_dataset",
]
