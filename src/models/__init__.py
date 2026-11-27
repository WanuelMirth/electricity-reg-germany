"""Forecasting model interfaces, rolling backtest runner, baselines, and autoregressive models."""

from src.models.autoregressive import (
    RidgeARXForecaster,
    RidgeARXPyFunc,
    SARIMAXForecaster,
    SARIMAXPyFunc,
    SplineARXForecaster,
    SplineARXPyFunc,
)
from src.models.base import BacktestResult, BaseForecaster, run_rolling_backtest
from src.models.baselines import (
    DayBeforeBaseline,
    LagBaselineForecaster,
    LagBaselinePyFunc,
    SeasonalNaiveBaseline,
)
from src.models.trees import TreeForecaster

__all__ = [
    "BacktestResult",
    "BaseForecaster",
    "DayBeforeBaseline",
    "LagBaselineForecaster",
    "LagBaselinePyFunc",
    "RidgeARXForecaster",
    "RidgeARXPyFunc",
    "SARIMAXForecaster",
    "SARIMAXPyFunc",
    "SeasonalNaiveBaseline",
    "TreeForecaster",
    "SplineARXForecaster",
    "SplineARXPyFunc",
    "run_rolling_backtest",
]
