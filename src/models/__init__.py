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
from src.models.icl import (
    CONTEXT_PRESETS,
    DEFAULT_CONTEXT_PRESETS,
    ICL_MODEL_SPECS,
    ICL_MODELS,
    ICLForecaster,
    ICLPyFunc,
)
from src.models.trees import TreeForecaster

__all__ = [
    "BacktestResult",
    "BaseForecaster",
    "CONTEXT_PRESETS",
    "DEFAULT_CONTEXT_PRESETS",
    "DayBeforeBaseline",
    "ICLForecaster",
    "ICLPyFunc",
    "ICL_MODELS",
    "ICL_MODEL_SPECS",
    "LagBaselineForecaster",
    "LagBaselinePyFunc",
    "RidgeARXForecaster",
    "RidgeARXPyFunc",
    "SARIMAXForecaster",
    "SARIMAXPyFunc",
    "SeasonalNaiveBaseline",
    "SplineARXForecaster",
    "SplineARXPyFunc",
    "TreeForecaster",
    "run_rolling_backtest",
]
