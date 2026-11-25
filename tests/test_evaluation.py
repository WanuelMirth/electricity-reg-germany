"""Tests for dataset splits, leak-free rolling-origin iteration, metrics, and baselines."""

from __future__ import annotations

import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import compute_forecast_metrics, compute_seasonal_naive_scale
from src.evaluation.splits import (
    LEAKY_TARGET_COLS,
    get_split_frames,
    iter_daily_origins,
)
from src.models.base import BaseForecaster, run_rolling_backtest
from src.models.baselines import DayBeforeBaseline, SeasonalNaiveBaseline


def _make_synthetic_hourly(start: str, end_exclusive: str) -> pd.DataFrame:
    ts_local = pd.date_range(start, end_exclusive, freq="h", tz="Europe/Berlin", inclusive="left")
    ts_utc = ts_local.tz_convert("UTC")
    date_local = pd.to_datetime(ts_local.tz_localize(None)).normalize()
    hours = ts_local.hour.to_numpy(dtype=int)
    dow = ts_local.day_of_week.to_numpy(dtype=int)

    # Encode day-of-year and local hour into load so lag alignment mistakes are obvious.
    doy = ts_local.day_of_year.to_numpy(dtype=float)
    load = 40_000.0 + doy * 100.0 + hours * 10.0

    return pd.DataFrame(
        {
            "timestamp_utc": ts_utc,
            "timestamp_local": ts_local,
            "date_local": date_local,
            "grid_load_mwh": load,
            "grid_load_incl_pumped_mwh": load + 500.0,
            "pumped_storage_mwh": np.full(len(ts_utc), 500.0),
            "residual_load_mwh": load - 10_000.0,
            "lag_1y": load - 1000.0,
            "hour": hours,
            "day_of_week": dow,
            "is_weekend": (dow >= 5).astype(int),
            "holiday_weight": np.zeros(len(ts_utc), dtype=float),
        }
    )


def test_split_frames_locks_2024_by_default() -> None:
    dates = pd.date_range("2022-12-20", "2024-01-10", freq="D")
    df = pd.DataFrame({"date_local": dates, "grid_load_mwh": np.arange(len(dates), dtype=float)})

    with pytest.raises(ValueError, match="locked"):
        get_split_frames(df, split="test_2024")

    train_df, eval_full_df, start_date, end_date = get_split_frames(df, split="val_2023")
    assert start_date == pd.Timestamp("2023-01-01")
    assert end_date == pd.Timestamp("2023-12-31")
    assert pd.to_datetime(train_df["date_local"]).dt.year.max() == 2022
    assert pd.to_datetime(eval_full_df["date_local"]).dt.year.max() == 2023

    train_val_df, test_full_df, test_start, test_end = get_split_frames(
        df, split="test_2024", allow_test=True
    )
    assert test_start == pd.Timestamp("2024-01-01")
    assert test_end == pd.Timestamp("2024-12-31")
    assert pd.to_datetime(train_val_df["date_local"]).dt.year.max() == 2023
    assert pd.to_datetime(test_full_df["date_local"]).dt.year.max() == 2024


def test_iter_daily_origins_no_leakage_and_dst_counts() -> None:
    spring_df = _make_synthetic_hourly("2023-03-20", "2023-03-29")
    autumn_df = _make_synthetic_hourly("2023-10-22", "2023-11-01")

    spring_counts: dict[pd.Timestamp, int] = {}
    for origin, history_df, target_df in iter_daily_origins(
        spring_df, pd.Timestamp("2023-03-25"), pd.Timestamp("2023-03-28")
    ):
        assert history_df["date_local"].max() < origin
        assert (target_df["date_local"] == origin).all()
        spring_counts[origin] = len(target_df)
    assert spring_counts[pd.Timestamp("2023-03-26")] == 23

    autumn_counts: dict[pd.Timestamp, int] = {}
    for origin, history_df, target_df in iter_daily_origins(
        autumn_df, pd.Timestamp("2023-10-28"), pd.Timestamp("2023-10-31")
    ):
        assert history_df["date_local"].max() < origin
        assert (target_df["date_local"] == origin).all()
        autumn_counts[origin] = len(target_df)
    assert autumn_counts[pd.Timestamp("2023-10-29")] == 25


def test_baselines_dst_alignment_and_zero_same_day_leakage() -> None:
    # Autumn DST transition: 2023-10-29 has 25 hours (02:00 occurs twice).
    autumn_df = _make_synthetic_hourly("2023-10-15", "2023-11-02")

    # Sentinel on 2023-10-29 00:00 catches any naive .shift(24) same-day leakage at 23:00.
    oct29_midnight = autumn_df["timestamp_local"] == pd.Timestamp(
        "2023-10-29 00:00", tz="Europe/Berlin"
    )
    oct28_23h = autumn_df["timestamp_local"] == pd.Timestamp(
        "2023-10-28 23:00", tz="Europe/Berlin"
    )
    oct22_23h = autumn_df["timestamp_local"] == pd.Timestamp(
        "2023-10-22 23:00", tz="Europe/Berlin"
    )
    autumn_df.loc[oct29_midnight, "grid_load_mwh"] = 999_999.0
    autumn_df.loc[oct28_23h, "grid_load_mwh"] = 41_234.0
    autumn_df.loc[oct22_23h, "grid_load_mwh"] = 38_765.0

    train_dummy = autumn_df.iloc[:24]
    day_before = run_rolling_backtest(
        DayBeforeBaseline("hourly"),
        train_dummy,
        autumn_df,
        pd.Timestamp("2023-10-29"),
        pd.Timestamp("2023-10-30"),
    ).predictions_df
    week_before = run_rolling_backtest(
        SeasonalNaiveBaseline("hourly"),
        train_dummy,
        autumn_df,
        pd.Timestamp("2023-10-29"),
        pd.Timestamp("2023-10-30"),
    ).predictions_df

    oct29_preds_1d = day_before.loc[day_before["date_local"] == pd.Timestamp("2023-10-29")]
    oct29_preds_7d = week_before.loc[week_before["date_local"] == pd.Timestamp("2023-10-29")]
    assert len(oct29_preds_1d) == 25
    assert np.isfinite(oct29_preds_1d["y_pred"]).all()
    assert oct29_preds_1d.loc[oct29_preds_1d["hour"] == 23, "y_pred"].iloc[0] == 41_234.0
    assert oct29_preds_7d.loc[oct29_preds_7d["hour"] == 23, "y_pred"].iloc[0] == 38_765.0

    # On 2023-10-30 (day after 25h day), hour 0 should match 2023-10-29 hour 0 (999_999.0).
    oct30_preds_1d = day_before.loc[day_before["date_local"] == pd.Timestamp("2023-10-30")]
    assert oct30_preds_1d.loc[oct30_preds_1d["hour"] == 0, "y_pred"].iloc[0] == 999_999.0

    # Spring DST transition: 2023-03-26 misses 02:00 (23 hours); 2023-03-27 must align hour 3+.
    spring_df = _make_synthetic_hourly("2023-03-18", "2023-04-04")
    mar26_h1 = spring_df.loc[
        (spring_df["date_local"] == pd.Timestamp("2023-03-26")) & (spring_df["hour"] == 1),
        "grid_load_mwh",
    ].iloc[0]
    mar26_h3 = spring_df.loc[
        (spring_df["date_local"] == pd.Timestamp("2023-03-26")) & (spring_df["hour"] == 3),
        "grid_load_mwh",
    ].iloc[0]

    spring_1d = run_rolling_backtest(
        DayBeforeBaseline("hourly"),
        spring_df.iloc[:24],
        spring_df,
        pd.Timestamp("2023-03-26"),
        pd.Timestamp("2023-03-28"),
    ).predictions_df
    assert np.isfinite(spring_1d["y_pred"]).all()
    mar27 = spring_1d.loc[spring_1d["date_local"] == pd.Timestamp("2023-03-27")]
    assert mar27.loc[mar27["hour"] == 2, "y_pred"].iloc[0] == pytest.approx(
        (mar26_h1 + mar26_h3) / 2.0
    )
    assert mar27.loc[mar27["hour"] == 3, "y_pred"].iloc[0] == pytest.approx(mar26_h3)


class _SpyForecaster(BaseForecaster):
    name = "spy"
    family = "test"
    task = "daily"

    def __init__(self) -> None:
        self.seen_future_columns: list[set[str]] = []

    def get_params(self) -> dict[str, str]:
        return {}

    def fit(self, train_df: pd.DataFrame) -> _SpyForecaster:
        return self

    def build_day_features(
        self,
        history_df: pd.DataFrame,
        future_df: pd.DataFrame,
    ) -> pd.DataFrame:
        self.seen_future_columns.append(set(future_df.columns))
        return pd.DataFrame({"x": np.ones(len(future_df))}, index=future_df.index)

    def predict_features(self, features_df: pd.DataFrame) -> np.ndarray:
        return features_df["x"].to_numpy(dtype=float)

    def to_pyfunc_model(self) -> mlflow.pyfunc.PythonModel:
        raise NotImplementedError


def test_run_rolling_backtest_strips_leaky_target_columns() -> None:
    dates = pd.date_range("2023-01-01", "2023-01-10", freq="D")
    df = pd.DataFrame(
        {
            "date_local": dates,
            "grid_load_mwh": np.full(len(dates), 100.0),
            "grid_load_incl_pumped_mwh": np.full(len(dates), 105.0),
            "pumped_storage_mwh": np.full(len(dates), 5.0),
            "residual_load_mwh": np.full(len(dates), 50.0),
            "lag_1y": np.full(len(dates), 98.0),
            "day_of_week": dates.day_of_week,
            "is_weekend": (dates.day_of_week >= 5).astype(int),
            "holiday_weight": np.zeros(len(dates)),
            "temp_c_mean": np.full(len(dates), 5.0),
        }
    )
    spy = _SpyForecaster()
    result = run_rolling_backtest(
        spy,
        df.iloc[:3],
        df,
        pd.Timestamp("2023-01-05"),
        pd.Timestamp("2023-01-08"),
    )
    assert len(result.predictions_df) == 4
    for cols in spy.seen_future_columns:
        assert cols.isdisjoint(set(LEAKY_TARGET_COLS))
        assert "temp_c_mean" in cols


def test_compute_forecast_metrics_and_mase_scale_hand_calculated() -> None:
    # Daily training values [10, 12, ..., 24]: diff at lag 7 = 14 -> scale = 14.0
    train_df = pd.DataFrame({"grid_load_mwh": np.arange(10.0, 26.0, 2.0)})
    scale = compute_seasonal_naive_scale(train_df, task="daily")
    assert scale == pytest.approx(14.0)

    preds = pd.DataFrame(
        {
            "y_true": [100.0, 200.0, 100.0],
            "y_pred": [110.0, 180.0, 100.0],
            "is_weekend": [0, 1, 0],
            "holiday_weight": [0.0, 0.0, 1.0],
        }
    )
    # Abs errors: [10, 20, 0] -> MAE=10.0; Sq errors: [100, 400, 0] -> RMSE=sqrt(500/3)
    # MAPE: mean([10/100, 20/200, 0]) * 100 = 20 / 3
    # sMAPE: mean([20/210, 40/380, 0]) * 100
    metrics = compute_forecast_metrics(preds, mase_scale=5.0, prefix="val_")
    expected_smape = np.mean([20.0 / 210.0, 40.0 / 380.0, 0.0]) * 100.0

    assert metrics["val_mae"] == pytest.approx(10.0)
    assert metrics["val_rmse"] == pytest.approx(np.sqrt(500.0 / 3.0))
    assert metrics["val_mape"] == pytest.approx(20.0 / 3.0)
    assert metrics["val_smape"] == pytest.approx(expected_smape)
    assert metrics["val_mase"] == pytest.approx(2.0)
    assert metrics["val_mae_weekday"] == pytest.approx(10.0)
    assert metrics["val_mae_weekend"] == pytest.approx(20.0)
    assert metrics["val_mae_holiday"] == pytest.approx(0.0)
