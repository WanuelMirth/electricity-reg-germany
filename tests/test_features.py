"""Tests for leak-free day-ahead tree lag builders and baseline consistency."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.splits import get_split_frames, iter_daily_origins, load_feature_dataset
from src.features import (
    DAILY_TREE_FEATURE_COLS,
    DAILY_TREE_LAG_COLS,
    HOURLY_TREE_FEATURE_COLS,
    HOURLY_TREE_LAG_COLS,
    add_day_ahead_lags,
    build_tree_features,
    compute_single_day_lags,
)


def _make_hourly_frame(start: str, end_exclusive: str) -> pd.DataFrame:
    ts_local = pd.date_range(start, end_exclusive, freq="h", tz="Europe/Berlin", inclusive="left")
    ts_utc = ts_local.tz_convert("UTC")
    date_local = pd.to_datetime(ts_local.tz_localize(None)).normalize()
    hours = ts_local.hour.to_numpy(dtype=int)
    doy = ts_local.day_of_year.to_numpy(dtype=float)
    load = 35_000.0 + doy * 120.0 + hours * 25.0 + (ts_local.day_of_week.to_numpy() * 50.0)
    return pd.DataFrame(
        {
            "timestamp_utc": ts_utc,
            "timestamp_local": ts_local,
            "date_local": date_local,
            "grid_load_mwh": load,
            "hour": hours,
            "day_of_week": ts_local.day_of_week.to_numpy(dtype=int),
            "day_of_year": ts_local.day_of_year.to_numpy(dtype=int),
        }
    )


def _make_daily_frame(start: str, end_inclusive: str) -> pd.DataFrame:
    dates = pd.date_range(start, end_inclusive, freq="D")
    load = 1_000_000.0 + np.arange(len(dates), dtype=float) * 5_000.0
    return pd.DataFrame(
        {
            "date_local": dates,
            "grid_load_mwh": load,
            "day_of_week": dates.day_of_week.to_numpy(dtype=int),
            "day_of_year": dates.day_of_year.to_numpy(dtype=int),
        }
    )


def test_add_day_ahead_lags_zero_same_day_leakage_hourly_and_daily() -> None:
    hourly = _make_hourly_frame("2023-10-15", "2023-11-02")
    clean_lags = add_day_ahead_lags(hourly, task="hourly")

    # Corrupt all hours of one target day at a time (normal weekday + 25h autumn DST Sunday).
    for target_date in (pd.Timestamp("2023-10-25"), pd.Timestamp("2023-10-29")):
        corrupted = hourly.copy()
        mask_h = corrupted["date_local"] == target_date
        corrupted.loc[mask_h, "grid_load_mwh"] = 9_999_999.0
        corrupted_lags = add_day_ahead_lags(corrupted, task="hourly")

        pd.testing.assert_frame_equal(
            clean_lags.loc[mask_h, HOURLY_TREE_LAG_COLS],
            corrupted_lags.loc[mask_h, HOURLY_TREE_LAG_COLS],
        )

    daily = _make_daily_frame("2023-01-01", "2023-01-20")
    clean_daily = add_day_ahead_lags(daily, task="daily")
    corrupted_daily = daily.copy()
    mask_d = corrupted_daily["date_local"] == pd.Timestamp("2023-01-15")
    corrupted_daily.loc[mask_d, "grid_load_mwh"] = 9_999_999.0
    corrupted_daily_lags = add_day_ahead_lags(corrupted_daily, task="daily")

    pd.testing.assert_frame_equal(
        clean_daily.loc[mask_d, DAILY_TREE_LAG_COLS],
        corrupted_daily_lags.loc[mask_d, DAILY_TREE_LAG_COLS],
    )


def test_vectorized_lags_match_single_day_lags_across_dst() -> None:
    # Spring DST (2023-03-26 has 23h) + Autumn DST (2023-10-29 has 25h).
    for start, end, eval_start, eval_end in (
        ("2023-03-10", "2023-04-06", "2023-03-20", "2023-04-05"),
        ("2023-10-15", "2023-11-08", "2023-10-25", "2023-11-07"),
    ):
        hourly = _make_hourly_frame(start, end)
        vec_df = add_day_ahead_lags(hourly, task="hourly")

        for origin, history_df, target_df in iter_daily_origins(
            hourly, pd.Timestamp(eval_start), pd.Timestamp(eval_end)
        ):
            single_lags = compute_single_day_lags(history_df, target_df, task="hourly")
            vec_slice = vec_df.loc[target_df.index, HOURLY_TREE_LAG_COLS]
            assert not vec_slice.isna().any().any(), f"Unexpected NaNs on {origin.date()}"
            np.testing.assert_allclose(
                vec_slice.to_numpy(dtype=float),
                single_lags[HOURLY_TREE_LAG_COLS].to_numpy(dtype=float),
                rtol=1e-12,
                atol=1e-12,
            )

    daily = _make_daily_frame("2023-01-01", "2023-01-25")
    vec_daily = add_day_ahead_lags(daily, task="daily")
    for origin, history_df, target_df in iter_daily_origins(
        daily, pd.Timestamp("2023-01-10"), pd.Timestamp("2023-01-25")
    ):
        single_lags = compute_single_day_lags(history_df, target_df, task="daily")
        vec_slice = vec_daily.loc[target_df.index, DAILY_TREE_LAG_COLS]
        assert not vec_slice.isna().any().any(), f"Unexpected NaNs on {origin.date()}"
        np.testing.assert_allclose(
            vec_slice.to_numpy(dtype=float),
            single_lags[DAILY_TREE_LAG_COLS].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
        )


def test_build_tree_features_on_real_dataset_shapes_and_no_nans_in_val_2023() -> None:
    for task, expected_cols in (
        ("hourly", HOURLY_TREE_FEATURE_COLS),
        ("daily", DAILY_TREE_FEATURE_COLS),
    ):
        df = load_feature_dataset(task)
        _, eval_full_df, val_start, _ = get_split_frames(df, split="val_2023")
        feat_df = build_tree_features(eval_full_df, task=task)

        for col in expected_cols:
            assert col in feat_df.columns

        val_2023 = feat_df.loc[feat_df["date_local"] >= val_start, expected_cols]
        assert not val_2023.isna().any().any()
        assert np.isfinite(val_2023.to_numpy(dtype=float)).all()


def test_add_day_ahead_lags_rejects_invalid_task() -> None:
    df = _make_daily_frame("2023-01-01", "2023-01-10")
    with pytest.raises(ValueError, match="Unsupported task"):
        add_day_ahead_lags(df, task="weekly")  # type: ignore[arg-type]
