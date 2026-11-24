"""Dataset splits and leak-free day-ahead rolling-origin backtest iterators."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pandas as pd

from src.data_loader import find_repo_root

TaskType = Literal["hourly", "daily"]
SplitType = Literal["val_2023", "test_2024"]

TARGET_COL = "grid_load_mwh"
LEAKY_TARGET_COLS: tuple[str, ...] = (
    "grid_load_mwh",
    "grid_load_incl_pumped_mwh",
    "pumped_storage_mwh",
    "residual_load_mwh",
    "lag_1y",
)

TRAIN_START = pd.Timestamp("2019-01-01")
TRAIN_END = pd.Timestamp("2022-12-31")
VAL_START = pd.Timestamp("2023-01-01")
VAL_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2024-12-31")


def load_feature_dataset(task: TaskType, repo_root: Path | None = None) -> pd.DataFrame:
    """Load processed hourly or daily feature CSV with normalized timestamps."""
    if task not in {"hourly", "daily"}:
        raise ValueError(f"Unsupported task '{task}'. Expected 'hourly' or 'daily'.")

    root = repo_root or find_repo_root()
    filename = "hourly_features.csv" if task == "hourly" else "daily_features.csv"
    path = root / "data" / "processed" / filename
    df = pd.read_csv(path)

    df["date_local"] = pd.to_datetime(df["date_local"]).dt.normalize()
    if task == "hourly":
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        # Reconstruct Berlin local timestamps from UTC to avoid mixed-offset object dtypes.
        df["timestamp_local"] = df["timestamp_utc"].dt.tz_convert("Europe/Berlin")
        df = df.sort_values("timestamp_utc").reset_index(drop=True)
    else:
        df = df.sort_values("date_local").reset_index(drop=True)

    return df


def get_split_frames(
    df: pd.DataFrame,
    split: SplitType = "val_2023",
    *,
    allow_test: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Return `(train_df, eval_full_df, start_date, end_date)` with strict 2024 isolation."""
    if split not in {"val_2023", "test_2024"}:
        raise ValueError(f"Unsupported split '{split}'. Expected 'val_2023' or 'test_2024'.")
    if split == "test_2024" and not allow_test:
        raise ValueError(
            "Test split 'test_2024' is locked during local development. "
            "Pass allow_test=True only for final locked evaluation."
        )

    dates = pd.to_datetime(df["date_local"]).dt.normalize()
    working = df.copy()
    working["date_local"] = dates

    if split == "val_2023":
        train_df = working.loc[dates <= TRAIN_END].reset_index(drop=True)
        eval_full_df = working.loc[dates <= VAL_END].reset_index(drop=True)
        return train_df, eval_full_df, VAL_START, VAL_END

    train_val_df = working.loc[dates <= VAL_END].reset_index(drop=True)
    eval_full_df = working.loc[dates <= TEST_END].reset_index(drop=True)
    return train_val_df, eval_full_df, TEST_START, TEST_END


def iter_daily_origins(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> Iterator[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]]:
    """Yield `(origin_date, history_df, target_df)` one calendar day at a time.

    Guarantees that every row in `history_df` strictly precedes `origin_date`
    (`date_local < origin_date`) and `target_df` contains only `origin_date`.
    """
    if df.empty:
        raise ValueError("Cannot iterate origins over an empty DataFrame.")

    dates = pd.to_datetime(df["date_local"]).dt.normalize()
    if not dates.is_monotonic_increasing:
        raise ValueError("DataFrame must be sorted in non-decreasing 'date_local' order.")

    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()

    origin_mask = (dates >= start_ts) & (dates <= end_ts)
    origin_dates = pd.DatetimeIndex(dates.loc[origin_mask].unique()).sort_values()
    if origin_dates.empty:
        raise ValueError(
            f"No origin dates found between {start_ts.date()} and {end_ts.date()}."
        )

    for origin_date in origin_dates:
        day_start = int(dates.searchsorted(origin_date, side="left"))
        day_end = int(dates.searchsorted(origin_date, side="right"))

        history_df = df.iloc[:day_start]
        target_df = df.iloc[day_start:day_end]

        if history_df.empty:
            raise ValueError(f"No historical rows strictly before {origin_date.date()}.")
        if target_df.empty:
            raise ValueError(f"No target rows found for origin date {origin_date.date()}.")
        if pd.Timestamp(history_df["date_local"].iloc[-1]).normalize() >= origin_date:
            raise ValueError(f"Leakage detected: history contains rows on/after {origin_date}.")

        yield origin_date, history_df, target_df
