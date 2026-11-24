"""Feature engineering for German hourly and daily electricity load forecasting.

Configured for strict full-year 2024 forecasting (train 2019-2023, test 2024)
with zero 2024 target leakage and a concise, interpretable feature set.
"""

from __future__ import annotations

from typing import Literal

import holidays
import numpy as np
import pandas as pd

# Approximate share of German population/industrial load by federal state (sums to 1.0).
STATE_WEIGHTS: dict[str, float] = {
    "NW": 0.215,
    "BY": 0.158,
    "BW": 0.134,
    "NI": 0.096,
    "HE": 0.075,
    "SN": 0.049,
    "RP": 0.049,
    "BE": 0.044,
    "SH": 0.035,
    "BB": 0.030,
    "ST": 0.026,
    "TH": 0.025,
    "HH": 0.022,
    "MV": 0.019,
    "SL": 0.012,
    "HB": 0.008,
}
_TOTAL_STATE_WEIGHT = sum(STATE_WEIGHTS.values())
STATE_WEIGHTS = {k: v / _TOTAL_STATE_WEIGHT for k, v in STATE_WEIGHTS.items()}


def build_german_calendar_features(dates: pd.Series) -> pd.DataFrame:
    """Build German state-weighted holidays, bridge days, and Christmas shutdown flags."""
    unique_dates = pd.Series(pd.to_datetime(dates).dt.normalize().unique()).sort_values()
    years = sorted(unique_dates.dt.year.unique().tolist())

    national_de = holidays.country_holidays("DE", years=years)
    state_cals = {
        subdiv: holidays.country_holidays("DE", subdiv=subdiv, years=years)
        for subdiv in STATE_WEIGHTS
    }

    rows = []
    for dt in unique_dates:
        d = dt.date()
        rows.append(
            {
                "date_local": dt,
                "is_national_holiday": int(d in national_de),
                "holiday_weight": float(
                    sum(w for subdiv, w in STATE_WEIGHTS.items() if d in state_cals[subdiv])
                ),
            }
        )

    cal = pd.DataFrame(rows)
    dow = cal["date_local"].dt.day_of_week  # 0=Mon .. 6=Sun

    # Bridge day (Brueckentag): Friday after Thursday holiday, or Monday before Tuesday holiday.
    prev_holiday = cal["holiday_weight"].shift(1, fill_value=0.0) >= 0.5
    next_holiday = cal["holiday_weight"].shift(-1, fill_value=0.0) >= 0.5
    is_workday = (dow < 5) & (cal["holiday_weight"] < 0.5)
    cal["is_bridge_day"] = (
        is_workday & (((dow == 4) & prev_holiday) | ((dow == 0) & next_holiday))
    ).astype(int)

    month = cal["date_local"].dt.month
    day = cal["date_local"].dt.day
    cal["is_christmas_shutdown"] = (
        ((month == 12) & (day >= 24)) | ((month == 1) & (day <= 1))
    ).astype(int)

    return cal


def _add_calendar_time_features(df: pd.DataFrame, date_col: str = "date_local") -> pd.DataFrame:
    """Add continuous trend time and basic calendar indices."""
    out = df.copy()
    dt = pd.to_datetime(out[date_col])
    out["time_years"] = (dt - pd.Timestamp("2019-01-01")).dt.total_seconds() / (365.25 * 86400.0)
    out["year"] = dt.dt.year
    out["month"] = dt.dt.month
    out["day_of_week"] = dt.dt.day_of_week
    out["day_of_year"] = dt.dt.day_of_year
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    return out


def build_hourly_features(
    smard_hourly: pd.DataFrame,
    weather_hourly: pd.DataFrame,
    target_col: str = "grid_load_mwh",
) -> pd.DataFrame:
    """Merge SMARD hourly load with weather, German calendar, and 2024-leakage-free lags."""
    df = smard_hourly.merge(weather_hourly, on="timestamp_utc", how="left")
    if df["temp_c"].isna().any():
        raise ValueError("Missing weather values after merging on timestamp_utc.")

    cal = build_german_calendar_features(df["date_local"])
    df = df.merge(cal, on="date_local", how="left")
    df = _add_calendar_time_features(df, date_col="date_local")

    local_dt = df["timestamp_local"].dt.tz_convert("Europe/Berlin")
    df["hour"] = local_dt.dt.hour

    # 53-week (371-day) weekday-aligned lag kept only for the Naive 1y Prior baseline.
    df["lag_1y"] = df[target_col].shift(371 * 24)
    return df


def build_daily_features(
    smard_daily: pd.DataFrame,
    weather_daily: pd.DataFrame,
    target_col: str = "grid_load_mwh",
) -> pd.DataFrame:
    """Merge SMARD daily load with weather, German calendar, and 2024-leakage-free lags."""
    df = smard_daily.merge(weather_daily, on="date_local", how="left")
    if df["temp_c_mean"].isna().any():
        raise ValueError("Missing daily weather values after merging on date_local.")

    cal = build_german_calendar_features(df["date_local"])
    df = df.merge(cal, on="date_local", how="left")
    df = _add_calendar_time_features(df, date_col="date_local")

    # 53-week (371-day) weekday-aligned lag kept only for the Naive 1y Prior baseline.
    df["lag_1y"] = df[target_col].shift(371)
    return df


HOURLY_FULL_YEAR_COLS = [
    "time_years",
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "holiday_weight",
    "is_bridge_day",
    "is_christmas_shutdown",
    "temp_c",
    "solar_radiation_wm2",
    "wind_speed_kmh",
]

DAILY_FULL_YEAR_COLS = [
    "time_years",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "holiday_weight",
    "is_bridge_day",
    "is_christmas_shutdown",
    "temp_c_mean",
    "solar_radiation_wm2_mean",
    "wind_speed_kmh_mean",
]

HOURLY_TREE_LAG_COLS = [
    "lag_24h",
    "lag_48h",
    "lag_168h",
    "rolling_mean_24h",
    "rolling_mean_168h",
]
DAILY_TREE_LAG_COLS = ["lag_1d", "lag_2d", "lag_7d", "rolling_mean_7d"]

HOURLY_CYCLICAL_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos"]
DAILY_CYCLICAL_COLS = ["dow_sin", "dow_cos", "doy_sin", "doy_cos"]

# ICL models infer target lags directly from raw history and only receive future covariates.
HOURLY_ICL_COVARIATE_COLS = list(HOURLY_FULL_YEAR_COLS)
DAILY_ICL_COVARIATE_COLS = list(DAILY_FULL_YEAR_COLS)

HOURLY_TREE_FEATURE_COLS = [
    *HOURLY_FULL_YEAR_COLS,
    *HOURLY_CYCLICAL_COLS,
    *HOURLY_TREE_LAG_COLS,
]
DAILY_TREE_FEATURE_COLS = [
    *DAILY_FULL_YEAR_COLS,
    *DAILY_CYCLICAL_COLS,
    *DAILY_TREE_LAG_COLS,
]

_TaskType = Literal["hourly", "daily"]


def _validate_task(task: _TaskType) -> None:
    if task not in {"hourly", "daily"}:
        raise ValueError(f"Unsupported task '{task}'. Expected 'hourly' or 'daily'.")


def _normalize_date_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.normalize()
    return pd.to_datetime(series).dt.normalize()


def lookup_day_lag(
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    *,
    task: _TaskType,
    lag_days: int,
    target_col: str = "grid_load_mwh",
) -> np.ndarray:
    """Look up `lag_days` prior load strictly inside `history_df` with DST-safe alignment."""
    _validate_task(task)
    if lag_days < 1:
        raise ValueError(f"lag_days must be >= 1, got {lag_days}.")
    if future_df.empty:
        raise ValueError("future_df cannot be empty.")

    origin_date = pd.Timestamp(future_df["date_local"].iloc[0]).normalize()
    hist_dates = _normalize_date_series(history_df["date_local"])
    if not history_df.empty and (hist_dates >= origin_date).any():
        raise ValueError(
            f"Leakage detected: history_df contains rows on/after {origin_date.date()}."
        )

    ref_date = origin_date - pd.Timedelta(days=lag_days)
    ref_rows = history_df.loc[hist_dates == ref_date]
    if ref_rows.empty:
        raise ValueError(
            f"Missing reference history for {ref_date.date()} "
            f"(origin={origin_date.date()}, lag_days={lag_days})."
        )

    if task == "daily":
        ref_value = float(ref_rows[target_col].iloc[-1])
        return np.full(len(future_df), ref_value, dtype=float)

    # Match Europe/Berlin local hour rather than fixed 24*k rows so spring (23h)
    # and autumn (25h) DST days stay aligned without same-day leakage.
    hour_means = {
        int(hour): float(val)
        for hour, val in ref_rows.groupby("hour")[target_col].mean().items()
    }
    day_hours = pd.Series(hour_means, dtype=float).reindex(range(24)).interpolate(
        limit_direction="both"
    )
    return day_hours.reindex(future_df["hour"].astype(int)).to_numpy(dtype=float)


def _safe_lookup_day_lag(
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    *,
    task: _TaskType,
    lag_days: int,
    target_col: str,
) -> np.ndarray:
    try:
        return lookup_day_lag(
            history_df,
            future_df,
            task=task,
            lag_days=lag_days,
            target_col=target_col,
        )
    except ValueError as exc:
        if "Missing reference history" in str(exc):
            return np.full(len(future_df), np.nan, dtype=float)
        raise


def compute_single_day_lags(
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    *,
    task: _TaskType,
    target_col: str = "grid_load_mwh",
) -> pd.DataFrame:
    """Compute leak-free day-ahead lag and rolling features for one origin day."""
    _validate_task(task)
    if future_df.empty:
        raise ValueError("future_df cannot be empty.")

    origin_date = pd.Timestamp(future_df["date_local"].iloc[0]).normalize()
    hist_dates = _normalize_date_series(history_df["date_local"])
    if not history_df.empty and (hist_dates >= origin_date).any():
        raise ValueError(
            f"Leakage detected: history_df contains rows on/after {origin_date.date()}."
        )

    if task == "hourly":
        prev_day = history_df.loc[hist_dates == (origin_date - pd.Timedelta(days=1))]
        rm_24h = float(prev_day[target_col].mean()) if not prev_day.empty else np.nan

        win_mask = (hist_dates >= origin_date - pd.Timedelta(days=7)) & (
            hist_dates <= origin_date - pd.Timedelta(days=1)
        )
        win_rows = history_df.loc[win_mask]
        rm_168h = (
            float(win_rows[target_col].mean())
            if hist_dates.loc[win_rows.index].nunique() == 7
            else np.nan
        )

        return pd.DataFrame(
            {
                "lag_24h": _safe_lookup_day_lag(
                    history_df, future_df, task="hourly", lag_days=1, target_col=target_col
                ),
                "lag_48h": _safe_lookup_day_lag(
                    history_df, future_df, task="hourly", lag_days=2, target_col=target_col
                ),
                "lag_168h": _safe_lookup_day_lag(
                    history_df, future_df, task="hourly", lag_days=7, target_col=target_col
                ),
                "rolling_mean_24h": np.full(len(future_df), rm_24h, dtype=float),
                "rolling_mean_168h": np.full(len(future_df), rm_168h, dtype=float),
            },
            index=future_df.index,
        )

    win_mask = (hist_dates >= origin_date - pd.Timedelta(days=7)) & (
        hist_dates <= origin_date - pd.Timedelta(days=1)
    )
    win_rows = history_df.loc[win_mask]
    rm_7d = (
        float(win_rows[target_col].mean())
        if hist_dates.loc[win_rows.index].nunique() == 7
        else np.nan
    )
    return pd.DataFrame(
        {
            "lag_1d": _safe_lookup_day_lag(
                history_df, future_df, task="daily", lag_days=1, target_col=target_col
            ),
            "lag_2d": _safe_lookup_day_lag(
                history_df, future_df, task="daily", lag_days=2, target_col=target_col
            ),
            "lag_7d": _safe_lookup_day_lag(
                history_df, future_df, task="daily", lag_days=7, target_col=target_col
            ),
            "rolling_mean_7d": np.full(len(future_df), rm_7d, dtype=float),
        },
        index=future_df.index,
    )


def add_day_ahead_lags(
    df: pd.DataFrame,
    *,
    task: _TaskType,
    target_col: str = "grid_load_mwh",
) -> pd.DataFrame:
    """Add vectorized, DST-safe, day-ahead target lags and rolling means to `df`.

    For every forecast day `d`, all lags and rolling statistics use strictly
    observations from `date_local <= d - 1` so target-day load never leaks.
    """
    _validate_task(task)
    out = df.copy()
    if out.empty:
        cols = HOURLY_TREE_LAG_COLS if task == "hourly" else DAILY_TREE_LAG_COLS
        for col in cols:
            out[col] = np.nan
        return out

    out_dates = _normalize_date_series(out["date_local"])
    full_dates = pd.date_range(out_dates.min(), out_dates.max(), freq="D").astype(out_dates.dtype)

    if task == "hourly":
        out_hours = out["hour"].astype(int)
        hourly_ref = (
            out.assign(_date=out_dates, _hour=out_hours)
            .groupby(["_date", "_hour"], sort=False)[target_col]
            .mean()
        )
        filled_ref = (
            hourly_ref.unstack("_hour")
            .reindex(columns=range(24))
            .interpolate(axis=1, limit_direction="both")
            .stack()
        )

        for lag_days, col_name in ((1, "lag_24h"), (2, "lag_48h"), (7, "lag_168h")):
            lookup_dates = (out_dates - pd.Timedelta(days=lag_days)).astype(out_dates.dtype)
            idx = pd.MultiIndex.from_arrays([lookup_dates, out_hours])
            out[col_name] = filled_ref.reindex(idx).to_numpy(dtype=float)

        daily_agg = (
            out.assign(_date=out_dates)
            .groupby("_date", sort=True)[target_col]
            .agg(sum="sum", count="count")
            .reindex(full_dates)
        )
        prev_sum = daily_agg["sum"].shift(1)
        prev_cnt = daily_agg["count"].shift(1)
        rm_24h_map = prev_sum / prev_cnt
        rm_168h_map = (
            prev_sum.rolling(7, min_periods=7).sum()
            / prev_cnt.rolling(7, min_periods=7).sum()
        )
        out["rolling_mean_24h"] = out_dates.map(rm_24h_map).to_numpy(dtype=float)
        out["rolling_mean_168h"] = out_dates.map(rm_168h_map).to_numpy(dtype=float)
        return out

    daily_series = (
        out.assign(_date=out_dates)
        .groupby("_date", sort=True)[target_col]
        .last()
        .reindex(full_dates)
    )
    for lag_days, col_name in ((1, "lag_1d"), (2, "lag_2d"), (7, "lag_7d")):
        out[col_name] = out_dates.map(daily_series.shift(lag_days)).to_numpy(dtype=float)
    out["rolling_mean_7d"] = out_dates.map(
        daily_series.shift(1).rolling(7, min_periods=7).mean()
    ).to_numpy(dtype=float)
    return out


def add_cyclical_calendar_features(df: pd.DataFrame, *, task: _TaskType) -> pd.DataFrame:
    """Add sine/cosine encodings for hour-of-day (hourly only), day-of-week, and day-of-year."""
    _validate_task(task)
    out = df.copy()
    norm_dates = _normalize_date_series(out["date_local"])

    dow_series = out["day_of_week"] if "day_of_week" in out.columns else norm_dates.dt.day_of_week
    doy_series = out["day_of_year"] if "day_of_year" in out.columns else norm_dates.dt.day_of_year
    dow = dow_series.to_numpy(dtype=float)
    doy = doy_series.to_numpy(dtype=float)

    if task == "hourly":
        hour = out["hour"].to_numpy(dtype=float)
        out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

    out["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    out["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)
    return out


def build_tree_features(
    df: pd.DataFrame,
    *,
    task: _TaskType,
    target_col: str = "grid_load_mwh",
) -> pd.DataFrame:
    """Build cyclical calendar encodings and leak-free day-ahead lag features for tree models."""
    out = add_cyclical_calendar_features(df, task=task)
    return add_day_ahead_lags(out, task=task, target_col=target_col)
