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


AR_HEATING_BASE_C = 15.5
AR_COOLING_BASE_C = 22.0
AR_FOURIER_ORDER = 3

AR_DOW_DUMMY_COLS = [f"dow_{i}" for i in range(1, 7)]
AR_FOURIER_COLS = [
    f"doy_{fn}_{k}" for k in range(1, AR_FOURIER_ORDER + 1) for fn in ("sin", "cos")
]
PREV_DAY_PROFILE_COLS = [f"prev_day_h{h:02d}" for h in range(24)]

HOURLY_AR_EXOG_BASE_COLS = [
    "hdd",
    "cdd",
    "wind_chill_hdd",
    "solar_radiation_wm2",
    "wind_speed_kmh",
    "holiday_weight",
    "holiday_x_weekday",
    "is_bridge_day",
    "is_christmas_shutdown",
    *AR_FOURIER_COLS,
]
DAILY_AR_EXOG_BASE_COLS = [
    "hdd",
    "cdd",
    "wind_chill_hdd",
    "solar_radiation_wm2_mean",
    "wind_speed_kmh_mean",
    "holiday_weight",
    "holiday_x_weekday",
    "is_bridge_day",
    "is_christmas_shutdown",
    *AR_FOURIER_COLS,
]
# Collinearity-pruned daily exogenous covariates for SARIMAX MLE stability
# (omits `holiday_weight` which overlaps `holiday_x_weekday`, and `wind_chill_hdd`).
DAILY_SARIMAX_EXOG_BASE_COLS = [
    "hdd",
    "cdd",
    "solar_radiation_wm2_mean",
    "wind_speed_kmh_mean",
    "holiday_x_weekday",
    "is_bridge_day",
    "is_christmas_shutdown",
    *AR_FOURIER_COLS,
]

HOURLY_ARX_LAG_COLS = [
    "lag_24h",
    "lag_48h",
    "lag_72h",
    "lag_168h",
    "lag_336h",
    "lag_504h",
    "rolling_mean_24h",
    "rolling_mean_168h",
    "prev_day_min",
    "prev_day_max",
    "prev_day_last_hour",
    "temp_lag_24h",
    "lag_1d_holiday_weight",
    "lag_7d_holiday_weight",
    "lag_1d_is_weekend",
]
DAILY_ARX_LAG_COLS = [
    "lag_1d",
    "lag_2d",
    "lag_3d",
    "lag_4d",
    "lag_5d",
    "lag_6d",
    "lag_7d",
    "lag_14d",
    "lag_21d",
    "rolling_mean_7d",
    "rolling_mean_28d",
    "rolling_std_7d",
    "temp_lag_1d",
    "lag_1d_holiday_weight",
    "lag_7d_holiday_weight",
    "lag_1d_is_weekend",
]

HOURLY_ARX_FEATURE_COLS = [
    *HOURLY_AR_EXOG_BASE_COLS,
    *AR_DOW_DUMMY_COLS,
    *HOURLY_ARX_LAG_COLS,
]
DAILY_ARX_FEATURE_COLS = [
    *DAILY_AR_EXOG_BASE_COLS,
    *AR_DOW_DUMMY_COLS,
    *DAILY_ARX_LAG_COLS,
]

HOURLY_SPLINE_WEATHER_COLS = ["temp_c", "solar_radiation_wm2", "wind_speed_kmh"]
DAILY_SPLINE_WEATHER_COLS = [
    "temp_c_mean",
    "temp_c_min",
    "temp_c_max",
    "solar_radiation_wm2_mean",
    "wind_speed_kmh_mean",
]

HOURLY_SPLINE_ARX_FEATURE_COLS = [
    *HOURLY_SPLINE_WEATHER_COLS,
    "day_of_year",
    "holiday_weight",
    "holiday_x_weekday",
    "is_bridge_day",
    "is_christmas_shutdown",
    *AR_DOW_DUMMY_COLS,
    *HOURLY_ARX_LAG_COLS,
    *PREV_DAY_PROFILE_COLS,
]
DAILY_SPLINE_ARX_FEATURE_COLS = [
    *DAILY_SPLINE_WEATHER_COLS,
    "day_of_year",
    "holiday_weight",
    "holiday_x_weekday",
    "is_bridge_day",
    "is_christmas_shutdown",
    *AR_DOW_DUMMY_COLS,
    *DAILY_ARX_LAG_COLS,
]


def add_ar_exogenous_features(df: pd.DataFrame, *, task: _TaskType) -> pd.DataFrame:
    """Add degree-day weather basis, annual Fourier harmonics, and calendar dummies."""
    _validate_task(task)
    out = df.copy()
    norm_dates = _normalize_date_series(out["date_local"])

    temp_col = "temp_c" if task == "hourly" else "temp_c_mean"
    wind_col = "wind_speed_kmh" if task == "hourly" else "wind_speed_kmh_mean"
    temp = out[temp_col].to_numpy(dtype=float)
    wind = out[wind_col].to_numpy(dtype=float)

    hdd = np.maximum(0.0, AR_HEATING_BASE_C - temp)
    cdd = np.maximum(0.0, temp - AR_COOLING_BASE_C)
    out["hdd"] = hdd
    out["cdd"] = cdd
    out["wind_chill_hdd"] = hdd * wind

    if task == "daily":
        if "temp_c_min" not in out.columns:
            out["temp_c_min"] = temp
        if "temp_c_max" not in out.columns:
            out["temp_c_max"] = temp

    dow_series = out["day_of_week"] if "day_of_week" in out.columns else norm_dates.dt.day_of_week
    doy_series = out["day_of_year"] if "day_of_year" in out.columns else norm_dates.dt.day_of_year
    dow = dow_series.to_numpy(dtype=int)
    doy = doy_series.to_numpy(dtype=float)
    out["day_of_year"] = doy

    for day_idx in range(1, 7):
        out[f"dow_{day_idx}"] = (dow == day_idx).astype(float)

    is_weekend = (
        out["is_weekend"].to_numpy(dtype=float)
        if "is_weekend" in out.columns
        else (dow >= 5).astype(float)
    )
    holiday_w = out["holiday_weight"].to_numpy(dtype=float)
    out["holiday_x_weekday"] = holiday_w * (1.0 - is_weekend)

    for k in range(1, AR_FOURIER_ORDER + 1):
        angle = 2.0 * np.pi * k * doy / 365.25
        out[f"doy_sin_{k}"] = np.sin(angle)
        out[f"doy_cos_{k}"] = np.cos(angle)

    return out


def _build_day_hour_matrix(
    df: pd.DataFrame,
    dates: pd.Series,
    value_col: str,
) -> pd.DataFrame:
    """Build a DST-safe `(date_local x hour 0..23)` matrix with intra-day interpolation."""
    hours = df["hour"].astype(int)
    grouped = (
        df.assign(_date=dates, _hour=hours)
        .groupby(["_date", "_hour"], sort=False)[value_col]
        .mean()
    )
    return (
        grouped.unstack("_hour")
        .reindex(columns=range(24))
        .interpolate(axis=1, limit_direction="both")
    )


def add_arx_lags(
    df: pd.DataFrame,
    *,
    task: _TaskType,
    target_col: str = "grid_load_mwh",
    include_prev_day_profile: bool = False,
) -> pd.DataFrame:
    """Add vectorized, DST-safe multi-week autoregressive lags strictly using `date_local <= d-1`.

    Includes lag-holiday/weekend indicators so linear and spline ARX models can offset
    holiday dips embedded in lagged load without propagating errors forward.
    """
    _validate_task(task)
    out = df.copy()
    if out.empty:
        cols = list(HOURLY_ARX_LAG_COLS if task == "hourly" else DAILY_ARX_LAG_COLS)
        if task == "hourly" and include_prev_day_profile:
            cols.extend(PREV_DAY_PROFILE_COLS)
        for col in cols:
            out[col] = np.nan
        return out

    out_dates = _normalize_date_series(out["date_local"])
    full_dates = pd.date_range(out_dates.min(), out_dates.max(), freq="D").astype(out_dates.dtype)

    daily_holiday = (
        out.assign(_date=out_dates)
        .groupby("_date", sort=True)["holiday_weight"]
        .first()
        .reindex(full_dates)
    )
    daily_weekend = (
        out.assign(_date=out_dates)
        .groupby("_date", sort=True)["is_weekend"]
        .first()
        .reindex(full_dates)
    )
    out["lag_1d_holiday_weight"] = out_dates.map(daily_holiday.shift(1)).to_numpy(dtype=float)
    out["lag_7d_holiday_weight"] = out_dates.map(daily_holiday.shift(7)).to_numpy(dtype=float)
    out["lag_1d_is_weekend"] = out_dates.map(daily_weekend.shift(1)).to_numpy(dtype=float)

    if task == "hourly":
        out_hours = out["hour"].astype(int)
        load_matrix = _build_day_hour_matrix(out, out_dates, target_col)
        filled_load = load_matrix.stack()

        for lag_days, col_name in (
            (1, "lag_24h"),
            (2, "lag_48h"),
            (3, "lag_72h"),
            (7, "lag_168h"),
            (14, "lag_336h"),
            (21, "lag_504h"),
        ):
            lookup_dates = (out_dates - pd.Timedelta(days=lag_days)).astype(out_dates.dtype)
            idx = pd.MultiIndex.from_arrays([lookup_dates, out_hours])
            out[col_name] = filled_load.reindex(idx).to_numpy(dtype=float)

        temp_matrix = _build_day_hour_matrix(out, out_dates, "temp_c")
        filled_temp = temp_matrix.stack()
        prev_dates = (out_dates - pd.Timedelta(days=1)).astype(out_dates.dtype)
        temp_idx = pd.MultiIndex.from_arrays([prev_dates, out_hours])
        out["temp_lag_24h"] = filled_temp.reindex(temp_idx).to_numpy(dtype=float)

        daily_agg = (
            out.assign(_date=out_dates)
            .groupby("_date", sort=True)[target_col]
            .agg(sum="sum", count="count", min="min", max="max")
            .reindex(full_dates)
        )
        prev_sum = daily_agg["sum"].shift(1)
        prev_cnt = daily_agg["count"].shift(1)
        out["rolling_mean_24h"] = out_dates.map(prev_sum / prev_cnt).to_numpy(dtype=float)
        out["rolling_mean_168h"] = out_dates.map(
            prev_sum.rolling(7, min_periods=7).sum() / prev_cnt.rolling(7, min_periods=7).sum()
        ).to_numpy(dtype=float)
        out["prev_day_min"] = out_dates.map(daily_agg["min"].shift(1)).to_numpy(dtype=float)
        out["prev_day_max"] = out_dates.map(daily_agg["max"].shift(1)).to_numpy(dtype=float)

        last_hour_series = load_matrix[23].reindex(full_dates)
        out["prev_day_last_hour"] = out_dates.map(last_hour_series.shift(1)).to_numpy(dtype=float)

        if include_prev_day_profile:
            profile_aligned = load_matrix.reindex(prev_dates)
            for h in range(24):
                out[f"prev_day_h{h:02d}"] = profile_aligned[h].to_numpy(dtype=float)

        return out

    daily_series = (
        out.assign(_date=out_dates)
        .groupby("_date", sort=True)[target_col]
        .last()
        .reindex(full_dates)
    )
    daily_temp = (
        out.assign(_date=out_dates)
        .groupby("_date", sort=True)["temp_c_mean"]
        .last()
        .reindex(full_dates)
    )
    for lag_days, col_name in (
        (1, "lag_1d"),
        (2, "lag_2d"),
        (3, "lag_3d"),
        (4, "lag_4d"),
        (5, "lag_5d"),
        (6, "lag_6d"),
        (7, "lag_7d"),
        (14, "lag_14d"),
        (21, "lag_21d"),
    ):
        out[col_name] = out_dates.map(daily_series.shift(lag_days)).to_numpy(dtype=float)

    shifted = daily_series.shift(1)
    out["rolling_mean_7d"] = out_dates.map(
        shifted.rolling(7, min_periods=7).mean()
    ).to_numpy(dtype=float)
    out["rolling_mean_28d"] = out_dates.map(
        shifted.rolling(28, min_periods=28).mean()
    ).to_numpy(dtype=float)
    out["rolling_std_7d"] = out_dates.map(
        shifted.rolling(7, min_periods=7).std(ddof=0)
    ).to_numpy(dtype=float)
    out["temp_lag_1d"] = out_dates.map(daily_temp.shift(1)).to_numpy(dtype=float)
    return out


def compute_single_day_arx_features(
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    *,
    task: _TaskType,
    target_col: str = "grid_load_mwh",
    include_prev_day_profile: bool = False,
) -> pd.DataFrame:
    """Compute leak-free day-ahead ARX exogenous + lag features for one origin day."""
    _validate_task(task)
    if future_df.empty:
        raise ValueError("future_df cannot be empty.")

    origin_date = pd.Timestamp(future_df["date_local"].iloc[0]).normalize()
    hist_dates = _normalize_date_series(history_df["date_local"])
    if not history_df.empty and (hist_dates >= origin_date).any():
        raise ValueError(
            f"Leakage detected: history_df contains rows on/after {origin_date.date()}."
        )

    out = add_ar_exogenous_features(future_df, task=task)
    n_rows = len(future_df)

    prev_1d_date = origin_date - pd.Timedelta(days=1)
    prev_7d_date = origin_date - pd.Timedelta(days=7)
    prev_1d_rows = history_df.loc[hist_dates == prev_1d_date]
    prev_7d_rows = history_df.loc[hist_dates == prev_7d_date]

    lag_1d_hw = float(prev_1d_rows["holiday_weight"].iloc[0]) if not prev_1d_rows.empty else np.nan
    lag_7d_hw = float(prev_7d_rows["holiday_weight"].iloc[0]) if not prev_7d_rows.empty else np.nan
    lag_1d_wknd = float(prev_1d_rows["is_weekend"].iloc[0]) if not prev_1d_rows.empty else np.nan

    out["lag_1d_holiday_weight"] = np.full(n_rows, lag_1d_hw, dtype=float)
    out["lag_7d_holiday_weight"] = np.full(n_rows, lag_7d_hw, dtype=float)
    out["lag_1d_is_weekend"] = np.full(n_rows, lag_1d_wknd, dtype=float)

    if task == "hourly":
        for lag_days, col_name in (
            (1, "lag_24h"),
            (2, "lag_48h"),
            (3, "lag_72h"),
            (7, "lag_168h"),
            (14, "lag_336h"),
            (21, "lag_504h"),
        ):
            out[col_name] = _safe_lookup_day_lag(
                history_df,
                future_df,
                task="hourly",
                lag_days=lag_days,
                target_col=target_col,
            )

        out["temp_lag_24h"] = _safe_lookup_day_lag(
            history_df,
            future_df,
            task="hourly",
            lag_days=1,
            target_col="temp_c",
        )

        rm_24h = float(prev_1d_rows[target_col].mean()) if not prev_1d_rows.empty else np.nan
        prev_min = float(prev_1d_rows[target_col].min()) if not prev_1d_rows.empty else np.nan
        prev_max = float(prev_1d_rows[target_col].max()) if not prev_1d_rows.empty else np.nan

        win_mask = (hist_dates >= prev_7d_date) & (hist_dates <= prev_1d_date)
        win_rows = history_df.loc[win_mask]
        rm_168h = (
            float(win_rows[target_col].mean())
            if hist_dates.loc[win_rows.index].nunique() == 7
            else np.nan
        )

        if not prev_1d_rows.empty:
            hour_means = {
                int(hour): float(val)
                for hour, val in prev_1d_rows.groupby("hour")[target_col].mean().items()
            }
            prev_profile = (
                pd.Series(hour_means, dtype=float)
                .reindex(range(24))
                .interpolate(limit_direction="both")
            )
            prev_last_hour = float(prev_profile.iloc[23])
        else:
            prev_profile = pd.Series(np.nan, index=range(24), dtype=float)
            prev_last_hour = np.nan

        out["rolling_mean_24h"] = np.full(n_rows, rm_24h, dtype=float)
        out["rolling_mean_168h"] = np.full(n_rows, rm_168h, dtype=float)
        out["prev_day_min"] = np.full(n_rows, prev_min, dtype=float)
        out["prev_day_max"] = np.full(n_rows, prev_max, dtype=float)
        out["prev_day_last_hour"] = np.full(n_rows, prev_last_hour, dtype=float)

        if include_prev_day_profile:
            for h in range(24):
                out[f"prev_day_h{h:02d}"] = np.full(
                    n_rows, float(prev_profile.iloc[h]), dtype=float
                )

        return out

    for lag_days, col_name in (
        (1, "lag_1d"),
        (2, "lag_2d"),
        (3, "lag_3d"),
        (4, "lag_4d"),
        (5, "lag_5d"),
        (6, "lag_6d"),
        (7, "lag_7d"),
        (14, "lag_14d"),
        (21, "lag_21d"),
    ):
        out[col_name] = _safe_lookup_day_lag(
            history_df,
            future_df,
            task="daily",
            lag_days=lag_days,
            target_col=target_col,
        )

    out["temp_lag_1d"] = _safe_lookup_day_lag(
        history_df,
        future_df,
        task="daily",
        lag_days=1,
        target_col="temp_c_mean",
    )

    win_7_mask = (hist_dates >= prev_7d_date) & (hist_dates <= prev_1d_date)
    win_7_rows = history_df.loc[win_7_mask]
    if hist_dates.loc[win_7_rows.index].nunique() == 7:
        win_7_vals = (
            win_7_rows.assign(_date=hist_dates.loc[win_7_rows.index])
            .groupby("_date", sort=True)[target_col]
            .last()
            .to_numpy(dtype=float)
        )
        rm_7d = float(np.mean(win_7_vals))
        rstd_7d = float(np.std(win_7_vals, ddof=0))
    else:
        rm_7d = np.nan
        rstd_7d = np.nan

    win_28_mask = (hist_dates >= origin_date - pd.Timedelta(days=28)) & (hist_dates <= prev_1d_date)
    win_28_rows = history_df.loc[win_28_mask]
    if hist_dates.loc[win_28_rows.index].nunique() == 28:
        win_28_vals = (
            win_28_rows.assign(_date=hist_dates.loc[win_28_rows.index])
            .groupby("_date", sort=True)[target_col]
            .last()
            .to_numpy(dtype=float)
        )
        rm_28d = float(np.mean(win_28_vals))
    else:
        rm_28d = np.nan

    out["rolling_mean_7d"] = np.full(n_rows, rm_7d, dtype=float)
    out["rolling_mean_28d"] = np.full(n_rows, rm_28d, dtype=float)
    out["rolling_std_7d"] = np.full(n_rows, rstd_7d, dtype=float)
    return out


def build_arx_features(
    df: pd.DataFrame,
    *,
    task: _TaskType,
    target_col: str = "grid_load_mwh",
    include_prev_day_profile: bool = False,
) -> pd.DataFrame:
    """Build exogenous features and leak-free day-ahead lags for autoregressive models."""
    out = add_ar_exogenous_features(df, task=task)
    return add_arx_lags(
        out,
        task=task,
        target_col=target_col,
        include_prev_day_profile=include_prev_day_profile,
    )
