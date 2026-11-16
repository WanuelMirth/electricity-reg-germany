"""Feature engineering for German hourly and daily electricity load forecasting.

Configured for strict full-year 2024 forecasting (train 2019-2023, test 2024)
with zero 2024 target leakage and a concise, interpretable feature set.
"""

from __future__ import annotations

import holidays
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
