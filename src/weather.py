"""Fetch, weight, and cache German historical weather covariates from Open-Meteo."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from src.data_loader import find_repo_root

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Five major German population & industrial hubs with approximate regional weights.
GERMAN_WEATHER_HUBS: list[dict[str, float | str]] = [
    {"name": "essen_ruhr", "lat": 51.4556, "lon": 7.0116, "weight": 0.30},
    {"name": "munich", "lat": 48.1351, "lon": 11.5820, "weight": 0.20},
    {"name": "frankfurt", "lat": 50.1109, "lon": 8.6821, "weight": 0.20},
    {"name": "berlin", "lat": 52.5200, "lon": 13.4050, "weight": 0.15},
    {"name": "hamburg", "lat": 53.5511, "lon": 9.9937, "weight": 0.15},
]

HOURLY_WEATHER_VARS = [
    "temperature_2m",
    "apparent_temperature",
    "shortwave_radiation",
    "wind_speed_10m",
    "cloud_cover",
]


def _fetch_hub_hourly(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    timeout: int = 30,
) -> pd.DataFrame:
    """Download hourly historical weather for a single coordinate in UTC."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_WEATHER_VARS),
        "timezone": "UTC",
    }
    resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()["hourly"]

    df = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(payload["time"], utc=True),
            "temp_c": pd.to_numeric(payload["temperature_2m"], errors="coerce"),
            "apparent_temp_c": pd.to_numeric(payload["apparent_temperature"], errors="coerce"),
            "solar_radiation_wm2": pd.to_numeric(payload["shortwave_radiation"], errors="coerce"),
            "wind_speed_kmh": pd.to_numeric(payload["wind_speed_10m"], errors="coerce"),
            "cloud_cover_pct": pd.to_numeric(payload["cloud_cover"], errors="coerce"),
        }
    )
    return df


def load_or_fetch_hourly_weather(
    repo_root: Path | None = None,
    start_date: str = "2018-12-31",
    end_date: str = "2025-01-01",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load cached population-weighted German hourly weather or fetch from Open-Meteo."""
    root = repo_root or find_repo_root()
    cache_path = root / "data" / "external" / "weather_germany_hourly_2019_2024.csv"

    if cache_path.is_file() and not force_refresh:
        df = pd.read_csv(cache_path)
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        return df

    total_weight = sum(float(h["weight"]) for h in GERMAN_WEATHER_HUBS)
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(f"Weather hub weights must sum to 1.0, got {total_weight}.")

    weighted_frames: list[pd.DataFrame] = []
    value_cols = [
        "temp_c",
        "apparent_temp_c",
        "solar_radiation_wm2",
        "wind_speed_kmh",
        "cloud_cover_pct",
    ]

    for hub in GERMAN_WEATHER_HUBS:
        hub_df = _fetch_hub_hourly(
            lat=float(hub["lat"]),
            lon=float(hub["lon"]),
            start_date=start_date,
            end_date=end_date,
        ).set_index("timestamp_utc")
        weighted_frames.append(hub_df[value_cols] * float(hub["weight"]))

    combined = sum(weighted_frames[1:], start=weighted_frames[0]).reset_index()
    combined = combined.sort_values("timestamp_utc").reset_index(drop=True)

    # Interpolate any isolated boundary NaNs if present.
    combined[value_cols] = combined[value_cols].interpolate(limit_direction="both")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(cache_path, index=False)
    return combined


def aggregate_weather_daily(hourly_weather: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly UTC weather into German local calendar-date summaries."""
    df = hourly_weather.copy()
    local_ts = df["timestamp_utc"].dt.tz_convert("Europe/Berlin")
    df["date_local"] = local_ts.dt.tz_localize(None).dt.normalize()

    daily = (
        df.groupby("date_local", as_index=False)
        .agg(
            temp_c_mean=("temp_c", "mean"),
            temp_c_min=("temp_c", "min"),
            temp_c_max=("temp_c", "max"),
            apparent_temp_c_mean=("apparent_temp_c", "mean"),
            solar_radiation_wm2_mean=("solar_radiation_wm2", "mean"),
            solar_radiation_whm2_sum=("solar_radiation_wm2", "sum"),
            wind_speed_kmh_mean=("wind_speed_kmh", "mean"),
            cloud_cover_pct_mean=("cloud_cover_pct", "mean"),
        )
        .sort_values("date_local")
        .reset_index(drop=True)
    )
    return daily
