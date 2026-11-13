"""Load and validate SMARD hourly and daily electricity load datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HOURLY_FILENAME = "Realisierter_Stromverbrauch_201901010000_202501010000_Stunde.csv"
DAILY_FILENAME = "Realisierter_Stromverbrauch_201901010000_202501010000_Tag.csv"

COLUMN_RENAME_MAP = {
    "Netzlast [MWh] Berechnete Auflösungen": "grid_load_mwh",
    "Netzlast inkl. Pumpspeicher [MWh] Berechnete Auflösungen": "grid_load_incl_pumped_mwh",
    "Pumpspeicher [MWh] Berechnete Auflösungen": "pumped_storage_mwh",
    "Residuallast [MWh] Berechnete Auflösungen": "residual_load_mwh",
}

TARGET_COL = "grid_load_mwh"


def find_repo_root(start: Path | None = None) -> Path:
    """Locate the repository root containing `data/` and `pyproject.toml`."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "data").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate repository root containing `data/` and `pyproject.toml`."
    )


def _parse_german_number(series: pd.Series) -> pd.Series:
    """Convert German-formatted numeric strings (e.g. '43.273,75') to float64."""
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="raise").astype("float64")


def load_smard_hourly(repo_root: Path | None = None) -> pd.DataFrame:
    """Load and parse the hourly SMARD dataset with DST-safe UTC & local timestamps."""
    root = repo_root or find_repo_root()
    path = root / "data" / HOURLY_FILENAME
    raw = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")

    start_naive = pd.to_datetime(raw["Datum von"], format="%d.%m.%Y %H:%M", errors="raise")
    # SMARD stores Europe/Berlin local time in chronological order; ambiguous="infer"
    # resolves the repeated autumn DST hour (02:00) cleanly into unique UTC instants.
    start_local = start_naive.dt.tz_localize(
        "Europe/Berlin", ambiguous="infer", nonexistent="raise"
    )
    start_utc = start_local.dt.tz_convert("UTC")

    df = pd.DataFrame(
        {
            "timestamp_utc": start_utc,
            "timestamp_local": start_local,
            "date_local": start_local.dt.tz_localize(None).dt.normalize(),
        }
    )
    for src_col, dst_col in COLUMN_RENAME_MAP.items():
        df[dst_col] = _parse_german_number(raw[src_col])

    if not df["timestamp_utc"].is_monotonic_increasing:
        raise ValueError("Hourly UTC timestamps are not monotonically increasing.")
    if df["timestamp_utc"].duplicated().any():
        raise ValueError("Duplicate UTC timestamps found in hourly SMARD data.")
    if df[TARGET_COL].isna().any():
        raise ValueError("Missing values found in hourly target column.")

    return df.reset_index(drop=True)


def load_smard_daily(repo_root: Path | None = None) -> pd.DataFrame:
    """Load and parse the daily SMARD dataset indexed by German local calendar date."""
    root = repo_root or find_repo_root()
    path = root / "data" / DAILY_FILENAME
    raw = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")

    date_local = pd.to_datetime(raw["Datum von"], format="%d.%m.%Y", errors="raise")

    df = pd.DataFrame({"date_local": date_local})
    for src_col, dst_col in COLUMN_RENAME_MAP.items():
        df[dst_col] = _parse_german_number(raw[src_col])

    if not df["date_local"].is_monotonic_increasing:
        raise ValueError("Daily dates are not monotonically increasing.")
    if df["date_local"].duplicated().any():
        raise ValueError("Duplicate dates found in daily SMARD data.")
    if df[TARGET_COL].isna().any():
        raise ValueError("Missing values found in daily target column.")

    return df.reset_index(drop=True)
