"""Descriptive consumption comparisons; no observations are removed or imputed."""

import numpy as np
import pandas as pd


def add_comparisons(frame: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """Compare adjacent intervals and the same local period seven days earlier.

    Input: ordered Europe/Berlin-aware start_local and numeric consumption_mwh.
    Ambiguous autumn local hours are excluded from weekly comparisons on either
    side, rather than arbitrarily choosing one of the two physical hours.
    """
    if resolution not in {"hourly", "daily"}:
        raise ValueError("resolution must be hourly or daily")
    df = frame.copy().reset_index(drop=True)
    start = df["start_local"]
    if str(start.dt.tz) != "Europe/Berlin":
        raise ValueError("start_local must use the Europe/Berlin timezone")
    if start.isna().any() or not start.is_monotonic_increasing or start.duplicated().any():
        raise ValueError("Start instants must be valid, ordered and unique")
    values = df["consumption_mwh"]
    if not np.isfinite(values).all():
        raise ValueError("Consumption must contain finite numeric values")
    local = pd.DatetimeIndex(start.dt.tz_localize(None))
    if resolution == "daily" and not (local == local.normalize()).all():
        raise ValueError("Daily intervals must start at local midnight")

    df["local_date"] = local.normalize()
    df["year"] = local.year
    df["month"] = local.month
    df["hour"] = local.hour
    df["weekday"] = local.day_of_week
    df["day_type"] = np.where(local.day_of_week < 5, "Weekday", "Weekend")
    if resolution == "daily":
        next_midnight = (local + pd.Timedelta(days=1)).tz_localize("Europe/Berlin")
        df["interval_hours"] = (next_midnight - pd.DatetimeIndex(start)).total_seconds() / 3600
        adjacent = pd.Series(local, index=df.index).diff().eq(pd.Timedelta(days=1))
    else:
        df["interval_hours"] = 1.0
        adjacent = start.diff().eq(pd.Timedelta(hours=1))
    df["average_load_mw"] = values / df["interval_hours"]
    for col in ["consumption_mwh", "average_load_mw"]:
        df[f"previous_{col}"] = df[col].shift(1).where(adjacent)
        df[f"change_{col}"] = df[col] - df[f"previous_{col}"]

    ambiguous = local.duplicated(keep=False)
    for col in ["consumption_mwh", "average_load_mw"]:
        reference = pd.Series(df[col].to_numpy()[~ambiguous], index=local[~ambiguous])
        previous_week = reference.reindex(local - pd.Timedelta(days=7)).to_numpy(
            dtype=float, copy=True
        )
        previous_week[ambiguous] = np.nan
        df[f"week_reference_{col}"] = previous_week
        df[f"week_change_{col}"] = df[col] - previous_week
    return df


def grouped_iqr(frame: pd.DataFrame, resolution: str, minimum_group_size: int = 8):
    """Add descriptive IQR bounds within year/month/day-type(/hour) groups.

    Full development-period groups are used. These are retrospective EDA flags,
    not a detector suitable for online forecasting or held-out evaluation.
    """
    if resolution not in {"hourly", "daily"}:
        raise ValueError("resolution must be hourly or daily")
    df = frame.copy()
    groups = ["year", "month", "day_type"]
    if resolution == "hourly":
        groups.append("hour")
    grouped = df.groupby(groups)["average_load_mw"]
    df["group_count"] = grouped.transform("count")
    df["group_q1_mw"] = grouped.transform(lambda x: x.quantile(0.25))
    df["group_q3_mw"] = grouped.transform(lambda x: x.quantile(0.75))
    width = df["group_q3_mw"] - df["group_q1_mw"]
    eligible = df["group_count"].ge(minimum_group_size) & width.gt(0)
    df["iqr_lower_mw"] = (df["group_q1_mw"] - 1.5 * width).where(eligible)
    df["iqr_upper_mw"] = (df["group_q3_mw"] + 1.5 * width).where(eligible)
    df["iqr_screened"] = eligible
    df["iqr_candidate"] = eligible & (
        df["average_load_mw"].lt(df["iqr_lower_mw"])
        | df["average_load_mw"].gt(df["iqr_upper_mw"])
    )
    return df
