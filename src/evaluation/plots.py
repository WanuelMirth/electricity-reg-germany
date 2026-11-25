"""Diagnostic validation plots logged as MLflow artifacts."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.splits import TaskType

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def create_validation_plots(
    predictions_df: pd.DataFrame,
    task: TaskType,
    model_name: str,
) -> dict[str, plt.Figure]:
    """Create standard diagnostic figures for a validation backtest run."""
    df = predictions_df.copy()
    df["date_local"] = pd.to_datetime(df["date_local"]).dt.normalize()
    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=float)
    df["abs_error"] = np.abs(y_true - y_pred)

    figures: dict[str, plt.Figure] = {}
    figures["actual_vs_pred_sample.png"] = _plot_actual_vs_pred_sample(df, task, model_name)
    figures["mae_by_weekday.png"] = _plot_mae_by_weekday(df, task, model_name)
    figures["daily_mae_over_time.png"] = _plot_daily_mae_over_time(df, task, model_name)
    return figures


def _plot_actual_vs_pred_sample(
    df: pd.DataFrame,
    task: TaskType,
    model_name: str,
) -> plt.Figure:
    if task == "hourly":
        # Two weeks in spring including weekends and Easter holidays (2023-03-27 to 2023-04-10).
        start, end = pd.Timestamp("2023-03-27"), pd.Timestamp("2023-04-10")
        sample = df.loc[(df["date_local"] >= start) & (df["date_local"] <= end)]
        if sample.empty:
            sample = df.iloc[: min(len(df), 14 * 24)]
        x_vals = pd.to_datetime(sample["timestamp_local"])
        subtitle = "2-Week Hourly Sample (Incl. Weekends & Easter)"
    else:
        start, end = pd.Timestamp("2023-03-01"), pd.Timestamp("2023-04-30")
        sample = df.loc[(df["date_local"] >= start) & (df["date_local"] <= end)]
        if sample.empty:
            sample = df.iloc[: min(len(df), 60)]
        x_vals = sample["date_local"]
        subtitle = "2-Month Daily Sample (Mar–Apr)"

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(x_vals, sample["y_true"] / 1000.0, label="Actual Load", linewidth=1.6, color="#1b4332")
    ax.plot(
        x_vals,
        sample["y_pred"] / 1000.0,
        label=f"Predicted ({model_name})",
        linewidth=1.4,
        linestyle="--",
        color="#d97706",
    )
    ax.set_title(f"{model_name} ({task}) — {subtitle}")
    ax.set_ylabel("Grid Load (GWh)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def _plot_mae_by_weekday(
    df: pd.DataFrame,
    task: TaskType,
    model_name: str,
) -> plt.Figure:
    mae_by_dow = (
        df.groupby("day_of_week")["abs_error"].mean().reindex(range(7), fill_value=0.0)
    )
    colors = ["#2d6a4f" if dow < 5 else "#e76f51" for dow in range(7)]

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    bars = ax.bar(WEEKDAY_NAMES, mae_by_dow.to_numpy() / 1000.0, color=colors, edgecolor="#222")
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.set_title(f"{model_name} ({task}) — Validation MAE by Day of Week")
    ax.set_ylabel("MAE (GWh)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_daily_mae_over_time(
    df: pd.DataFrame,
    task: TaskType,
    model_name: str,
) -> plt.Figure:
    daily_mae = df.groupby("date_local", as_index=False)["abs_error"].mean()
    rolling_7d = daily_mae["abs_error"].rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(
        daily_mae["date_local"],
        daily_mae["abs_error"] / 1000.0,
        alpha=0.45,
        linewidth=0.9,
        color="#457b9d",
        label="Daily MAE",
    )
    ax.plot(
        daily_mae["date_local"],
        rolling_7d / 1000.0,
        linewidth=1.8,
        color="#1d3557",
        label="7-Day Rolling Mean",
    )
    ax.set_title(f"{model_name} ({task}) — Daily Validation MAE Across 2023")
    ax.set_ylabel("MAE (GWh)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
