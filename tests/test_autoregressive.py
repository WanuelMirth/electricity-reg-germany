"""Tests for autoregressive features, Ridge-ARX, Spline-ARX, daily SARIMAX, and MLflow PyFunc."""

from __future__ import annotations

import gc
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.evaluation.splits import iter_daily_origins
from src.features import (
    DAILY_ARX_FEATURE_COLS,
    DAILY_ARX_LAG_COLS,
    HOURLY_ARX_FEATURE_COLS,
    HOURLY_ARX_LAG_COLS,
    PREV_DAY_PROFILE_COLS,
    add_ar_exogenous_features,
    add_arx_lags,
    build_arx_features,
    compute_single_day_arx_features,
)
from src.models.autoregressive import (
    RidgeARXForecaster,
    SARIMAXForecaster,
    SplineARXForecaster,
    _sarimax_exog_cols,
)
from src.models.base import run_rolling_backtest
from src.tracking.mlflow_utils import (
    MLflowConfig,
    apply_quality_gate_aliases,
    configure_mlflow,
    evaluate_quality_gate,
    log_and_register_pyfunc_model,
    log_forecast_outputs,
    log_pyfunc_model,
    render_leaderboard,
    save_validation_summary,
    start_forecast_run,
)


def _make_synthetic_hourly(start: str, end_exclusive: str, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts_local = pd.date_range(start, end_exclusive, freq="h", tz="Europe/Berlin", inclusive="left")
    ts_utc = ts_local.tz_convert("UTC")
    date_local = pd.to_datetime(ts_local.tz_localize(None)).normalize()

    hours = ts_local.hour.to_numpy(dtype=int)
    dow = ts_local.day_of_week.to_numpy(dtype=int)
    doy = ts_local.day_of_year.to_numpy(dtype=int)
    is_weekend = (dow >= 5).astype(int)
    time_years = (date_local - pd.Timestamp("2019-01-01")).total_seconds().to_numpy() / (
        365.25 * 86400.0
    )

    holiday_weight = np.where((ts_local.month == 5) & (ts_local.day == 1), 1.0, 0.0)
    is_bridge = np.zeros(len(ts_utc), dtype=int)
    is_xmas = ((ts_local.month == 12) & (ts_local.day >= 24)).astype(int)

    temp_c = (
        10.0
        - 9.0 * np.cos(2.0 * np.pi * doy / 365.25)
        - 3.5 * np.cos(2.0 * np.pi * (hours - 14) / 24.0)
        + rng.normal(0.0, 1.5, len(ts_utc))
    )
    solar = np.maximum(
        0.0,
        450.0
        * np.sin(np.clip((hours - 6) / 12.0, 0.0, 1.0) * np.pi)
        * (0.6 - 0.4 * np.cos(2.0 * np.pi * doy / 365.25)),
    )
    wind = np.clip(12.0 + rng.normal(0.0, 3.0, len(ts_utc)), 1.0, 40.0)

    hdd = np.maximum(0.0, 15.5 - temp_c)
    load = (
        50_000.0
        + 7_000.0 * np.sin(2.0 * np.pi * (hours - 6) / 24.0)
        - 6_500.0 * is_weekend
        - 9_000.0 * holiday_weight * (1.0 - is_weekend)
        - 4_000.0 * is_xmas
        + 350.0 * hdd
        - 6.0 * solar
        - 400.0 * time_years
        + rng.normal(0.0, 350.0, len(ts_utc))
    )

    return pd.DataFrame(
        {
            "timestamp_utc": ts_utc,
            "timestamp_local": ts_local,
            "date_local": date_local,
            "grid_load_mwh": load,
            "grid_load_incl_pumped_mwh": load + 500.0,
            "pumped_storage_mwh": np.full(len(ts_utc), 500.0),
            "residual_load_mwh": load - 10_000.0,
            "lag_1y": load - 500.0,
            "temp_c": temp_c,
            "solar_radiation_wm2": solar,
            "wind_speed_kmh": wind,
            "holiday_weight": holiday_weight,
            "is_bridge_day": is_bridge,
            "is_christmas_shutdown": is_xmas,
            "time_years": time_years,
            "hour": hours,
            "day_of_week": dow,
            "day_of_year": doy,
            "is_weekend": is_weekend,
        }
    )


def _make_synthetic_daily(start: str, end_inclusive: str, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end_inclusive, freq="D")
    n = len(dates)
    dow = dates.day_of_week.to_numpy(dtype=int)
    doy = dates.day_of_year.to_numpy(dtype=int)
    is_weekend = (dow >= 5).astype(int)
    time_years = (dates - pd.Timestamp("2019-01-01")).total_seconds().to_numpy() / (
        365.25 * 86400.0
    )

    holiday_weight = np.where((dates.month == 5) & (dates.day == 1), 1.0, 0.0)
    is_bridge = np.zeros(n, dtype=int)
    is_xmas = ((dates.month == 12) & (dates.day >= 24)).astype(int)

    temp_mean = 10.0 - 9.0 * np.cos(2.0 * np.pi * doy / 365.25) + rng.normal(0.0, 2.0, n)
    temp_min = temp_mean - 3.5
    temp_max = temp_mean + 3.5
    solar_mean = np.clip(
        130.0 - 90.0 * np.cos(2.0 * np.pi * doy / 365.25) + rng.normal(0.0, 15.0, n),
        5.0,
        350.0,
    )
    wind_mean = np.clip(12.0 + rng.normal(0.0, 3.0, n), 1.0, 35.0)
    hdd = np.maximum(0.0, 15.5 - temp_mean)

    # Generate AR(1) + seasonal AR(7) autocorrelated errors for realistic SARIMAX testing.
    eps = rng.normal(0.0, 8_000.0, n)
    u = np.zeros(n, dtype=float)
    for t in range(n):
        ar1 = 0.55 * u[t - 1] if t >= 1 else 0.0
        ar7 = 0.25 * u[t - 7] if t >= 7 else 0.0
        u[t] = ar1 + ar7 + eps[t]

    load = (
        1_300_000.0
        - 180_000.0 * is_weekend
        - 220_000.0 * holiday_weight * (1.0 - is_weekend)
        - 140_000.0 * is_xmas
        + 9_500.0 * hdd
        - 180.0 * solar_mean
        - 10_000.0 * time_years
        + u
    )

    return pd.DataFrame(
        {
            "date_local": dates,
            "grid_load_mwh": load,
            "grid_load_incl_pumped_mwh": load + 10_000.0,
            "pumped_storage_mwh": np.full(n, 10_000.0),
            "residual_load_mwh": load - 200_000.0,
            "lag_1y": load - 15_000.0,
            "temp_c_mean": temp_mean,
            "temp_c_min": temp_min,
            "temp_c_max": temp_max,
            "solar_radiation_wm2_mean": solar_mean,
            "wind_speed_kmh_mean": wind_mean,
            "holiday_weight": holiday_weight,
            "is_bridge_day": is_bridge,
            "is_christmas_shutdown": is_xmas,
            "time_years": time_years,
            "day_of_week": dow,
            "day_of_year": doy,
            "is_weekend": is_weekend,
        }
    )


@pytest.fixture()
def isolated_mlflow(tmp_path: Path):
    """Provide an isolated SQLite MLflow client/config without touching repo `mlruns/`."""
    prev_tracking = mlflow.get_tracking_uri()
    prev_registry = mlflow.get_registry_uri()

    db_path = (tmp_path / "mlflow.db").resolve()
    artifact_dir = (tmp_path / "artifacts").resolve()
    uri = f"sqlite:///{db_path.as_posix()}"

    config = MLflowConfig(
        tracking_uri=uri,
        registry_uri=uri,
        artifact_root=artifact_dir.as_uri(),
        experiment_name="test-autoregressive",
    )
    client = configure_mlflow(config)
    try:
        yield client, config
    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run()
        try:
            client._tracking_client.store.engine.dispose()
        except AttributeError:
            pass
        try:
            client._get_registry_client().store.engine.dispose()
        except AttributeError:
            pass
        mlflow.set_tracking_uri(prev_tracking)
        mlflow.set_registry_uri(prev_registry)
        gc.collect()


def test_arx_lags_zero_same_day_leakage_hourly_and_daily() -> None:
    hourly = _make_synthetic_hourly("2023-09-20", "2023-11-02")
    clean_hourly = add_arx_lags(hourly, task="hourly", include_prev_day_profile=True)
    hourly_check_cols = [*HOURLY_ARX_LAG_COLS, *PREV_DAY_PROFILE_COLS]

    # Normal weekday + 25-hour autumn DST Sunday (2023-10-29).
    for target_date in (pd.Timestamp("2023-10-25"), pd.Timestamp("2023-10-29")):
        corrupted = hourly.copy()
        mask = corrupted["date_local"] == target_date
        corrupted.loc[mask, "grid_load_mwh"] = 9_999_999.0
        corrupted_lags = add_arx_lags(corrupted, task="hourly", include_prev_day_profile=True)
        pd.testing.assert_frame_equal(
            clean_hourly.loc[mask, hourly_check_cols],
            corrupted_lags.loc[mask, hourly_check_cols],
        )

    daily = _make_synthetic_daily("2023-01-01", "2023-03-10")
    clean_daily = add_arx_lags(daily, task="daily")
    corrupted_daily = daily.copy()
    mask_d = corrupted_daily["date_local"] == pd.Timestamp("2023-02-20")
    corrupted_daily.loc[mask_d, "grid_load_mwh"] = 9_999_999.0
    corrupted_daily_lags = add_arx_lags(corrupted_daily, task="daily")
    pd.testing.assert_frame_equal(
        clean_daily.loc[mask_d, DAILY_ARX_LAG_COLS],
        corrupted_daily_lags.loc[mask_d, DAILY_ARX_LAG_COLS],
    )


def test_vectorized_arx_matches_single_day_across_spring_and_autumn_dst() -> None:
    hourly_cols = [*HOURLY_ARX_FEATURE_COLS, *PREV_DAY_PROFILE_COLS]
    for start, end, eval_start, eval_end in (
        ("2023-02-25", "2023-04-04", "2023-03-23", "2023-04-02"),
        ("2023-10-01", "2023-11-06", "2023-10-26", "2023-11-04"),
    ):
        hourly = _make_synthetic_hourly(start, end)
        vec_df = build_arx_features(hourly, task="hourly", include_prev_day_profile=True)
        for origin, history_df, target_df in iter_daily_origins(
            hourly, pd.Timestamp(eval_start), pd.Timestamp(eval_end)
        ):
            single = compute_single_day_arx_features(
                history_df,
                target_df.drop(columns=["grid_load_mwh"]),
                task="hourly",
                include_prev_day_profile=True,
            )
            vec_slice = vec_df.loc[target_df.index, hourly_cols]
            assert not vec_slice.isna().any().any(), f"Unexpected NaNs on {origin.date()}"
            np.testing.assert_allclose(
                vec_slice.to_numpy(dtype=float),
                single[hourly_cols].to_numpy(dtype=float),
                rtol=1e-12,
                atol=1e-12,
            )

    daily = _make_synthetic_daily("2023-01-01", "2023-03-01")
    vec_daily = build_arx_features(daily, task="daily")
    for origin, history_df, target_df in iter_daily_origins(
        daily, pd.Timestamp("2023-02-05"), pd.Timestamp("2023-02-25")
    ):
        single = compute_single_day_arx_features(
            history_df,
            target_df.drop(columns=["grid_load_mwh"]),
            task="daily",
        )
        vec_slice = vec_daily.loc[target_df.index, DAILY_ARX_FEATURE_COLS]
        assert not vec_slice.isna().any().any(), f"Unexpected NaNs on {origin.date()}"
        np.testing.assert_allclose(
            vec_slice.to_numpy(dtype=float),
            single[DAILY_ARX_FEATURE_COLS].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
        )


@pytest.mark.parametrize("model_cls", [RidgeARXForecaster, SplineARXForecaster])
@pytest.mark.parametrize("tune", [False, True])
def test_ridge_and_spline_arx_hourly_dst_backtest_and_pyfunc_parity(model_cls, tune: bool) -> None:
    # Span across both spring (23h) and autumn (25h) DST transitions.
    df = _make_synthetic_hourly("2022-01-01", "2023-11-03")
    train_df = df.loc[df["date_local"] <= pd.Timestamp("2023-02-28")].reset_index(drop=True)

    model = model_cls("hourly", tune=tune)
    # Test spring 23h day (2023-03-26) and surrounding days.
    spring_res = run_rolling_backtest(
        model,
        train_df,
        df,
        pd.Timestamp("2023-03-25"),
        pd.Timestamp("2023-03-27"),
    )
    preds = spring_res.predictions_df
    assert len(preds.loc[preds["date_local"] == pd.Timestamp("2023-03-26")]) == 23
    assert np.isfinite(preds["y_pred"]).all()

    # Verify autumn 25h day (2023-10-29) via direct predict_day + PyFunc parity.
    oct29 = pd.Timestamp("2023-10-29")
    hist = df.loc[df["date_local"] < oct29]
    fut = df.loc[df["date_local"] == oct29].drop(columns=["grid_load_mwh"])
    feats = model.build_day_features(hist, fut)
    assert len(feats) == 25
    in_mem = model.predict_features(feats)
    pyfunc_out = model.to_pyfunc_model().predict(None, feats)
    np.testing.assert_allclose(in_mem, pyfunc_out, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("model_cls", [RidgeARXForecaster, SplineARXForecaster])
def test_ridge_and_spline_arx_daily_backtest_and_pyfunc_parity(model_cls) -> None:
    df = _make_synthetic_daily("2021-01-01", "2023-03-15")
    train_df = df.loc[df["date_local"] <= pd.Timestamp("2022-12-31")].reset_index(drop=True)

    model = model_cls("daily", tune=True)
    res = run_rolling_backtest(
        model,
        train_df,
        df,
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2023-02-15"),
    )
    assert len(res.predictions_df) == 46
    assert np.isfinite(res.predictions_df["y_pred"]).all()
    assert model.cv_best_mae_ is not None

    pyfunc = model.to_pyfunc_model()
    np.testing.assert_allclose(
        pyfunc.predict(None, res.input_example),
        res.sample_output,
        rtol=1e-12,
        atol=1e-12,
    )


def test_sarimax_rejects_hourly_task() -> None:
    with pytest.raises(ValueError, match="daily"):
        SARIMAXForecaster("hourly")


@pytest.mark.parametrize(
    ("order", "seasonal_order"),
    [
        ((1, 0, 1), (1, 0, 1, 7)),
        ((1, 0, 1), (1, 1, 1, 7)),
        ((1, 1, 1), (1, 1, 1, 7)),
    ],
)
def test_sarimax_stateless_filter_matches_statsmodels_apply(
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> None:
    """Verify our compact SARIMAX innovations filter matches `statsmodels.apply().forecast(1)`."""
    df = _make_synthetic_daily("2021-01-01", "2023-03-15", seed=123)
    train_df = df.loc[df["date_local"] <= pd.Timestamp("2022-12-31")].reset_index(drop=True)

    model = SARIMAXForecaster(
        "daily",
        order=order,
        seasonal_order=seasonal_order,
        residual_window=56,
        tune=False,
        maxiter=100,
    )
    model.fit(train_df)
    assert model.state_ is not None
    state = model.state_

    # Confirm D >= 1 drops collinear weekday dummies automatically.
    if seasonal_order[1] >= 1:
        assert "dow_1" not in state.exog_cols
    else:
        assert "dow_1" in state.exog_cols

    # Reconstruct exact statsmodels SARIMAX object on the full history up to 2023-02-20
    # with the exact same fitted parameters to compare 1-step-ahead forecast.
    target_date = pd.Timestamp("2023-02-20")
    history_df = df.loc[df["date_local"] < target_date].reset_index(drop=True)
    future_df = df.loc[df["date_local"] == target_date].drop(columns=["grid_load_mwh"])

    feats = model.build_day_features(history_df, future_df)
    our_pred = float(model.predict_features(feats)[0])
    pyfunc_pred = float(model.to_pyfunc_model().predict(None, feats)[0])
    assert our_pred == pytest.approx(pyfunc_pred, rel=1e-12, abs=1e-12)

    # Build native statsmodels reference using identical fitted parameters from `state`.
    train_exog_df = add_ar_exogenous_features(train_df, task="daily")
    exog_cols = _sarimax_exog_cols(seasonal_order)

    def _make_design(frame: pd.DataFrame) -> pd.DataFrame:
        ex_scaled = (frame[exog_cols].to_numpy(dtype=float) - state.exog_mean) / state.exog_scale
        if state.has_const:
            mat = np.column_stack([np.ones(len(ex_scaled), dtype=float), ex_scaled])
            return pd.DataFrame(mat, columns=["_const", *exog_cols])
        return pd.DataFrame(ex_scaled, columns=list(exog_cols))

    y_train_s = (train_df["grid_load_mwh"].to_numpy(dtype=float) - state.y_mean) / state.y_std
    sm_model = SARIMAX(
        endog=y_train_s,
        exog=_make_design(train_exog_df),
        order=order,
        seasonal_order=seasonal_order,
        trend="n",
        concentrate_scale=True,
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    sm_fit = sm_model.fit(method="lbfgs", maxiter=100, disp=False)
    if not sm_fit.mle_retvals.get("converged", False):
        sm_alt = sm_model.fit(start_params=sm_fit.params, method="bfgs", maxiter=80, disp=False)
        if np.isfinite(sm_alt.aic) and sm_alt.aic <= sm_fit.aic:
            sm_fit = sm_alt

    hist_exog_df = add_ar_exogenous_features(history_df, task="daily")
    fut_exog_df = add_ar_exogenous_features(future_df, task="daily")
    y_hist_s = (history_df["grid_load_mwh"].to_numpy(dtype=float) - state.y_mean) / state.y_std
    exog_hist_design = _make_design(hist_exog_df)
    exog_fut_design = _make_design(fut_exog_df)

    # 1. Exact numerical identity against statsmodels.apply() on the same 56-day window.
    sm_win = sm_fit.apply(endog=y_hist_s[-56:], exog=exog_hist_design.iloc[-56:])
    sm_win_pred = state.y_mean + state.y_std * float(
        sm_win.forecast(steps=1, exog=exog_fut_design).iloc[0]
    )
    assert our_pred == pytest.approx(sm_win_pred, rel=1e-9, abs=1e-5)

    # 2. Close agreement with full 780-day history (D=1 near-unit-root MA decays slowly over weeks).
    sm_full = sm_fit.apply(endog=y_hist_s, exog=exog_hist_design)
    sm_full_pred = state.y_mean + state.y_std * float(
        sm_full.forecast(steps=1, exog=exog_fut_design).iloc[0]
    )
    rel_tol = 0.001 if seasonal_order[1] == 0 else 0.01
    assert abs(our_pred - sm_full_pred) / abs(sm_full_pred) < rel_tol


def test_sarimax_daily_tuned_backtest_zero_leakage() -> None:
    df = _make_synthetic_daily("2021-01-01", "2023-02-10", seed=99)
    train_df = df.loc[df["date_local"] <= pd.Timestamp("2022-12-31")].reset_index(drop=True)

    model = SARIMAXForecaster(
        "daily",
        tune=True,
        candidate_orders=(
            ((1, 0, 1), (1, 0, 1, 7)),
            ((1, 0, 1), (1, 1, 1, 7)),
        ),
        maxiter=60,
    )
    res = run_rolling_backtest(
        model,
        train_df,
        df,
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2023-01-20"),
    )
    assert len(res.predictions_df) == 20
    assert np.isfinite(res.predictions_df["y_pred"]).all()
    assert model.cv_best_mae_ is not None

    # Corrupting target day's load must leave that day's forecast untouched.
    corrupted = df.copy()
    target_day = pd.Timestamp("2023-01-15")
    corrupted.loc[corrupted["date_local"] == target_day, "grid_load_mwh"] = 99_999_999.0
    hist = corrupted.loc[corrupted["date_local"] < target_day]
    fut = corrupted.loc[corrupted["date_local"] == target_day].drop(columns=["grid_load_mwh"])
    pred_corrupted = model.predict_day(hist, fut)
    pred_clean = res.predictions_df.loc[
        res.predictions_df["date_local"] == target_day, "y_pred"
    ].to_numpy(dtype=float)
    np.testing.assert_allclose(pred_corrupted, pred_clean, rtol=1e-12, atol=1e-12)


def test_autoregressive_mlflow_pyfunc_roundtrip_and_databricks_config(
    isolated_mlflow,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, config = isolated_mlflow

    # Verify Databricks default experiment path behavior without breaking local defaults.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "databricks")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    dbx_cfg = MLflowConfig.from_env()
    assert dbx_cfg.experiment_name == "/Shared/electricity-load-germany"

    df = _make_synthetic_daily("2021-01-01", "2023-01-20", seed=11)
    train_df = df.loc[df["date_local"] <= pd.Timestamp("2022-12-31")].reset_index(drop=True)

    models = [
        RidgeARXForecaster("daily", tune=False),
        SplineARXForecaster("daily", tune=False),
        SARIMAXForecaster(
            "daily",
            order=(1, 0, 1),
            seasonal_order=(1, 0, 1, 7),
            tune=False,
            maxiter=60,
        ),
    ]
    model_name = config.registered_model_name("daily")

    for idx, model in enumerate(models, start=1):
        bt = run_rolling_backtest(
            model,
            train_df,
            df,
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-01-10"),
        )
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])

        val_mae = 40_000.0 - idx * 1_000.0
        with start_forecast_run(
            f"{model.name}_daily_test",
            task="daily",
            split="val_2023",
            model_family=model.family,
            model_name=model.name,
            params=model.get_params(),
        ):
            log_forecast_outputs(bt.predictions_df, {"val_mae": val_mae}, {"sample.png": fig})
            mv = log_and_register_pyfunc_model(
                client,
                pyfunc_model=model.to_pyfunc_model(),
                registered_model_name=model_name,
                input_example=bt.input_example,
                sample_output=bt.sample_output,
                pip_requirements=model.pip_requirements(),
                version_tags={"model_name": model.name, "val_mae": f"{val_mae:.2f}"},
            )
        gate = evaluate_quality_gate(client, model_name, candidate_metric=val_mae)
        apply_quality_gate_aliases(client, model_name, str(mv.version), gate, auto_promote=True)

        loaded = mlflow.pyfunc.load_model(f"models:/{model_name}/{mv.version}")
        reloaded_preds = loaded.predict(bt.input_example)
        np.testing.assert_allclose(reloaded_preds, bt.sample_output, rtol=1e-10, atol=1e-10)

    # Verify leaderboard integration with baseline JSON + autoregressive JSON.
    val_dir = tmp_path / "results" / "val_2023"
    save_validation_summary(
        [
            {
                "model_name": "seasonal_naive_week_before",
                "model_family": "baseline",
                "metrics": {"val_mae": 49920.8, "val_rmse": 81771.9, "val_mase": 0.912},
            }
        ],
        task="daily",
        split="val_2023",
        primary_metric="val_mae",
        output_path=val_dir / "baselines_daily.json",
        git_metadata={"git_commit": "aaa1111", "git_dirty": "false"},
    )
    save_validation_summary(
        [
            {
                "model_name": "spline_arx",
                "model_family": "autoregressive",
                "metrics": {"val_mae": 21000.0, "val_rmse": 30000.0, "val_mase": 0.384},
            }
        ],
        task="daily",
        split="val_2023",
        primary_metric="val_mae",
        output_path=val_dir / "autoregressive_daily.json",
        git_metadata={"git_commit": "bbb2222", "git_dirty": "false"},
    )
    lb_path = render_leaderboard(val_dir, split="val_2023")
    lb_text = lb_path.read_text(encoding="utf-8")
    assert "`spline_arx`" in lb_text
    assert lb_text.index("`spline_arx`") < lb_text.index("`seasonal_naive_week_before`")


def test_log_pyfunc_model_without_registration_and_partial_summary_merge(
    isolated_mlflow,
    tmp_path: Path,
) -> None:
    """Verify test-split PyFunc logging leaves registry aliases untouched and partial runs merge."""
    from scripts.run_autoregressive import _merge_summary_records

    client, config = isolated_mlflow
    df = _make_synthetic_daily("2021-01-01", "2023-01-20", seed=21)
    train_df = df.loc[df["date_local"] <= pd.Timestamp("2022-12-31")].reset_index(drop=True)

    model = RidgeARXForecaster("daily", tune=False)
    bt = run_rolling_backtest(
        model,
        train_df,
        df,
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2023-01-08"),
    )

    with start_forecast_run(
        "ridge_arx_daily_test_2024",
        task="daily",
        split="test_2024",
        model_family=model.family,
        model_name=model.name,
        params=model.get_params(),
    ):
        model_info = log_pyfunc_model(
            pyfunc_model=model.to_pyfunc_model(),
            input_example=bt.input_example,
            sample_output=bt.sample_output,
            pip_requirements=model.pip_requirements(),
        )

    loaded = mlflow.pyfunc.load_model(model_info.model_uri)
    np.testing.assert_allclose(loaded.predict(bt.input_example), bt.sample_output, rtol=1e-10)
    assert client.search_registered_models() == []

    summary_path = tmp_path / "autoregressive_daily.json"
    save_validation_summary(
        [{"model_name": "ridge_arx", "metrics": {"val_mae": 25000.0}}],
        task="daily",
        split="val_2023",
        primary_metric="val_mae",
        output_path=summary_path,
        git_metadata={"git_commit": "aaa1111", "git_dirty": "false"},
    )
    merged = _merge_summary_records(
        summary_path,
        [{"model_name": "sarimax", "metrics": {"val_mae": 19000.0}}],
    )
    assert {rec["model_name"] for rec in merged} == {"ridge_arx", "sarimax"}
    updated = _merge_summary_records(
        summary_path,
        [{"model_name": "ridge_arx", "metrics": {"val_mae": 24000.0}}],
    )
    assert updated == [{"model_name": "ridge_arx", "metrics": {"val_mae": 24000.0}}]
