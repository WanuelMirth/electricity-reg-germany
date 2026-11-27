"""Offline unit and integration tests for ICL models (`TimeH2O-v1` and `Timestral-209M`)."""

from __future__ import annotations

import builtins
import gc
import json
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException

from src.evaluation.splits import LEAKY_TARGET_COLS, TARGET_COL
from src.features import DAILY_ICL_COVARIATE_COLS, HOURLY_ICL_COVARIATE_COLS
from src.models.base import run_rolling_backtest
from src.models.icl import (
    ICL_MODEL_SPECS,
    ICL_MODELS,
    ICLForecaster,
    build_icl_context_frame,
    load_icl_backend,
    resolve_context_steps,
    split_icl_context_frame,
)
from src.tracking.mlflow_utils import MLflowConfig, configure_mlflow


def _make_icl_frame(task: str, start: str, end_exclusive: str) -> pd.DataFrame:
    if task == "hourly":
        local = pd.date_range(start, end_exclusive, freq="h", tz="Europe/Berlin", inclusive="left")
        dates = pd.to_datetime(local.tz_localize(None)).normalize()
        df = pd.DataFrame(
            {
                "timestamp_local": local,
                "timestamp_utc": local.tz_convert("UTC"),
                "date_local": dates,
                "hour": local.hour,
            }
        )
    else:
        dates = pd.date_range(start, end_exclusive, freq="D", inclusive="left")
        df = pd.DataFrame({"date_local": dates})

    df["time_years"] = (dates - pd.Timestamp("2019-01-01")).days / 365.25
    df["day_of_week"] = dates.day_of_week
    df["month"] = dates.month
    df["day_of_year"] = dates.day_of_year
    df["is_weekend"] = (dates.day_of_week >= 5).astype(int)
    df["holiday_weight"] = np.where((dates.month == 5) & (dates.day == 1), 1.0, 0.0)
    df["is_bridge_day"] = 0
    df["is_christmas_shutdown"] = ((dates.month == 12) & (dates.day >= 24)).astype(int)

    n = len(df)
    idx = np.arange(n, dtype=float)
    if task == "hourly":
        df["temp_c"] = 10.0 + 5.0 * np.sin(idx / 24.0)
        df["solar_radiation_wm2"] = np.maximum(0.0, 200.0 * np.sin(idx / 12.0))
        df["wind_speed_kmh"] = 12.0 + np.cos(idx / 18.0)
        base_load = 50_000.0 + df["hour"] * 250.0 - df["is_weekend"] * 5_000.0
    else:
        df["temp_c_mean"] = 10.0 + 5.0 * np.sin(idx / 30.0)
        df["solar_radiation_wm2_mean"] = 120.0 + 40.0 * np.cos(idx / 30.0)
        df["wind_speed_kmh_mean"] = 12.0 + np.cos(idx / 10.0)
        base_load = 1_200_000.0 - df["is_weekend"] * 120_000.0

    df[TARGET_COL] = base_load + np.sin(idx / 7.0) * 500.0
    df["grid_load_incl_pumped_mwh"] = df[TARGET_COL] + 400.0
    df["pumped_storage_mwh"] = 400.0
    df["residual_load_mwh"] = df[TARGET_COL] - 5_000.0
    df["lag_1y"] = df[TARGET_COL] - 250.0
    return df


class FakeICLBackend:
    """Deterministic offline ICL backend recording input shapes."""

    def __init__(self, *, bias_by_long_context: bool = False) -> None:
        self.calls: list[tuple[int, tuple[int, int], int]] = []
        self.bias_by_long_context = bias_by_long_context

    def predict(
        self,
        target_history: np.ndarray,
        future_covariates: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        self.calls.append((len(target_history), future_covariates.shape, int(horizon)))
        base = float(np.mean(target_history[-min(len(target_history), 24) :]))
        fut_cov = future_covariates[:, -horizon:]
        cov_effect = np.sum(fut_cov, axis=0) * 0.5
        if self.bias_by_long_context and len(target_history) >= 200:
            # Used to make `1y` strictly outperform `3m` in context-selection tests.
            return base + cov_effect
        offset = 500_000.0 if self.bias_by_long_context else 0.0
        return base + cov_effect + offset


@pytest.fixture()
def isolated_icl_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = (tmp_path / "mlflow.db").resolve()
    artifacts_dir = (tmp_path / "artifacts").resolve()
    tracking_uri = f"sqlite:///{db_path.as_posix()}"

    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", artifacts_dir.as_uri())
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test-icl-experiment")
    monkeypatch.delenv("MLFLOW_UC_PREFIX", raising=False)

    config = MLflowConfig.from_env(tmp_path)
    client = configure_mlflow(config)
    yield client, config
    while mlflow.active_run() is not None:
        mlflow.end_run()
    gc.collect()


@pytest.mark.parametrize("task", ["hourly", "daily"])
@pytest.mark.parametrize("model_name", ICL_MODELS)
def test_icl_models_fit_predict_and_covariate_contract(task: str, model_name: str) -> None:
    df = _make_icl_frame(task, "2022-11-01", "2023-01-10")
    history = df.loc[df["date_local"] < "2023-01-05"].reset_index(drop=True)
    target = df.loc[df["date_local"] == "2023-01-05"].reset_index(drop=True)

    backend = FakeICLBackend()
    model = ICLForecaster(
        task,  # type: ignore[arg-type]
        model_name,
        context_preset="3m",
        device="cpu",
        backend=backend,
    )
    assert model.family == "icl"
    expected_covs = (
        HOURLY_ICL_COVARIATE_COLS if task == "hourly" else DAILY_ICL_COVARIATE_COLS
    )
    assert model.covariate_cols == expected_covs
    assert not set(model.covariate_cols).intersection(LEAKY_TARGET_COLS)

    model.fit(history)
    features_df = model.build_day_features(history, target.drop(columns=[TARGET_COL]))
    preds = model.predict_features(features_df)

    assert len(preds) == len(target)
    assert np.isfinite(preds).all()
    assert len(backend.calls) == 1
    hist_len, cov_shape, horizon = backend.calls[0]
    assert horizon == len(target)
    assert cov_shape == (len(expected_covs), hist_len + horizon)
    json.dumps(model.get_params(), allow_nan=False)


@pytest.mark.parametrize("task", ["hourly", "daily"])
def test_icl_zero_leakage_and_future_target_invariance(task: str) -> None:
    df = _make_icl_frame(task, "2022-12-01", "2023-01-10")
    origin_date = pd.Timestamp("2023-01-05")
    history = df.loc[df["date_local"] < origin_date].reset_index(drop=True)
    target = df.loc[df["date_local"] == origin_date].reset_index(drop=True)

    model = ICLForecaster(task, "timeh2o_v1", context_preset="3m", backend=FakeICLBackend())
    model.fit(history)

    clean_features = model.build_day_features(history, target.drop(columns=list(LEAKY_TARGET_COLS)))
    corrupted = target.copy()
    corrupted[TARGET_COL] = 1e12
    corrupted["residual_load_mwh"] = -1e12
    corrupted_features = model.build_day_features(history, corrupted)

    pd.testing.assert_frame_equal(clean_features, corrupted_features)
    np.testing.assert_allclose(
        model.predict_features(clean_features),
        model.predict_features(corrupted_features),
    )

    leaky_history = df.loc[df["date_local"] <= origin_date].reset_index(drop=True)
    with pytest.raises(ValueError, match="Leakage detected"):
        build_icl_context_frame(
            leaky_history,
            target.drop(columns=[TARGET_COL]),
            task=task,  # type: ignore[arg-type]
            max_history_steps=91,
        )


def test_context_preset_truncation_and_model_caps() -> None:
    h2o_max = ICL_MODEL_SPECS["timeh2o_v1"].max_context
    assert resolve_context_steps("hourly", "3m", h2o_max) == 2_184
    assert resolve_context_steps("hourly", "1y", h2o_max) == 8_760
    assert resolve_context_steps("hourly", "max", h2o_max) == 16_384
    assert (
        resolve_context_steps("hourly", "max", ICL_MODEL_SPECS["timestral_209m"].max_context)
        == 32_768
    )
    assert resolve_context_steps("daily", "3m", h2o_max) == 91
    assert resolve_context_steps("daily", "1y", h2o_max) == 365
    assert resolve_context_steps("daily", "max", h2o_max) == 16_384

    daily = _make_icl_frame("daily", "2020-01-01", "2023-01-05")
    hist = daily.loc[daily["date_local"] < "2023-01-01"]
    fut = daily.loc[daily["date_local"] == "2023-01-01"].drop(columns=[TARGET_COL])

    frame_3m = build_icl_context_frame(hist, fut, task="daily", max_history_steps=91)
    assert len(frame_3m) == 91 + 1
    assert frame_3m[TARGET_COL].iloc[:-1].notna().all()
    assert pd.isna(frame_3m[TARGET_COL].iloc[-1])

    frame_max = build_icl_context_frame(hist, fut, task="daily", max_history_steps=16_384)
    assert len(frame_max) == len(hist) + 1


@pytest.mark.parametrize(
    "start,end_excl,origin,expected_hours",
    [
        ("2023-03-10", "2023-03-28", "2023-03-26", 23),
        ("2023-10-10", "2023-10-31", "2023-10-29", 25),
    ],
)
def test_hourly_backtest_handles_dst_23h_and_25h_days(
    start: str,
    end_excl: str,
    origin: str,
    expected_hours: int,
) -> None:
    train = _make_icl_frame("hourly", "2022-11-01", "2023-01-01")
    eval_df = _make_icl_frame("hourly", start, end_excl)

    model = ICLForecaster("hourly", "timestral_209m", context_preset="3m", backend=FakeICLBackend())
    result = run_rolling_backtest(
        model,
        train,
        eval_df,
        pd.Timestamp(origin),
        pd.Timestamp(origin),
    )

    assert len(result.predictions_df) == expected_hours
    assert np.isfinite(result.predictions_df["y_pred"]).all()
    assert len(result.input_example) == 48 + expected_hours
    assert len(result.sample_output) == expected_hours


def test_icl_pyfunc_save_load_roundtrip_and_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_backend = FakeICLBackend()
    monkeypatch.setattr("src.models.icl.load_icl_backend", lambda *args, **kwargs: fake_backend)

    df = _make_icl_frame("daily", "2022-01-01", "2023-01-05")
    history = df.loc[df["date_local"] < "2023-01-02"]
    target = df.loc[df["date_local"] == "2023-01-02"].drop(columns=[TARGET_COL])

    model = ICLForecaster("daily", "timeh2o_v1", context_preset="3m", device="cpu")
    model.fit(history)
    features_df = model.build_day_features(history, target)
    expected = model.predict_features(features_df)
    input_example, sample_output = model.make_input_example(features_df, expected)

    model_dir = tmp_path / "icl_pyfunc_model"
    mlflow.pyfunc.save_model(
        path=str(model_dir),
        python_model=model.to_pyfunc_model(),
        input_example=input_example,
        pip_requirements=model.pip_requirements(),
    )

    loaded = mlflow.pyfunc.load_model(str(model_dir))
    reloaded = loaded.predict(features_df)
    np.testing.assert_allclose(reloaded, expected, rtol=1e-12, atol=1e-12)
    assert len(sample_output) == 1

    with pytest.raises(MlflowException, match="missing inputs"):
        loaded.predict(features_df.drop(columns=["temp_c_mean"]))

    with pytest.raises(ValueError, match="observed history rows before NaN"):
        bad = features_df.copy()
        bad.loc[5, TARGET_COL] = np.nan
        split_icl_context_frame(bad, model.covariate_cols)

    with pytest.raises(ValueError, match="trailing horizon rows"):
        no_horizon = features_df.iloc[:-1].copy()
        split_icl_context_frame(no_horizon, model.covariate_cols)


def test_run_icl_selects_best_context_logs_mlflow_and_renders_leaderboard(
    isolated_icl_mlflow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.run_icl as run_icl_mod

    client, config = isolated_icl_mlflow
    df = _make_icl_frame("daily", "2021-06-01", "2023-01-11")

    monkeypatch.setattr(run_icl_mod, "load_feature_dataset", lambda task, repo_root=None: df.copy())
    # Mock load_icl_backend too because mlflow.pyfunc.log_model evaluates input_example.
    monkeypatch.setattr(
        "src.models.icl.load_icl_backend",
        lambda *args, **kwargs: FakeICLBackend(bias_by_long_context=True),
    )

    out_dir = tmp_path / "results" / "val_2023"
    written = run_icl_mod.run_icl(
        task="daily",
        models=["timeh2o_v1"],
        context_lengths=["3m", "1y"],
        split="val_2023",
        device="cpu",
        auto_promote=True,
        output_dir=out_dir,
        progress_every=0,
        backend_factory=lambda *args, **kwargs: FakeICLBackend(bias_by_long_context=True),
    )

    assert len(written) == 1
    summary_path = out_dir / "icl_daily.json"
    assert summary_path.is_file()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(payload["models"]) == 1

    rec = payload["models"][0]
    assert rec["model_name"] == "timeh2o_v1"
    assert rec["model_family"] == "icl"
    assert rec["params"]["selected_context_preset"] == "1y"
    assert rec["params"]["tested_context_presets"] == "3m,1y"
    sweep = rec["params"]["context_sweep"]
    assert [entry["context_preset"] for entry in sweep] == ["3m", "1y"]
    assert sweep[1]["val_mae"] < sweep[0]["val_mae"]
    assert rec["is_champion"] is True

    # Verify nested MLflow child runs + tuning/context_sweep.json artifact exist.
    exp = client.get_experiment_by_name(config.experiment_name)
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    run_names = {r.info.run_name for r in runs}
    assert "timeh2o_v1_daily_val_2023" in run_names
    assert "timeh2o_v1_3m_daily_val_2023" in run_names
    assert "timeh2o_v1_1y_daily_val_2023" in run_names

    parent_run_id = rec["run_id"]
    artifacts = {a.path for a in client.list_artifacts(parent_run_id, path="tuning")}
    assert "tuning/context_sweep.json" in artifacts

    leaderboard_path = out_dir.parent / "validation_leaderboard.md"
    assert leaderboard_path.is_file()
    assert "`timeh2o_v1`" in leaderboard_path.read_text(encoding="utf-8")

    # Dry-run with --max-origins must not log to MLflow or overwrite official JSON summary.
    run_count_before = len(client.search_runs(experiment_ids=[exp.experiment_id]))
    dry_dir = tmp_path / "dry_results" / "val_2023"
    dry_written = run_icl_mod.run_icl(
        task="daily",
        models=["timestral_209m"],
        context_lengths=["3m"],
        split="val_2023",
        device="cpu",
        output_dir=dry_dir,
        max_origins=3,
        progress_every=0,
        backend_factory=lambda *args, **kwargs: FakeICLBackend(),
    )
    assert dry_written == []
    assert not (dry_dir / "icl_daily.json").exists()
    assert len(client.search_runs(experiment_ids=[exp.experiment_id])) == run_count_before


def test_missing_optional_dependencies_raise_clear_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {"h2ofm", "timestral"}:
            raise ImportError(f"Simulated missing package: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ImportError, match="uv sync --extra icl"):
        load_icl_backend("timeh2o_v1", device="cpu")
    with pytest.raises(ImportError, match="uv sync --extra icl"):
        load_icl_backend("timestral_209m", device="cpu")
