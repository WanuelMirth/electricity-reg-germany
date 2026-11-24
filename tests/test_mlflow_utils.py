"""Tests for portable MLflow tracking, registry quality gates, aliases, and summaries."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest

from src.models.baselines import SeasonalNaiveBaseline
from src.tracking.mlflow_utils import (
    MLflowConfig,
    apply_quality_gate_aliases,
    configure_mlflow,
    evaluate_quality_gate,
    get_champion_challenger_aliases,
    log_and_register_pyfunc_model,
    log_forecast_outputs,
    promote_challenger_to_champion,
    render_leaderboard,
    save_validation_summary,
    start_forecast_run,
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
        experiment_name="test-electricity-load",
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


def _register_dummy_candidate(
    client,
    config: MLflowConfig,
    *,
    model_name: str,
    run_label: str,
    val_mae: float,
) -> str:
    baseline = SeasonalNaiveBaseline("daily")
    input_example = pd.DataFrame({"lag_7d_mwh": [1_100_000.0, 1_150_000.0]})
    sample_output = np.array([1_100_000.0, 1_150_000.0], dtype=float)
    preds_df = pd.DataFrame(
        {
            "date_local": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "y_true": [1_110_000.0, 1_160_000.0],
            "y_pred": sample_output,
        }
    )
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    with start_forecast_run(
        run_label,
        task="daily",
        split="val_2023",
        model_family="baseline",
        model_name=run_label,
        params={"lag_days": 7},
    ):
        log_forecast_outputs(preds_df, {"val_mae": val_mae}, {"sample.png": fig})
        mv = log_and_register_pyfunc_model(
            client,
            pyfunc_model=baseline.to_pyfunc_model(),
            registered_model_name=model_name,
            input_example=input_example,
            sample_output=sample_output,
            pip_requirements=baseline.pip_requirements(),
            version_tags={"model_name": run_label, "val_mae": f"{val_mae:.2f}"},
        )
    return str(mv.version)


def test_pyfunc_logging_registration_and_reload(isolated_mlflow) -> None:
    client, config = isolated_mlflow
    model_name = config.registered_model_name("daily")
    ver = _register_dummy_candidate(
        client,
        config,
        model_name=model_name,
        run_label="seasonal_naive_week_before",
        val_mae=50_000.0,
    )
    gate = evaluate_quality_gate(client, model_name, candidate_metric=50_000.0)
    assert gate.status == "initial_champion"
    aliases = apply_quality_gate_aliases(client, model_name, ver, gate, auto_promote=True)
    assert aliases == {"champion": ver, "challenger": None}

    loaded = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
    test_input = pd.DataFrame({"lag_7d_mwh": [1_234_567.0, 1_300_000.0]})
    preds = loaded.predict(test_input)
    np.testing.assert_allclose(preds, [1_234_567.0, 1_300_000.0])

    mv = client.get_model_version_by_alias(model_name, "champion")
    assert mv.tags["quality_gate"] == "initial_champion"
    run = client.get_run(mv.run_id)
    assert run.data.metrics["val_mae"] == pytest.approx(50_000.0)
    assert run.data.tags["task"] == "daily"


def test_quality_gate_auto_promote_and_runner_up(isolated_mlflow) -> None:
    client, config = isolated_mlflow
    model_name = config.registered_model_name("hourly")

    v1 = _register_dummy_candidate(
        client, config, model_name=model_name, run_label="m1", val_mae=1000.0
    )
    g1 = evaluate_quality_gate(client, model_name, candidate_metric=1000.0)
    assert g1.status == "initial_champion"
    assert apply_quality_gate_aliases(client, model_name, v1, g1, auto_promote=True) == {
        "champion": v1,
        "challenger": None,
    }

    # v2 beats v1 -> auto-promoted to @champion, v1 moves to @challenger.
    v2 = _register_dummy_candidate(
        client, config, model_name=model_name, run_label="m2", val_mae=750.0
    )
    g2 = evaluate_quality_gate(client, model_name, candidate_metric=750.0)
    assert g2.status == "passed"
    assert g2.passed is True
    assert g2.improvement_pct == pytest.approx(25.0)
    assert apply_quality_gate_aliases(client, model_name, v2, g2, auto_promote=True) == {
        "champion": v2,
        "challenger": v1,
    }

    # v3 is worse than both @champion (v2) and @challenger (v1) -> aliases stay unchanged.
    v3 = _register_dummy_candidate(
        client, config, model_name=model_name, run_label="m3", val_mae=1200.0
    )
    g3 = evaluate_quality_gate(client, model_name, candidate_metric=1200.0)
    assert g3.status == "failed"
    assert g3.passed is False
    assert apply_quality_gate_aliases(client, model_name, v3, g3, auto_promote=True) == {
        "champion": v2,
        "challenger": v1,
    }

    # v4 fails against @champion (v2=750) but beats @challenger (v1=1000) -> becomes @challenger.
    v4 = _register_dummy_candidate(
        client, config, model_name=model_name, run_label="m4", val_mae=850.0
    )
    g4 = evaluate_quality_gate(client, model_name, candidate_metric=850.0)
    assert g4.status == "failed"
    assert apply_quality_gate_aliases(client, model_name, v4, g4, auto_promote=True) == {
        "champion": v2,
        "challenger": v4,
    }


def test_quality_gate_manual_mode_and_explicit_promotion(isolated_mlflow) -> None:
    client, config = isolated_mlflow
    model_name = config.registered_model_name("daily")

    v1 = _register_dummy_candidate(
        client, config, model_name=model_name, run_label="baseline_day", val_mae=1000.0
    )
    g1 = evaluate_quality_gate(client, model_name, candidate_metric=1000.0)
    apply_quality_gate_aliases(client, model_name, v1, g1, auto_promote=False)

    # With auto_promote=False, passing the gate assigns @challenger without promoting yet.
    v2 = _register_dummy_candidate(
        client, config, model_name=model_name, run_label="baseline_week", val_mae=700.0
    )
    g2 = evaluate_quality_gate(client, model_name, candidate_metric=700.0)
    assert g2.passed is True
    assert apply_quality_gate_aliases(client, model_name, v2, g2, auto_promote=False) == {
        "champion": v1,
        "challenger": v2,
    }

    # Failing the gate does not overwrite @challenger in strict manual mode.
    v3 = _register_dummy_candidate(
        client, config, model_name=model_name, run_label="worse_model", val_mae=1100.0
    )
    g3 = evaluate_quality_gate(client, model_name, candidate_metric=1100.0)
    assert apply_quality_gate_aliases(client, model_name, v3, g3, auto_promote=False) == {
        "champion": v1,
        "challenger": v2,
    }

    # Explicit promotion moves v2 -> @champion and retains v1 as @challenger.
    promoted = promote_challenger_to_champion(
        client, model_name, keep_previous_as_challenger=True
    )
    assert promoted == {"champion": v2, "challenger": v1}
    assert get_champion_challenger_aliases(client, model_name) == {"champion": v2, "challenger": v1}
    assert "promoted_at_utc" in client.get_model_version(model_name, v2).tags


def test_save_validation_summary_sorted_json(tmp_path: Path) -> None:
    out_path = tmp_path / "results" / "val_2023" / "baselines_hourly.json"
    records = [
        {"model_name": "worse", "metrics": {"val_mae": 2500.0}},
        {"model_name": "better", "metrics": {"val_mae": 900.0}},
    ]
    written = save_validation_summary(
        records,
        task="hourly",
        split="val_2023",
        primary_metric="val_mae",
        output_path=out_path,
        git_metadata={"git_commit": "abc1234", "git_dirty": "false"},
    )
    assert written == out_path
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["task"] == "hourly"
    assert data["split"] == "val_2023"
    assert data["git_commit"] == "abc1234"
    assert [m["model_name"] for m in data["models"]] == ["better", "worse"]


def test_render_leaderboard_combines_families_and_supports_test_split(tmp_path: Path) -> None:
    val_dir = tmp_path / "results" / "val_2023"
    save_validation_summary(
        [
            {
                "model_name": "seasonal_naive_week_before",
                "model_family": "baseline",
                "metrics": {
                    "val_mae": 2315.26,
                    "val_rmse": 3741.52,
                    "val_mape": 4.59,
                    "val_smape": 4.53,
                    "val_mase": 0.928,
                    "val_mae_weekday": 2155.1,
                    "val_mae_weekend": 1783.3,
                    "val_mae_holiday": 10969.0,
                },
            }
        ],
        task="hourly",
        split="val_2023",
        primary_metric="val_mae",
        output_path=val_dir / "baselines_hourly.json",
        git_metadata={"git_commit": "111aaaa", "git_dirty": "false"},
    )
    save_validation_summary(
        [
            {
                "model_name": "lightgbm",
                "model_family": "tree",
                "metrics": {
                    "val_mae": 1100.5,
                    "val_rmse": 1650.0,
                    "val_mape": 2.15,
                    "val_smape": 2.12,
                    "val_mase": 0.441,
                    "val_mae_weekday": 1050.0,
                    "val_mae_weekend": 980.0,
                    "val_mae_holiday": 2200.0,
                },
            }
        ],
        task="hourly",
        split="val_2023",
        primary_metric="val_mae",
        output_path=val_dir / "trees_hourly.json",
        git_metadata={"git_commit": "222bbbb", "git_dirty": "false"},
    )
    save_validation_summary(
        [
            {
                "model_name": "seasonal_naive_week_before",
                "model_family": "baseline",
                "metrics": {
                    "val_mae": 49920.8,
                    "val_rmse": 81771.9,
                    "val_mape": 4.12,
                    "val_smape": 4.07,
                    "val_mase": 0.912,
                },
            }
        ],
        task="daily",
        split="val_2023",
        primary_metric="val_mae",
        output_path=val_dir / "baselines_daily.json",
        git_metadata={"git_commit": "111aaaa", "git_dirty": "false"},
    )

    val_leaderboard = render_leaderboard(val_dir, split="val_2023")
    assert val_leaderboard == tmp_path / "results" / "validation_leaderboard.md"
    content = val_leaderboard.read_text(encoding="utf-8")
    assert "# Validation Leaderboard (2023)" in content
    assert "## Hourly (H = 24h)" in content
    assert "## Daily (H = 1d)" in content
    assert content.index("`lightgbm`") < content.index("`seasonal_naive_week_before`")
    assert "`trees_hourly.json`" in content
    assert "`baselines_daily.json`" in content

    test_dir = tmp_path / "results" / "test_2024"
    save_validation_summary(
        [
            {
                "model_name": "timestral_209m",
                "model_family": "icl",
                "metrics": {
                    "test_mae": 950.0,
                    "test_rmse": 1400.0,
                    "test_mape": 1.90,
                    "test_smape": 1.88,
                    "test_mase": 0.380,
                },
            }
        ],
        task="hourly",
        split="test_2024",
        primary_metric="test_mae",
        output_path=test_dir / "icl_hourly.json",
        git_metadata={"git_commit": "333cccc", "git_dirty": "false"},
    )
    test_leaderboard = render_leaderboard(test_dir, split="test_2024")
    assert test_leaderboard == tmp_path / "results" / "test_leaderboard.md"
    test_content = test_leaderboard.read_text(encoding="utf-8")
    assert "# Test Leaderboard (2024)" in test_content
    assert "950.0" in test_content
