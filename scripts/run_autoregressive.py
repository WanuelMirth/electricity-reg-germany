"""Run day-ahead autoregressive forecasters (`Ridge-ARX`, `Spline-ARX`, daily `SARIMAX`)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import mlflow

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    compute_forecast_metrics,
    compute_seasonal_naive_scale,
)
from src.evaluation.plots import create_validation_plots  # noqa: E402
from src.evaluation.splits import (  # noqa: E402
    SplitType,
    TaskType,
    get_split_frames,
    load_feature_dataset,
)
from src.models.autoregressive import (  # noqa: E402
    RidgeARXForecaster,
    SARIMAXForecaster,
    SplineARXForecaster,
)
from src.models.base import BaseForecaster, run_rolling_backtest  # noqa: E402
from src.tracking.mlflow_utils import (  # noqa: E402
    MLflowConfig,
    apply_quality_gate_aliases,
    configure_mlflow,
    evaluate_quality_gate,
    get_champion_challenger_aliases,
    get_git_metadata,
    log_and_register_pyfunc_model,
    log_forecast_outputs,
    save_validation_summary,
    start_forecast_run,
)

ALL_MODEL_KEYS = ("ridge_arx", "spline_arx", "sarimax")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for autoregressive validation/test runs."""
    parser = argparse.ArgumentParser(
        description="Evaluate day-ahead autoregressive forecasters and log to MLflow."
    )
    parser.add_argument(
        "--task",
        choices=["hourly", "daily", "all"],
        default="all",
        help="Forecasting task resolution to evaluate (default: all).",
    )
    parser.add_argument(
        "--models",
        choices=["ridge_arx", "spline_arx", "sarimax", "all"],
        default="all",
        help="Model subset to run (default: all; sarimax runs on daily only).",
    )
    parser.add_argument(
        "--split",
        choices=["val_2023", "test_2024"],
        default="val_2023",
        help="Evaluation split (default: val_2023; test_2024 requires --allow-test).",
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        default=False,
        help="Unlock final 2024 test evaluation (use only for frozen final evaluation).",
    )
    parser.add_argument(
        "--tune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run strict train-only inner-year CV tuning (default: True).",
    )
    parser.add_argument(
        "--auto-promote",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically promote models passing the val_2023 quality gate (default: True).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for git-tracked JSON summaries (default: results/<split>).",
    )
    return parser.parse_args()


def _build_models_for_task(
    task: TaskType,
    selected: str,
    *,
    tune: bool,
) -> list[BaseForecaster]:
    if selected == "sarimax" and task == "hourly":
        raise ValueError(
            "SARIMAXForecaster is daily-only. "
            "Run with --task daily or --models all/ridge_arx/spline_arx."
        )

    models: list[BaseForecaster] = []
    if selected in {"all", "ridge_arx"}:
        models.append(RidgeARXForecaster(task, tune=tune))
    if selected in {"all", "spline_arx"}:
        models.append(SplineARXForecaster(task, tune=tune))
    if selected in {"all", "sarimax"} and task == "daily":
        models.append(SARIMAXForecaster("daily", tune=tune))
    return models


def _format_alias_label(version: str | None, final_aliases: dict[str, str | None]) -> str:
    if version is None:
        return "-"
    if final_aliases.get("champion") == version:
        return "@champion"
    if final_aliases.get("challenger") == version:
        return "@challenger"
    return "-"


def _print_task_summary(
    task: TaskType,
    split: SplitType,
    registered_model_name: str,
    records: list[dict[str, Any]],
    final_aliases: dict[str, str | None],
    summary_path: Path,
    metric_prefix: str,
) -> None:
    primary_key = f"{metric_prefix}mae"
    print(
        f"\n=== {task.upper()} AUTOREGRESSIVE SUMMARY ({split}) "
        f"| Registry: {registered_model_name} ==="
    )
    header = (
        f"{'Model':<22} {'Ver':>3} {'MAE':>9} {'RMSE':>9} "
        f"{'MAPE%':>7} {'MASE':>6} {'HolMAE':>9} {'Time(s)':>7} {'Gate':<16} {'Alias':<11}"
    )
    print(header)
    print("-" * len(header))
    for rec in sorted(records, key=lambda r: r["metrics"][primary_key]):
        m = rec["metrics"]
        ver = rec.get("registered_version")
        ver_str = str(ver) if ver is not None else "-"
        alias = _format_alias_label(ver, final_aliases)
        hol = m.get(f"{metric_prefix}mae_holiday", float("nan"))
        elapsed = rec.get("runtime_sec", float("nan"))
        gate = rec.get("quality_gate", "n/a")
        print(
            f"{rec['model_name']:<22} {ver_str:>3} {m[f'{metric_prefix}mae']:>9.2f} "
            f"{m[f'{metric_prefix}rmse']:>9.2f} {m[f'{metric_prefix}mape']:>7.2f} "
            f"{m[f'{metric_prefix}mase']:>6.3f} {hol:>9.2f} {elapsed:>7.1f} "
            f"{gate:<16} {alias:<11}"
        )
    print(f"Summary JSON: {summary_path}")


def run_task_autoregressive(
    task: TaskType,
    *,
    client: Any,
    config: MLflowConfig,
    models_choice: str,
    split: SplitType,
    allow_test: bool,
    tune: bool,
    auto_promote: bool,
    output_dir: Path,
    git_meta: dict[str, str],
) -> Path:
    """Run autoregressive models for one task and log/register them in MLflow."""
    df = load_feature_dataset(task, repo_root=REPO_ROOT)
    train_df, eval_full_df, start_date, end_date = get_split_frames(
        df,
        split=split,
        allow_test=allow_test,
    )
    mase_scale = compute_seasonal_naive_scale(train_df, task)

    models = _build_models_for_task(task, models_choice, tune=tune)
    registered_model_name = config.registered_model_name(task)
    metric_prefix = "val_" if split == "val_2023" else "test_"
    primary_metric = f"{metric_prefix}mae"
    predictions_filename = (
        "val_predictions.csv" if split == "val_2023" else "test_predictions.csv"
    )
    records: list[dict[str, Any]] = []

    for model in models:
        t0 = time.perf_counter()
        backtest = run_rolling_backtest(
            model=model,
            train_df=train_df,
            eval_full_df=eval_full_df,
            start_date=start_date,
            end_date=end_date,
        )
        runtime_sec = float(time.perf_counter() - t0)

        metrics = compute_forecast_metrics(
            backtest.predictions_df,
            mase_scale=mase_scale,
            prefix=metric_prefix,
        )
        figures = create_validation_plots(
            backtest.predictions_df,
            task=task,
            model_name=model.name,
        )

        with start_forecast_run(
            f"{model.name}_{task}_{split}",
            task=task,
            split=split,
            model_family=model.family,
            model_name=model.name,
            params=model.get_params(),
        ) as run:
            log_forecast_outputs(
                backtest.predictions_df,
                metrics,
                figures,
                predictions_filename=predictions_filename,
            )
            mlflow.log_metric("runtime_sec", runtime_sec)

            version_tags = {
                "model_name": model.name,
                "model_family": model.family,
                "task": task,
                "split": split,
                primary_metric: f"{metrics[primary_metric]:.4f}",
                f"{metric_prefix}mase": f"{metrics[f'{metric_prefix}mase']:.4f}",
                "git_commit": git_meta["git_commit"],
            }

            registered_version: str | None = None
            gate_status = "skipped_test_split"
            gate_improvement_pct: float | None = None
            aliases: dict[str, str | None] = {"champion": None, "challenger": None}

            if split == "val_2023":
                mv = log_and_register_pyfunc_model(
                    client,
                    pyfunc_model=model.to_pyfunc_model(),
                    registered_model_name=registered_model_name,
                    input_example=backtest.input_example,
                    sample_output=backtest.sample_output,
                    pip_requirements=model.pip_requirements(),
                    version_tags=version_tags,
                )
                registered_version = str(mv.version)

                gate = evaluate_quality_gate(
                    client,
                    registered_model_name,
                    candidate_metric=metrics[config.primary_metric],
                    metric_name=config.primary_metric,
                )
                gate_status = gate.status
                gate_improvement_pct = gate.improvement_pct
                mlflow.set_tags({"quality_gate": gate.status})
                if gate.improvement_pct is not None:
                    mlflow.log_metric("gate_improvement_pct", gate.improvement_pct)

                aliases = apply_quality_gate_aliases(
                    client,
                    registered_model_name,
                    registered_version,
                    gate,
                    auto_promote=auto_promote,
                )
            else:
                mlflow.set_tags({"quality_gate": gate_status})

            records.append(
                {
                    "model_name": model.name,
                    "model_family": model.family,
                    "run_id": run.info.run_id,
                    "registered_model_name": registered_model_name,
                    "registered_version": registered_version,
                    "quality_gate": gate_status,
                    "gate_improvement_pct": gate_improvement_pct,
                    "runtime_sec": round(runtime_sec, 3),
                    "params": model.get_params(),
                    "metrics": metrics,
                    "aliases_after_run": aliases,
                }
            )

    final_aliases = (
        get_champion_challenger_aliases(client, registered_model_name)
        if split == "val_2023"
        else {"champion": None, "challenger": None}
    )
    for rec in records:
        ver = rec["registered_version"]
        rec["is_champion"] = ver is not None and final_aliases.get("champion") == str(ver)
        rec["is_challenger"] = ver is not None and final_aliases.get("challenger") == str(ver)

    summary_path = save_validation_summary(
        records,
        task=task,
        split=split,
        primary_metric=primary_metric,
        output_path=output_dir / f"autoregressive_{task}.json",
        git_metadata=git_meta,
    )
    _print_task_summary(
        task,
        split,
        registered_model_name,
        records,
        final_aliases,
        summary_path,
        metric_prefix,
    )
    return summary_path


def run_autoregressive(
    *,
    task: str = "all",
    models: str = "all",
    split: SplitType = "val_2023",
    allow_test: bool = False,
    tune: bool = True,
    auto_promote: bool = True,
    output_dir: Path | None = None,
) -> list[Path]:
    """Programmatic entrypoint for local CLI or Azure Databricks notebook execution."""
    config = MLflowConfig.from_env(REPO_ROOT)
    client = configure_mlflow(config)
    git_meta = get_git_metadata(REPO_ROOT)

    dest_dir = output_dir or (REPO_ROOT / "results" / split)
    tasks: list[TaskType] = ["hourly", "daily"] if task == "all" else [task]  # type: ignore[list-item]

    written_paths: list[Path] = []
    for current_task in tasks:
        if current_task == "hourly" and models == "sarimax":
            raise ValueError(
                "SARIMAXForecaster is daily-only. "
                "Use --task daily or --models all/ridge_arx/spline_arx."
            )
        written_paths.append(
            run_task_autoregressive(
                current_task,
                client=client,
                config=config,
                models_choice=models,
                split=split,
                allow_test=allow_test,
                tune=tune,
                auto_promote=auto_promote,
                output_dir=dest_dir,
                git_meta=git_meta,
            )
        )

    print("\nInspect runs & registry in MLflow UI:")
    print(f"  uv run mlflow ui --backend-store-uri {config.tracking_uri}")
    return written_paths


def main() -> None:
    """CLI entrypoint for autoregressive validation/test runs."""
    args = parse_args()
    run_autoregressive(
        task=args.task,
        models=args.models,
        split=args.split,
        allow_test=args.allow_test,
        tune=args.tune,
        auto_promote=args.auto_promote,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
