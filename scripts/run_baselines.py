"""Run day-ahead baselines on the 2023 validation set and log to MLflow."""

from __future__ import annotations

import argparse
import sys
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
    TaskType,
    get_split_frames,
    load_feature_dataset,
)
from src.models.base import BaseForecaster, run_rolling_backtest  # noqa: E402
from src.models.baselines import DayBeforeBaseline, SeasonalNaiveBaseline  # noqa: E402
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


def parse_args() -> argparse.Namespace:
    """Parse command-line options for baseline validation runs."""
    parser = argparse.ArgumentParser(
        description="Evaluate day-ahead baseline forecasters on the 2023 validation set."
    )
    parser.add_argument(
        "--task",
        choices=["hourly", "daily", "all"],
        default="all",
        help="Forecasting task resolution to evaluate (default: all).",
    )
    parser.add_argument(
        "--auto-promote",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically promote models passing the quality gate to @champion (default: True).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "val_2023",
        help="Directory for git-tracked JSON validation summaries.",
    )
    return parser.parse_args()


def _format_alias_label(version: str, final_aliases: dict[str, str | None]) -> str:
    if final_aliases.get("champion") == version:
        return "@champion"
    if final_aliases.get("challenger") == version:
        return "@challenger"
    return "-"


def _print_task_summary(
    task: TaskType,
    registered_model_name: str,
    records: list[dict[str, Any]],
    final_aliases: dict[str, str | None],
    summary_path: Path,
) -> None:
    print(f"\n=== {task.upper()} VALIDATION SUMMARY (2023) | Registry: {registered_model_name} ===")
    header = (
        f"{'Model':<28} {'Ver':>3} {'MAE':>9} {'RMSE':>9} "
        f"{'MAPE%':>7} {'MASE':>6} {'WkndMAE':>9} {'Gate':<16} {'Alias':<11}"
    )
    print(header)
    print("-" * len(header))
    for rec in sorted(records, key=lambda r: r["metrics"]["val_mae"]):
        m = rec["metrics"]
        ver = str(rec["registered_version"])
        alias = _format_alias_label(ver, final_aliases)
        wknd = m.get("val_mae_weekend", float("nan"))
        print(
            f"{rec['model_name']:<28} {ver:>3} {m['val_mae']:>9.2f} {m['val_rmse']:>9.2f} "
            f"{m['val_mape']:>7.2f} {m['val_mase']:>6.3f} {wknd:>9.2f} "
            f"{rec['quality_gate']:<16} {alias:<11}"
        )
    print(f"Summary JSON: {summary_path}")


def run_task_baselines(
    task: TaskType,
    *,
    client: Any,
    config: MLflowConfig,
    auto_promote: bool,
    output_dir: Path,
    git_meta: dict[str, str],
) -> None:
    """Run all baseline models for one task and log/register them in MLflow."""
    df = load_feature_dataset(task, repo_root=REPO_ROOT)
    train_df, eval_full_df, start_date, end_date = get_split_frames(
        df,
        split="val_2023",
        allow_test=False,
    )
    mase_scale = compute_seasonal_naive_scale(train_df, task)

    models: list[BaseForecaster] = [
        DayBeforeBaseline(task),
        SeasonalNaiveBaseline(task),
    ]
    registered_model_name = config.registered_model_name(task)
    records: list[dict[str, Any]] = []

    for model in models:
        backtest = run_rolling_backtest(
            model=model,
            train_df=train_df,
            eval_full_df=eval_full_df,
            start_date=start_date,
            end_date=end_date,
        )
        metrics = compute_forecast_metrics(
            backtest.predictions_df,
            mase_scale=mase_scale,
            prefix="val_",
        )
        figures = create_validation_plots(
            backtest.predictions_df,
            task=task,
            model_name=model.name,
        )

        with start_forecast_run(
            f"{model.name}_{task}_val_2023",
            task=task,
            split="val_2023",
            model_family=model.family,
            model_name=model.name,
            params=model.get_params(),
        ) as run:
            log_forecast_outputs(backtest.predictions_df, metrics, figures)

            version_tags = {
                "model_name": model.name,
                "model_family": model.family,
                "task": task,
                "split": "val_2023",
                "val_mae": f"{metrics['val_mae']:.4f}",
                "val_mase": f"{metrics['val_mase']:.4f}",
                "git_commit": git_meta["git_commit"],
            }
            mv = log_and_register_pyfunc_model(
                client,
                pyfunc_model=model.to_pyfunc_model(),
                registered_model_name=registered_model_name,
                input_example=backtest.input_example,
                sample_output=backtest.sample_output,
                pip_requirements=model.pip_requirements(),
                version_tags=version_tags,
            )

            gate = evaluate_quality_gate(
                client,
                registered_model_name,
                candidate_metric=metrics[config.primary_metric],
                metric_name=config.primary_metric,
            )
            mlflow.set_tags({"quality_gate": gate.status})
            if gate.improvement_pct is not None:
                mlflow.log_metric("gate_improvement_pct", gate.improvement_pct)

            aliases = apply_quality_gate_aliases(
                client,
                registered_model_name,
                str(mv.version),
                gate,
                auto_promote=auto_promote,
            )

            records.append(
                {
                    "model_name": model.name,
                    "model_family": model.family,
                    "run_id": run.info.run_id,
                    "registered_model_name": registered_model_name,
                    "registered_version": str(mv.version),
                    "quality_gate": gate.status,
                    "gate_improvement_pct": gate.improvement_pct,
                    "params": model.get_params(),
                    "metrics": metrics,
                    "aliases_after_run": aliases,
                }
            )

    final_aliases = get_champion_challenger_aliases(client, registered_model_name)
    for rec in records:
        ver = str(rec["registered_version"])
        rec["is_champion"] = final_aliases.get("champion") == ver
        rec["is_challenger"] = final_aliases.get("challenger") == ver

    summary_path = save_validation_summary(
        records,
        task=task,
        split="val_2023",
        primary_metric=config.primary_metric,
        output_path=output_dir / f"baselines_{task}.json",
        git_metadata=git_meta,
    )
    _print_task_summary(task, registered_model_name, records, final_aliases, summary_path)


def main() -> None:
    """Entrypoint for baseline validation runs."""
    args = parse_args()
    config = MLflowConfig.from_env(REPO_ROOT)
    client = configure_mlflow(config)
    git_meta = get_git_metadata(REPO_ROOT)

    tasks: list[TaskType] = ["hourly", "daily"] if args.task == "all" else [args.task]
    for task in tasks:
        run_task_baselines(
            task,
            client=client,
            config=config,
            auto_promote=args.auto_promote,
            output_dir=args.output_dir,
            git_meta=git_meta,
        )

    print("\nInspect runs & registry in MLflow UI:")
    print(f"  uv run mlflow ui --backend-store-uri {config.tracking_uri}")


if __name__ == "__main__":
    main()
