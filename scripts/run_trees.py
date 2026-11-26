"""Tune on training years, evaluate frozen tree models on 2023, and log to MLflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Local training does not need MLflow's external usage reporting.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

import mlflow  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    compute_forecast_metrics,
    compute_seasonal_naive_scale,
)
from src.evaluation.plots import create_validation_plots  # noqa: E402
from src.evaluation.splits import get_split_frames, load_feature_dataset  # noqa: E402
from src.models.base import run_rolling_backtest  # noqa: E402
from src.models.trees import TREE_MODELS, TreeForecaster  # noqa: E402
from src.tracking.mlflow_utils import (  # noqa: E402
    MLflowConfig,
    apply_quality_gate_aliases,
    configure_mlflow,
    evaluate_quality_gate,
    get_champion_challenger_aliases,
    get_git_metadata,
    log_and_register_pyfunc_model,
    log_forecast_outputs,
    render_leaderboard,
    save_validation_summary,
    start_forecast_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["hourly", "daily", "all"], default="all")
    parser.add_argument("--models", choices=[*TREE_MODELS, "all"], nargs="+", default=["all"])
    parser.add_argument("--tune", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=2, help="CPU threads per estimator.")
    parser.add_argument("--auto-promote", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "val_2023")
    args = parser.parse_args()
    if args.n_trials < 1 or args.n_jobs < 1:
        parser.error("--n-trials and --n-jobs must be positive")
    if "all" in args.models and len(args.models) > 1:
        parser.error("Use --models all alone, or specify individual model names")
    return args


def run_trees(args: argparse.Namespace) -> None:
    config = MLflowConfig.from_env(REPO_ROOT)
    client = configure_mlflow(config)
    git_meta = get_git_metadata(REPO_ROOT)
    tasks = ["hourly", "daily"] if args.task == "all" else [args.task]
    model_names = list(TREE_MODELS) if args.models == ["all"] else list(dict.fromkeys(args.models))
    weather_mode = "observed_historical"
    print("Weather: historical observations, not archived day-ahead forecasts.", flush=True)
    for task in tasks:
        df = load_feature_dataset(task, repo_root=REPO_ROOT)
        train_df, eval_df, start, end = get_split_frames(df, split="val_2023", allow_test=False)
        scale = compute_seasonal_naive_scale(train_df, task)
        registry = config.registered_model_name(task)
        summary_path = args.output_dir / f"trees_{task}.json"
        records = []
        if summary_path.exists() and args.models != ["all"]:
            records = json.loads(summary_path.read_text())["models"]
        for name in model_names:
            model = TreeForecaster(
                task,
                name,
                tune=args.tune,
                n_trials=args.n_trials,
                seed=args.seed,
                n_jobs=args.n_jobs,
            )
            print(f"\n{task}: {model.name} | tuning={args.tune}", flush=True)
            with start_forecast_run(
                f"{model.name}_{task}_val_2023",
                task=task,
                split="val_2023",
                model_family=model.family,
                model_name=model.name,
                params={},
                extra_tags={"weather_mode": weather_mode, "forecast_origin": "local_midnight"},
            ) as run:
                started = time.perf_counter()
                result = run_rolling_backtest(model, train_df, eval_df, start, end)
                runtime = time.perf_counter() - started
                params = model.get_params()
                mlflow.log_params(params)
                metrics = compute_forecast_metrics(result.predictions_df, scale, prefix="val_")
                metrics["val_r2"] = float(
                    r2_score(result.predictions_df["y_true"], result.predictions_df["y_pred"])
                )
                metrics["runtime_seconds"] = runtime
                figures = create_validation_plots(result.predictions_df, task, model.name)
                log_forecast_outputs(result.predictions_df, metrics, figures)
                mlflow.log_dict({"trials": model.trials_}, "tuning/trials.json")
                mlflow.log_dict({"features": model.feature_cols}, "features/columns.json")
                mlflow.log_table(model.feature_importance(), "features/importance.json")
                for trial in model.trials_:
                    if trial["inner_mae"] is not None:
                        mlflow.log_metric("inner_mae", trial["inner_mae"], step=trial["number"])
                mv = log_and_register_pyfunc_model(
                    client,
                    pyfunc_model=model.to_pyfunc_model(),
                    registered_model_name=registry,
                    input_example=result.input_example,
                    sample_output=result.sample_output,
                    pip_requirements=model.pip_requirements(),
                    version_tags={
                        "model_name": model.name,
                        "model_family": model.family,
                        "task": task,
                        "split": "val_2023",
                        "val_mae": str(metrics["val_mae"]),
                        "weather_mode": weather_mode,
                        "git_commit": git_meta["git_commit"],
                    },
                )
                gate = evaluate_quality_gate(client, registry, metrics[config.primary_metric])
                mlflow.set_tag("quality_gate", gate.status)
                if gate.improvement_pct is not None:
                    mlflow.log_metric("gate_improvement_pct", gate.improvement_pct)
                aliases = apply_quality_gate_aliases(
                    client,
                    registry,
                    str(mv.version),
                    gate,
                    auto_promote=args.auto_promote,
                )
                record = {
                    "model_name": model.name,
                    "model_family": model.family,
                    "run_id": run.info.run_id,
                    "registered_model_name": registry,
                    "registered_version": str(mv.version),
                    "quality_gate": gate.status,
                    "gate_improvement_pct": gate.improvement_pct,
                    "params": params,
                    "metrics": metrics,
                    "aliases_after_run": aliases,
                }
            records = [r for r in records if r["model_name"] != model.name] + [record]
            # Checkpoint each completed model, so interruption does not lose earlier results.
            final_aliases = get_champion_challenger_aliases(client, registry)
            for rec in records:
                rec["is_champion"] = final_aliases["champion"] == str(rec["registered_version"])
                rec["is_challenger"] = final_aliases["challenger"] == str(rec["registered_version"])
            save_validation_summary(
                records,
                task=task,
                split="val_2023",
                primary_metric=config.primary_metric,
                output_path=summary_path,
                git_metadata=git_meta,
            )
            print(
                f"{model.name}: MAE={metrics['val_mae']:,.2f} MWh, "
                f"RMSE={metrics['val_rmse']:,.2f}, R2={metrics['val_r2']:.4f} "
                f"({runtime:.1f}s) | {gate.status}",
                flush=True,
            )
        print(f"Saved {summary_path}", flush=True)
    leaderboard = render_leaderboard(args.output_dir, split="val_2023")
    print(f"Leaderboard: {leaderboard}")
    print(f"MLflow: uv run mlflow ui --backend-store-uri {config.tracking_uri} --port 5002")


if __name__ == "__main__":
    run_trees(parse_args())
