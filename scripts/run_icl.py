"""Evaluate zero-shot ICL models (`TimeH2O-v1`, `Timestral-209M`) across context lengths."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

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
from src.models.base import run_rolling_backtest  # noqa: E402
from src.models.icl import (  # noqa: E402
    CONTEXT_PRESETS,
    DEFAULT_CONTEXT_PRESETS,
    ICL_MODEL_SPECS,
    ICL_MODELS,
    WEATHER_MODE,
    ICLBackend,
    ICLForecaster,
    load_icl_backend,
)
from src.tracking.mlflow_utils import (  # noqa: E402
    MLflowConfig,
    apply_quality_gate_aliases,
    configure_mlflow,
    evaluate_quality_gate,
    get_champion_challenger_aliases,
    get_git_metadata,
    log_and_register_pyfunc_model,
    log_forecast_outputs,
    log_pyfunc_model,
    render_leaderboard,
    save_validation_summary,
    start_forecast_run,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for ICL validation and locked test runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=["hourly", "daily", "all"],
        default="all",
        help="Forecasting task resolution to evaluate (default: all).",
    )
    parser.add_argument(
        "--models",
        choices=[*ICL_MODELS, "all"],
        nargs="+",
        default=["all"],
        help="ICL model(s) to evaluate (default: all).",
    )
    parser.add_argument(
        "--context-lengths",
        choices=list(CONTEXT_PRESETS),
        nargs="+",
        default=list(DEFAULT_CONTEXT_PRESETS),
        help="Context window presets to sweep (default: 3m 1y).",
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
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Torch inference device (default: auto).",
    )
    parser.add_argument(
        "--patch-ensemble",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable multi-patch ensembling for TimeH2O-v1 (default: True).",
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
    parser.add_argument(
        "--max-origins",
        type=int,
        default=None,
        help="Optional dry-run limit on the number of origin days (skips MLflow/summary/registry).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=30,
        help="Log origin progress every N days (0 disables; default: 30).",
    )
    args = parser.parse_args()
    if "all" in args.models and len(args.models) > 1:
        parser.error("Use --models all alone, or specify individual model names.")
    if args.max_origins is not None and args.max_origins < 1:
        parser.error("--max-origins must be >= 1.")
    if args.progress_every < 0:
        parser.error("--progress-every must be >= 0.")
    return args


def _reset_peak_cuda_memory() -> None:
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _read_and_clear_peak_cuda_memory_mb() -> float | None:
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return None
    peak_bytes = float(torch.cuda.max_memory_allocated())
    torch.cuda.empty_cache()
    return round(peak_bytes / (1024.0 * 1024.0), 2)


def _merge_summary_records(
    existing_path: Path,
    new_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Upsert `new_records` by `model_name` into an existing summary JSON if present."""
    if not existing_path.is_file():
        return list(new_records)
    existing_payload = json.loads(existing_path.read_text(encoding="utf-8"))
    by_name: dict[str, dict[str, Any]] = {
        rec["model_name"]: rec for rec in existing_payload.get("models", [])
    }
    for rec in new_records:
        by_name[rec["model_name"]] = rec
    return list(by_name.values())


def _resolve_effective_end_date(
    eval_full_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    max_origins: int | None,
) -> pd.Timestamp:
    if max_origins is None:
        return end_date
    dates = pd.to_datetime(eval_full_df["date_local"]).dt.normalize()
    eligible = pd.DatetimeIndex(
        dates.loc[(dates >= start_date) & (dates <= end_date)].unique()
    ).sort_values()
    if eligible.empty:
        raise ValueError(
            f"No origin dates found between {start_date.date()} and {end_date.date()}."
        )
    return pd.Timestamp(eligible[min(max_origins, len(eligible)) - 1]).normalize()


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
        f"\n=== {task.upper()} ICL SUMMARY ({split}) "
        f"| Registry: {registered_model_name} ==="
    )
    header = (
        f"{'Model':<18} {'Ctx':>4} {'Ver':>3} {'MAE':>9} {'RMSE':>9} "
        f"{'MAPE%':>7} {'MASE':>6} {'Time(s)':>8} {'Gate':<16} {'Alias':<11}"
    )
    print(header)
    print("-" * len(header))
    for rec in sorted(records, key=lambda r: r["metrics"][primary_key]):
        m = rec["metrics"]
        p = rec["params"]
        ver = rec.get("registered_version")
        ver_str = str(ver) if ver is not None else "-"
        alias = _format_alias_label(ver, final_aliases)
        ctx = str(p.get("selected_context_preset", "-"))
        elapsed = rec.get("runtime_sec", float("nan"))
        gate = rec.get("quality_gate", "n/a")
        print(
            f"{rec['model_name']:<18} {ctx:>4} {ver_str:>3} {m[primary_key]:>9.2f} "
            f"{m[f'{metric_prefix}rmse']:>9.2f} {m[f'{metric_prefix}mape']:>7.2f} "
            f"{m[f'{metric_prefix}mase']:>6.3f} {elapsed:>8.1f} {gate:<16} {alias:<11}"
        )
    print(f"Summary JSON: {summary_path}")


def run_task_icl(
    task: TaskType,
    *,
    client: Any,
    config: MLflowConfig,
    model_names: list[str],
    context_lengths: list[str],
    split: SplitType,
    allow_test: bool,
    device: str,
    patch_ensemble: bool,
    auto_promote: bool,
    output_dir: Path,
    git_meta: dict[str, str],
    max_origins: int | None,
    progress_every: int,
    shared_backends: dict[str, ICLBackend],
    backend_factory: Callable[..., ICLBackend],
) -> Path | None:
    """Evaluate ICL models across context presets for one task and log/register winners."""
    df = load_feature_dataset(task, repo_root=REPO_ROOT)
    train_df, eval_full_df, start_date, end_date = get_split_frames(
        df,
        split=split,
        allow_test=allow_test,
    )
    effective_end_date = _resolve_effective_end_date(
        eval_full_df, start_date, end_date, max_origins
    )
    is_dry_run = max_origins is not None

    mase_scale = compute_seasonal_naive_scale(train_df, task)
    registered_model_name = config.registered_model_name(task)
    metric_prefix = "val_" if split == "val_2023" else "test_"
    primary_metric = f"{metric_prefix}mae"
    predictions_filename = (
        "val_predictions.csv" if split == "val_2023" else "test_predictions.csv"
    )
    summary_path = output_dir / f"icl_{task}.json"
    is_subset = set(model_names) != set(ICL_MODELS)
    records_to_save: list[dict[str, Any]] = (
        _merge_summary_records(summary_path, []) if (not is_dry_run and is_subset) else []
    )
    run_records: list[dict[str, Any]] = []

    for model_name in model_names:
        spec = ICL_MODEL_SPECS[model_name]
        if model_name not in shared_backends:
            print(f"\nLoading {spec.model_id} (device={device})...", flush=True)
            shared_backends[model_name] = backend_factory(
                model_name,
                device=device,
                patch_ensemble=patch_ensemble,
            )
        shared_backend = shared_backends[model_name]

        print(
            f"\n{task}: {model_name} | split={split} | contexts={context_lengths}"
            + (f" | dry_run_origins={max_origins}" if is_dry_run else ""),
            flush=True,
        )

        parent_tags = {
            "weather_mode": WEATHER_MODE,
            "forecast_origin": "local_midnight",
            "license": spec.license,
            "model_id": spec.model_id,
        }
        if is_dry_run:
            parent_tags["dry_run_max_origins"] = str(max_origins)

        parent_ctx = (
            nullcontext(None)
            if is_dry_run
            else start_forecast_run(
                f"{model_name}_{task}_{split}",
                task=task,
                split=split,
                model_family="icl",
                model_name=model_name,
                params={},
                extra_tags=parent_tags,
            )
        )
        with parent_ctx as parent_run:
            sweep_entries: list[dict[str, Any]] = []
            best_outcome: dict[str, Any] | None = None

            for context_preset in context_lengths:
                model = ICLForecaster(
                    task,
                    model_name,
                    context_preset=context_preset,
                    device=device,
                    patch_ensemble=patch_ensemble,
                    backend=shared_backend,
                    progress_every=progress_every,
                )
                child_ctx = (
                    nullcontext(None)
                    if is_dry_run
                    else mlflow.start_run(
                        run_name=f"{model_name}_{context_preset}_{task}_{split}",
                        nested=True,
                    )
                )
                with child_ctx as child_run:
                    if not is_dry_run:
                        mlflow.set_tags(
                            {
                                "task": task,
                                "split": split,
                                "model_family": model.family,
                                "model_name": model.name,
                                "context_preset": context_preset,
                                "weather_mode": WEATHER_MODE,
                                "forecast_origin": "local_midnight",
                                "python_version": platform.python_version(),
                                **git_meta,
                            }
                        )
                    _reset_peak_cuda_memory()
                    t0 = time.perf_counter()
                    backtest = run_rolling_backtest(
                        model=model,
                        train_df=train_df,
                        eval_full_df=eval_full_df,
                        start_date=start_date,
                        end_date=effective_end_date,
                    )
                    runtime_sec = float(time.perf_counter() - t0)
                    peak_vram_mb = _read_and_clear_peak_cuda_memory_mb()

                    child_params = model.get_params()
                    metrics = compute_forecast_metrics(
                        backtest.predictions_df,
                        mase_scale=mase_scale,
                        prefix=metric_prefix,
                    )
                    metrics[f"{metric_prefix}r2"] = float(
                        r2_score(
                            backtest.predictions_df["y_true"],
                            backtest.predictions_df["y_pred"],
                        )
                    )
                    metrics["runtime_sec"] = round(runtime_sec, 3)
                    if peak_vram_mb is not None:
                        metrics["peak_gpu_memory_mb"] = peak_vram_mb

                    if not is_dry_run:
                        mlflow.log_params(child_params)
                        figures = create_validation_plots(
                            backtest.predictions_df,
                            task=task,
                            model_name=f"{model.name}_{context_preset}",
                        )
                        log_forecast_outputs(
                            backtest.predictions_df,
                            metrics,
                            figures,
                            predictions_filename=predictions_filename,
                        )

                    vram_str = (
                        f", VRAM={peak_vram_mb:.0f}MB" if peak_vram_mb is not None else ""
                    )
                    print(
                        f"  [{model_name} | {task} | ctx={context_preset} "
                        f"({model.effective_context_max_} steps)] "
                        f"MAE={metrics[primary_metric]:,.2f} MWh, "
                        f"RMSE={metrics[f'{metric_prefix}rmse']:,.2f}, "
                        f"MASE={metrics[f'{metric_prefix}mase']:.3f} "
                        f"({runtime_sec:.1f}s{vram_str})",
                        flush=True,
                    )

                    sweep_entry = {
                        "context_preset": context_preset,
                        "max_history_steps": model.max_history_steps,
                        "effective_context_min": model.effective_context_min_,
                        "effective_context_max": model.effective_context_max_,
                        primary_metric: float(metrics[primary_metric]),
                        f"{metric_prefix}rmse": float(metrics[f"{metric_prefix}rmse"]),
                        f"{metric_prefix}mase": float(metrics[f"{metric_prefix}mase"]),
                        "runtime_sec": round(runtime_sec, 3),
                        "peak_gpu_memory_mb": peak_vram_mb,
                        "child_run_id": child_run.info.run_id if child_run is not None else None,
                    }
                    sweep_entries.append(sweep_entry)

                    if (
                        best_outcome is None
                        or metrics[primary_metric] < best_outcome["metrics"][primary_metric]
                    ):
                        best_outcome = {
                            "model": model,
                            "params": child_params,
                            "metrics": metrics,
                            "backtest": backtest,
                            "runtime_sec": runtime_sec,
                            "peak_vram_mb": peak_vram_mb,
                        }

            if is_dry_run:
                continue

            assert best_outcome is not None
            assert parent_run is not None
            winner_model: ICLForecaster = best_outcome["model"]
            winner_metrics: dict[str, float] = best_outcome["metrics"]
            winner_backtest = best_outcome["backtest"]
            total_sweep_runtime = round(sum(e["runtime_sec"] for e in sweep_entries), 3)

            parent_params = {
                **best_outcome["params"],
                "selected_context_preset": winner_model.context_preset,
                "tested_context_presets": ",".join(context_lengths),
            }
            mlflow.log_params(parent_params)
            mlflow.log_dict({"context_sweep": sweep_entries}, "tuning/context_sweep.json")

            winner_figures = create_validation_plots(
                winner_backtest.predictions_df,
                task=task,
                model_name=winner_model.name,
            )
            log_forecast_outputs(
                winner_backtest.predictions_df,
                winner_metrics,
                winner_figures,
                predictions_filename=predictions_filename,
            )
            mlflow.log_metric("total_sweep_runtime_sec", total_sweep_runtime)

            registered_version: str | None = None
            gate_status = "skipped_test_split"
            gate_improvement_pct: float | None = None
            aliases: dict[str, str | None] = {"champion": None, "challenger": None}

            version_tags = {
                "model_name": winner_model.name,
                "model_family": winner_model.family,
                "task": task,
                "split": split,
                "selected_context_preset": winner_model.context_preset,
                primary_metric: f"{winner_metrics[primary_metric]:.4f}",
                f"{metric_prefix}mase": f"{winner_metrics[f'{metric_prefix}mase']:.4f}",
                "weather_mode": WEATHER_MODE,
                "git_commit": git_meta["git_commit"],
            }
            if split == "val_2023":
                mv = log_and_register_pyfunc_model(
                    client,
                    pyfunc_model=winner_model.to_pyfunc_model(),
                    registered_model_name=registered_model_name,
                    input_example=winner_backtest.input_example,
                    sample_output=winner_backtest.sample_output,
                    pip_requirements=winner_model.pip_requirements(),
                    version_tags=version_tags,
                )
                registered_version = str(mv.version)

                gate = evaluate_quality_gate(
                    client,
                    registered_model_name,
                    candidate_metric=winner_metrics[config.primary_metric],
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
                log_pyfunc_model(
                    pyfunc_model=winner_model.to_pyfunc_model(),
                    input_example=winner_backtest.input_example,
                    sample_output=winner_backtest.sample_output,
                    pip_requirements=winner_model.pip_requirements(),
                )
                mlflow.set_tags({"quality_gate": gate_status})

            record = {
                "model_name": winner_model.name,
                "model_family": winner_model.family,
                "run_id": parent_run.info.run_id,
                "registered_model_name": registered_model_name,
                "registered_version": registered_version,
                "quality_gate": gate_status,
                "gate_improvement_pct": gate_improvement_pct,
                "runtime_sec": round(best_outcome["runtime_sec"], 3),
                "total_sweep_runtime_sec": total_sweep_runtime,
                "params": {
                    **parent_params,
                    "context_sweep": sweep_entries,
                },
                "metrics": winner_metrics,
                "aliases_after_run": aliases,
            }
            run_records.append(record)

        if not is_dry_run:
            records_to_save = _merge_summary_records_in_memory(records_to_save, record)
            final_aliases = (
                get_champion_challenger_aliases(client, registered_model_name)
                if split == "val_2023"
                else {"champion": None, "challenger": None}
            )
            for rec in records_to_save:
                ver = rec.get("registered_version")
                rec["is_champion"] = ver is not None and final_aliases.get("champion") == str(ver)
                rec["is_challenger"] = (
                    ver is not None and final_aliases.get("challenger") == str(ver)
                )
            save_validation_summary(
                records_to_save,
                task=task,
                split=split,
                primary_metric=primary_metric,
                output_path=summary_path,
                git_metadata=git_meta,
            )

    if is_dry_run:
        print(
            f"\nDry run complete ({task}, max_origins={max_origins}): "
            "skipped MLflow logging, JSON summary, and Model Registry updates."
        )
        return None

    final_aliases = (
        get_champion_challenger_aliases(client, registered_model_name)
        if split == "val_2023"
        else {"champion": None, "challenger": None}
    )
    _print_task_summary(
        task,
        split,
        registered_model_name,
        run_records,
        final_aliases,
        summary_path,
        metric_prefix,
    )
    return summary_path


def _merge_summary_records_in_memory(
    records: list[dict[str, Any]],
    new_record: dict[str, Any],
) -> list[dict[str, Any]]:
    return [r for r in records if r["model_name"] != new_record["model_name"]] + [new_record]


def run_icl(
    *,
    task: str = "all",
    models: list[str] | tuple[str, ...] = ("all",),
    context_lengths: list[str] | tuple[str, ...] = DEFAULT_CONTEXT_PRESETS,
    split: SplitType = "val_2023",
    allow_test: bool = False,
    device: str = "auto",
    patch_ensemble: bool = True,
    auto_promote: bool = True,
    output_dir: Path | None = None,
    max_origins: int | None = None,
    progress_every: int = 30,
    backend_factory: Callable[..., ICLBackend] | None = None,
) -> list[Path]:
    """Programmatic entrypoint for local CLI or Azure Databricks notebook execution."""
    model_list = (
        list(ICL_MODELS) if list(models) == ["all"] else list(dict.fromkeys(models))
    )
    for model_name in model_list:
        if model_name not in ICL_MODEL_SPECS:
            raise ValueError(f"Unsupported ICL model '{model_name}'. Expected one of {ICL_MODELS}.")

    ctx_list = list(dict.fromkeys(context_lengths))
    for ctx in ctx_list:
        if ctx not in CONTEXT_PRESETS:
            raise ValueError(f"Unsupported context preset '{ctx}'. Expected {CONTEXT_PRESETS}.")

    config = MLflowConfig.from_env(REPO_ROOT)
    client = None if max_origins is not None else configure_mlflow(config)
    git_meta = get_git_metadata(REPO_ROOT)
    dest_dir = output_dir or (REPO_ROOT / "results" / split)
    tasks: list[TaskType] = ["hourly", "daily"] if task == "all" else [task]  # type: ignore[list-item]
    factory = backend_factory or load_icl_backend

    shared_backends: dict[str, ICLBackend] = {}
    written_paths: list[Path] = []
    for current_task in tasks:
        written = run_task_icl(
            current_task,
            client=client,
            config=config,
            model_names=model_list,
            context_lengths=ctx_list,
            split=split,
            allow_test=allow_test,
            device=device,
            patch_ensemble=patch_ensemble,
            auto_promote=auto_promote,
            output_dir=dest_dir,
            git_meta=git_meta,
            max_origins=max_origins,
            progress_every=progress_every,
            shared_backends=shared_backends,
            backend_factory=factory,
        )
        if written is not None:
            written_paths.append(written)

    if max_origins is None:
        if written_paths:
            leaderboard = render_leaderboard(dest_dir, split=split)
            print(f"\nLeaderboard written: {leaderboard}")
        print("\nInspect runs & registry in MLflow UI:")
        print(f"  uv run mlflow ui --backend-store-uri {config.tracking_uri}")
    return written_paths


def main() -> None:
    """CLI entrypoint for ICL validation/test runs."""
    args = parse_args()
    run_icl(
        task=args.task,
        models=args.models,
        context_lengths=args.context_lengths,
        split=args.split,
        allow_test=args.allow_test,
        device=args.device,
        patch_ensemble=args.patch_ensemble,
        auto_promote=args.auto_promote,
        output_dir=args.output_dir,
        max_origins=args.max_origins,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
