"""Shared portable MLflow tracking, registry quality gates, and summary logging."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse
from urllib.request import url2pathname

import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from src.data_loader import find_repo_root
from src.evaluation.splits import SplitType, TaskType


@dataclass(frozen=True)
class MLflowConfig:
    """Configuration for local SQLite or remote Databricks MLflow tracking."""

    tracking_uri: str
    registry_uri: str
    artifact_root: str | None
    experiment_name: str
    uc_prefix: str = ""
    primary_metric: str = "val_mae"

    @classmethod
    def from_env(cls, repo_root: Path | None = None) -> MLflowConfig:
        """Build MLflow config from env vars or local SQLite defaults."""
        root = repo_root or find_repo_root()
        env_tracking = os.environ.get("MLFLOW_TRACKING_URI")
        if env_tracking:
            tracking_uri = env_tracking
            artifact_root = os.environ.get("MLFLOW_ARTIFACT_ROOT")
        else:
            db_path = (root / "mlruns" / "mlflow.db").resolve()
            tracking_uri = f"sqlite:///{db_path.as_posix()}"
            artifact_root = os.environ.get(
                "MLFLOW_ARTIFACT_ROOT",
                (root / "mlruns" / "artifacts").resolve().as_uri(),
            )

        registry_uri = os.environ.get("MLFLOW_REGISTRY_URI", tracking_uri)
        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "electricity-load-germany")
        uc_prefix = os.environ.get("MLFLOW_UC_PREFIX", "")
        return cls(
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
            artifact_root=artifact_root,
            experiment_name=experiment_name,
            uc_prefix=uc_prefix,
        )

    def registered_model_name(self, task: TaskType) -> str:
        """Return portable registered model name (local or Unity Catalog)."""
        if self.uc_prefix:
            return f"{self.uc_prefix}de_load_{task}"
        return f"de-load-{task}"


@dataclass(frozen=True)
class QualityGateResult:
    """Outcome of comparing a candidate model against current `@champion`."""

    status: Literal["initial_champion", "passed", "failed"]
    passed: bool
    metric_name: str
    champion_version: str | None
    champion_metric: float | None
    candidate_metric: float
    improvement_pct: float | None


def _ensure_local_dirs(config: MLflowConfig) -> None:
    if config.tracking_uri.startswith("sqlite:///"):
        raw_path = config.tracking_uri.removeprefix("sqlite:///")
        Path(raw_path).parent.mkdir(parents=True, exist_ok=True)
    if config.artifact_root and config.artifact_root.startswith("file:///"):
        parsed = urlparse(config.artifact_root)
        Path(url2pathname(parsed.path)).mkdir(parents=True, exist_ok=True)


def configure_mlflow(config: MLflowConfig) -> MlflowClient:
    """Configure MLflow tracking/registry URIs, create experiment if needed, return client."""
    _ensure_local_dirs(config)
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_registry_uri(config.registry_uri)

    client = MlflowClient(
        tracking_uri=config.tracking_uri,
        registry_uri=config.registry_uri,
    )
    if client.get_experiment_by_name(config.experiment_name) is None:
        client.create_experiment(
            config.experiment_name,
            artifact_location=config.artifact_root,
        )
    mlflow.set_experiment(config.experiment_name)
    return client


def get_git_metadata(repo_root: Path | None = None) -> dict[str, str]:
    """Read current git short commit hash and dirty status safely."""
    try:
        root = repo_root or find_repo_root()
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return {"git_commit": commit or "unknown", "git_dirty": "true" if status else "false"}
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"git_commit": "unknown", "git_dirty": "unknown"}


@contextmanager
def start_forecast_run(
    run_name: str,
    *,
    task: str,
    split: str,
    model_family: str,
    model_name: str,
    params: dict[str, Any],
    extra_tags: dict[str, str] | None = None,
) -> Iterator[mlflow.ActiveRun]:
    """Start an MLflow run with standard audit tags and model parameters."""
    tags: dict[str, str] = {
        "task": task,
        "split": split,
        "model_family": model_family,
        "model_name": model_name,
        "python_version": platform.python_version(),
        **get_git_metadata(),
    }
    if extra_tags:
        tags.update({k: str(v) for k, v in extra_tags.items()})

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(tags)
        if params:
            mlflow.log_params(params)
        yield run


def log_forecast_outputs(
    predictions_df: pd.DataFrame,
    metrics: dict[str, float],
    figures: dict[str, plt.Figure],
) -> None:
    """Log validation metrics, diagnostic plots, and predictions CSV to the active run."""
    if metrics:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

    for filename, fig in figures.items():
        try:
            mlflow.log_figure(fig, f"plots/{filename}")
        finally:
            plt.close(fig)

    with tempfile.TemporaryDirectory() as tmp_dir:
        pred_path = Path(tmp_dir) / "val_predictions.csv"
        predictions_df.to_csv(pred_path, index=False)
        mlflow.log_artifact(str(pred_path), artifact_path="predictions")


def log_and_register_pyfunc_model(
    client: MlflowClient,
    *,
    pyfunc_model: mlflow.pyfunc.PythonModel,
    registered_model_name: str,
    input_example: pd.DataFrame,
    sample_output: np.ndarray,
    pip_requirements: list[str],
    version_tags: dict[str, str],
) -> ModelVersion:
    """Log a PyFunc model with inferred signature, register it, and tag the version."""
    signature = infer_signature(input_example, sample_output)
    code_paths: list[str] | None = None
    try:
        src_dir = find_repo_root() / "src"
        if src_dir.is_dir():
            code_paths = [str(src_dir)]
    except FileNotFoundError:
        code_paths = None

    model_info = mlflow.pyfunc.log_model(
        name="model",
        python_model=pyfunc_model,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        code_paths=code_paths,
    )
    mv = mlflow.register_model(model_uri=model_info.model_uri, name=registered_model_name)
    for key, value in version_tags.items():
        client.set_model_version_tag(registered_model_name, str(mv.version), key, str(value))
    return mv


def get_model_version_by_alias(
    client: MlflowClient,
    model_name: str,
    alias: str,
) -> ModelVersion | None:
    """Return the ModelVersion for `alias`, or None if model/alias does not exist."""
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return None


def get_champion_challenger_aliases(
    client: MlflowClient,
    model_name: str,
) -> dict[str, str | None]:
    """Return current version numbers assigned to `@champion` and `@challenger`."""
    champion = get_model_version_by_alias(client, model_name, "champion")
    challenger = get_model_version_by_alias(client, model_name, "challenger")
    return {
        "champion": str(champion.version) if champion is not None else None,
        "challenger": str(challenger.version) if challenger is not None else None,
    }


def _get_version_metric(
    client: MlflowClient,
    mv: ModelVersion,
    metric_name: str,
) -> float | None:
    if mv.run_id:
        run = client.get_run(mv.run_id)
        if metric_name in run.data.metrics:
            return float(run.data.metrics[metric_name])
    tag_val = mv.tags.get(metric_name)
    return float(tag_val) if tag_val is not None else None


def _is_better(candidate: float, reference: float, *, lower_is_better: bool) -> bool:
    return candidate < reference if lower_is_better else candidate > reference


def evaluate_quality_gate(
    client: MlflowClient,
    model_name: str,
    candidate_metric: float,
    metric_name: str = "val_mae",
    lower_is_better: bool = True,
) -> QualityGateResult:
    """Compare candidate metric against the current `@champion` registered version."""
    champion = get_model_version_by_alias(client, model_name, "champion")
    if champion is None:
        return QualityGateResult(
            status="initial_champion",
            passed=True,
            metric_name=metric_name,
            champion_version=None,
            champion_metric=None,
            candidate_metric=float(candidate_metric),
            improvement_pct=None,
        )

    champ_metric = _get_version_metric(client, champion, metric_name)
    if champ_metric is None:
        raise ValueError(
            f"Current @champion (version {champion.version}) lacks metric '{metric_name}'."
        )

    passed = _is_better(
        float(candidate_metric),
        float(champ_metric),
        lower_is_better=lower_is_better,
    )
    if champ_metric == 0.0:
        improvement_pct = None
    elif lower_is_better:
        improvement_pct = (champ_metric - candidate_metric) / abs(champ_metric) * 100.0
    else:
        improvement_pct = (candidate_metric - champ_metric) / abs(champ_metric) * 100.0

    return QualityGateResult(
        status="passed" if passed else "failed",
        passed=passed,
        metric_name=metric_name,
        champion_version=str(champion.version),
        champion_metric=float(champ_metric),
        candidate_metric=float(candidate_metric),
        improvement_pct=float(improvement_pct) if improvement_pct is not None else None,
    )


def promote_challenger_to_champion(
    client: MlflowClient,
    model_name: str,
    *,
    keep_previous_as_challenger: bool = True,
) -> dict[str, str | None]:
    """Promote `@challenger` to `@champion` and optionally retain old champion as `@challenger`."""
    challenger = get_model_version_by_alias(client, model_name, "challenger")
    if challenger is None:
        raise ValueError(
            f"Registered model '{model_name}' does not have a @challenger version to promote."
        )

    previous_champion = get_model_version_by_alias(client, model_name, "champion")
    new_champion_ver = str(challenger.version)

    client.set_registered_model_alias(model_name, "champion", new_champion_ver)
    client.set_model_version_tag(
        model_name,
        new_champion_ver,
        "promoted_at_utc",
        datetime.now(timezone.utc).isoformat(),
    )

    new_challenger_ver: str | None = None
    if (
        keep_previous_as_challenger
        and previous_champion is not None
        and str(previous_champion.version) != new_champion_ver
    ):
        new_challenger_ver = str(previous_champion.version)
        client.set_registered_model_alias(model_name, "challenger", new_challenger_ver)
    else:
        try:
            client.delete_registered_model_alias(model_name, "challenger")
        except MlflowException:
            pass

    return {"champion": new_champion_ver, "challenger": new_challenger_ver}


def apply_quality_gate_aliases(
    client: MlflowClient,
    model_name: str,
    candidate_version: str,
    gate_result: QualityGateResult,
    *,
    auto_promote: bool = True,
    lower_is_better: bool = True,
) -> dict[str, str | None]:
    """Update `@champion` / `@challenger` aliases based on quality gate outcome."""
    candidate_ver = str(candidate_version)
    client.set_model_version_tag(model_name, candidate_ver, "quality_gate", gate_result.status)

    if gate_result.status == "initial_champion":
        client.set_registered_model_alias(model_name, "champion", candidate_ver)
    elif gate_result.passed:
        client.set_registered_model_alias(model_name, "challenger", candidate_ver)
        if auto_promote:
            promote_challenger_to_champion(
                client,
                model_name,
                keep_previous_as_challenger=True,
            )
            client.set_model_version_tag(model_name, candidate_ver, "promotion_mode", "auto")
    elif auto_promote:
        current_champion = get_model_version_by_alias(client, model_name, "champion")
        if current_champion is None or str(current_champion.version) != candidate_ver:
            current_challenger = get_model_version_by_alias(client, model_name, "challenger")
            if current_challenger is None:
                client.set_registered_model_alias(model_name, "challenger", candidate_ver)
            else:
                chall_metric = _get_version_metric(
                    client,
                    current_challenger,
                    gate_result.metric_name,
                )
                if chall_metric is None or _is_better(
                    gate_result.candidate_metric,
                    chall_metric,
                    lower_is_better=lower_is_better,
                ):
                    client.set_registered_model_alias(model_name, "challenger", candidate_ver)

    return get_champion_challenger_aliases(client, model_name)


def save_validation_summary(
    records: list[dict[str, Any]],
    *,
    task: str,
    split: str,
    primary_metric: str,
    output_path: Path,
    git_metadata: dict[str, str],
) -> Path:
    """Write a git-trackable validation summary JSON sorted by `primary_metric`."""
    sorted_records = sorted(
        records,
        key=lambda rec: float(rec["metrics"][primary_metric]),
    )
    payload = {
        "task": task,
        "split": split,
        "primary_metric": primary_metric,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_metadata.get("git_commit", "unknown"),
        "git_dirty": git_metadata.get("git_dirty", "unknown"),
        "models": sorted_records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def _fmt_num(val: float | None, *, decimals: int = 1, thousands: bool = True) -> str:
    if val is None or not np.isfinite(val):
        return "-"
    if thousands:
        return f"{val:,.{decimals}f}"
    return f"{val:.{decimals}f}"


def render_leaderboard(
    results_dir: Path,
    output_path: Path | None = None,
    *,
    split: SplitType = "val_2023",
) -> Path:
    """Build a combined Markdown leaderboard from `results_dir/*.json`."""
    if split not in {"val_2023", "test_2024"}:
        raise ValueError(f"Unsupported split '{split}'. Expected 'val_2023' or 'test_2024'.")

    default_filename = (
        "validation_leaderboard.md" if split == "val_2023" else "test_leaderboard.md"
    )
    dest = output_path or (results_dir.parent / default_filename)
    default_metric = "val_mae" if split == "val_2023" else "test_mae"

    json_paths = sorted(results_dir.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No JSON summary files found in '{results_dir}'.")

    grouped: dict[str, list[tuple[str, str, str, dict[str, Any]]]] = {
        "hourly": [],
        "daily": [],
    }
    primary_metric = default_metric

    for json_path in json_paths:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        task = payload["task"]
        entry_metric = payload.get("primary_metric", default_metric)
        primary_metric = entry_metric
        file_commit = payload.get("git_commit", "unknown")
        grouped.setdefault(task, [])
        for model_rec in payload.get("models", []):
            grouped[task].append((entry_metric, json_path.name, file_commit, model_rec))

    title = "Validation Leaderboard (2023)" if split == "val_2023" else "Test Leaderboard (2024)"
    lines = [
        f"# {title}",
        "",
        f"Primary metric: `{primary_metric}` (MWh, lower is better). "
        f"Generated from `results/{split}/*.json`.",
        "",
    ]

    section_titles = {"hourly": "Hourly (H = 24h)", "daily": "Daily (H = 1d)"}
    ordered_tasks = [t for t in ("hourly", "daily") if grouped.get(t)] + [
        t for t in sorted(grouped) if t not in {"hourly", "daily"} and grouped[t]
    ]

    for task in ordered_tasks:
        entries = sorted(
            grouped[task],
            key=lambda item: float(item[3]["metrics"][item[0]]),
        )
        lines.extend(
            [
                f"## {section_titles.get(task, task.capitalize())}",
                "",
                "| Rank | Model | Family | MAE (MWh) | RMSE (MWh) | MAPE (%) | sMAPE (%) "
                "| MASE | Weekday MAE | Weekend MAE | Holiday MAE | Source | Commit |",
                "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: "
                "| ---: | ---: | --- | --- |",
            ]
        )
        for rank, (entry_metric, source_file, file_commit, rec) in enumerate(entries, start=1):
            metric_prefix = (
                entry_metric[:-3] if entry_metric.endswith("_mae") else default_metric[:-3]
            )
            m = rec["metrics"]
            lines.append(
                f"| {rank} "
                f"| `{rec['model_name']}` "
                f"| {rec.get('model_family', '-')} "
                f"| {_fmt_num(m.get(f'{metric_prefix}mae'), decimals=1)} "
                f"| {_fmt_num(m.get(f'{metric_prefix}rmse'), decimals=1)} "
                f"| {_fmt_num(m.get(f'{metric_prefix}mape'), decimals=2, thousands=False)} "
                f"| {_fmt_num(m.get(f'{metric_prefix}smape'), decimals=2, thousands=False)} "
                f"| {_fmt_num(m.get(f'{metric_prefix}mase'), decimals=3, thousands=False)} "
                f"| {_fmt_num(m.get(f'{metric_prefix}mae_weekday'), decimals=1)} "
                f"| {_fmt_num(m.get(f'{metric_prefix}mae_weekend'), decimals=1)} "
                f"| {_fmt_num(m.get(f'{metric_prefix}mae_holiday'), decimals=1)} "
                f"| `{source_file}` "
                f"| `{file_commit}` |"
            )
        lines.append("")

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest
