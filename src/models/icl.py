"""In-context learning (ICL) foundation forecasters (`TimeH2O-v1` and `Timestral-209M`).

Both models perform zero-shot day-ahead forecasting directly from raw target history
and known-future covariates (`HOURLY_ICL_COVARIATE_COLS` / `DAILY_ICL_COVARIATE_COLS`),
without manual target lags or task-specific weight training.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

import mlflow.pyfunc
import numpy as np
import pandas as pd

from src.evaluation.splits import LEAKY_TARGET_COLS, TARGET_COL, TaskType
from src.features import DAILY_ICL_COVARIATE_COLS, HOURLY_ICL_COVARIATE_COLS
from src.models.base import BaseForecaster

ICL_MODELS: tuple[str, ...] = ("timeh2o_v1", "timestral_209m")
CONTEXT_PRESETS: tuple[str, ...] = ("3m", "1y", "max")
DEFAULT_CONTEXT_PRESETS: tuple[str, ...] = ("3m", "1y")

WEATHER_MODE = "observed_historical"


@dataclass(frozen=True)
class ICLModelSpec:
    """Static metadata and limits for a supported ICL foundation model."""

    name: str
    model_id: str
    max_context: int
    license: str
    supports_patch_ensemble: bool


ICL_MODEL_SPECS: dict[str, ICLModelSpec] = {
    "timeh2o_v1": ICLModelSpec(
        name="timeh2o_v1",
        model_id="h2oai/timeh2o-v1",
        max_context=16_384,
        license="H2O Model Non-Commercial License v1.0",
        supports_patch_ensemble=True,
    ),
    "timestral_209m": ICLModelSpec(
        name="timestral_209m",
        model_id="mistralai/Timestral-209M",
        max_context=32_768,
        license="Modified MIT (weights) / Apache 2.0 (code)",
        supports_patch_ensemble=False,
    ),
}


class ICLBackend(Protocol):
    """Minimal inference protocol implemented by TimeH2O, Timestral, and test fakes."""

    def predict(
        self,
        target_history: np.ndarray,
        future_covariates: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        """Return `horizon` point forecasts given `(L,)` history and `(C, L + H)` covariates."""


def resolve_context_steps(task: TaskType, context_preset: str, max_context: int) -> int:
    """Resolve a context preset (`3m`, `1y`, `max`) to a positive history step count."""
    if task not in {"hourly", "daily"}:
        raise ValueError(f"Unsupported task '{task}'. Expected 'hourly' or 'daily'.")
    if context_preset not in CONTEXT_PRESETS:
        raise ValueError(
            f"Unsupported context_preset '{context_preset}'. Expected one of {CONTEXT_PRESETS}."
        )
    if max_context < 1:
        raise ValueError(f"max_context must be >= 1, got {max_context}.")

    if context_preset == "3m":
        requested = 91 * 24 if task == "hourly" else 91
    elif context_preset == "1y":
        requested = 365 * 24 if task == "hourly" else 365
    else:
        requested = max_context
    return min(requested, max_context)


def icl_covariate_cols(task: TaskType) -> list[str]:
    """Return known-future covariate columns for ICL models (without target lags)."""
    if task == "hourly":
        return list(HOURLY_ICL_COVARIATE_COLS)
    if task == "daily":
        return list(DAILY_ICL_COVARIATE_COLS)
    raise ValueError(f"Unsupported task '{task}'. Expected 'hourly' or 'daily'.")


def _normalize_dates(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.normalize()
    return pd.to_datetime(series).dt.normalize()


def build_icl_context_frame(
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    *,
    task: TaskType,
    max_history_steps: int,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Build a leak-free chronological `[target_col, *covariate_cols]` context+horizon frame.

    History rows occupy the leading block with finite `target_col` values; future origin
    rows occupy the trailing `horizon` rows with `target_col = NaN` and known-future
    covariates only.
    """
    if history_df.empty:
        raise ValueError("history_df cannot be empty for ICL forecasting.")
    if future_df.empty:
        raise ValueError("future_df cannot be empty for ICL forecasting.")
    if max_history_steps < 1:
        raise ValueError(f"max_history_steps must be >= 1, got {max_history_steps}.")

    cov_cols = icl_covariate_cols(task)
    if "date_local" not in history_df.columns or "date_local" not in future_df.columns:
        raise ValueError("Both history_df and future_df must contain 'date_local'.")

    hist_dates = _normalize_dates(history_df["date_local"])
    fut_dates = _normalize_dates(future_df["date_local"])
    origin_date = pd.Timestamp(fut_dates.min()).normalize()
    if (hist_dates >= origin_date).any():
        raise ValueError(
            f"Leakage detected: history_df contains rows on/after {origin_date.date()}."
        )

    hist_tail = history_df.iloc[-max_history_steps:]
    missing_hist = set([target_col, *cov_cols]) - set(hist_tail.columns)
    if missing_hist:
        raise ValueError(f"history_df missing required ICL columns: {sorted(missing_hist)}")
    missing_fut = set(cov_cols) - set(future_df.columns)
    if missing_fut:
        raise ValueError(f"future_df missing required ICL columns: {sorted(missing_fut)}")

    hist_block = hist_tail.loc[:, [target_col, *cov_cols]].astype(float).copy()
    if not np.isfinite(hist_block.to_numpy()).all():
        raise ValueError("ICL history inputs contain missing or infinite values.")

    # Explicitly ignore any target/leaky columns on future_df by inserting NaN for target_col.
    fut_covs = future_df.loc[:, cov_cols].astype(float).copy()
    if not np.isfinite(fut_covs.to_numpy()).all():
        raise ValueError("ICL future covariates contain missing or infinite values.")
    fut_covs.insert(0, target_col, np.nan)

    return pd.concat([hist_block, fut_covs], ignore_index=True)


def split_icl_context_frame(
    model_input: Any,
    covariate_cols: list[str],
    *,
    target_col: str = TARGET_COL,
    max_history_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Validate and split a `[target_col, *covariate_cols]` frame into arrays.

    Returns `(target_history, future_covariates, horizon)` where:
    - `target_history` has shape `(L,)`
    - `future_covariates` has shape `(C, L + H)`
    - `horizon` is `H >= 1`
    """
    df = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
    required = [target_col, *covariate_cols]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"ICL model input missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("ICL model input cannot be empty.")

    target_vals = df[target_col].to_numpy(dtype=float)
    is_horizon = np.isnan(target_vals)
    if not np.any(~is_horizon):
        raise ValueError("ICL model input must contain at least one historical target row.")
    if not np.any(is_horizon):
        raise ValueError(
            f"ICL model input must contain trailing horizon rows with NaN in '{target_col}'."
        )

    first_horizon_idx = int(np.argmax(is_horizon))
    if not np.all(is_horizon[first_horizon_idx:]):
        raise ValueError(
            "ICL model input must place all observed history rows before NaN horizon rows."
        )

    hist_start = 0
    if max_history_steps is not None and first_horizon_idx > max_history_steps:
        hist_start = first_horizon_idx - max_history_steps

    trimmed = df.iloc[hist_start:].reset_index(drop=True)
    hist_len = first_horizon_idx - hist_start
    horizon = len(trimmed) - hist_len

    target_history = trimmed[target_col].iloc[:hist_len].to_numpy(dtype=float)
    if not np.isfinite(target_history).all():
        raise ValueError("ICL historical target contains missing or infinite values.")

    cov_matrix = trimmed.loc[:, covariate_cols].to_numpy(dtype=float).T
    if not np.isfinite(cov_matrix).all():
        raise ValueError("ICL covariate inputs contain missing or infinite values.")

    return target_history, cov_matrix, horizon


def _resolve_device(device: str) -> str:
    if device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(
            f"Unsupported device '{device}'. Expected one of ('auto', 'cpu', 'cuda', 'mps')."
        )
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _format_hf_auth_error(model_id: str, exc: Exception) -> RuntimeError:
    return RuntimeError(
        f"Failed to download or access Hugging Face model '{model_id}'. "
        "For gated models such as 'h2oai/timeh2o-v1', accept the model license on "
        "Hugging Face and set the HF_TOKEN environment variable (or run `hf auth login`). "
        f"Original error: {exc}"
    )


class _TimeH2OBackend:
    """Lazy wrapper around `h2ofm.TimeH2O`."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "auto",
        patch_ensemble: bool = True,
        local_files_only: bool = False,
    ) -> None:
        try:
            from h2ofm import TimeH2O
        except ImportError as exc:
            raise ImportError(
                "Package 'h2ofm' is required for TimeH2O-v1. "
                "Install optional ICL dependencies with: `uv sync --extra icl`."
            ) from exc

        resolved_device = _resolve_device(device)
        kwargs: dict[str, Any] = {"device": resolved_device}
        if local_files_only:
            kwargs["local_files_only"] = True
        try:
            self._model = TimeH2O.from_pretrained(model_id, **kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            if any(tok in msg for tok in ("gated", "401", "403", "token", "unauthorized")):
                raise _format_hf_auth_error(model_id, exc) from exc
            raise
        self.patch_ensemble = bool(patch_ensemble)

    def predict(
        self,
        target_history: np.ndarray,
        future_covariates: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        forecast = self._model.forecast(
            np.asarray(target_history, dtype=np.float32),
            horizon=int(horizon),
            future_covariates=np.asarray(future_covariates, dtype=np.float32),
            clip_nonnegative=True,
            patch_ensemble=self.patch_ensemble,
        )
        point = forecast.point
        if hasattr(point, "detach"):
            point = point.detach().cpu().numpy()
        return np.asarray(point, dtype=float).reshape(-1)


class _TimestralBackend:
    """Lazy wrapper around `timestral.TimestralForecaster`."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "auto",
        local_files_only: bool = False,
    ) -> None:
        try:
            from timestral import TimestralForecaster
        except ImportError as exc:
            raise ImportError(
                "Package 'timestral' is required for Timestral-209M. "
                "Install optional ICL dependencies with: `uv sync --extra icl`."
            ) from exc

        resolved_device = _resolve_device(device)
        try:
            self._model = TimestralForecaster.from_pretrained(
                model_id,
                device=resolved_device,
                local_files_only=local_files_only or os.environ.get("HF_HUB_OFFLINE") == "1",
            )
        except Exception as exc:
            msg = str(exc).lower()
            if any(tok in msg for tok in ("gated", "401", "403", "token", "unauthorized")):
                raise _format_hf_auth_error(model_id, exc) from exc
            raise

    def predict(
        self,
        target_history: np.ndarray,
        future_covariates: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        context = np.asarray(target_history, dtype=np.float32)[None, None, :]
        known = np.asarray(future_covariates, dtype=np.float32)[None, :, :]
        forecast = self._model.forecast(
            context,
            prediction_length=int(horizon),
            known_covariates=known,
            clip_nonnegative=True,
        )
        median = forecast.median
        if hasattr(median, "detach"):
            median = median.detach().cpu().numpy()
        preds = np.asarray(median, dtype=float).reshape(-1)
        return np.maximum(preds, 0.0)


def load_icl_backend(
    model_name: str,
    *,
    device: str = "auto",
    patch_ensemble: bool = True,
    local_files_only: bool = False,
) -> ICLBackend:
    """Instantiate the pretrained backend for `model_name`."""
    if model_name not in ICL_MODEL_SPECS:
        raise ValueError(f"Unsupported ICL model '{model_name}'. Expected one of {ICL_MODELS}.")
    spec = ICL_MODEL_SPECS[model_name]
    if model_name == "timeh2o_v1":
        return _TimeH2OBackend(
            spec.model_id,
            device=device,
            patch_ensemble=patch_ensemble,
            local_files_only=local_files_only,
        )
    return _TimestralBackend(
        spec.model_id,
        device=device,
        local_files_only=local_files_only,
    )


class ICLPyFunc(mlflow.pyfunc.PythonModel):
    """Lightweight MLflow PyFunc wrapper that loads pretrained ICL weights on demand."""

    def __init__(
        self,
        *,
        model_name: str,
        task: TaskType,
        context_preset: str,
        max_history_steps: int,
        covariate_cols: list[str],
        device: str = "auto",
        patch_ensemble: bool = True,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.task: TaskType = task
        self.context_preset = context_preset
        self.max_history_steps = int(max_history_steps)
        self.covariate_cols = list(covariate_cols)
        self.device = device
        self.patch_ensemble = bool(patch_ensemble)
        self.local_files_only = bool(local_files_only)
        self._backend: ICLBackend | None = None

    def _ensure_backend(self) -> ICLBackend:
        if self._backend is None:
            self._backend = load_icl_backend(
                self.model_name,
                device=self.device,
                patch_ensemble=self.patch_ensemble,
                local_files_only=self.local_files_only,
            )
        return self._backend

    def load_context(self, context: Any) -> None:
        """Load pretrained ICL model weights when MLflow loads this PyFunc artifact."""
        self._ensure_backend()

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Predict horizon steps from a `[TARGET_COL, *covariate_cols]` DataFrame."""
        target_history, future_covariates, horizon = split_icl_context_frame(
            model_input,
            self.covariate_cols,
            target_col=TARGET_COL,
            max_history_steps=self.max_history_steps,
        )
        backend = self._ensure_backend()
        preds = np.asarray(
            backend.predict(target_history, future_covariates, horizon),
            dtype=float,
        ).reshape(-1)
        if len(preds) != horizon:
            raise ValueError(
                f"{self.model_name} backend returned {len(preds)} predictions "
                f"for horizon={horizon}."
            )
        return preds


class ICLForecaster(BaseForecaster):
    """Zero-shot ICL forecaster (`TimeH2O-v1` or `Timestral-209M`) with rolling context."""

    family = "icl"

    def __init__(
        self,
        task: TaskType,
        model_name: str = "timeh2o_v1",
        *,
        context_preset: str = "1y",
        device: str = "auto",
        patch_ensemble: bool = True,
        local_files_only: bool = False,
        backend: ICLBackend | None = None,
        progress_every: int = 0,
    ) -> None:
        if task not in {"hourly", "daily"}:
            raise ValueError(f"Unsupported task '{task}'. Expected 'hourly' or 'daily'.")
        if model_name not in ICL_MODEL_SPECS:
            raise ValueError(f"Unsupported ICL model '{model_name}'. Expected one of {ICL_MODELS}.")

        self.task: TaskType = task
        self.model_name = model_name
        self.name = model_name
        self.spec = ICL_MODEL_SPECS[model_name]
        self.context_preset = context_preset
        self.max_history_steps = resolve_context_steps(task, context_preset, self.spec.max_context)
        self.covariate_cols = icl_covariate_cols(task)
        self.device = device
        self.patch_ensemble = bool(patch_ensemble)
        self.local_files_only = bool(local_files_only)
        self._backend: ICLBackend | None = backend
        self.progress_every = int(progress_every)

        self.train_rows_: int | None = None
        self.effective_context_min_: int | None = None
        self.effective_context_max_: int | None = None
        self.origins_predicted_: int = 0

    def _ensure_backend(self) -> ICLBackend:
        if self._backend is None:
            self._backend = load_icl_backend(
                self.model_name,
                device=self.device,
                patch_ensemble=self.patch_ensemble,
                local_files_only=self.local_files_only,
            )
        return self._backend

    def get_params(self) -> dict[str, Any]:
        """Return JSON-serializable ICL configuration and runtime context metadata."""
        params: dict[str, Any] = {
            "task": self.task,
            "model_name": self.model_name,
            "model_id": self.spec.model_id,
            "license": self.spec.license,
            "context_preset": self.context_preset,
            "max_history_steps": self.max_history_steps,
            "model_max_context": self.spec.max_context,
            "weather_mode": WEATHER_MODE,
            "point_estimate": "median",
            "covariate_cols": self.covariate_cols,
            "n_covariates": len(self.covariate_cols),
            "device": self.device,
        }
        if self.spec.supports_patch_ensemble:
            params["patch_ensemble"] = self.patch_ensemble
        if self.train_rows_ is not None:
            params["train_rows"] = self.train_rows_
        if self.effective_context_min_ is not None:
            params["effective_context_min"] = self.effective_context_min_
        if self.effective_context_max_ is not None:
            params["effective_context_max"] = self.effective_context_max_
        return params

    def fit(self, train_df: pd.DataFrame) -> ICLForecaster:
        """Validate historical training frame schema (ICL models are zero-shot)."""
        if train_df.empty:
            raise ValueError("Training data cannot be empty.")
        if "date_local" not in train_df.columns:
            raise ValueError("Training data must contain 'date_local'.")
        dates = _normalize_dates(train_df["date_local"])
        if not dates.is_monotonic_increasing:
            raise ValueError("Training data must be sorted by date_local.")

        required = [TARGET_COL, *self.covariate_cols]
        missing = set(required) - set(train_df.columns)
        if missing:
            raise ValueError(f"Training data missing required ICL columns: {sorted(missing)}")

        values = train_df.loc[:, required].astype(float).to_numpy()
        if not np.isfinite(values).all():
            raise ValueError("Training data contains missing or infinite values in ICL columns.")

        self.train_rows_ = len(train_df)
        self.effective_context_min_ = None
        self.effective_context_max_ = None
        self.origins_predicted_ = 0
        return self

    def build_day_features(
        self,
        history_df: pd.DataFrame,
        future_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build leak-free context + horizon input frame for one origin day."""
        # Guard against callers passing full target frames directly outside run_rolling_backtest.
        drop_cols = [col for col in LEAKY_TARGET_COLS if col in future_df.columns]
        clean_future = future_df.drop(columns=drop_cols) if drop_cols else future_df

        frame = build_icl_context_frame(
            history_df,
            clean_future,
            task=self.task,
            max_history_steps=self.max_history_steps,
            target_col=TARGET_COL,
        )
        hist_len = len(frame) - len(clean_future)
        if self.effective_context_min_ is None or hist_len < self.effective_context_min_:
            self.effective_context_min_ = hist_len
        if self.effective_context_max_ is None or hist_len > self.effective_context_max_:
            self.effective_context_max_ = hist_len
        return frame

    def predict_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """Generate point forecasts from a `[TARGET_COL, *covariate_cols]` context+horizon frame."""
        target_history, future_covariates, horizon = split_icl_context_frame(
            features_df,
            self.covariate_cols,
            target_col=TARGET_COL,
            max_history_steps=self.max_history_steps,
        )
        backend = self._ensure_backend()
        preds = np.asarray(
            backend.predict(target_history, future_covariates, horizon),
            dtype=float,
        ).reshape(-1)
        if len(preds) != horizon:
            raise ValueError(
                f"{self.name} backend returned {len(preds)} predictions for horizon={horizon}."
            )
        self.origins_predicted_ += 1
        if self.progress_every > 0 and self.origins_predicted_ % self.progress_every == 0:
            print(
                f"  [{self.model_name} | {self.task} | {self.context_preset}] "
                f"predicted {self.origins_predicted_} origins",
                flush=True,
            )
        return preds

    def make_input_example(
        self,
        features_df: pd.DataFrame,
        sample_output: np.ndarray,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Return a compact `(input_example, sample_output)` keeping a short trailing context."""
        is_horizon = features_df[TARGET_COL].isna().to_numpy()
        first_horizon_idx = int(np.argmax(is_horizon))
        keep_history = 48 if self.task == "hourly" else 14
        start_idx = max(0, first_horizon_idx - keep_history)
        compact = features_df.iloc[start_idx:].reset_index(drop=True)
        return compact, np.asarray(sample_output, dtype=float)

    def to_pyfunc_model(self) -> mlflow.pyfunc.PythonModel:
        """Return a lightweight MLflow PyFunc wrapper referencing the pretrained model ID."""
        return ICLPyFunc(
            model_name=self.model_name,
            task=self.task,
            context_preset=self.context_preset,
            max_history_steps=self.max_history_steps,
            covariate_cols=self.covariate_cols,
            device=self.device,
            patch_ensemble=self.patch_ensemble,
            local_files_only=self.local_files_only,
        )

    def pip_requirements(self) -> list[str]:
        """Return minimal pip dependencies required by `ICLPyFunc`."""
        pkg = "h2ofm" if self.model_name == "timeh2o_v1" else "timestral"
        return ["mlflow", "numpy", "pandas", pkg]
