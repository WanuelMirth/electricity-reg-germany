"""Tree integration checks for temporal boundaries, DST and serialized inference."""

from __future__ import annotations

import json

import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException

from src.evaluation.splits import LEAKY_TARGET_COLS, TARGET_COL
from src.features import build_tree_features
from src.models.base import run_rolling_backtest
from src.models.trees import TREE_MODELS, TreeForecaster


def make_frame(task: str, start: str, end: str) -> pd.DataFrame:
    if task == "hourly":
        local = pd.date_range(start, end, freq="h", tz="Europe/Berlin", inclusive="left")
        dates = local.tz_localize(None).normalize()
        df = pd.DataFrame(
            {
                "timestamp_local": local,
                "timestamp_utc": local.tz_convert("UTC"),
                "date_local": dates,
                "hour": local.hour,
            }
        )
    else:
        dates = pd.date_range(start, end, freq="D", inclusive="left")
        df = pd.DataFrame({"date_local": dates})
    df["day_of_week"] = dates.day_of_week
    df["day_of_year"] = dates.day_of_year
    df["month"] = dates.month
    df["is_weekend"] = (dates.day_of_week >= 5).astype(int)
    df["holiday_weight"] = 0.0
    df["is_bridge_day"] = 0
    df["is_christmas_shutdown"] = 0
    df["time_years"] = (dates - pd.Timestamp("2019-01-01")).days / 365.25
    for name in (
        "temp_c",
        "solar_radiation_wm2",
        "wind_speed_kmh",
        "temp_c_mean",
        "solar_radiation_wm2_mean",
        "wind_speed_kmh_mean",
    ):
        df[name] = 5 + np.cos(np.arange(len(df)) / 24)
    df[TARGET_COL] = 1000 + dates.day_of_year * 2 + dates.day_of_week * 20
    if task == "hourly":
        df[TARGET_COL] += df["hour"] * 5
    return df


def fast_params(name: str) -> dict:
    return {"iterations": 8, "depth": 3} if name == "catboost" else {"n_estimators": 8}


@pytest.mark.parametrize("task", ["hourly", "daily"])
@pytest.mark.parametrize("name", TREE_MODELS)
def test_all_tree_families_fit_predict_with_same_feature_contract(task: str, name: str) -> None:
    frame = make_frame(task, "2021-12-01", "2022-02-01")
    model = TreeForecaster(task, name, tune=False, model_params=fast_params(name), n_jobs=1)
    history = frame.loc[frame["date_local"] < "2022-01-25"]
    target = frame.loc[frame["date_local"] == "2022-01-25"]
    model.fit(history)
    inputs = model.build_day_features(history, target.drop(columns=[TARGET_COL]))
    prediction = model.predict_features(inputs)
    assert len(prediction) == len(target)
    assert np.isfinite(prediction).all()
    assert not set(model.feature_cols).intersection(LEAKY_TARGET_COLS)
    json.dumps(model.get_params(), allow_nan=False)
    np.testing.assert_allclose(prediction, model.to_pyfunc_model().predict(None, inputs))
    # The vectorized training builder must give identical inputs to origin-by-origin inference.
    expected = build_tree_features(frame, task=task).loc[target.index, model.feature_cols]
    pd.testing.assert_frame_equal(inputs, expected.astype(float))


def test_tuning_partition_and_linear_scaler_are_fit_on_past_only(monkeypatch) -> None:
    frame = make_frame("daily", "2021-11-01", "2023-01-01")
    model = TreeForecaster(
        "daily", "lightgbm_linear", n_trials=2, model_params={"n_estimators": 8}, n_jobs=1
    )
    created = []
    original = model._make_estimator

    def capture(params):
        estimator = original(params)
        created.append(estimator)
        return estimator

    monkeypatch.setattr(model, "_make_estimator", capture)
    model.fit(frame)
    features = build_tree_features(frame, task="daily")
    usable = frame["date_local"] >= pd.Timestamp("2021-11-08")
    inner = usable & (frame["date_local"].dt.year == 2021)
    expected_inner = features.loc[inner, model.feature_cols].mean().to_numpy()
    for estimator in created[:-1]:
        np.testing.assert_allclose(estimator.named_steps["scale"].mean_, expected_inner)
    np.testing.assert_allclose(
        created[-1].named_steps["scale"].mean_,
        features.loc[usable, model.feature_cols].mean().to_numpy(),
    )
    assert model.cv_train_end_ == "2021-12-31"
    assert model.cv_val_start_ == "2022-01-01"
    assert model.cv_val_end_ == "2022-12-31"
    assert len(model.trials_) == 2
    assert model.train_rows_ == len(frame) - 7


@pytest.mark.parametrize(
    "start,end,origin,hours",
    [
        ("2023-03-10", "2023-03-28", "2023-03-26", 23),
        ("2023-10-10", "2023-10-31", "2023-10-29", 25),
    ],
)
def test_hourly_backtest_and_target_invariance_on_dst_days(start, end, origin, hours) -> None:
    train = make_frame("hourly", "2021-12-01", "2022-02-01")
    evaluation = make_frame("hourly", start, end)
    model = TreeForecaster("hourly", tune=False, model_params={"n_estimators": 8}, n_jobs=1)
    result = run_rolling_backtest(
        model, train, evaluation, pd.Timestamp(origin), pd.Timestamp(origin)
    )
    assert len(result.predictions_df) == hours
    target = evaluation.loc[evaluation["date_local"] == origin]
    history = evaluation.loc[evaluation["date_local"] < origin]
    before = model.build_day_features(history, target)
    corrupted = target.copy()
    corrupted[TARGET_COL] = 1e12
    pd.testing.assert_frame_equal(before, model.build_day_features(history, corrupted))


def test_saved_mlflow_model_reproduces_predictions(tmp_path) -> None:
    frame = make_frame("daily", "2021-12-01", "2022-02-01")
    model = TreeForecaster("daily", tune=False, model_params={"n_estimators": 8}, n_jobs=1)
    history = frame.loc[frame["date_local"] < "2022-01-25"]
    target = frame.loc[frame["date_local"] == "2022-01-25"]
    model.fit(history)
    inputs = model.build_day_features(history, target)
    path = tmp_path / "model"
    mlflow.pyfunc.save_model(
        path=str(path),
        python_model=model.to_pyfunc_model(),
        input_example=inputs,
        pip_requirements=model.pip_requirements(),
    )
    loaded = mlflow.pyfunc.load_model(str(path))
    np.testing.assert_allclose(loaded.predict(inputs), model.predict_features(inputs))
    with pytest.raises(MlflowException, match="missing inputs"):
        loaded.predict(inputs.drop(columns=["lag_7d"]))


def test_missing_values_and_short_history_fail_clearly() -> None:
    frame = make_frame("daily", "2021-12-01", "2022-02-01")
    model = TreeForecaster("daily", tune=False)
    with pytest.raises(ValueError, match="seven training days"):
        model.fit(frame.iloc[:7])
    frame.loc[20, "temp_c_mean"] = np.nan
    with pytest.raises(ValueError, match="missing or infinite"):
        model.fit(frame)
