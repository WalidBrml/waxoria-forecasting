import json

import pandas as pd
import pytest

from src.forecasting.evaluate_multiple_series import write_outputs
from src.forecasting.rolling_origin import (
    EvaluationConfig,
    MODEL_NAMES,
    PREDICTION_COLUMNS,
    RF_NAME,
    run_evaluation,
    validate_predictions,
)

FAST = EvaluationConfig(horizon=7, n_windows=3, refit_every=3, n_estimators=5)


@pytest.fixture
def result(small_matrix):
    return run_evaluation(small_matrix, FAST, n_jobs=1, progress_every=0)


def test_prediction_schema_and_dtypes(result):
    p = result.predictions
    assert list(p.columns) == PREDICTION_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(p["date"])
    assert pd.api.types.is_datetime64_any_dtype(p["forecast_origin"])
    assert pd.api.types.is_integer_dtype(p["horizon"])
    assert pd.api.types.is_integer_dtype(p["forecast_window"])
    assert pd.api.types.is_float_dtype(p["predicted_sales"])
    assert not p[["actual_sales", "predicted_sales"]].isna().any().any()
    assert (p["predicted_sales"] >= 0).all()


def test_every_model_including_random_forest_is_saved(result):
    p = result.predictions
    assert set(p["model"].astype(str)) == set(MODEL_NAMES)
    assert RF_NAME in set(p["model"].astype(str))
    sizes = p.groupby("model", observed=True).size()
    assert sizes.nunique() == 1          # identical coverage for all models


def test_validator_rejects_inconsistent_outputs(result):
    p = result.predictions.copy()
    with pytest.raises(ValueError, match="Unexpected columns"):
        validate_predictions(p.drop(columns=["horizon"]))
    rf_rows = p.index[p["model"].astype(str) == RF_NAME]
    with pytest.raises(ValueError):                      # one model loses a row
        validate_predictions(p.drop(index=rf_rows[0]))
    with pytest.raises(ValueError, match="Model set"):   # Random Forest missing
        validate_predictions(p[p["model"].astype(str) != RF_NAME])
    broken = p.copy()
    broken.loc[broken.index[0], "predicted_sales"] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        validate_predictions(broken)


def test_written_files_round_trip_and_stay_dashboard_compatible(result, tmp_path):
    write_outputs(result, tmp_path, total_series=3)

    saved = pd.read_csv(tmp_path / "predictions_all_models.csv", parse_dates=["date", "forecast_origin"])
    assert list(saved.columns) == PREDICTION_COLUMNS
    assert len(saved) == len(result.predictions)
    assert RF_NAME in set(saved["model"])

    # The Streamlit dashboard reads multi_series_metrics.csv and does
    # groupby("model")[["MAE", "RMSE"]].mean(); that must keep working.
    metrics = pd.read_csv(tmp_path / "multi_series_metrics.csv")
    assert {"store", "item", "model", "MAE", "RMSE", "WAPE", "MASE"} <= set(metrics.columns)
    summary = metrics.groupby("model", as_index=False)[["MAE", "RMSE"]].mean()
    assert set(summary["model"]) == set(MODEL_NAMES)

    pooled = pd.read_csv(tmp_path / "model_summary.csv")
    assert {"model", "MAE", "RMSE", "WAPE", "MASE_mean", "n_series", "n_obs"} <= set(pooled.columns)

    report = json.loads((tmp_path / "evaluation_report.json").read_text())
    cov = report["coverage"]
    assert cov["series"] == 3 and cov["full_coverage"] is True
    assert cov["stores"] == 2 and cov["products"] == 2
    assert cov["forecast_windows"] == 3
    assert cov["observations_per_model"] == 3 * 3 * 7


def test_partial_run_is_flagged_in_the_report(result, tmp_path):
    write_outputs(result, tmp_path, total_series=500)
    cov = json.loads((tmp_path / "evaluation_report.json").read_text())["coverage"]
    assert cov["full_coverage"] is False and cov["series_in_dataset"] == 500
