
import pandas as pd
import pytest

from src.forecasting.baseline import evaluate_forecasts


def test_perfect_forecast_has_zero_error():
    actual = pd.Series([10, 20, 30])
    predicted = pd.Series([10, 20, 30])

    metrics = evaluate_forecasts(actual, predicted)

    assert metrics["MAE"] == 0
    assert metrics["RMSE"] == 0


def test_forecast_errors_are_calculated_correctly():
    actual = pd.Series([10, 20, 30])
    predicted = pd.Series([12, 18, 30])

    metrics = evaluate_forecasts(actual, predicted)

    assert metrics["MAE"] == pytest.approx(4 / 3)
    assert metrics["RMSE"] == pytest.approx((8 / 3) ** 0.5)


def test_worse_predictions_have_higher_error():
    actual = pd.Series([10, 20, 30])

    good = evaluate_forecasts(
        actual, pd.Series([11, 19, 30])
    )
    bad = evaluate_forecasts(
        actual, pd.Series([15, 25, 35])
    )

    assert bad["MAE"] > good["MAE"]
    assert bad["RMSE"] > good["RMSE"]
