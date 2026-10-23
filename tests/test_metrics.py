import numpy as np
import pandas as pd
import pytest

from src.forecasting.metrics import (
    bias,
    mae,
    mase,
    mase_scale,
    per_series_metrics,
    pooled_metrics,
    rmse,
    wape,
)


def long_frame(rows):
    """rows: (store, item, model, actual, predicted)"""
    return pd.DataFrame(
        rows, columns=["store", "item", "model", "actual_sales", "predicted_sales"]
    )


def scales(*pairs):
    return pd.DataFrame(
        [{"store": s, "item": i, "mase_scale": v} for s, i, v in pairs]
    )


def test_basic_metric_values():
    a, p = [10, 20, 30], [12, 18, 30]
    assert mae(a, p) == pytest.approx(4 / 3)
    assert rmse(a, p) == pytest.approx((8 / 3) ** 0.5)
    assert wape(a, p) == pytest.approx(4 / 60)
    assert bias(a, p) == pytest.approx(0.0)
    assert bias(a, [11, 21, 31]) == pytest.approx(3 / 60)


def test_mase_scale_and_mase():
    history = np.arange(30, dtype=float)             # |y_t - y_{t-7}| = 7
    assert mase_scale(history) == pytest.approx(7.0)
    assert mase([10, 10], [14, 10], scale=7.0) == pytest.approx(2.0 / 7.0)


def test_input_validation():
    with pytest.raises(ValueError):
        mae([1, 2], [1])
    with pytest.raises(ValueError):
        rmse([], [])


# ------------------------------------------------------- zero-demand cases

def test_zero_actuals_make_wape_and_bias_undefined_not_inf():
    a, p = [0, 0, 0], [1, 0, 2]
    assert np.isnan(wape(a, p))
    assert np.isnan(bias(a, p))
    assert mae(a, p) == pytest.approx(1.0)           # MAE/RMSE stay defined
    assert rmse(a, p) == pytest.approx((5 / 3) ** 0.5)


def test_perfect_forecast_of_zero_demand():
    assert mae([0, 0], [0, 0]) == 0
    assert np.isnan(wape([0, 0], [0, 0]))            # 0/0 is undefined, not 0


def test_mase_undefined_for_flat_or_short_history():
    assert mase_scale(np.zeros(30)) == 0.0
    assert np.isnan(mase([1, 2], [1, 2], scale=0.0))
    assert np.isnan(mase([1, 2], [1, 2], scale=float("nan")))
    assert np.isnan(mase_scale([1, 2, 3]))           # shorter than one season


# ----------------------------------------------------- pooled aggregation

def test_pooled_wape_is_ratio_of_sums_not_mean_of_ratios():
    df = long_frame(
        [
            (1, 1, "M", 100, 110),    # series A: |e| = 10, actual 100 -> WAPE 0.10
            (2, 1, "M", 1, 2),        # series B: |e| = 1,  actual 1   -> WAPE 1.00
        ]
    )
    ps = per_series_metrics(df, scales((1, 1, 5.0), (2, 1, 5.0)))
    pooled = pooled_metrics(ps).set_index("model").loc["M"]
    assert pooled["WAPE"] == pytest.approx(11 / 101)
    assert pooled["WAPE"] != pytest.approx((0.10 + 1.00) / 2)
    assert pooled["n_obs"] == 2 and pooled["n_series"] == 2


def test_zero_demand_series_is_handled_in_aggregation():
    df = long_frame(
        [
            (1, 1, "M", 0, 0), (1, 1, "M", 0, 0),          # all-zero series
            (2, 1, "M", 10, 12), (2, 1, "M", 10, 8),
        ]
    )
    ps = per_series_metrics(df, scales((1, 1, 0.0), (2, 1, 2.0)))
    zero = ps[ps.store == 1].iloc[0]
    assert zero["MAE"] == 0 and np.isnan(zero["WAPE"]) and np.isnan(zero["MASE"])
    ok = ps[ps.store == 2].iloc[0]
    assert ok["WAPE"] == pytest.approx(4 / 20) and ok["MASE"] == pytest.approx(1.0)

    pooled = pooled_metrics(ps).iloc[0]
    assert pooled["WAPE"] == pytest.approx(4 / 20)       # sums include the zero series safely
    assert pooled["series_wape_defined"] == 1
    assert pooled["series_mase_defined"] == 1
    assert pooled["MAE"] == pytest.approx(4 / 4 * 1.0)   # (0+0+2+2)/4


def test_pooled_wape_is_nan_when_every_actual_is_zero():
    df = long_frame([(1, 1, "M", 0, 1), (1, 1, "M", 0, 0)])
    pooled = pooled_metrics(per_series_metrics(df, scales((1, 1, 1.0)))).iloc[0]
    assert np.isnan(pooled["WAPE"]) and pooled["MAE"] == pytest.approx(0.5)


def test_series_best_counts_lowest_mae_per_series():
    df = long_frame(
        [
            (1, 1, "good", 10, 10), (1, 1, "bad", 10, 15),
            (2, 1, "good", 10, 11), (2, 1, "bad", 10, 10),
        ]
    )
    pooled = pooled_metrics(per_series_metrics(df, scales((1, 1, 1.0), (2, 1, 1.0))))
    best = pooled.set_index("model")["series_best"]
    assert best["good"] == 1 and best["bad"] == 1


def test_old_dashboard_columns_come_first():
    df = long_frame([(1, 1, "M", 10, 11)])
    ps = per_series_metrics(df, scales((1, 1, 1.0)))
    assert list(ps.columns[:5]) == ["store", "item", "model", "MAE", "RMSE"]
