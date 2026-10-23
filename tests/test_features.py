import numpy as np
import pandas as pd
import pytest

from src.features.demand_features import (
    BASELINE_COLUMNS,
    FEATURE_COLUMNS,
    build_origin_features,
    min_origin,
    seasonal_naive_forecast,
    seasonal_step,
    trailing_mean,
)
from tests.conftest import make_dates


@pytest.fixture
def ramp():
    """y[i] = i, so every feature has an obvious hand-computed value."""
    values = np.arange(80, dtype=float)
    return values, make_dates(len(values))


def row(frame, origin, h):
    return frame[(frame.origin_pos == origin) & (frame.horizon == h)].iloc[0]


# ---------------------------------------------------------------- features

def test_row_layout_and_columns(ramp):
    values, dates = ramp
    frame = build_origin_features(values, dates, np.array([40, 47]), horizon=7)
    assert len(frame) == 2 * 7
    assert set(FEATURE_COLUMNS) <= set(frame.columns)
    assert list(frame["horizon"][:7]) == [1, 2, 3, 4, 5, 6, 7]
    assert (frame["target_pos"] == frame["origin_pos"] + frame["horizon"]).all()
    assert not frame[FEATURE_COLUMNS].isna().any().any()


def test_feature_values_horizon_1(ramp):
    values, dates = ramp
    r = row(build_origin_features(values, dates, np.array([40]), 7), 40, 1)
    assert r["last_value"] == 40
    assert r["mean_7"] == pytest.approx(np.mean(range(34, 41)))      # 37
    assert r["mean_28"] == pytest.approx(np.mean(range(13, 41)))     # 26.5
    assert r["seasonal_lag_1"] == 34       # target 41 -> 41 - 7
    assert r["seasonal_lag_2"] == 27
    assert r["seasonal_mean_4"] == pytest.approx(np.mean([34, 27, 20, 13]))
    # position 41 is 41 days after a Monday -> Sunday (6)
    assert r["dow"] == dates[41].day_of_week == 6
    assert r["month"] == dates[41].month


def test_seasonal_lag_at_horizon_7_is_the_origin_day(ramp):
    values, dates = ramp
    r = row(build_origin_features(values, dates, np.array([40]), 7), 40, 7)
    assert r["seasonal_lag_1"] == 40       # target 47 -> 47 - 7 = origin


def test_longer_horizon_steps_back_whole_weeks(ramp):
    values, dates = ramp
    frame = build_origin_features(values, dates, np.array([40]), horizon=14)
    r = row(frame, 40, 10)                  # target 50, step = 14
    assert r["seasonal_lag_1"] == 36
    assert seasonal_step(10) == 14 and seasonal_step(7) == 7 and seasonal_step(8) == 14
    # every seasonal source is at or before the origin
    assert (frame["target_pos"] - np.array([seasonal_step(int(h)) for h in frame["horizon"]])
            <= frame["origin_pos"]).all()


def test_trailing_mean_uses_only_past_values():
    v = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    out = trailing_mean(v, 3)
    assert np.isnan(out[:2]).all()
    assert out[2] == 2 and out[7] == 7
    v2 = v.copy()
    v2[5:] = 1000
    assert np.array_equal(trailing_mean(v2, 3)[:5], out[:5], equal_nan=True)


def test_min_origin_and_input_validation(ramp):
    values, dates = ramp
    assert min_origin(7) == 27
    with pytest.raises(ValueError):
        build_origin_features(values, dates, np.array([min_origin(7) - 1]), 7)
    with pytest.raises(ValueError):
        build_origin_features(values, dates, np.array([len(values)]), 7)
    with pytest.raises(ValueError):          # calendar too short for targets
        build_origin_features(values, dates[:45], np.array([40]), 7)
    with pytest.raises(ValueError):
        min_origin(0)


# ----------------------------------------------------------------- leakage

@pytest.mark.parametrize("horizon", [1, 7, 14])
def test_features_do_not_change_when_the_future_changes(horizon):
    rng = np.random.default_rng(1)
    values = rng.integers(0, 50, 200).astype(float)
    dates = make_dates(200)
    for origin in (min_origin(horizon), 90, 150):
        future_changed = values.copy()
        future_changed[origin + 1:] = 9999.0
        a = build_origin_features(values, dates, np.array([origin]), horizon)
        b = build_origin_features(future_changed, dates, np.array([origin]), horizon)
        assert np.array_equal(a[FEATURE_COLUMNS].to_numpy(), b[FEATURE_COLUMNS].to_numpy())


def test_features_can_be_built_for_the_true_future():
    """Origin = last observed day; targets lie beyond the data (calendar only)."""
    values = np.arange(60, dtype=float)
    dates = make_dates(67)
    frame = build_origin_features(values, dates, np.array([59]), 7)
    assert len(frame) == 7 and not frame[FEATURE_COLUMNS].isna().any().any()


# ------------------------------------------------------- seasonal naive

def test_seasonal_naive_is_same_weekday_last_week(ramp):
    values, dates = ramp
    origins = np.array([40, 50])
    pred = seasonal_naive_forecast(values, origins, 7)
    target = np.repeat(origins, 7) + np.tile(np.arange(1, 8), 2)
    assert np.array_equal(pred, values[target - 7])
    # and the weekday really matches
    assert (dates[target].day_of_week == dates[target - 7].day_of_week).all()


def test_seasonal_naive_matches_feature_column_and_perfect_on_periodic_data():
    values = np.tile([5, 6, 7, 8, 9, 10, 11], 20).astype(float)
    dates = make_dates(len(values))
    origins = np.array([60, 61, 62])
    frame = build_origin_features(values, dates, origins, 7)
    pred = seasonal_naive_forecast(values, origins, 7)
    assert np.array_equal(pred, frame[BASELINE_COLUMNS["Seasonal naive"]].to_numpy())
    assert np.array_equal(pred, values[frame["target_pos"].to_numpy()])   # zero error


def test_baseline_columns_map_to_features():
    assert set(BASELINE_COLUMNS.values()) <= set(FEATURE_COLUMNS)
    assert set(BASELINE_COLUMNS) == {
        "Previous-day baseline", "7-day moving average", "Seasonal naive"
    }
