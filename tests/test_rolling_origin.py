import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.features.demand_features import min_origin
from src.forecasting import rolling_origin as ro
from src.forecasting.rolling_origin import (
    EvaluationConfig,
    MODEL_NAMES,
    PREDICTION_COLUMNS,
    RF_NAME,
    assert_chronological,
    check_features_ignore_future,
    evaluate_series,
    make_windows,
    matrix_from_frame,
    run_evaluation,
    training_origins,
    validate_predictions,
)
from tests.conftest import make_dates, weekly_series

FAST = EvaluationConfig(horizon=7, n_windows=4, refit_every=2, n_estimators=5)


# ------------------------------------------------- windows / chronology

def test_windows_are_consecutive_and_end_at_the_last_day():
    n_days = 400
    windows = make_windows(n_days, horizon=7, n_windows=12, refit_every=4)
    assert len(windows) == 12
    assert windows[-1].last_target_pos == n_days - 1
    assert windows[0].first_target_pos == n_days - 12 * 7
    for a, b in zip(windows, windows[1:]):
        assert b.first_target_pos == a.last_target_pos + 1
        assert b.origin_pos == a.origin_pos + 7
    assert [w.window_id for w in windows] == list(range(1, 13))


def test_refit_groups_share_the_cutoff_of_their_first_window():
    windows = make_windows(400, 7, 12, 4)
    cutoffs = [w.fit_cutoff_pos for w in windows]
    assert cutoffs[:4] == [windows[0].origin_pos] * 4
    assert cutoffs[4:8] == [windows[4].origin_pos] * 4
    assert len(set(cutoffs)) == 3
    assert_chronological(windows, 7)


def test_every_training_label_is_on_or_before_the_cutoff():
    for w in make_windows(400, 7, 12, 4):
        origins = training_origins(w.fit_cutoff_pos, 7)
        assert origins[0] == min_origin(7)
        assert origins[-1] + 7 <= w.fit_cutoff_pos < w.first_target_pos


def test_chronology_check_catches_tampered_windows():
    windows = make_windows(400, 7, 4, 2)
    bad = list(windows)
    bad[1] = dataclasses.replace(bad[1], fit_cutoff_pos=bad[1].first_target_pos)
    with pytest.raises(AssertionError):
        assert_chronological(bad, 7)
    gap = list(windows)
    gap[2] = dataclasses.replace(gap[2], first_target_pos=gap[2].first_target_pos + 1)
    with pytest.raises(AssertionError):
        assert_chronological(gap, 7)


def test_windows_reject_impossible_setups():
    with pytest.raises(ValueError):
        make_windows(60, horizon=7, n_windows=12, refit_every=4)
    with pytest.raises(ValueError):
        make_windows(400, horizon=0, n_windows=4, refit_every=2)


# ------------------------------------------------------------- leakage

def test_end_to_end_predictions_ignore_data_after_the_first_origin():
    """Corrupt everything after window 1's origin: window-1 forecasts of
    every model (including the trained Random Forest) must not change."""
    values = weekly_series(400, seed=3)
    dates = make_dates(400)
    windows = make_windows(400, FAST.horizon, FAST.n_windows, FAST.refit_every)
    origin = windows[0].origin_pos

    corrupted = values.copy()
    corrupted[origin + 1:] = 5000.0

    clean = evaluate_series(values, dates, windows, FAST)
    dirty = evaluate_series(corrupted, dates, windows, dataclasses.replace(FAST, check_leakage=False))

    k = ["forecast_window", "horizon", "model"]
    a = clean[clean.forecast_window == 1].sort_values(k).reset_index(drop=True)
    b = dirty[dirty.forecast_window == 1].sort_values(k).reset_index(drop=True)
    assert len(a) == 4 * 7
    assert np.array_equal(a["predicted_sales"].to_numpy(), b["predicted_sales"].to_numpy())
    assert not np.array_equal(a["actual_sales"].to_numpy(), b["actual_sales"].to_numpy())


def test_leakage_detector_is_not_vacuous(monkeypatch):
    values = weekly_series(400)
    dates = make_dates(400)
    windows = make_windows(400, 7, 4, 2)
    check_features_ignore_future(values, dates, windows, 7)     # clean code passes

    real = ro.build_origin_features

    def leaky(values, dates, origins, horizon):
        frame = real(values, dates, origins, horizon)
        frame["last_value"] = values[frame["origin_pos"].to_numpy() + 1]   # peeks at T+1
        return frame

    monkeypatch.setattr(ro, "build_origin_features", leaky)
    with pytest.raises(AssertionError, match="Leakage"):
        check_features_ignore_future(values, dates, windows, 7)


# --------------------------------------------------------- one series

def test_evaluate_series_shape_columns_and_alignment():
    values = weekly_series(400, seed=1)
    dates = make_dates(400)
    windows = make_windows(400, FAST.horizon, FAST.n_windows, FAST.refit_every)
    out = evaluate_series(values, dates, windows, FAST)

    assert len(out) == FAST.n_windows * FAST.horizon * len(MODEL_NAMES)
    assert set(out["model"]) == set(MODEL_NAMES) and RF_NAME in set(out["model"])
    assert out["date"].min() == dates[windows[0].first_target_pos]
    assert out["date"].max() == dates[-1]
    # actuals are the true series values at those dates, identical for every model
    truth = pd.Series(values, index=dates)
    assert np.array_equal(out["actual_sales"].to_numpy(), truth.loc[out["date"]].to_numpy())
    # forecast_origin is exactly `horizon` days or fewer before the target
    lead = (out["date"] - out["forecast_origin"]).dt.days
    assert np.array_equal(lead.to_numpy(), out["horizon"].to_numpy())
    assert out["horizon"].between(1, 7).all()


def test_seasonal_naive_and_baselines_in_the_pipeline():
    values = weekly_series(400, seed=2, noise=False)       # perfectly weekly
    dates = make_dates(400)
    windows = make_windows(400, 7, 4, 2)
    out = evaluate_series(values, dates, windows, FAST)
    sn = out[out.model == "Seasonal naive"]
    assert np.array_equal(sn["predicted_sales"].to_numpy(), sn["actual_sales"].to_numpy())
    # weekday-matched: seasonal-naive prediction equals the value 7 days before the target
    truth = pd.Series(values, index=dates)
    assert np.array_equal(sn["predicted_sales"].to_numpy(), truth.loc[sn["date"] - pd.Timedelta(days=7)].to_numpy())
    pd_base = out[out.model == "Previous-day baseline"]
    expected = truth.loc[pd_base["forecast_origin"]].to_numpy()   # value on the origin day
    assert np.array_equal(pd_base["predicted_sales"].to_numpy(), expected)


def test_random_forest_learns_a_weekly_pattern():
    values = weekly_series(500, seed=4, noise=False)
    dates = make_dates(500)
    cfg = EvaluationConfig(horizon=7, n_windows=4, refit_every=4, n_estimators=20)
    out = evaluate_series(values, dates, make_windows(500, 7, 4, 4), cfg)
    rf = out[out.model == RF_NAME]
    assert np.abs(rf["predicted_sales"] - rf["actual_sales"]).mean() < 1.0


# ---------------------------------------------------------- all series

def test_run_evaluation_covers_every_series(small_matrix):
    result = run_evaluation(small_matrix, FAST, n_jobs=1, progress_every=0)
    p = result.predictions
    assert p[["store", "item"]].drop_duplicates().shape[0] == small_matrix.n_series
    assert len(p) == small_matrix.n_series * FAST.n_windows * FAST.horizon * 4
    assert result.scales["mase_scale"].notna().all()
    info = validate_predictions(p)
    assert info["rows_per_model"] == small_matrix.n_series * FAST.n_windows * FAST.horizon


def test_parallel_and_serial_runs_agree(small_matrix):
    serial = run_evaluation(small_matrix, FAST, n_jobs=1, progress_every=0).predictions
    parallel = run_evaluation(small_matrix, FAST, n_jobs=2, progress_every=0).predictions
    pd.testing.assert_frame_equal(serial, parallel)


def test_subset_filter_keeps_only_requested_series(small_matrix):
    sub = small_matrix.subset(stores=[1])
    assert sub.n_series == 2 and set(sub.keys["store"]) == {1}
    with pytest.raises(ValueError):
        small_matrix.subset(stores=[99])


# --------------------------------------------------------- data loading

def _frame(n_days=40, drop=None):
    dates = make_dates(n_days)
    rows = [(d, s, i, 1) for s in (1, 2) for i in (1, 2) for d in dates]
    df = pd.DataFrame(rows, columns=["date", "store", "item", "sales"])
    if drop is not None:
        df = df.drop(index=drop)
    return df


def test_matrix_from_complete_frame():
    m = matrix_from_frame(_frame())
    assert m.values.shape == (4, 40) and len(m.dates) == 40
    assert m.keys.to_dict("list") == {"store": [1, 1, 2, 2], "item": [1, 2, 1, 2]}


def test_missing_days_raise_instead_of_being_zero_filled():
    with pytest.raises(ValueError, match="missing"):
        matrix_from_frame(_frame(drop=[3]))
