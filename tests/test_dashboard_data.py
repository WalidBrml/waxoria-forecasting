"""Tests for app/data_access.py using small synthetic files (no real data needed)."""

import json

import numpy as np
import pandas as pd
import pytest

from app import data_access as da
from src.forecasting.evaluate_multiple_series import write_outputs
from src.forecasting.rolling_origin import EvaluationConfig, run_evaluation

FAST = EvaluationConfig(horizon=7, n_windows=3, refit_every=3, n_estimators=5)


def sales_frame(matrix) -> pd.DataFrame:
    rows = []
    for (store, item), values in zip(matrix.keys.itertuples(index=False), matrix.values):
        rows.append(pd.DataFrame({"date": matrix.dates, "store": store, "item": item, "sales": values.astype(int)}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def project(tmp_path, small_matrix):
    """A complete set of Waxoria files written to a temp directory."""
    result = run_evaluation(small_matrix, FAST, n_jobs=1, progress_every=0)
    reports = tmp_path / "reports"
    write_outputs(result, reports, total_series=small_matrix.n_series)
    sales_path = tmp_path / "sales_clean.csv"
    sales_frame(small_matrix).to_csv(sales_path, index=False)
    paths = {
        "sales": sales_path,
        "predictions": reports / "predictions_all_models.csv",
        "per_series": reports / "multi_series_metrics.csv",
        "summary": reports / "model_summary.csv",
        "report": reports / "evaluation_report.json",
    }
    return paths, result


# ------------------------------------------------------------ file handling

def test_missing_files_are_reported_not_raised(tmp_path):
    paths = {k: tmp_path / f"nope_{k}" for k in da.PATHS}
    data = da.load_all(paths)
    assert not data.has_sales and not data.has_predictions and data.report is None
    for status in data.status.values():
        assert status.state == "Missing" and "python -m" in status.message


def test_invalid_schema_and_malformed_json_are_flagged(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("date,store\n2020-01-01,1\n")
    obj, status = da.load_one("sales", bad_csv)
    assert obj is None and status.state == "Invalid" and "item" in status.message

    empty = tmp_path / "empty.csv"
    empty.write_text("date,store,item,sales\n")
    obj, status = da.load_one("sales", empty)
    assert obj is None and status.state == "Invalid"

    bad_json = tmp_path / "r.json"
    bad_json.write_text("{not json")
    obj, status = da.load_one("report", bad_json)
    assert obj is None and status.state == "Invalid"
    no_cov = tmp_path / "r2.json"
    no_cov.write_text(json.dumps({"x": 1}))
    assert da.load_one("report", no_cov)[0] is None


def test_load_all_on_complete_project(project):
    paths, result = project
    data = da.load_all(paths)
    assert all(s.ok for s in data.status.values())
    assert data.has_sales and data.has_predictions
    assert data.models() == ["Random Forest", "Seasonal naive", "7-day moving average", "Previous-day baseline"]
    assert len(data.scales) == 3 and data.scales["mase_scale"].notna().all()
    assert data.report["coverage"]["series"] == 3


def test_file_signature_changes_when_a_file_changes(project):
    paths, _ = project
    before = da.file_signature(paths)
    paths["summary"].write_text(paths["summary"].read_text() + "\n")
    assert da.file_signature(paths) != before


# -------------------------------------------------------------- filtering

def test_filters_apply_to_sales_and_predictions(project):
    paths, _ = project
    data = da.load_all(paths)
    f = da.default_filters(data)
    assert f.stores == (1, 2) and f.start == data.sales["date"].min()
    only1 = da.Filters(f.start, f.end, (1,))
    assert set(da.filter_sales(data.sales, only1)["store"]) == {1}
    assert set(da.filter_predictions(data.predictions, only1)["store"]) == {1}
    assert da.filter_sales(data.sales, da.Filters(f.start, f.end, ())).empty
    one = da.filter_predictions(data.predictions, f, models=["Random Forest"], store=2, item=1)
    assert set(one["model"].astype(str)) == {"Random Forest"} and len(one) == 3 * 7
    late = da.Filters(data.predictions["date"].max(), f.end, f.stores)
    assert da.filter_predictions(data.predictions, late)["date"].nunique() == 1


def test_unfiltered_dashboard_metrics_equal_the_saved_files(project):
    paths, _ = project
    data = da.load_all(paths)
    per_series, pooled = da.metric_tables(data.predictions, data.scales)
    saved = pd.read_csv(paths["per_series"])
    merged = per_series.merge(saved, on=["store", "item", "model"], suffixes=("", "_saved"))
    assert len(merged) == len(saved) == len(per_series)
    for col in ("MAE", "RMSE", "WAPE", "MASE", "Bias"):
        np.testing.assert_allclose(merged[col], merged[f"{col}_saved"], rtol=1e-4, atol=1e-5)
    # predictions are saved with 4 decimals, so recomputed values agree to ~1e-5 (not bit-exact)
    saved_pooled = pd.read_csv(paths["summary"]).set_index("model")
    for _, row in pooled.iterrows():
        assert row["MAE"] == pytest.approx(saved_pooled.loc[row["model"], "MAE"], rel=1e-4)
        assert row["WAPE"] == pytest.approx(saved_pooled.loc[row["model"], "WAPE"], rel=1e-4)


def test_filtered_metrics_use_only_the_filtered_rows(project):
    paths, _ = project
    data = da.load_all(paths)
    f = da.default_filters(data)
    sub = da.filter_predictions(data.predictions, da.Filters(f.start, f.end, (1,)))
    _, pooled = da.metric_tables(sub, data.scales)
    assert pooled["n_series"].eq(2).all() and pooled["n_obs"].eq(2 * 3 * 7).all()
    rf = sub[sub["model"] == "Random Forest"]
    assert pooled.set_index("model").loc["Random Forest", "MAE"] == pytest.approx(
        (rf["predicted_sales"] - rf["actual_sales"]).abs().mean()
    )
    empty_ps, empty_pool = da.metric_tables(sub.iloc[0:0], data.scales)
    assert empty_ps.empty and empty_pool.empty


def test_metric_tables_without_scales_leave_mase_undefined(project):
    paths, _ = project
    data = da.load_all(paths)
    per_series, pooled = da.metric_tables(data.predictions, None)
    assert per_series["MASE"].isna().all() and per_series["MAE"].notna().all()


def test_pooled_by_store_is_ratio_of_sums(project):
    paths, _ = project
    data = da.load_all(paths)
    per_series, _ = da.metric_tables(data.predictions, data.scales)
    by_store = da.pooled_by_store(per_series)
    row = by_store[(by_store["store"] == 1) & (by_store["model"] == "Random Forest")].iloc[0]
    g = per_series[(per_series["store"] == 1) & (per_series["model"] == "Random Forest")]
    assert row["WAPE"] == pytest.approx(g["sum_abs_err"].sum() / g["sum_actual"].sum())


def test_metrics_by_horizon_and_window_errors(project):
    paths, _ = project
    pred = da.load_all(paths).predictions
    hz = da.metrics_by_horizon(pred)
    assert set(hz["horizon"]) == set(range(1, 8)) and set(hz["model"].astype(str)) == set(pred["model"].astype(str))
    assert (hz["n_obs"] == 3 * 3).all()          # 3 series x 3 windows per horizon and model
    wins = da.window_errors(pred[pred["store"] == 1])
    assert list(wins["forecast_window"]) == [1, 2, 3]


# ----------------------------------------------------- coverage / labels

def test_coverage_matches_report_and_mismatches_are_detected(project):
    paths, _ = project
    data = da.load_all(paths)
    cov = da.coverage_from_predictions(data.predictions)
    assert cov["series"] == 3 and cov["stores"] == 2 and cov["products"] == 2
    assert cov["forecast_windows"] == 3 and cov["observations_per_model"] == 63
    assert da.report_mismatches(data.report, cov) == []
    wrong = json.loads(json.dumps(data.report))
    wrong["coverage"]["series"] = 500
    assert any("series" in m for m in da.report_mismatches(wrong, cov))
    assert da.report_mismatches(None, cov) == []
    empty = da.coverage_from_predictions(data.predictions.iloc[0:0])
    assert empty["series"] == 0 and empty["test_start"] is None


def test_horizon_labels_show_weekday_only_when_it_is_constant():
    dates = pd.to_datetime(["2020-01-06", "2020-01-07"])          # Mon, Tue
    constant = pd.DataFrame({"horizon": [1, 2], "date": dates,
                             "forecast_origin": pd.to_datetime(["2020-01-05"] * 2)})
    assert da.horizon_labels(constant) == {1: "1 · Mon", 2: "2 · Tue"}
    assert da.origin_weekdays(constant) == ["Sunday"]
    mixed = pd.DataFrame({"horizon": [1, 1], "date": dates,
                          "forecast_origin": pd.to_datetime(["2020-01-05", "2020-01-06"])})
    assert da.horizon_labels(mixed) == {1: "1"}
    assert da.horizon_labels(constant.iloc[0:0]) == {}


# ------------------------------------------------------------ sales stats

def test_yoy_growth_requires_two_full_years():
    dates = pd.date_range("2018-01-01", "2019-12-31", freq="D")
    values = np.where(dates.year == 2018, 10, 12)
    df = pd.DataFrame({"date": dates, "store": 1, "item": 1, "sales": values})
    assert da.yoy_growth(df, pd.Timestamp("2019-12-31")) == pytest.approx(0.2)
    assert da.yoy_growth(df[df["date"] >= "2019-01-01"], pd.Timestamp("2019-12-31")) is None
    assert da.yoy_growth(df.iloc[0:0], pd.Timestamp("2019-12-31")) is None


def test_series_history_and_rf_error_lookup(project):
    paths, _ = project
    data = da.load_all(paths)
    h = da.series_history(data.sales, 1, 2)
    assert h.index.freqstr == "D" and len(h) == 400 and not h.isna().any()
    err = da.rf_forecast_error_std(data.per_series, 1, 1)
    saved = pd.read_csv(paths["per_series"])
    assert err == pytest.approx(saved[(saved.store == 1) & (saved.item == 1) & (saved.model == "Random Forest")]["RMSE"].iloc[0])
    assert da.rf_forecast_error_std(data.per_series, 99, 99) is None
    assert da.rf_forecast_error_std(None, 1, 1) is None


def test_mase_scale_uses_history_up_to_the_first_origin(project):
    paths, _ = project
    data = da.load_all(paths)
    first_origin = data.predictions["forecast_origin"].min()
    scales = da.compute_mase_scales(data.sales, first_origin)
    row = scales[(scales.store == 1) & (scales.item == 1)].iloc[0]
    y = da.series_history(data.sales, 1, 1)
    y = y[y.index <= first_origin].to_numpy()
    assert row["mase_scale"] == pytest.approx(np.mean(np.abs(y[7:] - y[:-7])))
