"""Regression tests added by the audit (no Streamlit, no real data needed)."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import data_access as da
from src.forecasting.baseline import daily_series
from src.forecasting.evaluate_multiple_series import write_outputs
from src.forecasting.rolling_origin import EvaluationConfig, run_evaluation
from src.inventory.recommender import explain_risk, recommend_inventory

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- MAE reduction

def test_mae_reduction_sign_and_edge_cases():
    assert da.mae_reduction(8.0, 10.0) == pytest.approx(0.2)      # better model -> positive
    assert da.mae_reduction(12.0, 10.0) == pytest.approx(-0.2)    # worse model -> negative
    assert da.mae_reduction(5.0, 0.0) is None
    assert da.mae_reduction(5.0, float("nan")) is None


def test_mae_reduction_matches_saved_summary_when_present():
    path = da.PATHS["summary"]
    if not path.exists():
        pytest.skip("saved model_summary.csv not present")
    s = pd.read_csv(path).set_index("model")
    rf, best = s.loc["Random Forest", "MAE"], s.drop("Random Forest")["MAE"].min()
    assert da.mae_reduction(rf, best) == pytest.approx(1 - rf / best)
    assert 0 < da.mae_reduction(rf, best) < 1


# ------------------------------------------------------------------ risk wording

def _history(n=120):
    idx = pd.date_range(end="2020-01-12", periods=n, freq="D")
    rng = np.random.default_rng(3)
    return pd.Series(20 + 5 * np.sin(np.arange(n) * 2 * np.pi / 7) + rng.normal(0, 3, n), index=idx).clip(lower=0)


def test_stockout_probability_at_reorder_point_is_about_one_minus_service_level():
    h = _history()
    probe = recommend_inventory(h, current_stock=0, service_level=0.95)
    at_rop = recommend_inventory(h, current_stock=probe.reorder_point, service_level=0.95)
    assert at_rop.stockout_risk == "High"
    # The old wording ("lead-time demand is likely to exceed what is on hand") was wrong:
    assert at_rop.lead_time_stockout_probability <= 0.05 + 1e-9
    assert at_rop.lead_time_stockout_probability > 0.03        # and it is close to the 5% tolerance


def test_explain_risk_wording_matches_the_maths():
    text = explain_risk("High", 0.95)
    assert "likely" not in text.lower()
    assert "5.0%" in text
    assert "99.0%" not in text and "1.0%" in explain_risk("High", 0.99)
    assert explain_risk("Low", 0.95) and explain_risk("Medium", 0.95)
    with pytest.raises(KeyError):
        explain_risk("Unknown", 0.95)


# ----------------------------------------------------------- zero-demand caption

def _sales(values_by_series):
    rows = []
    for (store, item), vals in values_by_series.items():
        rows.append(pd.DataFrame({"date": pd.date_range("2020-01-01", periods=len(vals)),
                                  "store": store, "item": item, "sales": vals}))
    return pd.concat(rows, ignore_index=True)


def test_zero_demand_caption_is_truthful():
    clean = _sales({(1, 1): [0, 1, 2], (1, 2): [3, 0, 1]})
    assert da.zero_demand_summary(clean) == (2, 0)
    assert "No series is entirely zero" in da.zero_demand_caption(clean)

    dead = _sales({(1, 1): [0, 0, 0], (1, 2): [3, 0, 1]})
    assert da.zero_demand_summary(dead) == (4, 1)
    caption = da.zero_demand_caption(dead)
    assert "No series is entirely zero" not in caption
    assert "1 series have no sales at all" in caption


# ------------------------------------------------------- baseline: no zero-filling

def test_daily_series_refuses_to_zero_fill_missing_days():
    df = _sales({(1, 1): [5, 6, 7, 8]})
    ok = daily_series(df, 1, 1)
    assert list(ok) == [5.0, 6.0, 7.0, 8.0]
    gappy = df.drop(index=2)                       # remove one day
    with pytest.raises(ValueError, match="missing days"):
        daily_series(gappy, 1, 1)


# --------------------------------------------- no pandas deprecations from our code

def _deprecation_categories():
    cats = [FutureWarning, DeprecationWarning, PendingDeprecationWarning]
    try:  # pandas >= 3 warns about changes planned for 4.0 with this class
        from pandas.errors import Pandas4Warning
        cats.append(Pandas4Warning)
    except ImportError:
        pass
    return tuple(cats)


def test_project_code_raises_no_deprecation_warnings(small_matrix, tmp_path):
    """Exercise the pipeline, metrics, data access and recommender; no deprecated
    API may be triggered from project files. Pandas attributes its warnings to the
    calling line, so a warning whose file lives in this repo is our own usage.
    Warnings are recorded, never silenced."""
    cats = _deprecation_categories()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_evaluation(small_matrix, EvaluationConfig(horizon=7, n_windows=3, refit_every=3,
                                                              n_estimators=5), n_jobs=1, progress_every=0)
        write_outputs(result, tmp_path / "reports", total_series=small_matrix.n_series)
        per_series, pooled = da.metric_tables(result.predictions, result.scales)
        da.metrics_by_horizon(result.predictions)
        da.window_errors(result.predictions)
        da.horizon_labels(result.predictions)
        da.origin_weekdays(result.predictions)
        da.pooled_by_store(per_series)
        da.coverage_from_predictions(result.predictions)
        da.order_models(pooled)
        hist = _history()
        recommend_inventory(hist, current_stock=50, lead_time_days=7, service_level=0.95)
        da.zero_demand_caption(_sales({(1, 1): [0, 1, 2]}))
    ours = [w for w in caught
            if issubclass(w.category, cats) and str(ROOT) in str(Path(w.filename).resolve())]
    assert not ours, [f"{w.category.__name__}: {w.message} ({w.filename}:{w.lineno})" for w in ours]


# ------------------------------------------------ weekday API: no deprecated spelling

def test_no_project_file_uses_the_deprecated_dayofweek_attribute():
    """Static guard: covers DatetimeIndex, Timestamp and .dt usage in every project .py file,
    including code paths no test executes."""
    import ast
    offenders = []
    for path in ROOT.rglob("*.py"):
        if any(part in {".venv", "venv", "__pycache__", ".git"} for part in path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "dayofweek":
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, offenders


def test_weekday_values_are_monday_zero_in_features_and_inventory():
    from src.features.demand_features import build_origin_features
    from src.inventory.recommender import _weekday_means

    dates = pd.date_range("2020-01-06", periods=60, freq="D")      # 2020-01-06 is a Monday
    values = np.arange(60, dtype=float)
    frame = build_origin_features(values, dates, np.array([41]), horizon=7)
    expected = [dates[41 + h].day_name() for h in range(1, 8)]
    names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    assert [names[int(d)] for d in frame["dow"]] == expected
    assert frame["dow"].tolist() == [(41 + h) % 7 for h in range(1, 8)]

    # weekday profile: demand equals 10 + weekday number, so the profile must be exactly that
    idx = pd.date_range("2020-01-06", periods=70, freq="D")
    s = pd.Series([10.0 + d for d in idx.day_of_week], index=idx)
    assert list(_weekday_means(s)) == [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    # the day after a Sunday is a Monday -> first expected day uses the Monday mean
    from src.inventory.recommender import expected_demand_path
    path, method = expected_demand_path(s, 3, lookback_days=None)
    assert s.index[-1].day_name() == "Sunday"
    assert method == "weekday profile"
    assert list(path) == [10.0, 11.0, 12.0]
