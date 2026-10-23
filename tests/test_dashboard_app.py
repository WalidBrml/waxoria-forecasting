"""Runs the real Streamlit app headlessly (streamlit.testing.AppTest).

These tests execute every page of app/dashboard.py. Tests that need the real
project data are skipped automatically when the files are not present.
"""

import datetime as dt
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from app import data_access as da  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")
PAGES = [
    "Executive Overview",
    "Sales Intelligence",
    "Demand Forecasting",
    "Model Performance",
    "Inventory Intelligence",
    "Data & Methodology",
]
have_data = all(p.exists() for p in da.PATHS.values())
needs_data = pytest.mark.skipif(not have_data, reason="project data/report files not present")


def new_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def go(at: AppTest, page: str) -> AppTest:
    return at.sidebar.radio[0].set_value(page).run()


def text_of(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown)


@needs_data
@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_errors(page):
    at = new_app()
    go(at, page)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error


@needs_data
def test_inventory_result_persists_across_reruns_and_pages():
    at = new_app()
    go(at, "Inventory Intelligence")
    assert "inventory_result" not in at.session_state
    at.button(key="inv_calc").click().run()
    assert not at.exception
    first = at.session_state["inventory_result"]["rec"]
    assert "Order " in text_of(at) or "No order needed" in text_of(at)

    go(at, "Executive Overview")
    go(at, "Inventory Intelligence")
    assert at.session_state["inventory_result"]["rec"] == first       # survived a page round trip
    assert len(at.get("download_button")) >= 1                        # download still offered

    at.number_input(key="keep_inv_stock").set_value(9999).run()
    assert any("changed since" in w.value for w in at.warning)         # stale notice, result kept
    assert at.session_state["inventory_result"]["inputs"]["stock"] != 9999


@needs_data
def test_forecasting_selection_and_filters():
    at = new_app()
    go(at, "Demand Forecasting")
    at.selectbox(key="keep_fc_store").set_value(7).run()
    at.selectbox(key="keep_fc_item").set_value(33).run()
    at.multiselect(key="keep_fc_models").set_value(list(da.MODEL_ORDER)).run()
    assert not at.exception and len(at.get("plotly_chart")) == 2

    at.multiselect(key="keep_fc_models").set_value([]).run()
    assert not at.exception and len(at.get("plotly_chart")) == 0       # asks for a model, no crash

    at.multiselect(key="keep_fc_models").set_value(["Random Forest"]).run()
    at.sidebar.date_input(key="keep_dates").set_value((dt.date(2014, 1, 1), dt.date(2014, 12, 31))).run()
    assert not at.exception
    assert "No predictions in the selected date range" in text_of(at)  # graceful empty state


@needs_data
def test_store_filter_empty_selection_is_handled_on_every_page():
    at = new_app()
    at.sidebar.multiselect(key="keep_stores").set_value([]).run()
    for page in PAGES:
        go(at, page)
        assert not at.exception, (page, [e.value for e in at.exception])


def test_missing_files_degrade_gracefully(monkeypatch, tmp_path):
    for key in da.PATHS:
        monkeypatch.setitem(da.PATHS, key, tmp_path / f"missing_{key}")
    at = new_app()
    assert not at.exception
    for page in PAGES:
        go(at, page)
        assert not at.exception, (page, [e.value for e in at.exception])
    go(at, "Demand Forecasting")
    assert "Predictions not found" in text_of(at)
    go(at, "Data & Methodology")
    assert any("missing or invalid" in w.value for w in at.warning)


@needs_data
def test_predictions_missing_only_affects_forecast_pages(monkeypatch, tmp_path):
    monkeypatch.setitem(da.PATHS, "predictions", tmp_path / "gone.csv")
    at = new_app()
    go(at, "Executive Overview")
    assert not at.exception and "Predictions not found" in text_of(at)
    go(at, "Sales Intelligence")
    assert not at.exception and len(at.get("plotly_chart")) >= 4         # sales pages unaffected
    go(at, "Inventory Intelligence")
    assert not at.exception
