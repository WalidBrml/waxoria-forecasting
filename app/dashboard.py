"""Waxoria dashboard entry point.

Run from the project root:
    python -m streamlit run app/dashboard.py

Six pages share one sidebar (navigation + date and store filters). All data
comes from the saved project files; pages degrade gracefully if a file is
missing. Page code lives in app/views/, data access in app/data_access.py.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Waxoria | Inventory Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app import data_access as da  # noqa: E402
from app import ui  # noqa: E402
from app.views import (  # noqa: E402
    forecasting,
    inventory,
    methodology,
    overview,
    performance,
    sales,
)

PAGES = {
    "Executive Overview": overview,
    "Sales Intelligence": sales,
    "Demand Forecasting": forecasting,
    "Model Performance": performance,
    "Inventory Intelligence": inventory,
    "Data & Methodology": methodology,
}


@st.cache_resource(show_spinner="Loading Waxoria data...")
def get_data(signature: tuple) -> da.AppData:
    """Loaded once per file change; treated as read-only by every page."""
    return da.load_all()


def keep_widget_state() -> None:
    """Streamlit drops the state of widgets that were not drawn in a run, so
    inputs would reset when the user switches page. Re-assigning every
    ``keep_*`` key on each run preserves them."""
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith("keep_"):
            st.session_state[key] = st.session_state[key]


def sidebar(data: da.AppData) -> tuple[str, da.Filters]:
    st.sidebar.markdown(
        '<div class="wx-brand">Wax<span>oria</span></div>'
        '<div class="wx-tag">Intelligent inventory &amp; demand forecasting</div>',
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "Navigate", list(PAGES), key="keep_page", label_visibility="collapsed"
    )
    st.sidebar.divider()
    st.sidebar.markdown("**Filters**")

    defaults = da.default_filters(data)
    if not data.has_sales:
        st.sidebar.caption("Filters unavailable: clean sales data not found.")
        return page, defaults

    all_stores = list(defaults.stores)
    first, last = defaults.start.date(), defaults.end.date()

    def reset_filters():
        st.session_state["keep_dates"] = (first, last)
        st.session_state["keep_stores"] = all_stores

    ui.init_state("keep_dates", (first, last))
    ui.sanitize_multi("keep_stores", all_stores, fallback=all_stores)
    dates = st.sidebar.date_input(
        "Date range", min_value=first, max_value=last, key="keep_dates"
    )
    stores = st.sidebar.multiselect(
        "Stores", all_stores, format_func=lambda s: f"Store {s}", key="keep_stores"
    )
    st.sidebar.button("Reset filters", on_click=reset_filters, width="stretch")

    if isinstance(dates, (tuple, list)) and len(dates) == 2:
        start, end = pd.Timestamp(dates[0]), pd.Timestamp(dates[1])
    else:
        start, end = defaults.start, defaults.end
        st.sidebar.caption("Pick an end date to apply the range.")
    if not stores:
        st.sidebar.warning("Select at least one store.")

    ready = sum(s.ok for s in data.status.values())
    st.sidebar.divider()
    st.sidebar.caption(f"Data files ready: {ready} of {len(data.status)}")
    return page, da.Filters(start, end, tuple(stores))


def main() -> None:
    keep_widget_state()
    ui.inject_css()
    data = get_data(da.file_signature())
    page, filters = sidebar(data)
    PAGES[page].render(data, filters)
    st.caption(
        "Waxoria is a portfolio prototype. Forecasts and recommendations are "
        "experimental decision support and need validation before operational use."
    )


main()
