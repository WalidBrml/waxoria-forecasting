"""Small reusable Streamlit components for the Waxoria dashboard."""

from __future__ import annotations

import html
import math

import pandas as pd
import streamlit as st

from app import theme


# ---------------------------------------------------------------- formatting

def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def fmt_int(x) -> str:
    return "n/a" if _missing(x) else f"{int(round(x)):,}"


def fmt_num(x, digits: int = 2) -> str:
    return "n/a" if _missing(x) else f"{x:,.{digits}f}"


def fmt_pct(x, digits: int = 1, signed: bool = False) -> str:
    if _missing(x):
        return "n/a"
    return f"{x * 100:+.{digits}f}%" if signed else f"{x * 100:.{digits}f}%"


# ---------------------------------------------------------------- components

def inject_css() -> None:
    st.markdown(theme.CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    sub = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f'<div class="wx-hero"><h1>{html.escape(title)}</h1>{sub}</div>',
        unsafe_allow_html=True,
    )


def section(title: str, caption: str = "") -> None:
    cap = f"<p>{html.escape(caption)}</p>" if caption else ""
    st.markdown(
        f'<div class="wx-section"><h3>{html.escape(title)}</h3>{cap}</div>',
        unsafe_allow_html=True,
    )


def kpi_card_html(label: str, value: str, sub: str = "", tone: str = "") -> str:
    sub_html = f'<div class="wx-kpi-sub">{html.escape(sub)}</div>' if sub else ""
    return (
        f'<div class="wx-kpi {tone}"><div class="wx-kpi-label">{html.escape(label)}</div>'
        f'<div class="wx-kpi-value">{html.escape(value)}</div>{sub_html}</div>'
    )


def kpi_row(cards: list[dict]) -> None:
    """cards: dicts with label, value and optional sub / tone."""
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        col.markdown(
            kpi_card_html(card["label"], card["value"], card.get("sub", ""), card.get("tone", "")),
            unsafe_allow_html=True,
        )


def note(text: str) -> None:
    st.markdown(f'<div class="wx-note">{html.escape(text)}</div>', unsafe_allow_html=True)


def badge(text: str, color: str) -> str:
    return f'<span class="wx-badge" style="background:{color}">{html.escape(text)}</span>'


def empty_state(title: str, detail: str, command: str | None = None) -> None:
    st.markdown(
        f'<div class="wx-empty"><h4>{html.escape(title)}</h4><p>{html.escape(detail)}</p></div>',
        unsafe_allow_html=True,
    )
    if command:
        st.code(command, language="powershell")


def download_df(label: str, df: pd.DataFrame, filename: str, key: str) -> None:
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def show_table(df: pd.DataFrame, **kwargs) -> None:
    st.dataframe(df, hide_index=True, width="stretch", **kwargs)


# ------------------------------------------------------- widget-state helpers

def init_state(key: str, value) -> None:
    """Set an initial value once; widgets then take no ``default``/``value``
    argument, which avoids Streamlit's 'default value but also set via
    Session State' warning."""
    st.session_state.setdefault(key, value)


def sanitize_choice(key: str, options: list) -> None:
    """Drop a stored single choice that is no longer among the options."""
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]


def sanitize_multi(key: str, options: list, fallback: list | None = None) -> None:
    """Keep only stored multi-select values that are still valid options."""
    if key in st.session_state:
        kept = [v for v in st.session_state[key] if v in options]
        st.session_state[key] = kept
    elif fallback is not None:
        st.session_state[key] = [v for v in fallback if v in options]
