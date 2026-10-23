"""Visual identity for the Waxoria dashboard: restrained navy and blue."""

from __future__ import annotations

import plotly.graph_objects as go

NAVY = "#0B1F3A"
NAVY_2 = "#14335E"
BLUE = "#2F6FDE"
BLUE_MID = "#5FA8D3"
BLUE_LIGHT = "#8DB4F0"
SKY = "#E6EEFB"
INK = "#14233B"
MUTED = "#5B6B82"
GRID = "#E4E9F2"
BG = "#F5F7FB"
CARD = "#FFFFFF"
BORDER = "#DCE3EE"
SLATE = "#7C8DA6"
SILVER = "#9AA7BC"

GOOD = "#0F7B4F"
WARN = "#B45309"
BAD = "#B42318"

ACTUAL_COLOR = NAVY
MODEL_COLORS = {
    "Random Forest": BLUE,
    "Seasonal naive": BLUE_MID,
    "7-day moving average": SLATE,
    "Previous-day baseline": SILVER,
}
MODEL_DASH = {
    "Random Forest": "solid",
    "Seasonal naive": "dash",
    "7-day moving average": "dot",
    "Previous-day baseline": "dashdot",
}
FONT = "Inter, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

RISK_COLORS = {"High": BAD, "Medium": WARN, "Low": GOOD}


def wrap_model(name: str) -> str:
    """Two-line axis label so long model names stay horizontal."""
    return {
        "Random Forest": "Random<br>Forest",
        "Seasonal naive": "Seasonal<br>naive",
        "7-day moving average": "7-day moving<br>average",
        "Previous-day baseline": "Previous-day<br>baseline",
    }.get(name, name)


def model_color(model: str) -> str:
    return MODEL_COLORS.get(model, BLUE_LIGHT)


def style_fig(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    """Apply the shared Plotly look to every chart."""
    fig.update_layout(
        title={"text": title, "x": 0.0, "xanchor": "left",
               "font": {"size": 15, "color": NAVY}} if title else None,
        font={"family": FONT, "size": 12, "color": INK},
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        height=height,
        margin={"l": 48, "r": 24, "t": 92 if title else 40, "b": 44},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0,
                "title": {"text": ""}, "font": {"size": 11}},
        hoverlabel={"bgcolor": NAVY, "font": {"color": "white", "family": FONT}},
        colorway=[BLUE, NAVY_2, BLUE_MID, SLATE, BLUE_LIGHT, SILVER],
    )
    fig.update_xaxes(showgrid=False, linecolor=BORDER, ticks="outside",
                     tickcolor=BORDER, title_font={"size": 12, "color": MUTED})
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=BORDER,
                     title_font={"size": 12, "color": MUTED})
    return fig


CSS = f"""
<style>
.block-container {{ padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1320px; }}
footer {{ visibility: hidden; }}
h1, h2, h3, h4 {{ color: {NAVY}; letter-spacing: -0.01em; }}
section[data-testid="stSidebar"] {{ background: {SKY}; border-right: 1px solid {BORDER}; }}
.wx-brand {{ font-size: 1.45rem; font-weight: 800; color: {NAVY}; letter-spacing: -0.02em; line-height: 1.1; }}
.wx-brand span {{ color: {BLUE}; }}
.wx-tag {{ font-size: .74rem; color: {MUTED}; margin: .1rem 0 .9rem 0; }}
.wx-hero {{ background: linear-gradient(135deg, {NAVY} 0%, {NAVY_2} 100%); border-radius: 12px;
  padding: 1.1rem 1.4rem; margin-bottom: 1.1rem; }}
.wx-hero h1 {{ color: #FFFFFF; font-size: 1.55rem; margin: 0; padding: 0; font-weight: 700; }}
.wx-hero p {{ color: #C7D6F0; margin: .25rem 0 0 0; font-size: .92rem; }}
.wx-kpi {{ background: {CARD}; border: 1px solid {BORDER}; border-left: 4px solid {BLUE};
  border-radius: 10px; padding: .85rem 1rem; min-height: 112px; box-sizing: border-box; }}
.wx-kpi.good {{ border-left-color: {GOOD}; }} .wx-kpi.warn {{ border-left-color: {WARN}; }}
.wx-kpi.bad {{ border-left-color: {BAD}; }} .wx-kpi.navy {{ border-left-color: {NAVY}; }}
.wx-kpi-label {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; color: {MUTED}; font-weight: 600; }}
.wx-kpi-value {{ font-size: 1.65rem; font-weight: 750; color: {NAVY}; line-height: 1.25; margin-top: .15rem; }}
.wx-kpi-sub {{ font-size: .78rem; color: {MUTED}; margin-top: .1rem; }}
.wx-note {{ background: {SKY}; border: 1px solid {BORDER}; border-radius: 10px; padding: .75rem 1rem;
  color: {INK}; font-size: .88rem; margin: .4rem 0 .8rem 0; }}
.wx-empty {{ background: {CARD}; border: 1px dashed {SILVER}; border-radius: 12px; padding: 1.4rem 1.2rem;
  text-align: center; margin: 1rem 0; }}
.wx-empty h4 {{ margin: 0 0 .3rem 0; }} .wx-empty p {{ color: {MUTED}; margin: 0; font-size: .9rem; }}
.wx-badge {{ display: inline-block; padding: .15rem .6rem; border-radius: 999px; font-size: .78rem;
  font-weight: 650; color: #fff; }}
.wx-section {{ margin: 1.3rem 0 .3rem 0; }}
.wx-section h3 {{ margin: 0; font-size: 1.12rem; }} .wx-section p {{ margin: .1rem 0 0 0; color: {MUTED}; font-size: .85rem; }}
div[data-testid="stPlotlyChart"] {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; padding: .25rem; }}
</style>
"""
