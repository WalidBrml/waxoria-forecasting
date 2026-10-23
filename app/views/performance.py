"""Page 4: Model Performance."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import data_access as da
from app import theme, ui

METRIC_HELP = {
    "MAE": "Mean absolute error in units per day (lower is better).",
    "RMSE": "Root mean squared error; penalises large misses more than MAE.",
    "WAPE": "Σ|error| ÷ Σ actual over all observations (ratio of sums, not a mean of ratios).",
    "MASE": "MAE ÷ in-sample seasonal-naive (lag-7) MAE. Below 1 beats that benchmark.",
    "Bias": "Σ(predicted − actual) ÷ Σ actual. Negative = under-forecasting.",
}


@st.cache_data(show_spinner=False)
def _tables(_pred: pd.DataFrame, _scales, key: tuple):
    """Recompute metrics for a filter; ``key`` is the cache key."""
    return da.metric_tables(_pred, _scales)


def render(data: da.AppData, f: da.Filters) -> None:
    ui.page_header(
        "Model Performance",
        "How the four models compare on the same series, dates and horizons",
    )
    if not data.has_predictions:
        ui.empty_state("Predictions not found", data.status["predictions"].message,
                       da.HOW_TO_CREATE["predictions"])
        return

    full_cov = da.coverage_from_predictions(data.predictions)
    pred_f = da.filter_predictions(data.predictions, f)
    if pred_f.empty:
        ui.empty_state(
            "No evaluated predictions in the selected filters",
            f"Evaluation covers {full_cov['test_start']} to {full_cov['test_end']}. "
            "Include that period and at least one store in the sidebar.",
        )
        return

    key = (f.start, f.end, f.stores, da.file_signature())
    per_series, pooled = _tables(pred_f, data.scales, key)
    pooled = da.order_models(pooled)
    per_series = da.order_models(per_series)
    cov = da.coverage_from_predictions(pred_f)
    filtered = cov != full_cov

    # ------------------------------------------------------------- coverage
    ui.section("Evaluation coverage",
               "Filtered view" if filtered else "Full evaluation, no filters narrowing it")
    ui.kpi_row(
        [
            {"label": "Stores", "value": f"{cov['stores']}"},
            {"label": "Products", "value": f"{cov['products']}"},
            {"label": "Series", "value": f"{cov['series']:,}", "tone": "navy"},
            {"label": "Forecast windows", "value": f"{cov['forecast_windows']}"},
            {"label": "Observations per model", "value": f"{cov['observations_per_model']:,}",
             "sub": f"{cov['test_start']} to {cov['test_end']}"},
        ]
    )
    mism = da.report_mismatches(data.report, full_cov)
    if data.report is None:
        st.caption("evaluation_report.json not found, so coverage is computed from the predictions file only.")
    elif mism:
        st.warning("evaluation_report.json disagrees with the predictions file: " + "; ".join(mism))
    else:
        st.caption(
            f"Matches evaluation_report.json: {full_cov['series']} series, "
            f"{full_cov['forecast_windows']} windows, {full_cov['observations_per_model']:,} observations per model. "
            + ("Metrics below are recomputed for the current filters." if filtered else "")
        )
    if data.report:
        st.markdown(f"**Forecast type.** {data.report.get('forecast_type', '')}")

    # -------------------------------------------------------------- table
    ui.section("Metrics by model", "Pooled over every observation unless noted; best value highlighted")
    cols = ["model", "MAE", "RMSE", "WAPE", "Bias", "MASE_mean", "MASE_median", "series_best"]
    table = pooled[cols].rename(columns={"model": "Model", "MASE_mean": "MASE (mean of series)",
                                         "MASE_median": "MASE (median)",
                                         "series_best": "Series won (lowest MAE)"})
    fmt = {"MAE": "{:.3f}", "RMSE": "{:.3f}", "WAPE": "{:.1%}", "Bias": "{:+.1%}",
           "MASE (mean of series)": "{:.3f}", "MASE (median)": "{:.3f}"}
    styled = (table.style.format(fmt)
              .highlight_min(subset=["MAE", "RMSE", "WAPE", "MASE (mean of series)", "MASE (median)"],
                             color=theme.SKY)
              .highlight_max(subset=["Series won (lowest MAE)"], color=theme.SKY))
    st.dataframe(styled, hide_index=True, width="stretch")
    origin_days = da.origin_weekdays(pred_f)
    if len(origin_days) == 1 and "Previous-day baseline" in set(pooled["model"]):
        ui.note(
            f"Reading the baselines: every forecast is issued on a {origin_days[0]}. The Previous-day baseline holds "
            f"that day's sales for all 7 forecast days, so it is a weak benchmark here (see its bias). Seasonal naive "
            "and the 7-day moving average are the fairer comparisons."
        )
    with st.expander("Metric definitions"):
        for name, text in METRIC_HELP.items():
            st.markdown(f"- **{name}**: {text}")
        undefined_w = int((pooled["series_wape_defined"] < pooled["n_series"]).sum())
        undefined_m = int((pooled["series_mase_defined"] < pooled["n_series"]).sum())
        st.markdown(
            f"- Zero-demand edge cases: WAPE/Bias are undefined (shown as n/a) when a series has no demand in the "
            f"window, and MASE is undefined when its seasonal-naive scale is 0. In this view, models with "
            f"undefined WAPE: {undefined_w}; with undefined MASE: {undefined_m}."
        )

    # --------------------------------------------------------------- charts
    ui.section("Accuracy comparison")
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="MAE", x=[theme.wrap_model(m) for m in pooled["model"]], y=pooled["MAE"], marker_color=theme.BLUE,
                             text=[f"{v:.2f}" for v in pooled["MAE"]], textposition="outside"))
        fig.add_trace(go.Bar(name="RMSE", x=[theme.wrap_model(m) for m in pooled["model"]], y=pooled["RMSE"], marker_color=theme.NAVY,
                             text=[f"{v:.2f}" for v in pooled["RMSE"]], textposition="outside"))
        fig.update_layout(barmode="group")
        fig.update_yaxes(title="Units per day")
        st.plotly_chart(theme.style_fig(fig, "MAE and RMSE (lower is better)", 360),
                        width="stretch", key="perf_mae")
    with b:
        fig = go.Figure(go.Bar(
            x=[theme.wrap_model(m) for m in pooled["model"]], y=pooled["WAPE"] * 100,
            marker_color=[theme.model_color(m) for m in pooled["model"]],
            text=[f"{v:.1%}" for v in pooled["WAPE"]], textposition="outside"))
        fig.update_yaxes(title="WAPE (%)")
        st.plotly_chart(theme.style_fig(fig, "Pooled WAPE (lower is better)", 360),
                        width="stretch", key="perf_wape")

    a, b = st.columns(2)
    with a:
        fig = go.Figure(go.Bar(
            x=[theme.wrap_model(m) for m in pooled["model"]], y=pooled["MASE_mean"],
            marker_color=[theme.model_color(m) for m in pooled["model"]],
            text=[ui.fmt_num(v, 2) for v in pooled["MASE_mean"]], textposition="outside"))
        fig.add_hline(y=1.0, line_dash="dash", line_color=theme.NAVY)
        fig.update_yaxes(title="MASE (mean over series)")
        st.plotly_chart(theme.style_fig(fig, "MASE (dashed line = 1.0 benchmark)", 340),
                        width="stretch", key="perf_mase")
    with b:
        colors = [theme.BAD if v > 0 else theme.BLUE for v in pooled["Bias"].fillna(0)]
        fig = go.Figure(go.Bar(x=[theme.wrap_model(m) for m in pooled["model"]], y=pooled["Bias"] * 100, marker_color=colors,
                               text=[ui.fmt_pct(v, signed=True) for v in pooled["Bias"]],
                               textposition="outside"))
        fig.add_hline(y=0, line_color=theme.NAVY, line_width=1)
        fig.update_yaxes(title="Bias (%)")
        st.plotly_chart(theme.style_fig(fig, "Bias (red = over-forecast, blue = under-forecast)", 340),
                        width="stretch", key="perf_bias")

    # -------------------------------------------------------------- horizon
    ui.section("Error by forecast horizon", "MAE by number of days ahead of the forecast origin")
    hz = da.metrics_by_horizon(pred_f)
    labels = da.horizon_labels(pred_f)
    origin_days = da.origin_weekdays(pred_f)
    if len(origin_days) == 1:
        ui.note(
            f"Every forecast origin in this backtest is a {origin_days[0]}, so horizon and target weekday are the "
            "same thing (axis labels show both). Differences across horizons therefore mix forecast distance with "
            "weekday demand patterns and should not be read as pure error growth with distance."
        )
    fig = go.Figure()
    for m in pooled["model"]:
        g = hz[hz["model"] == m].sort_values("horizon")
        fig.add_trace(go.Scatter(x=g["horizon"], y=g["MAE"], mode="lines+markers", name=m,
                                 line={"color": theme.model_color(m), "dash": theme.MODEL_DASH.get(m, "solid"),
                                       "width": 2.2}))
    fig.update_xaxes(title="Days ahead (horizon) · target weekday", tickmode="array",
                     tickvals=list(labels), ticktext=list(labels.values()))
    fig.update_yaxes(title="MAE (units per day)")
    st.plotly_chart(theme.style_fig(fig, None, 340), width="stretch", key="perf_horizon")

    # ------------------------------------------------------ distribution/store
    ui.section("Where the models differ")
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        for m in pooled["model"]:
            vals = per_series.loc[per_series["model"] == m, "MASE"].dropna()
            fig.add_trace(go.Box(y=vals, name=theme.wrap_model(m), marker_color=theme.model_color(m), boxpoints=False))
        fig.add_hline(y=1.0, line_dash="dash", line_color=theme.NAVY)
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title="Per-series MASE")
        st.plotly_chart(theme.style_fig(fig, "Spread of per-series MASE", 380),
                        width="stretch", key="perf_box")
    with b:
        by_store = da.pooled_by_store(per_series)
        fig = go.Figure()
        for m in pooled["model"]:
            g = by_store[by_store["model"] == m].sort_values("store")
            fig.add_trace(go.Bar(x=[f"S{s}" for s in g["store"]], y=g["WAPE"] * 100, name=m,
                                 marker_color=theme.model_color(m)))
        fig.update_layout(barmode="group")
        fig.update_yaxes(title="WAPE (%)")
        st.plotly_chart(theme.style_fig(fig, "Pooled WAPE by store", 380),
                        width="stretch", key="perf_store")

    # ---------------------------------------------------------- hardest series
    ui.section("Hardest series", "Where the selected model's error is largest relative to demand")
    pick = st.selectbox("Model", list(pooled["model"]), key="keep_perf_model")
    hard = (per_series[per_series["model"] == pick]
            .sort_values("WAPE", ascending=False)
            .head(15)[["store", "item", "MAE", "RMSE", "WAPE", "MASE", "Bias", "n_obs"]])
    ui.show_table(hard, column_config={
        "MAE": st.column_config.NumberColumn(format="%.2f"),
        "RMSE": st.column_config.NumberColumn(format="%.2f"),
        "WAPE": st.column_config.NumberColumn(format="%.3f"),
        "MASE": st.column_config.NumberColumn(format="%.3f"),
        "Bias": st.column_config.NumberColumn(format="%.3f"),
    })

    # ------------------------------------------------------------ downloads
    ui.section("Downloads", "Exports reflect the current filters")
    d1, d2 = st.columns(2)
    with d1:
        ui.download_df("Model summary (CSV)", pooled, "waxoria_model_summary_filtered.csv", "dl_perf_summary")
    with d2:
        out = per_series.drop(columns=[c for c in per_series.columns if c.startswith("sum_")])
        ui.download_df("Per-series metrics (CSV)", out, "waxoria_series_metrics_filtered.csv", "dl_perf_series")
    if st.checkbox(f"Prepare predictions download ({len(pred_f):,} rows)", key="keep_perf_prepare"):
        ui.download_df("Predictions, all models (CSV)", pred_f.sort_values(["store", "item", "date", "model"]),
                       "waxoria_predictions_filtered.csv", "dl_perf_pred")
