
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = ROOT / "reports" / "forecast_predictions.csv"
OUTPUT = ROOT / "reports" / "figures" / "forecast_comparison.html"


def main() -> None:
    if not PREDICTIONS.exists():
        raise FileNotFoundError(
            "Run the baseline forecasting script first."
        )

    df = pd.read_csv(PREDICTIONS, parse_dates=["date"])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["actual_sales"],
            mode="lines",
            name="Actual sales",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["baseline_prediction"],
            mode="lines",
            name="Previous-day baseline",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["moving_average_prediction"],
            mode="lines",
            name="7-day moving average",
        )
    )

    fig.update_layout(
        title="Waxoria | Demand Forecast Comparison",
        xaxis_title="Date",
        yaxis_title="Units sold",
        template="plotly_white",
        hovermode="x unified",
        legend_title="Series",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUTPUT)

    print(f"Forecast chart saved to: {OUTPUT}")
    print("Open the HTML file in your browser to inspect the chart.")


if __name__ == "__main__":
    main()
