# Waxoria — Demand Forecasting & Inventory Intelligence

🚀 **[Launch the Live Dashboard](https://waxoria-forecasting.streamlit.app)**

Waxoria is a Python-based retail analytics project combining demand forecasting, sales analytics, model evaluation, and inventory recommendations in an interactive Streamlit dashboard.

## Dashboard Features

- **Executive Overview:** Sales KPIs and forecasting performance.
- **Sales Intelligence:** Sales trends, seasonality, and store/product rankings.
- **Demand Forecasting:** Compare actual demand with multiple forecasting models.
- **Model Performance:** Evaluate MAE, RMSE, WAPE, bias, and forecast-horizon errors.
- **Inventory Intelligence:** Calculate reorder points, safety stock, target stock, and order quantities.
- **Data & Methodology:** Dataset coverage, evaluation methodology, metrics, and limitations.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Plotly
- Pytest

## Project Structure

- `app/` — Dashboard and application pages
- `data/` — Input and processed datasets
- `reports/` — Forecasting outputs and evaluation results
- `src/` — Forecasting, feature engineering, metrics, and inventory logic
- `tests/` — Automated tests

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the Dashboard Locally

```powershell
python -m streamlit run app/dashboard.py
```

## Run Tests

```powershell
python -m pytest -q
```

## Limitations

Waxoria is a portfolio prototype. Forecast performance depends on the dataset and evaluation period. Inventory recommendations depend on assumptions about stock levels, lead times, demand variability, and service levels.

Validate results with real operational data before using recommendations for purchasing decisions.

## Author

**Walid** — Data Science and Machine Learning portfolio project.
