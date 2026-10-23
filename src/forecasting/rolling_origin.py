"""Chronological rolling-origin evaluation for Waxoria.

Design
------
* Horizon H (default 7): at each forecast origin T the models predict the
  next H days at once ("direct multi-horizon"). This is a multi-day forecast,
  not one-day-ahead; with ``horizon=1`` it reduces to one-day-ahead.
* The last ``n_windows * H`` days of the data are the test period, cut into
  consecutive, non-overlapping windows. Window j has origin
  ``T_j = last_day - (n_windows - j) * H`` and targets ``T_j+1 .. T_j+H``.
* Models are refit every ``refit_every`` windows on an expanding history.
  A fit group's cutoff is the origin of its first window; training uses only
  origins ``o`` with ``o + H <= cutoff`` so every training TARGET is <= cutoff.
  Test targets are always > cutoff. Within a group, windows use the same
  fitted model but features recomputed from the newest *observed* history,
  exactly as a deployed system retrained monthly and forecasting weekly.
* All four models (3 baselines + Random Forest) are scored on identical
  (series, date, horizon) rows because they share one feature table.
* Fitted models are discarded immediately; only predictions are kept.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.features.demand_features import (
    BASELINE_COLUMNS,
    FEATURE_COLUMNS,
    build_origin_features,
    min_origin,
)
from src.forecasting.metrics import mase_scale

RF_NAME = "Random Forest"
MODEL_NAMES = [*BASELINE_COLUMNS.keys(), RF_NAME]

PREDICTION_COLUMNS = [
    "store",
    "item",
    "forecast_window",
    "forecast_origin",
    "horizon",
    "date",
    "actual_sales",
    "model",
    "predicted_sales",
]


@dataclass(frozen=True)
class EvaluationConfig:
    horizon: int = 7
    n_windows: int = 12
    refit_every: int = 4
    n_estimators: int = 100
    max_depth: int = 8
    min_samples_leaf: int = 3
    random_state: int = 42
    check_leakage: bool = True


@dataclass(frozen=True)
class Window:
    window_id: int
    origin_pos: int
    fit_cutoff_pos: int
    first_target_pos: int
    last_target_pos: int


@dataclass
class SalesMatrix:
    """Dense, validated panel: one row per series, one column per day."""

    keys: pd.DataFrame            # columns: store, item (row i <-> values[i])
    dates: pd.DatetimeIndex
    values: np.ndarray            # shape (n_series, n_days), float64

    @property
    def n_series(self) -> int:
        return self.values.shape[0]

    def subset(self, stores=None, items=None) -> "SalesMatrix":
        mask = np.ones(len(self.keys), dtype=bool)
        if stores is not None:
            mask &= self.keys["store"].isin(list(stores)).to_numpy()
        if items is not None:
            mask &= self.keys["item"].isin(list(items)).to_numpy()
        if not mask.any():
            raise ValueError("The store/item filter selected no series.")
        return SalesMatrix(
            self.keys[mask].reset_index(drop=True), self.dates, self.values[mask]
        )


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_sales_matrix(path: Path) -> SalesMatrix:
    """Read the clean CSV once into a (series x day) matrix.

    Raises instead of silently zero-filling when a series has missing days:
    imputing 0 would invent demand data and bias every metric.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Clean data not found: {path}\nRun: python -m src.data.validate_data"
        )
    df = pd.read_csv(
        path,
        usecols=["date", "store", "item", "sales"],
        dtype={"store": "int16", "item": "int16", "sales": "float64"},
        parse_dates=["date"],
    )
    return matrix_from_frame(df)


def matrix_from_frame(df: pd.DataFrame) -> SalesMatrix:
    df = df.sort_values(["store", "item", "date"], kind="stable")
    dates = pd.DatetimeIndex(np.sort(df["date"].unique()))
    n_days = len(dates)
    if n_days > 1 and (np.diff(dates) != np.timedelta64(1, "D")).any():
        raise ValueError("The calendar has gaps; expected consecutive daily dates.")

    sizes = df.groupby(["store", "item"], sort=True).size()
    if (sizes != n_days).any():
        bad = sizes[sizes != n_days]
        raise ValueError(
            f"{len(bad)} series do not have all {n_days} days "
            "(missing dates). Fix the data instead of zero-filling."
        )
    keys = sizes.index.to_frame(index=False)
    values = df["sales"].to_numpy(dtype=float).reshape(len(keys), n_days)
    # Verify the reshape really lines up with the calendar for every series.
    reshaped_dates = df["date"].to_numpy().reshape(len(keys), n_days)
    if not (reshaped_dates == dates.to_numpy()[None, :]).all():
        raise ValueError("Series are not aligned on a common daily calendar.")
    return SalesMatrix(keys=keys, dates=dates, values=values)


# --------------------------------------------------------------------------
# Windows and chronological splits
# --------------------------------------------------------------------------

def make_windows(
    n_days: int, horizon: int, n_windows: int, refit_every: int
) -> list[Window]:
    if min(horizon, n_windows, refit_every) < 1:
        raise ValueError("horizon, n_windows and refit_every must be >= 1")
    first_origin = n_days - 1 - n_windows * horizon
    # The first fit needs history for the features plus at least one
    # training origin whose labels end at or before the first cutoff.
    if first_origin - horizon < min_origin(horizon):
        raise ValueError("Not enough history before the test period.")

    windows = []
    for j in range(n_windows):
        origin = first_origin + j * horizon
        group_start = (j // refit_every) * refit_every
        cutoff = first_origin + group_start * horizon
        windows.append(
            Window(
                window_id=j + 1,
                origin_pos=origin,
                fit_cutoff_pos=cutoff,
                first_target_pos=origin + 1,
                last_target_pos=origin + horizon,
            )
        )
    return windows


def training_origins(cutoff_pos: int, horizon: int) -> np.ndarray:
    """Origins whose entire label span ends at or before ``cutoff_pos``."""
    start = min_origin(horizon)
    stop = cutoff_pos - horizon          # inclusive: o + H <= cutoff
    if stop < start:
        raise ValueError("No training origins before the cutoff.")
    return np.arange(start, stop + 1)


def assert_chronological(windows: list[Window], horizon: int) -> None:
    """Raise if any split could leak future information into training."""
    for w in windows:
        train_last_target = training_origins(w.fit_cutoff_pos, horizon)[-1] + horizon
        if train_last_target > w.fit_cutoff_pos:
            raise AssertionError("training label extends past the cutoff")
        if w.fit_cutoff_pos > w.origin_pos:
            raise AssertionError("fit cutoff is after the forecast origin")
        if w.first_target_pos <= w.fit_cutoff_pos:
            raise AssertionError("test target is not after the fit cutoff")
    for a, b in zip(windows, windows[1:]):
        if b.first_target_pos != a.last_target_pos + 1:
            raise AssertionError("windows must be consecutive and non-overlapping")


# --------------------------------------------------------------------------
# Leakage self-check
# --------------------------------------------------------------------------

def check_features_ignore_future(
    values: np.ndarray, dates: pd.DatetimeIndex, windows: list[Window], horizon: int
) -> None:
    """Features at each origin must not change if later values are replaced."""
    rng = np.random.default_rng(0)
    for w in windows:
        corrupted = values.copy()
        corrupted[w.origin_pos + 1:] = rng.uniform(1e6, 2e6, len(corrupted) - w.origin_pos - 1)
        origin = np.array([w.origin_pos])
        a = build_origin_features(values, dates, origin, horizon)
        b = build_origin_features(corrupted, dates, origin, horizon)
        if not np.array_equal(a[FEATURE_COLUMNS].to_numpy(), b[FEATURE_COLUMNS].to_numpy()):
            raise AssertionError(
                f"Leakage: features at origin {w.origin_pos} depend on future data"
            )


# --------------------------------------------------------------------------
# One series
# --------------------------------------------------------------------------

def evaluate_series(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    windows: list[Window],
    config: EvaluationConfig,
) -> pd.DataFrame:
    """Predict every window for one series with all four models.

    Returns a long table with columns
    ``forecast_window, forecast_origin, horizon, date, actual_sales, model,
    predicted_sales`` (store/item are added by the caller).
    """
    values = np.asarray(values, dtype=float)
    H = config.horizon
    if config.check_leakage:
        check_features_ignore_future(values, dates, windows, H)

    frames = []
    cutoffs = sorted({w.fit_cutoff_pos for w in windows})
    for cutoff in cutoffs:
        group = [w for w in windows if w.fit_cutoff_pos == cutoff]

        train_o = training_origins(cutoff, H)
        train = build_origin_features(values, dates, train_o, H)
        if train["target_pos"].max() > cutoff:
            raise AssertionError("training target after cutoff")
        y_train = values[train["target_pos"].to_numpy()]

        model = RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            random_state=config.random_state,
            n_jobs=1,
        )
        model.fit(train[FEATURE_COLUMNS].to_numpy(), y_train)

        test_o = np.array([w.origin_pos for w in group])
        test = build_origin_features(values, dates, test_o, H)
        rf_pred = model.predict(test[FEATURE_COLUMNS].to_numpy())
        del model, train  # never retain fitted models

        window_by_origin = {w.origin_pos: w for w in group}
        base = pd.DataFrame(
            {
                "forecast_window": test["origin_pos"].map(
                    lambda o: window_by_origin[o].window_id
                ).to_numpy(),
                "forecast_origin": dates[test["origin_pos"].to_numpy()],
                "horizon": test["horizon"].to_numpy().astype(int),
                "date": dates[test["target_pos"].to_numpy()],
                "actual_sales": values[test["target_pos"].to_numpy()],
            }
        )
        preds = {name: test[col].to_numpy() for name, col in BASELINE_COLUMNS.items()}
        preds[RF_NAME] = rf_pred
        for name, p in preds.items():
            frames.append(base.assign(model=name, predicted_sales=p))

    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# All series
# --------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    predictions: pd.DataFrame
    scales: pd.DataFrame
    windows: list[Window]
    config: EvaluationConfig
    seconds: float = 0.0
    notes: dict = field(default_factory=dict)


def _one(i, values, dates, windows, config):
    return i, evaluate_series(values, dates, windows, config)


def run_evaluation(
    matrix: SalesMatrix,
    config: EvaluationConfig = EvaluationConfig(),
    n_jobs: int = 1,
    progress_every: int = 25,
) -> EvaluationResult:
    """Evaluate every series in ``matrix`` (nothing is skipped)."""
    from joblib import Parallel, delayed

    start = time.time()
    n_days = matrix.values.shape[1]
    windows = make_windows(n_days, config.horizon, config.n_windows, config.refit_every)
    assert_chronological(windows, config.horizon)

    first_origin = windows[0].origin_pos
    scales = matrix.keys.copy()
    scales["mase_scale"] = [
        mase_scale(row[: first_origin + 1]) for row in matrix.values
    ]

    tasks = (
        delayed(_one)(i, matrix.values[i], matrix.dates, windows, config)
        for i in range(matrix.n_series)
    )
    results: list[pd.DataFrame | None] = [None] * matrix.n_series
    done = 0
    for i, frame in Parallel(n_jobs=n_jobs, return_as="generator_unordered")(tasks):
        key = matrix.keys.iloc[i]
        frame.insert(0, "item", int(key["item"]))
        frame.insert(0, "store", int(key["store"]))
        results[i] = frame
        done += 1
        if progress_every and (done % progress_every == 0 or done == matrix.n_series):
            print(f"  evaluated {done}/{matrix.n_series} series "
                  f"({time.time() - start:.0f}s)", flush=True)

    predictions = pd.concat(results, ignore_index=True)
    predictions["model"] = predictions["model"].astype("category")
    predictions = predictions[PREDICTION_COLUMNS].sort_values(
        ["store", "item", "forecast_window", "horizon", "model"], kind="stable"
    ).reset_index(drop=True)

    return EvaluationResult(
        predictions=predictions,
        scales=scales,
        windows=windows,
        config=config,
        seconds=time.time() - start,
    )


# --------------------------------------------------------------------------
# Output validation
# --------------------------------------------------------------------------

def validate_predictions(
    predictions: pd.DataFrame, expected_models: list[str] = MODEL_NAMES
) -> dict:
    """Check schema and that every model was scored on identical rows."""
    if list(predictions.columns) != PREDICTION_COLUMNS:
        raise ValueError(f"Unexpected columns: {list(predictions.columns)}")
    if predictions[["actual_sales", "predicted_sales"]].isna().any().any():
        raise ValueError("NaN in actual_sales/predicted_sales")
    if (predictions["predicted_sales"] < 0).any():
        raise ValueError("Negative predicted_sales")
    models = set(predictions["model"].astype(str).unique())
    if models != set(expected_models):
        raise ValueError(f"Model set mismatch: {sorted(models)}")
    if predictions.duplicated(["store", "item", "date", "model"]).any():
        raise ValueError("Duplicate (store, item, date, model) rows")

    key = ["store", "item", "date"]
    per_model = predictions.groupby("model", observed=True).size()
    if per_model.nunique() != 1:
        raise ValueError(f"Models have different row counts: {per_model.to_dict()}")
    ref = predictions[predictions["model"].astype(str) == expected_models[0]]
    ref_keys = ref[key].reset_index(drop=True)
    for name in expected_models[1:]:
        other = predictions[predictions["model"].astype(str) == name][key].reset_index(drop=True)
        if not ref_keys.equals(other):
            raise ValueError(f"{name} was not scored on the same dates as {expected_models[0]}")
    actual_by_model = predictions.groupby("model", observed=True)["actual_sales"].sum()
    if actual_by_model.nunique() != 1:
        raise ValueError("actual_sales differ between models")
    return {"rows": len(predictions), "rows_per_model": int(per_model.iloc[0])}
