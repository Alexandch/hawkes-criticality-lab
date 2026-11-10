from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


def calendar_design(index: pd.DatetimeIndex, hour_of_day: bool = False) -> np.ndarray:
    weekday = index.dayofweek.to_numpy()
    month = index.month.to_numpy()
    columns = [np.ones(len(index), dtype=float)]
    columns.extend((weekday == value).astype(float) for value in range(1, 7))
    columns.extend((month == value).astype(float) for value in range(2, 13))
    if hour_of_day:
        hour = index.hour.to_numpy()
        columns.extend((hour == value).astype(float) for value in range(1, 24))
    return np.column_stack(columns)


def fit_calendar_poisson(
    interval_counts: pd.Series, exposure_days: float = 1.0, hour_of_day: bool = False
) -> np.ndarray:
    """Fit rates in events/day, accounting for the duration of each equal-width bin."""
    if not np.isfinite(exposure_days) or exposure_days <= 0:
        raise ValueError("exposure_days must be finite and positive")
    counts = interval_counts.astype(float)
    y = counts.to_numpy()
    if len(y) == 0 or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("counts must be nonempty, finite, and nonnegative")
    design = calendar_design(counts.index, hour_of_day=hour_of_day)
    initial = np.zeros(design.shape[1], dtype=float)
    initial[0] = np.log(max(y.mean() / exposure_days, 1e-12))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        log_rate = design @ parameters
        expected_count = exposure_days * np.exp(np.clip(log_rate, -30.0, 30.0))
        loss = np.sum(expected_count - y * (log_rate + np.log(exposure_days)) + gammaln(y + 1.0))
        gradient = design.T @ (expected_count - y)
        return float(loss), gradient

    fit = minimize(
        objective, initial, jac=True, method="L-BFGS-B", bounds=[(-20, 20)] * len(initial),
        options={"maxiter": 500, "ftol": 1e-11},
    )
    if not fit.success:
        raise ValueError(f"calendar Poisson GLM did not converge: {fit.message}")
    return fit.x


def calendar_poisson_backtest(
    daily_counts: pd.Series, split_date: str
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """Fit calendar-only Poisson baselines before split_date and score after it."""
    counts = daily_counts.sort_index().astype(float)
    split = pd.Timestamp(split_date)
    train = counts.index < split
    test = counts.index >= split
    if not train.any() or not test.any():
        raise ValueError("split_date must leave at least one training day and one test day")
    y_train = counts.loc[train].to_numpy()
    y_test = counts.loc[test].to_numpy()
    test_index = counts.index[test]

    train_dow = counts.index[train].dayofweek
    train_month = counts.index[train].month
    test_dow = test_index.dayofweek
    test_month = test_index.month
    if not set(test_month).issubset(set(train_month)):
        raise ValueError("training data must cover every calendar month present in the test period")
    train_design = calendar_design(counts.index[train])
    test_design = calendar_design(test_index)

    predictions = {
        "homogeneous": np.full(len(y_test), y_train.mean()),
        "weekday": np.array([y_train[train_dow == day].mean() for day in test_dow]),
        "month": np.array([y_train[train_month == month].mean() for month in test_month]),
    }

    calendar_fit = fit_calendar_poisson(counts.loc[train])
    predictions["weekday_month"] = np.exp(np.clip(test_design @ calendar_fit, -30.0, 30.0))

    rows: dict[str, dict[str, float]] = {}
    forecast_frame = pd.DataFrame({"observed_count": y_test}, index=test_index)
    for name, forecast in predictions.items():
        forecast = np.maximum(forecast, 1e-12)
        nll = np.sum(forecast - y_test * np.log(forecast) + gammaln(y_test + 1.0))
        rows[name] = {
            "test_days": float(len(y_test)),
            "test_negative_log_likelihood": float(nll),
            "test_nll_per_day": float(nll / len(y_test)),
            "improvement_vs_homogeneous": float("nan"),
        }
        forecast_frame[f"predicted_{name}"] = forecast

    homogeneous_nll = rows["homogeneous"]["test_negative_log_likelihood"]
    for metrics in rows.values():
        metrics["improvement_vs_homogeneous"] = homogeneous_nll - metrics["test_negative_log_likelihood"]
    forecast_frame.index.name = "date"
    return rows, forecast_frame.reset_index()
