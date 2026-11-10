from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize

from .window_features import HawkesHistory, conditional_window_data

try:
    from numba import njit
except ImportError:
    njit = None


def background_design(days: np.ndarray, training_days: int) -> np.ndarray:
    days = np.asarray(days, dtype=float)
    columns = [np.ones(len(days))]
    columns.extend((days % 7 == weekday).astype(float) for weekday in range(1, 7))
    for harmonic in (1, 2):
        phase = 2 * np.pi * harmonic * days / 365.25
        columns.extend([np.sin(phase), np.cos(phase)])
    columns.append((days - (training_days - 1) / 2) / training_days)
    return np.column_stack(columns)


def fit_background_forecast(prefix_counts: np.ndarray, total_days: int) -> tuple[np.ndarray, bool]:
    """Fit an exogenous Poisson GLM on the prefix; future counts are not accepted."""
    counts = np.asarray(prefix_counts, dtype=float)
    if counts.ndim != 1 or len(counts) < 30 or not np.isfinite(counts).all() or (counts < 0).any() or total_days < len(counts):
        raise ValueError("need finite nonnegative prefix counts, >=30 training days and a matching forecast horizon")
    if counts.sum() == 0:
        return np.full(total_days, np.nan), False
    design = background_design(np.arange(total_days), len(counts))
    training = design[:len(counts)]
    initial = np.zeros(design.shape[1])
    initial[0] = np.log(counts.mean())
    penalty = np.full(len(initial), 1e-5)
    penalty[0] = 0.0

    def objective(parameters):
        log_rate = training @ parameters
        expected = np.exp(log_rate)
        value = np.sum(expected - counts * log_rate) + 0.5 * np.dot(penalty * parameters, parameters)
        gradient = training.T @ (expected - counts) + penalty * parameters
        return float(value), gradient

    fit = minimize(
        objective, initial, jac=True, method="L-BFGS-B",
        bounds=[(-20, 20)] + [(-5, 5)] * (len(initial) - 1),
        options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-6},
    )
    if not fit.success or not np.isfinite(fit.fun):
        return np.full(total_days, np.nan), False
    rates = np.exp(design @ fit.x)
    if not np.isfinite(rates).all() or (rates <= 0).any():
        return np.full(total_days, np.nan), False
    return rates, True


def _shaped_background_loss_impl(parameters, baseline_at_events, memory, counts, total, history_integral):
    level, n = parameters
    value = level * total + n * history_integral - total
    gradient = np.array([float(total), history_integral])
    for i in range(len(memory)):
        relative_intensity = level * baseline_at_events[i] + n * memory[i]
        value -= counts[i] * np.log(relative_intensity)
        weight = counts[i] / relative_intensity
        gradient[0] -= weight * baseline_at_events[i]
        gradient[1] -= weight * memory[i]
    return value / total, gradient / total


def _fixed_background_loss_impl(n, baseline_at_events, memory, counts, total, history_integral, baseline_integral):
    value = baseline_integral + n * history_integral - total
    gradient = history_integral
    for i in range(len(memory)):
        relative_intensity = baseline_at_events[i] + n * memory[i]
        value -= counts[i] * np.log(relative_intensity)
        gradient -= counts[i] * memory[i] / relative_intensity
    return value / total, gradient / total


shaped_background_loss = njit(cache=True)(_shaped_background_loss_impl) if njit is not None else _shaped_background_loss_impl
fixed_background_loss = njit(cache=True)(_fixed_background_loss_impl) if njit is not None else _fixed_background_loss_impl


def fit_background_window(
    history: HawkesHistory, start: int, end: int, daily_background: np.ndarray, fixed_level: bool,
) -> dict[str, object]:
    background = np.asarray(daily_background, dtype=float)
    if start != int(start) or end != int(end) or background.ndim != 1 or not 0 <= start < end <= len(background):
        raise ValueError("need whole-day window inside the baseline horizon")
    empty = {"n_hat": np.nan, "baseline_mean_hat": np.nan, "fit_valid": False, "failure_reason": ""}
    values = background[start:end]
    if not np.isfinite(values).all() or (values <= 0).any():
        return {**empty, "failure_reason": "background_forecast_failed"}
    memory, counts, total, integral, rate = conditional_window_data(history, start, end)
    if total < 3:
        return {**empty, "failure_reason": "insufficient_events"}
    left, right = np.searchsorted(history.times, [start, end], side="left")
    day_indices = np.floor(history.times[left:right]).astype(int)
    background_mean = float(values.mean())
    if fixed_level:
        at_events = background[day_indices] / rate
        baseline_integral = float(values.sum())

        def derivative(n):
            return float(fixed_background_loss(n, at_events, memory, counts, total, integral, baseline_integral)[1])

        if derivative(0.0) >= 0:
            n = 0.0
        elif derivative(0.995) <= 0:
            n = 0.995
        else:
            n = float(brentq(derivative, 0.0, 0.995, xtol=1e-9))
        baseline_hat = background_mean
    else:
        at_events = background[day_indices] / background_mean
        fit = minimize(
            shaped_background_loss, np.array([1.0, 0.0]),
            args=(at_events, memory, counts, total, integral), jac=True,
            method="L-BFGS-B", bounds=[(1e-8, 5.0), (0.0, 0.995)],
            options={"maxiter": 250, "ftol": 1e-12, "gtol": 1e-8},
        )
        if not fit.success or not np.isfinite(fit.fun):
            return {**empty, "failure_reason": "optimization_failed"}
        n = float(fit.x[1])
        baseline_hat = float(fit.x[0] * rate)
    return {"n_hat": n, "baseline_mean_hat": baseline_hat, "fit_valid": True, "failure_reason": ""}
