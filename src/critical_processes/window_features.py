from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .detector import feature_sequence_windows, transform_detector_features
from .negative_controls import stage_features_from_counts

try:
    from numba import njit
except ImportError:
    njit = None


def _history_impl(times, multiplicities, decay):
    memory = np.zeros(len(times))
    for i in range(1, len(times)):
        memory[i] = (memory[i - 1] + multiplicities[i - 1]) * np.exp(-decay * (times[i] - times[i - 1]))
    return memory


def _fixed_decay_loss_impl(parameters, normalized_history, multiplicities, event_count, history_integral):
    baseline_fraction, ratio = parameters
    loss = baseline_fraction * event_count + ratio * history_integral - event_count
    gradient = np.array([float(event_count), history_integral])
    for i in range(len(normalized_history)):
        relative_intensity = baseline_fraction + ratio * normalized_history[i]
        loss -= multiplicities[i] * np.log(relative_intensity)
        weight = multiplicities[i] / relative_intensity
        gradient[0] -= weight
        gradient[1] -= weight * normalized_history[i]
    return loss, gradient


_history = njit(cache=True)(_history_impl) if njit is not None else _history_impl
fixed_decay_loss = njit(cache=True)(_fixed_decay_loss_impl) if njit is not None else _fixed_decay_loss_impl


@dataclass
class HawkesHistory:
    times: np.ndarray
    multiplicities: np.ndarray
    memory: np.ndarray
    decay: float


def prepare_hawkes_history(events: np.ndarray, decay: float = 1.0) -> HawkesHistory:
    events = np.asarray(events, dtype=float)
    if events.ndim != 1 or not np.isfinite(events).all() or (events < 0).any() or not np.isfinite(decay) or decay <= 0:
        raise ValueError("events must be finite nonnegative times and decay positive")
    times, counts = np.unique(events, return_counts=True)
    return HawkesHistory(times, counts, _history(times, counts, decay), decay)


def conditional_window_data(history: HawkesHistory, start: float, end: float):
    if not np.isfinite(start) or not np.isfinite(end) or not 0 <= start < end:
        raise ValueError("window boundaries must satisfy 0 <= start < end")
    left, right = np.searchsorted(history.times, [start, end], side="left")

    def memory_at(time, index):
        if index == 0:
            return 0.0
        return float((history.memory[index - 1] + history.multiplicities[index - 1]) * np.exp(-history.decay * (time - history.times[index - 1])))

    counts = history.multiplicities[left:right]
    total = int(counts.sum())
    integral = max(0.0, memory_at(start, left) - memory_at(end, right) + total)
    rate = total / (end - start)
    scaled_history = history.decay * history.memory[left:right] / max(rate, 1e-12)
    return scaled_history, counts, total, integral, rate


def fit_hawkes_window(history: HawkesHistory, start: float, end: float) -> np.ndarray:
    """Return n, capped curvature SE, baseline/rate, LL gain/event, and validity.

    Decay is fixed; the window likelihood is concave in baseline and n. Past
    observed events contribute to both event intensities and the compensator.
    The SE is an information feature, not a boundary-valid confidence interval.
    """
    memory, counts, total, integral, rate = conditional_window_data(history, start, end)
    fallback = np.array([0.0, 2.0, 1.0 if total else 0.0, 0.0, 0.0])
    if total < 3:
        return fallback
    fit = minimize(
        fixed_decay_loss, np.array([1.0, 0.0]), args=(memory, counts, total, integral),
        method="L-BFGS-B", jac=True, bounds=[(1e-8, 5.0), (0.0, 0.995)],
        options={"maxiter": 150, "ftol": 1e-10, "gtol": 1e-6},
    )
    if not fit.success or not np.isfinite(fit.fun) or fit.fun > 1e-5:
        return fallback
    a, n = fit.x
    weights = counts / (a + n * memory) ** 2
    h_aa = float(weights.sum())
    h_an = float(np.dot(weights, memory))
    h_nn = float(np.dot(weights, memory * memory))
    profile_information = h_nn - h_an * h_an / max(h_aa, 1e-12)
    error = min(2.0, 1.0 / np.sqrt(max(profile_information, 1e-12)))
    return np.array([n, error, a, max(0.0, -float(fit.fun) / total), 1.0])


def window_feature_names(lookback: int = 3) -> list[str]:
    names = [f"lag_{lag}_{name}" for lag in range(lookback - 1, -1, -1) for name in ("log_rate", "fano", "acf1", "cluster_per_count")]
    names.extend(f"lag_{lag}_{name}" for lag in range(lookback - 1, -1, -1) for name in ("n_mle", "n_se", "baseline_fraction", "ll_gain_per_event", "fit_valid"))
    names.extend(f"long_{name}" for name in ("n_mle", "n_se", "baseline_fraction", "ll_gain_per_event", "fit_valid"))
    names.extend(["n_slope_per_stage", "n_projected", "short_minus_long_n"])
    return names


def build_window_features(
    counts: np.ndarray, events: np.ndarray, stage_days: int = 90,
    lookback: int = 3, forecast_stages: int = 2, decay: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build strictly causal decision features, without access to true regimes."""
    counts = np.asarray(counts)
    events = np.asarray(events, dtype=float)
    if lookback < 2 or forecast_stages < 1 or counts.ndim != 1 or events.ndim != 1 or not np.isfinite(events).all() or (events < 0).any() or (events >= len(counts)).any():
        raise ValueError("need lookback >= 2, positive forecast horizon and events inside count horizon")
    observed = np.bincount(np.floor(events).astype(int), minlength=len(counts))
    if not np.array_equal(observed, counts):
        raise ValueError("event timestamps must agree with daily counts")
    stage_features = stage_features_from_counts(counts, stage_days)
    raw = feature_sequence_windows(stage_features[np.newaxis], lookback)[0]
    normalized = transform_detector_features(raw, version=2)
    history = prepare_hawkes_history(events, decay)
    short = np.vstack([fit_hawkes_window(history, i * stage_days, (i + 1) * stage_days) for i in range(len(stage_features))])
    centred = np.arange(lookback, dtype=float) - (lookback - 1) / 2
    rows, diagnostics = [], []
    for decision, index in enumerate(range(lookback - 1, len(stage_features))):
        end = (index + 1) * stage_days
        long = fit_hawkes_window(history, end - lookback * stage_days, end)
        recent = short[index - lookback + 1:index + 1]
        slope = float(np.dot(centred, recent[:, 0]) / np.dot(centred, centred))
        projected = float(recent[-1, 0] + forecast_stages * slope)
        rows.append(np.concatenate([normalized[decision], recent.reshape(-1), long, [slope, projected, recent[-1, 0] - long[0]]]))
        diagnostics.append([end, recent[-1, 0], long[0], slope, projected, recent[:, 4].min(), long[4]])
    return np.vstack(rows), short, np.asarray(diagnostics)


def hawkes_trend_scores(diagnostics: np.ndarray) -> np.ndarray:
    diagnostics = np.asarray(diagnostics, dtype=float)
    projection = np.maximum(diagnostics[..., 4], 0.0)
    valid = (diagnostics[..., 5] > 0) & (diagnostics[..., 6] > 0)
    return np.where(valid, projection / (1 + projection), 0.0)
