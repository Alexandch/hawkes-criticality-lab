from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

try:
    from numba import njit

    @njit(cache=True)
    def _hawkes_log_likelihood_numba(times, counts, horizon, baseline, branching_ratio, decay):
        memory = 0.0
        previous = 0.0
        log_sum = 0.0
        compensator = baseline * horizon
        for i in range(len(times)):
            time = times[i]
            memory *= np.exp(-decay * (time - previous))
            intensity = baseline + branching_ratio * decay * memory
            if intensity <= 0:
                return -np.inf
            log_sum += counts[i] * np.log(intensity)
            compensator += branching_ratio * counts[i] * (1.0 - np.exp(-decay * (horizon - time)))
            memory += counts[i]
            previous = time
        return log_sum - compensator
except ImportError:
    _hawkes_log_likelihood_numba = None


def poisson_mle(events: np.ndarray, horizon: float) -> dict[str, float]:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    return {"rate": float(len(events) / horizon), "log_likelihood": float(len(events) * np.log(max(len(events) / horizon, 1e-300)) - len(events))}


def hawkes_log_likelihood(events: np.ndarray, horizon: float, baseline: float, branching_ratio: float, decay: float) -> float:
    if horizon <= 0 or baseline <= 0 or decay <= 0 or not 0 <= branching_ratio < 1:
        return -np.inf
    events = np.asarray(events, dtype=float)
    if len(events) == 0:
        return -baseline * horizon
    times, counts = np.unique(events, return_counts=True)
    if _hawkes_log_likelihood_numba is not None:
        return float(_hawkes_log_likelihood_numba(times, counts, horizon, baseline, branching_ratio, decay))
    memory = 0.0
    previous = 0.0
    log_sum = 0.0
    compensator = baseline * horizon
    for time, count in zip(times, counts):
        memory *= np.exp(-decay * (time - previous))
        intensity = baseline + branching_ratio * decay * memory
        if intensity <= 0:
            return -np.inf
        log_sum += count * np.log(intensity)
        compensator += branching_ratio * count * (1.0 - np.exp(-decay * (horizon - time)))
        memory += count
        previous = time
    return float(log_sum - compensator)


def hawkes_mle(events: np.ndarray, horizon: float) -> dict[str, float]:
    """Maximum likelihood fit for baseline, branching ratio and exponential decay."""
    events = np.asarray(events, dtype=float)
    events = np.sort(events[(events >= 0) & (events < horizon)])
    if horizon <= 0 or len(events) < 3:
        raise ValueError("at least three events in a positive observation window are required")

    empirical_rate = max(len(events) / horizon, 1e-4)
    starts = [
        [np.log(empirical_rate * (1.0 - n)), n, np.log(beta)]
        for n, beta in ((0.2, 1.0), (0.6, 1.0), (0.85, 2.0))
    ]

    def objective(x: np.ndarray) -> float:
        baseline, decay = np.exp(x[0]), np.exp(x[2])
        return -hawkes_log_likelihood(events, horizon, baseline, x[1], decay)

    bounds = [(-15, 15), (0.0, 0.999), (-8, 8)]
    fits = [minimize(objective, start, method="L-BFGS-B", bounds=bounds) for start in starts]
    best = min(fits, key=lambda fit: fit.fun)
    return {
        "baseline": float(np.exp(best.x[0])),
        "branching_ratio": float(best.x[1]),
        "decay": float(np.exp(best.x[2])),
        "log_likelihood": float(-best.fun),
        "converged": float(best.success),
    }
