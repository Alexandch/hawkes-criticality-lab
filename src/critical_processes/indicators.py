from __future__ import annotations

import numpy as np


FEATURE_NAMES = ("mean_rate", "variance", "lag1_autocorrelation", "mean_cluster_size")


def event_counts(events: np.ndarray, horizon: float, bin_width: float) -> np.ndarray:
    if horizon <= 0 or bin_width <= 0:
        raise ValueError("horizon and bin_width must be positive")
    bins = np.arange(0.0, horizon + bin_width, bin_width)
    if len(bins) < 2:
        bins = np.array([0.0, horizon])
    return np.histogram(events, bins=bins)[0].astype(float)


def summarize_counts(counts: np.ndarray, bin_width: float, cluster_gap: float = 0.5) -> np.ndarray:
    """Return rate, variance, lag-1 ACF, and heuristic temporal cluster size."""
    counts = np.asarray(counts, dtype=float)
    if len(counts) == 0 or bin_width <= 0 or cluster_gap <= 0:
        raise ValueError("counts must be nonempty and widths positive")
    mean = float(np.mean(counts) / bin_width)
    variance = float(np.var(counts, ddof=1)) if len(counts) > 1 else 0.0
    if len(counts) > 2 and np.std(counts[:-1]) > 0 and np.std(counts[1:]) > 0:
        acf = float(np.corrcoef(counts[:-1], counts[1:])[0, 1])
    else:
        acf = 0.0
    # A cluster is a run of occupied bins separated by no more than cluster_gap.
    max_bin_distance = int(np.floor(cluster_gap / bin_width))
    occupied = np.flatnonzero(counts > 0)
    cluster_sizes: list[int] = []
    size = 0
    previous = -1
    for index in occupied:
        if previous >= 0 and index - previous > max_bin_distance:
            cluster_sizes.append(size)
            size = 0
        size += int(counts[index])
        previous = int(index)
    if size:
        cluster_sizes.append(size)
    mean_cluster_size = float(np.mean(cluster_sizes)) if cluster_sizes else 0.0
    return np.array([mean, variance, acf, mean_cluster_size], dtype=float)


def rolling_features(counts: np.ndarray, window: int, bin_width: float = 1.0) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    if window < 3 or len(counts) < window:
        raise ValueError("window must be >= 3 and no longer than the count series")
    return np.vstack([summarize_counts(counts[i - window:i], bin_width) for i in range(window, len(counts) + 1)])
