from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import beta


@lru_cache(maxsize=1024)
def tolerance_rank(sample_size, alpha=.05, failure_probability=.05):
    """Choose the order-statistic rank before inspecting the score values.

    For a continuous negative-maximum distribution, the tail beyond X_(k)
    has Beta(m+1-k,k) distribution. Strict exceedance is conservative for ties.
    """
    if sample_size < 1 or not 0 < alpha < 1 or not 0 < failure_probability < 1:
        raise ValueError("need positive sample size and probabilities in (0,1)")
    for rank in range(1, sample_size + 1):
        upper = float(beta.ppf(1 - failure_probability, sample_size + 1 - rank, rank))
        if upper <= alpha:
            return rank, upper
    return None, 0.


def calibrate_confident(scores, target, families, alpha=.05, failure_probability=.05):
    scores, families = np.asarray(scores, dtype=float), np.asarray(families)
    if scores.shape != target.labels.shape or len(scores) != len(families) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("finite scores in [0,1] must match targets and families")
    negative = target.eligible & ~target.labels
    maxima = np.max(np.where(negative, scores, -np.inf), axis=1)
    active_families = np.unique(families[np.isfinite(maxima)])
    if not len(active_families):
        raise ValueError("calibration needs negative decisions")
    delta = failure_probability / len(active_families)
    rows = []
    for family in np.unique(families):
        values = np.sort(maxima[(families == family) & np.isfinite(maxima)])
        rank, upper = tolerance_rank(len(values), alpha, delta) if len(values) else (None, np.nan)
        cutoff = float(np.nextafter(values[rank-1], np.inf)) if rank is not None else float(np.nextafter(1., np.inf))
        rows.append({"family": family, "negative_trajectories": len(values), "rank": rank,
            "family_failure_probability": delta if len(values) else np.nan,
            "tail_probability_upper_bound": upper, "family_cutoff": cutoff if len(values) else np.nan,
            "forced_abstention_insufficient_calibration": bool(len(values) and rank is None)})
    threshold = max(row["family_cutoff"] for row in rows if row["negative_trajectories"])
    for row in rows:
        values = maxima[(families == row["family"]) & np.isfinite(maxima)]
        row["threshold"] = threshold
        row["calibration_probability_false_alarm"] = float((values >= threshold).mean()) if len(values) else np.nan
    return threshold, pd.DataFrame(rows)
