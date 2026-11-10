"""Past-only local-trend filtering of nonoverlapping Hawkes MLE blocks.

The covariance is a working scale, not a Gaussian confidence statement. Bands
are calibrated separately on whole independent trajectories.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


N_EDGES = np.array([.5, .8, .9])
SE_EDGES = np.array([.05, .15, .5])
Q_GRID = (.01, .03, .1, .3)
DAMPING_GRID = (.5, .9)


@dataclass
class EstimatePath:
    n: np.ndarray
    slope: np.ndarray
    n_scale: np.ndarray
    slope_scale: np.ndarray
    valid: np.ndarray
    slope_valid: np.ndarray


def stage_measurements(batch, variant="calendar"):
    """Recover every short block once, including the first two warm-up blocks."""
    offset = {"constant": 16, "calendar": 39}[variant]
    base = batch.matrix[:, 0]
    first = np.stack([base[:, 0, offset + 5*j:offset + 5*j + 5] for j in (0, 1)], axis=1)
    return np.concatenate([first, base[:, :, offset + 10:offset + 15]], axis=1)


def _cells(n, se):
    return np.searchsorted(N_EDGES, n, side="right") * 4 + np.searchsorted(SE_EDGES, se, side="right")


def fit_measurement_noise(measurements, truth, shrinkage=100.):
    """Train-only empirical MSE by observable n/SE bins, shrunk to global MSE."""
    n, se, valid = measurements[:, :, 0], measurements[:, :, 1], measurements[:, :, 4] > 0
    if n.shape != truth.shape or not valid.any():
        raise ValueError("need matching truth and valid training measurements")
    errors = (n - truth) ** 2
    global_mse = float(errors[valid].mean())
    cells, rows = _cells(n, se), []
    for cell in range(16):
        values = errors[valid & (cells == cell)]
        variance = max(.0025, float((values.sum() + shrinkage * global_mse) / (len(values) + shrinkage)))
        rows.append({"cell": cell, "count": len(values), "empirical_mse": float(values.mean()) if len(values) else None,
                     "working_variance": variance})
    return {"global_mse": global_mse, "shrinkage": shrinkage, "variance_floor": .0025,
            "n_edges": N_EDGES.tolist(), "se_edges": SE_EDGES.tolist(), "cells": rows}


def measurement_variance(measurements, noise):
    values = np.array([row["working_variance"] for row in noise["cells"]])
    return values[_cells(measurements[:, :, 0], measurements[:, :, 1])]


def filter_n_history(measurements, noise, q=.03, damping=.9, max_missing=2):
    """Forward recursion only. No backward smoothing, no true n at prediction."""
    z = np.asarray(measurements, dtype=float)
    if z.ndim != 3 or z.shape[2] != 5 or not np.isfinite(z).all():
        raise ValueError("need finite trajectory-by-stage-by-five measurements")
    if q <= 0 or not 0 <= damping <= 1 or max_missing < 0:
        raise ValueError("invalid process noise, damping or missing limit")
    count, steps = z.shape[:2]
    r = measurement_variance(z, noise)
    state, covariance = np.zeros((count, 2)), np.zeros((count, 2, 2))
    initialized, missing = np.zeros(count, dtype=bool), np.zeros(count, dtype=int)
    transition = np.array([[1., damping], [0., damping]])
    process = q*q * np.array([[1/3, .5], [.5, 1.]])
    results = [np.zeros((count, steps)) for _ in range(4)]
    valid_path = np.zeros((count, steps), dtype=bool)
    for step in range(steps):
        state = state @ transition.T
        covariance = transition[None] @ covariance @ transition.T[None] + process[None]
        observed = z[:, step, 4] > 0
        new = observed & ~initialized
        old = observed & initialized
        if old.any():
            p = covariance[old].copy()
            gain = p[:, :, 0] / (p[:, 0, 0] + r[old, step])[:, None]
            state[old] += gain * (z[old, step, 0] - state[old, 0])[:, None]
            # Joseph form retains positive semidefiniteness under roundoff.
            a = np.broadcast_to(np.eye(2), p.shape).copy()
            a[:, :, 0] -= gain
            covariance[old] = a @ p @ a.transpose(0, 2, 1) + r[old, step, None, None] * gain[:, :, None] * gain[:, None, :]
        if new.any():
            state[new, 0], state[new, 1] = z[new, step, 0], 0.
            covariance[new] = 0.
            covariance[new, 0, 0], covariance[new, 1, 1] = r[new, step], .04
        initialized |= observed
        missing = np.where(observed, 0, missing + 1)
        # Bounds make this an approximate filter, hence empirical band calibration.
        state[:, 0] = np.clip(state[:, 0], 0., .995)
        state[:, 1] = np.clip(state[:, 1], -.5, .5)
        results[0][:, step], results[1][:, step] = state[:, 0], state[:, 1]
        results[2][:, step] = np.sqrt(np.maximum(covariance[:, 0, 0], 1e-6))
        results[3][:, step] = np.sqrt(np.maximum(covariance[:, 1, 1], 1e-6))
        valid_path[:, step] = initialized & (missing <= max_missing)
    return EstimatePath(*results, valid_path, valid_path.copy())


def raw_estimate_path(batch, variant="calendar", window=None, windows=(90, 360)):
    short = stage_measurements(batch, variant)
    n = short[:, 2:, 0].copy()
    se = np.maximum(short[:, 2:, 1], .05)
    valid = short[:, 2:, 4] > 0
    if window is not None:
        fit = batch.fits[:, list(windows).index(window), :, int(variant == "calendar")]
        n, se, valid = fit[:, :, 0].copy(), np.maximum(fit[:, :, 1], .05), fit[:, :, 4] > 0
    slope = (short[:, 2:, 0] - short[:, :-2, 0]) / 2
    slope_scale = np.sqrt(short[:, 2:, 1]**2 + short[:, :-2, 1]**2) / 2
    slope_scale = np.maximum(slope_scale, .025)
    # A missing earlier block invalidates the trend, but not current n.
    trend_valid = np.lib.stride_tricks.sliding_window_view(short[:, :, 4] > 0, 3, axis=1).all(axis=-1)
    return EstimatePath(n, slope, se, slope_scale, valid, trend_valid)


def trim_path(path, start=2):
    return EstimatePath(**{key: value[:, start:] for key, value in path.__dict__.items()})


def true_slope(ratios):
    """Reference is the causal three-block secant, per stage, not a derivative."""
    return (ratios[:, 2:] - ratios[:, :-2]) / 2


def trajectory_error_summary(path, batch, method, stage_days=90):
    truth, slope = batch.ratios[:, 2:], true_slope(batch.ratios)
    rows = []
    for index, family in enumerate(batch.families):
        valid = path.valid[index]
        n_error, s_error = path.n[index] - truth[index], path.slope[index] - slope[index]
        rows.append({"trajectory_id": batch.ids[index], "family": family, "target_rate": batch.rates[index],
            "method": method, "valid_fraction": float(valid.mean()),
            "mae": float(np.abs(n_error[valid]).mean()) if valid.any() else np.nan,
            "bias": float(n_error[valid].mean()) if valid.any() else np.nan,
            "slope_mae": float(np.abs(s_error[path.slope_valid[index]]).mean()) if path.slope_valid[index].any() else np.nan})
    return pd.DataFrame(rows)


def delay_rows(path, batch, method, stage_days=90, critical=.9):
    """Abrupt upward changes; all missed crossings remain capped/censored rows."""
    days = np.arange(3, batch.ratios.shape[1] + 1) * stage_days
    rows = []
    for index in np.flatnonzero(batch.families == "hawkes_abrupt"):
        crossing = int(np.flatnonzero(batch.ratios[index] >= critical)[0] * stage_days)
        cap = batch.ratios.shape[1] * stage_days - crossing + stage_days
        hits = np.flatnonzero((days > crossing) & path.valid[index] & (path.n[index] >= critical))
        detected = bool(len(hits))
        delay = int(days[hits[0]] - crossing) if detected else cap
        rows.append({"trajectory_id": batch.ids[index], "method": method, "crossing_day": crossing,
                     "detected": detected, "delay_or_censor_cap_days": delay, "cap_days": cap})
    return pd.DataFrame(rows)


def validation_quality(path, batch, method, stage_days=90):
    errors = trajectory_error_summary(path, batch, method, stage_days)
    family = errors.groupby("family").mae.mean()
    near_high = errors[errors.family.isin(["hawkes_high", "hawkes_near_threshold"])].mae.mean()
    delay = delay_rows(path, batch, method, stage_days)
    return {"method": method, "macro_mae": float(family.mean()), "maximum_family_mae": float(family.max()),
        "near_high_mae": float(near_high), "mean_capped_abrupt_delay_days": float(delay.delay_or_censor_cap_days.mean()),
        "valid_fraction": float(errors.valid_fraction.mean())}


def choose_filter(candidates):
    """Fixed validation rule; deterministic ties prefer faster, less damped filter."""
    return candidates.sort_values(["macro_mae", "mean_capped_abrupt_delay_days", "q", "damping"],
                                  ascending=[True, True, False, False]).iloc[0].to_dict()


def extension_gate(filtered, baseline, stage_days=90):
    checks = {
        "near_high_mae_improves_at_least_5pct": filtered["near_high_mae"] <= .95 * baseline["near_high_mae"],
        "macro_mae_no_more_than_5pct_worse": filtered["macro_mae"] <= 1.05 * baseline["macro_mae"],
        "maximum_family_mae_no_more_than_002_worse": filtered["maximum_family_mae"] <= baseline["maximum_family_mae"] + .02,
        "mean_capped_abrupt_delay_no_more_than_one_stage_worse": filtered["mean_capped_abrupt_delay_days"] <= baseline["mean_capped_abrupt_delay_days"] + stage_days,
    }
    return {"passed": bool(all(checks.values())), "checks": {key: bool(value) for key, value in checks.items()},
            "decision_source": "validation only, before holdout generation"}


def calibrate_trajectory_bands(path, batch, alpha=.05):
    """Separate nominal coverage for n and slope over all valid monitored times.

    Each calibration observation is a whole trajectory maximum. The common
    cutoff is the maximum of family-specific conformal order statistics.
    No conditional-on-time, joint-n-and-slope, or real-data guarantee is claimed.
    """
    rows, cutoffs = [], {}
    for target, values, truth, scale in (("n", path.n, batch.ratios[:, 2:], path.n_scale),
                                        ("slope", path.slope, true_slope(batch.ratios), path.slope_scale)):
        valid = path.valid if target == "n" else path.slope_valid
        error = np.where(valid, np.abs(values - truth) / scale, -np.inf)
        maxima = error.max(axis=1)
        cutoffs[target] = 0.
        cutoffs[target + "_full_domain"] = False
        for family in np.unique(batch.families):
            vals = np.sort(maxima[(batch.families == family) & np.isfinite(maxima)])
            rank = math.ceil((len(vals) + 1) * (1 - alpha))
            insufficient = rank > len(vals)
            cutoff = None if insufficient else float(vals[rank - 1])
            if insufficient:
                cutoffs[target + "_full_domain"] = True
            else:
                cutoffs[target] = max(cutoffs[target], cutoff)
            rows.append({"target": target, "family": family, "trajectories_with_valid_estimates": len(vals),
                         "rank": rank, "family_multiplier": cutoff, "insufficient_calibration_full_domain": insufficient})
    return cutoffs, pd.DataFrame(rows)


def coverage_rows(path, batch, method, cutoffs):
    records = []
    for target, values, truth, scale in (("n", path.n, batch.ratios[:, 2:], path.n_scale),
                                        ("slope", path.slope, true_slope(batch.ratios), path.slope_scale)):
        half = cutoffs[target] * scale
        lower, upper = values - half, values + half
        if target == "n":
            lower, upper = np.maximum(lower, 0.), np.minimum(upper, 1.)
        if cutoffs[target + "_full_domain"]:
            lower = np.full_like(values, 0. if target == "n" else -.5)
            upper = np.full_like(values, 1. if target == "n" else .5)
        covered = (truth >= lower - 1e-12) & (truth <= upper + 1e-12)
        for index in range(len(values)):
            valid = path.valid[index] if target == "n" else path.slope_valid[index]
            records.append({"trajectory_id": batch.ids[index], "family": batch.families[index], "method": method,
                "target": target, "valid_windows": int(valid.sum()), "covered_windows": int(covered[index, valid].sum()),
                "trajectory_covered": bool(covered[index, valid].all()) if valid.any() else None,
                "mean_width": float((upper-lower)[index, valid].mean()) if valid.any() else np.nan})
    return pd.DataFrame(records)


def estimate_scores(path, task, short_valid, horizon=2):
    if task == "state":
        values, valid = path.n, path.valid
    else:
        values = np.maximum(path.n + horizon * path.slope, 0.)
        values = values / (1 + values)
        valid = path.valid & path.slope_valid & short_valid
    return np.where(valid, values, 0.), valid
