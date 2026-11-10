from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .n_accuracy import empirical_auc
from .negative_controls import wilson_interval


@dataclass
class TaskTargets:
    labels: np.ndarray
    eligible: np.ndarray
    decision_days: np.ndarray
    current_n: np.ndarray
    first_crossing: np.ndarray


def state_warning_targets(ratios, stage_days=90, lookback=3, forecast_stages=2, critical_level=0.9):
    """State at t- versus first crossing strictly after t with a full future horizon."""
    ratios = np.asarray(ratios, dtype=float)
    if ratios.ndim != 2 or not np.isfinite(ratios).all() or ((ratios < 0) | (ratios >= 1)).any():
        raise ValueError("need finite trajectory-by-stage ratios in [0,1)")
    if not 1 <= lookback <= ratios.shape[1] or stage_days < 1 or forecast_stages < 1 or not 0 < critical_level < 1:
        raise ValueError("invalid lookback, stage duration, horizon or high-n threshold")
    critical = ratios >= critical_level
    first = np.where(critical.any(axis=1), critical.argmax(axis=1) * stage_days, np.inf)
    days = np.arange(lookback, ratios.shape[1] + 1) * stage_days
    current = ratios[:, lookback - 1:]
    state = TaskTargets(current >= critical_level, np.ones_like(current, dtype=bool), days, current, first)
    lead = first[:, None] - days[None, :]
    full_horizon = days + forecast_stages * stage_days <= ratios.shape[1] * stage_days
    eligible = (lead > 0) & full_horizon[None, :]
    warning = TaskTargets(eligible & (lead <= forecast_stages * stage_days), eligible, days, current, first)
    return {"state": state, "warning": warning}


def calibrate_task_threshold(scores, targets: TaskTargets, families, alpha=0.05):
    """Use independent negative trajectory maxima; allow an all-abstain threshold."""
    scores = np.asarray(scores, dtype=float)
    families = np.asarray(families)
    if scores.shape != targets.labels.shape or len(families) != len(scores):
        raise ValueError("scores and family identifiers must match targets")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any() or not 0 < alpha < 1:
        raise ValueError("need scores in [0,1] and alpha in (0,1)")
    negative = targets.eligible & ~targets.labels
    maxima = np.max(np.where(negative, scores, -np.inf), axis=1)
    rows = []
    for family in np.unique(families):
        members = families == family
        values = np.sort(maxima[members & np.isfinite(maxima)])
        if not len(values):
            rows.append({"family": family, "negative_trajectories": 0, "excluded_no_negative_decisions": int(members.sum()),
                         "rank": np.nan, "negative_max_quantile": np.nan})
            continue
        rank = math.ceil((len(values) + 1) * (1 - alpha))
        if rank > len(values):
            raise ValueError("too few negative calibration trajectories for the requested alpha")
        rows.append({"family": family, "negative_trajectories": len(values),
                     "excluded_no_negative_decisions": int(members.sum()) - len(values),
                     "rank": rank, "negative_max_quantile": float(values[rank - 1])})
    quantiles = [row["negative_max_quantile"] for row in rows if row["negative_trajectories"]]
    if not quantiles:
        raise ValueError("calibration requires negative decisions")
    threshold = float(np.nextafter(max(quantiles), np.inf))
    for row in rows:
        selected = (families == row["family"]) & np.isfinite(maxima)
        row["calibration_probability_false_alarm"] = float((maxima[selected] >= threshold).mean()) if selected.any() else np.nan
        row["threshold"] = threshold
        row["all_abstain_threshold"] = threshold > 1
    return threshold, pd.DataFrame(rows)


def evaluate_task_scores(scores, valid, targets: TaskTargets, families, ids, rates, threshold, task,
                         bootstrap_seed=20261116, bootstrap_samples=500):
    scores, valid = np.asarray(scores, dtype=float), np.asarray(valid, dtype=bool)
    labels, eligible = targets.labels, targets.eligible
    families, ids, rates = np.asarray(families), np.asarray(ids), np.asarray(rates)
    if task not in ("state", "warning") or scores.shape != labels.shape or valid.shape != labels.shape:
        raise ValueError("need state/warning scores and validity matching targets")
    if len(families) != len(scores) or len(ids) != len(scores) or len(rates) != len(scores):
        raise ValueError("trajectory metadata must match scores")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any() or (scores[~valid] != 0).any():
        raise ValueError("scores must be finite in [0,1]; abstentions must have zero score")
    if not 0 < threshold <= np.nextafter(1., np.inf) or bootstrap_samples < 1:
        raise ValueError("invalid threshold or bootstrap count")
    alarm = valid & (scores >= threshold)
    masks = (eligible & labels & alarm, eligible & ~labels & alarm,
             eligible & labels & ~alarm, eligible & ~labels & ~alarm)
    counts = np.stack([mask.sum(axis=1) for mask in masks], axis=1)
    tp, fp, fn, tn = counts.sum(axis=0).tolist()
    trajectories = []
    for i in range(len(scores)):
        positive = eligible[i] & labels[i]
        negative = eligible[i] & ~labels[i]
        hits = np.flatnonzero(positive & alarm[i])
        alerts = np.flatnonzero(eligible[i] & alarm[i])
        first = int(alerts[0]) if len(alerts) else None
        first_hit_day = int(targets.decision_days[hits[0]]) if len(hits) else None
        crossing = float(targets.first_crossing[i])
        measurable_onset = task == "state" and np.isfinite(crossing) and crossing > 0
        trajectories.append({
            "trajectory_id": ids[i], "family": families[i], "target_rate": rates[i],
            "eligible_windows": int(eligible[i].sum()), "positive_windows": int(positive.sum()),
            "negative_windows": int(negative.sum()), "valid_windows": int((eligible[i] & valid[i]).sum()),
            "true_positive": int(counts[i, 0]), "false_positive": int(counts[i, 1]),
            "false_negative": int(counts[i, 2]), "true_negative": int(counts[i, 3]),
            "has_positive": bool(positive.any()), "has_negative": bool(negative.any()),
            "detected": bool(len(hits)) if positive.any() else None,
            "clean_detection": bool(first is not None and labels[i, first]) if positive.any() else None,
            "false_alarm": bool((negative & alarm[i]).any()) if negative.any() else None,
            "first_crossing_day": crossing if np.isfinite(crossing) else None,
            "initially_high": bool(crossing == 0),
            "first_alarm_day": int(targets.decision_days[first]) if first is not None else None,
            "first_hit_day": first_hit_day,
            "warning_lead_days": crossing - first_hit_day if task == "warning" and first_hit_day is not None else None,
            "state_onset_delay_days": first_hit_day - crossing if measurable_onset and first_hit_day is not None else None,
        })
    trajectory_frame = pd.DataFrame(trajectories)
    family_rows = []
    for family, frame in trajectory_frame.groupby("family", sort=True):
        negatives = frame[frame.has_negative]
        positives = frame[frame.has_positive]
        false_count, detected_count = int(negatives.false_alarm.sum()), int(positives.detected.sum())
        fa_ci = wilson_interval(false_count, len(negatives)) if len(negatives) else (np.nan, np.nan)
        hit_ci = wilson_interval(detected_count, len(positives)) if len(positives) else (np.nan, np.nan)
        family_rows.append({
            "family": family, "trajectories": len(frame), "negative_trajectories": len(negatives),
            "positive_trajectories": len(positives), "false_alarm_trajectories": false_count,
            "detected_trajectories": detected_count,
            "probability_false_alarm": false_count / len(negatives) if len(negatives) else np.nan,
            "false_alarm_ci95_lower": fa_ci[0], "false_alarm_ci95_upper": fa_ci[1],
            "event_detection_probability": detected_count / len(positives) if len(positives) else np.nan,
            "detection_ci95_lower": hit_ci[0], "detection_ci95_upper": hit_ci[1],
            "window_sensitivity": float(frame.true_positive.sum() / (frame.true_positive.sum() + frame.false_negative.sum())) if frame.positive_windows.sum() else np.nan,
        })
    family_frame = pd.DataFrame(family_rows)
    positives = trajectory_frame[trajectory_frame.has_positive]
    detected = int(positives.detected.sum())
    clean = int(positives.clean_detection.sum())
    detection_ci = wilson_interval(detected, len(positives)) if len(positives) else (None, None)
    clean_ci = wilson_interval(clean, len(positives)) if len(positives) else (None, None)
    rng = np.random.default_rng(bootstrap_seed)
    sensitivities, false_rates = [], []
    for _ in range(bootstrap_samples):
        a, b, c, d = counts[rng.integers(0, len(counts), len(counts))].sum(axis=0)
        if a + c:
            sensitivities.append(a / (a + c))
        if b + d:
            false_rates.append(b / (b + d))
    sensitivity_ci = np.quantile(sensitivities, [0.025, 0.975]).tolist() if sensitivities else [None, None]
    false_rate_ci = np.quantile(false_rates, [0.025, 0.975]).tolist() if false_rates else [None, None]
    auc_mask = eligible & valid
    auc = empirical_auc(scores[auc_mask & ~labels], scores[auc_mask & labels])
    lead = trajectory_frame.warning_lead_days.dropna()
    delay = trajectory_frame.state_onset_delay_days.dropna()
    metrics = {
        "task": task, "threshold": threshold, "all_abstain_threshold": threshold > 1,
        "eligible_windows": int(eligible.sum()), "positive_windows": tp + fn, "negative_windows": fp + tn,
        "score_valid_fraction_eligible": float(valid[eligible].mean()) if eligible.any() else None,
        "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
        "window_sensitivity": tp / (tp + fn) if tp + fn else None,
        "window_sensitivity_trajectory_bootstrap_ci95": sensitivity_ci,
        "window_false_alarm_rate": fp / (fp + tn) if fp + tn else None,
        "window_false_alarm_rate_trajectory_bootstrap_ci95": false_rate_ci,
        "precision": tp / (tp + fp) if tp + fp else None,
        "auc_valid_only": auc if np.isfinite(auc) else None,
        "positive_trajectories": len(positives), "detected_trajectories": detected,
        "event_detection_probability": detected / len(positives) if len(positives) else None,
        "event_detection_ci95": detection_ci,
        "clean_event_detection_probability": clean / len(positives) if len(positives) else None,
        "clean_detection_ci95": clean_ci,
        "median_warning_lead_days_detected_only": float(lead.median()) if len(lead) else None,
        "median_state_onset_delay_days_detected_only": float(delay.median()) if len(delay) else None,
        "maximum_family_false_alarm_probability": float(family_frame.probability_false_alarm.max()) if family_frame.probability_false_alarm.notna().any() else None,
    }
    trajectories_count, decisions_count = scores.shape
    decisions = pd.DataFrame({
        "trajectory_id": np.repeat(ids, decisions_count), "family": np.repeat(families, decisions_count),
        "decision_end_day": np.tile(targets.decision_days, trajectories_count),
        "current_n_audit_only": targets.current_n.ravel(), "score": scores.ravel(),
        "score_valid": valid.ravel(), "eligible": eligible.ravel(), "positive_target": labels.ravel(),
        "alarm": alarm.ravel(),
    })
    return metrics, family_frame, trajectory_frame, decisions
