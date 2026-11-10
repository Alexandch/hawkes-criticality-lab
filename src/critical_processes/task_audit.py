from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .negative_controls import SCENARIOS, poisson_control_rates, stage_features_from_counts
from .robust_training import FAMILIES, FAMILY_NAMES, TrajectoryBatch
from .simulation import simulate_staged_hawkes_counts
from .task_protocol import calibrate_task_threshold, evaluate_task_scores, state_warning_targets
from .window_benchmark import score_window_matrix
from .window_features import build_window_features, hawkes_trend_scores, window_feature_names


AUDIT_FAMILIES = FAMILIES + ("hawkes_high", "hawkes_falling", "hawkes_near_threshold", "hawkes_abrupt")
AUDIT_FAMILY_NAMES = {
    **FAMILY_NAMES, "hawkes_high": "Постоянно высокий n", "hawkes_falling": "Снижение из высокого n",
    "hawkes_near_threshold": "Постоянный n рядом с порогом", "hawkes_abrupt": "Внезапное пересечение",
}
SCORE_NAMES = {
    "count_mlp": "MLP: счётчики (заморожена)", "hawkes_mlp": "MLP: счётчики + MLE (заморожена)",
    "hawkes_trend": "Тренд оконного n", "current_mle": "MLE текущего n",
    "oracle_current_n": "Истинный текущий n (диагностика)",
}


@dataclass
class TaskAuditBatch:
    base: TrajectoryBatch
    windows: np.ndarray
    short_fits: np.ndarray
    diagnostics: np.ndarray
    parameter_seeds: np.ndarray
    simulator_seeds: np.ndarray


def audit_tasks(seed, sizes, config):
    rng = np.random.default_rng(seed)
    used, tasks = set(), {}

    def unique_seed():
        value = int(rng.integers(0, 2**32, dtype=np.uint64))
        while value in used:
            value = int(rng.integers(0, 2**32, dtype=np.uint64))
        used.add(value)
        return value

    for split, size in sizes.items():
        families = ("shared_prefix",) if split == "counterfactual" else AUDIT_FAMILIES
        tasks[split] = [(f"{split}/{family}/{rep}", family, unique_seed(), unique_seed(), config)
                        for family in families for rep in range(size)]
    return tasks


def simulate_audit_trajectory(task):
    identity, family, parameter_seed, simulator_seed, config = task
    rng, noise = np.random.default_rng(parameter_seed), np.random.default_rng(simulator_seed)
    stages, stage_days = config["stages"], config["stage_days"]
    critical, lookback, horizon = config["critical_level"], config["lookback"], config["forecast_stages"]
    rate = float(np.exp(rng.uniform(np.log(config["rate_min"]), np.log(config["rate_max"]))))
    anchors = [v for v in (2., 20., 50.) if config["rate_min"] <= v <= config["rate_max"]]
    if anchors and rng.random() < 0.3:
        rate = float(rng.choice(anchors))
    if family == "hawkes_rising":
        crossing = int(rng.integers(lookback + horizon + 1, stages - 1))
        before = np.linspace(rng.uniform(0.15, 0.5), rng.uniform(critical - 0.10, critical - 0.005), crossing)
        before = np.maximum.accumulate(np.clip(before + rng.normal(0, 0.005, crossing), 0.05, critical - 0.002))
        after = np.linspace(rng.uniform(critical + 0.005, min(critical + 0.04, 0.985)), 0.985, stages - crossing)
        ratios = np.r_[before, after]
    elif family == "hawkes_abrupt":
        crossing = int(rng.integers(lookback + horizon + 1, stages - 1))
        ratios = np.r_[np.full(crossing, rng.uniform(0.15, 0.5)), np.full(stages - crossing, rng.uniform(critical + 0.01, 0.98))]
    elif family == "hawkes_stable":
        ratios = np.full(stages, rng.uniform(0.05, critical - 0.01))
    elif family == "hawkes_subthreshold":
        ratios = np.linspace(rng.uniform(0.15, 0.5), rng.uniform(critical - 0.15, critical - 0.005), stages)
    elif family == "hawkes_near_threshold":
        ratios = np.full(stages, rng.uniform(critical - 0.04, critical - 0.001))
    elif family == "hawkes_high":
        ratios = np.full(stages, rng.uniform(critical + 0.01, 0.98))
    elif family == "hawkes_falling":
        transition = int(rng.integers(lookback + 1, stages - 2))
        ratios = np.r_[np.full(transition, rng.uniform(critical + 0.01, 0.98)),
                       np.linspace(critical - 0.04, 0.4, stages - transition)]
    elif family == "shared_prefix":
        stages = lookback + horizon + 1
        ratios = np.linspace(0.3, critical - 0.02, stages)
    elif family in SCENARIOS:
        ratios = np.zeros(stages)
    else:
        raise ValueError("unknown audit family")
    if family.startswith("hawkes_") or family == "shared_prefix":
        counts, events = simulate_staged_hawkes_counts(
            rate * (1 - ratios), ratios, stage_days, 1., simulator_seed,
            burn_in=8 / (1 - ratios[0]), record_events=True,
        )
    else:
        baseline = poisson_control_rates(family, rate, stages * stage_days, rng)
        counts = noise.poisson(baseline)
        days = np.repeat(np.arange(len(counts)), counts)
        events = np.sort(days + noise.random(len(days)))
    return counts, ratios, rate, events


def _audit_record(task):
    counts, ratios, rate, events = simulate_audit_trajectory(task)
    config = task[-1]
    windows, short, diagnostics = build_window_features(
        counts, events, config["stage_days"], config["lookback"], config["forecast_stages"], decay=1.,
    )
    return counts.astype(np.int32), ratios, rate, windows, short, diagnostics


def generate_audit_batch(tasks, workers=2):
    records = []
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        iterator = pool.map(_audit_record, tasks, chunksize=4) if pool else map(_audit_record, tasks)
        for i, record in enumerate(iterator, 1):
            records.append(record)
            if i % 200 == 0 or i == len(tasks):
                print(f"Task audit {tasks[0][0].split('/')[0]}: {i}/{len(tasks)} trajectories", flush=True)
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
    counts, ratios, rates, windows, short, diagnostics = zip(*records)
    base = TrajectoryBatch(
        np.stack([stage_features_from_counts(c, tasks[0][-1]["stage_days"]) for c in counts]),
        np.stack(ratios), np.asarray(rates), np.asarray([task[1] for task in tasks]),
        np.asarray([task[0] for task in tasks]), np.stack(counts),
    )
    return TaskAuditBatch(base, np.stack(windows), np.stack(short), np.stack(diagnostics),
                          np.asarray([task[2] for task in tasks]), np.asarray([task[3] for task in tasks]))


def save_audit_batch(batch, path):
    np.savez_compressed(path, **batch.base.__dict__, windows=batch.windows, short_fits=batch.short_fits,
                        diagnostics=batch.diagnostics, parameter_seeds=batch.parameter_seeds, simulator_seeds=batch.simulator_seeds)


def load_audit_batch(path):
    with np.load(path) as data:
        base = TrajectoryBatch(**{key: data[key].copy() for key in TrajectoryBatch.__dataclass_fields__})
        return TaskAuditBatch(base, **{key: data[key].copy() for key in TaskAuditBatch.__dataclass_fields__ if key != "base"})


def audit_scores(batch, model_dir, lookback=3):
    scores = {key: score_window_matrix(batch.windows, Path(model_dir) / f"{key}.pt") for key in ("count_mlp", "hawkes_mlp")}
    scores["hawkes_trend"] = hawkes_trend_scores(batch.diagnostics)
    current = batch.short_fits[:, lookback - 1:]
    valid = {key: np.ones_like(score, dtype=bool) for key, score in scores.items()}
    valid["hawkes_trend"] = (batch.diagnostics[:, :, 5] > 0) & (batch.diagnostics[:, :, 6] > 0)
    valid["current_mle"] = current[:, :, 4] > 0
    scores["current_mle"] = np.where(valid["current_mle"], current[:, :, 0], 0.)
    scores["oracle_current_n"] = batch.base.ratios[:, lookback - 1:].copy()
    valid["oracle_current_n"] = np.ones_like(scores["oracle_current_n"], dtype=bool)
    return scores, valid


def summarize_shared_prefix(scores, valid, thresholds, batch, config):
    """Two equally weighted possible futures share one observed prefix per pair."""
    prefix_stages = batch.base.ratios.shape[1]
    days = prefix_stages * config["stage_days"]
    rows, summary = [], []
    for method in SCORE_NAMES:
        score, usable = scores[method][:, -1], valid[method][:, -1]
        alarms = usable & (score >= thresholds[method]["warning"])
        for i, identity in enumerate(batch.base.ids):
            rows.append({
                "pair_id": identity, "method": method, "decision_day": days,
                "current_n_both_branches": batch.base.ratios[i, -1], "shared_score": score[i],
                "score_valid": usable[i], "alarm_both_branches": alarms[i],
                "warning_target_no_crossing": False, "warning_target_crossing": True,
                "future_crossing_day_positive_branch": days + config["stage_days"],
                "features_identical_by_shared_prefix": True,
            })
        summary.append({
            "method": method, "independent_prefix_pairs": len(score), "score_valid_fraction": usable.mean(),
            "threshold_warning": thresholds[method]["warning"],
            "sensitivity_crossing_branch": alarms.mean(), "false_alarm_probability_no_crossing_branch": alarms.mean(),
            "auc_paired_possible_futures": 0.5,
            "difference_in_detection_and_false_alarm_probability": 0.,
        })
    return pd.DataFrame(rows), pd.DataFrame(summary)


def run_task_audit(
    out="results/state_warning_audit", model_dir="results/window_detector_comparison/models",
    calibration_per_family=200, test_per_family=300, counterfactual_pairs=300,
    seed=20261116, workers=2, target_false_alarm_probability=0.05, bootstrap_samples=500,
    stage_days=90, stages=12, lookback=3, forecast_stages=2, critical_level=0.9, rate_min=0.2, rate_max=50.,
):
    import torch

    torch.set_num_threads(min(4, torch.get_num_threads()))
    if min(calibration_per_family, test_per_family, counterfactual_pairs, workers, bootstrap_samples) < 1 or seed < 0:
        raise ValueError("need positive dataset sizes, workers, bootstrap count and nonnegative seed")
    if not 0 < target_false_alarm_probability < 1 or calibration_per_family < np.ceil(1 / target_false_alarm_probability) - 1:
        raise ValueError("calibration size must support the false alarm target")
    if stage_days < 3 or lookback < 2 or forecast_stages < 1 or stages < lookback + forecast_stages + 3:
        raise ValueError("invalid stage duration, lookback or total horizon")
    if not 0.8 <= critical_level <= 0.95 or not 0 < rate_min <= rate_max:
        raise ValueError("invalid high-n threshold or rate bounds")
    config = dict(stage_days=stage_days, stages=stages, lookback=lookback, forecast_stages=forecast_stages,
                  critical_level=critical_level, rate_min=rate_min, rate_max=rate_max)
    model_dir, output = Path(model_dir), Path(out)
    if model_dir.resolve() == (output / "models").resolve():
        raise ValueError("output model snapshots must differ from source models")
    source_hashes, snapshots = {}, {}
    for method in ("count_mlp", "hawkes_mlp"):
        source = model_dir / f"{method}.pt"
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        if checkpoint.get("window_feature_names") != window_feature_names(lookback):
            raise ValueError("pretrained model feature schema does not match this audit")
        for key, expected in (("stage_horizon", stage_days), ("lookback", lookback),
                              ("forecast_stages", forecast_stages), ("critical_level", critical_level), ("fixed_decay_per_day", 1.)):
            if checkpoint.get(key) != expected:
                raise ValueError(f"pretrained model has incompatible {key}")
        snapshots[method] = source.read_bytes()
        source_hashes[method] = hashlib.sha256(snapshots[method]).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    for subdirectory in ("data", "models"):
        (output / subdirectory).mkdir(exist_ok=True)
    for method, snapshot in snapshots.items():
        (output / "models" / f"{method}.pt").write_bytes(snapshot)
    sizes = {"calibration": calibration_per_family, "test": test_per_family, "counterfactual": counterfactual_pairs}
    tasks = audit_tasks(seed, sizes, config)
    batches, hashes = {}, {}
    for split in sizes:
        batch = generate_audit_batch(tasks[split], workers)
        batches[split] = batch
        path = output / "data" / f"{split}.npz"
        save_audit_batch(batch, path)
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    protocol = dict(stage_days=stage_days, lookback=lookback, forecast_stages=forecast_stages, critical_level=critical_level)
    targets = {split: state_warning_targets(batches[split].base.ratios, **protocol) for split in ("calibration", "test")}
    scores, validity = {}, {}
    for split, batch in batches.items():
        scores[split], validity[split] = audit_scores(batch, output / "models", lookback)
        path = output / "data" / f"{split}_scores.npz"
        np.savez_compressed(path, **scores[split], **{method + "_valid": value for method, value in validity[split].items()})
        hashes[split + "_scores"] = hashlib.sha256(path.read_bytes()).hexdigest()
    tables, families, trajectories, metrics, thresholds = [], [], [], {}, {}
    for method in SCORE_NAMES:
        metrics[method], thresholds[method] = {}, {}
        for task in ("state", "warning"):
            threshold, table = calibrate_task_threshold(scores["calibration"][method], targets["calibration"][task],
                batches["calibration"].base.families, target_false_alarm_probability)
            thresholds[method][task] = threshold
            tables.append(table.assign(method=method, task=task))
            batch = batches["test"].base
            metric, family, trajectory, decisions = evaluate_task_scores(
                scores["test"][method], validity["test"][method], targets["test"][task], batch.families, batch.ids, batch.rates,
                threshold, task, bootstrap_seed=seed + 999, bootstrap_samples=bootstrap_samples,
            )
            metrics[method][task] = metric
            families.append(family.assign(method=method, task=task))
            trajectories.append(trajectory.assign(method=method, task=task))
            decisions.to_csv(output / f"{method}_{task}_test_scores.csv", index=False)
            print(f"{method}/{task}: sensitivity={metric['window_sensitivity']:.3f}; "
                  f"event recall={metric['event_detection_probability']:.3f}; max family false alarm={metric['maximum_family_false_alarm_probability']:.3f}", flush=True)
    pd.concat(tables, ignore_index=True).to_csv(output / "calibration.csv", index=False)
    pd.concat(families, ignore_index=True).to_csv(output / "family_test_metrics.csv", index=False)
    pd.concat(trajectories, ignore_index=True).to_csv(output / "trajectory_test_metrics.csv", index=False)
    counterfactual, counter_summary = summarize_shared_prefix(scores["counterfactual"], validity["counterfactual"], thresholds,
                                                            batches["counterfactual"], config)
    counterfactual.to_csv(output / "shared_prefix_scores.csv", index=False)
    counter_summary.to_csv(output / "shared_prefix_summary.csv", index=False)
    (output / "thresholds.json").write_text(json.dumps(thresholds, indent=2, allow_nan=False), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    manifest = {
        **config, "stage": "3_state_versus_first_future_crossing", "seed": seed, "workers": workers,
        "sizes_per_family": sizes, "families": AUDIT_FAMILIES, "methods": SCORE_NAMES,
        "total_independent_main_trajectories": len(tasks["calibration"]) + len(tasks["test"]),
        "counterfactual_pairs_are_not_independent_branches": True,
        "counterfactual_future_definition": "same prefix, then one further low stage; crossing at t+stage_days versus no crossing; equal branch weights",
        "state_target": "n(t-) >= critical_level in the last completed stage; no use of the next stage parameter",
        "warning_target": "first crossing tau with 0 < tau-t <= forecast_stages*stage_days, and t+H <= observation end",
        "no_warning_evaluation_at_or_after_first_crossing": True,
        "initially_high_excluded_from_warning_task": True,
        "false_alarm_unit": "at least one alarm at an eligible negative decision, per independent trajectory per family",
        "calibration_rule": "max family strict negative-max quantile, rank ceil((m+1)*(1-alpha)); no-negative families excluded explicitly",
        "thresholds_calibrated_separately_per_task": True,
        "target_false_alarm_probability": target_false_alarm_probability,
        "no_new_neural_training": True, "no_feature_or_model_selection_on_test": True,
        "source_model_directory": str(model_dir.resolve()), "source_model_sha256": source_hashes,
        "dataset_sha256": hashes, "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed + 999, "bootstrap_unit": "independent full trajectory",
        "auc": "pooled eligible valid decisions, descriptive point estimate; not an independent-window CI",
        "invalid_mle": "explicit abstention, zero detection score, counts as missed positive; n itself is not imputed",
        "known_decay_per_day": 1., "hawkes_kernel": "parent-time branching amplitude with continuous excitation across stage boundaries",
        "hawkes_baseline": "mu[k]=rate*(1-n[k]); constant local stationary mean, transients are not forced to have equal counts",
        "numpy_version": np.__version__, "torch_version": torch.__version__,
    }
    for method, digest in source_hashes.items():
        if hashlib.sha256((model_dir / f"{method}.pt").read_bytes()).hexdigest() != digest:
            raise RuntimeError("source model changed during the audit")
    (output / "run_config.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    from .task_report import write_task_report

    write_task_report(output)
    return metrics
