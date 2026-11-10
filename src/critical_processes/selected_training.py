from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .confident_calibration import calibrate_confident
from .n_accuracy import empirical_auc
from .selected_features import (
    FEATURE_GROUPS, generate_selection_batch, load_selection_batch, save_selection_batch,
    selection_feature_names, selection_tasks,
)
from .task_audit import AUDIT_FAMILIES
from .task_protocol import evaluate_task_scores, state_warning_targets
from .window_benchmark import fit_window_network, score_window_matrix


METHOD_NAMES = {
    "counts": "MLP: счётчики", "constant_mle": "MLP: обычный MLE", "calendar_mle": "MLP: календарный MLE",
    "stat_constant": "Статистика: обычный MLE", "stat_calendar": "Статистика: календарный MLE",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def selection_targets(batch, config):
    return state_warning_targets(batch.ratios, **{key: config[key] for key in ("stage_days", "lookback", "forecast_stages", "critical_level")})


def statistical_scores(batch, window_index, method, task):
    fit = batch.fits[:, window_index, :, 1 if method == "stat_calendar" else 0]
    valid = fit[:, :, 4] > 0
    if task == "state":
        scores = fit[:, :, 0]
    else:
        valid &= fit[:, :, 6] > 0
        projected = np.maximum(0., fit[:, :, 5])
        scores = projected / (1 + projected)
    return np.where(valid, scores, 0.), valid


def validation_metrics(scores, valid, target, families, threshold, task):
    alarm = valid & (scores >= threshold)
    positive, negative = target.labels & target.eligible, ~target.labels & target.eligible
    tp, fn = int((alarm & positive).sum()), int((~alarm & positive).sum())
    has_positive, has_negative = positive.any(axis=1), negative.any(axis=1)
    detected = (alarm & positive).any(axis=1)
    first = np.argmax(alarm & target.eligible, axis=1)
    has_alarm = (alarm & target.eligible).any(axis=1)
    clean = has_alarm & target.labels[np.arange(len(scores)), first]
    false = (alarm & negative).any(axis=1)
    family_false = [float(false[(families == f) & has_negative].mean()) for f in np.unique(families)
                    if ((families == f) & has_negative).any()]
    auc = empirical_auc(scores[negative & valid], scores[positive & valid])
    sensitivity = tp / (tp + fn) if tp + fn else 0.
    clean_recall = float(clean[has_positive].mean()) if has_positive.any() else 0.
    return {
        "utility": sensitivity if task == "state" else clean_recall,
        "window_sensitivity": sensitivity,
        "event_detection_probability": float(detected[has_positive].mean()) if has_positive.any() else 0.,
        "clean_recall": clean_recall, "maximum_family_false_alarm_probability": max(family_false),
        "auc_valid_only": float(auc) if np.isfinite(auc) else 0.,
    }


def choose_validation_candidate(frame, alpha=.05):
    """Fixed selection rule; test metrics are never an input."""
    feasible = frame[frame.maximum_family_false_alarm_probability <= alpha]
    if len(feasible):
        ordered = feasible.sort_values(["utility", "auc_valid_only", "window_days", "feature_count", "method"],
                                      ascending=[False, False, True, True, True])
    else:
        ordered = frame.sort_values(["maximum_family_false_alarm_probability", "utility", "auc_valid_only", "window_days", "feature_count", "method"],
                                    ascending=[True, False, False, True, True, True])
    selected = ordered.iloc[0].to_dict()
    selected["validation_feasible"] = bool(len(feasible))
    return selected


def score_selected_stream(batch, entry, output):
    if entry["method"].startswith("stat_"):
        return statistical_scores(batch, entry["window_index"], entry["method"], entry["task"])
    scores = [score_window_matrix(batch.matrix[:, entry["window_index"]], Path(output) / path) for path in entry["model_paths"]]
    result = np.mean(scores, axis=0)
    return result, np.ones_like(result, dtype=bool)


def _candidate_models(output, batches, targets, protocol, dataset_hashes):
    import torch

    rows, calibration_rows = [], []
    names = selection_feature_names()
    train, calibration, validation = [batches[split] for split in ("train", "validation_calibration", "validation")]
    for task in ("state", "warning"):
        for method, count in FEATURE_GROUPS.items():
            for window_index, window in enumerate(protocol["windows"]):
                cal_members, val_members = [], []
                for seed in protocol["model_seeds"]:
                    relative = f"models/{task}_{method}_w{window}_s{seed}.pt"
                    path = output / relative
                    checkpoint, history = fit_window_network(train.matrix[:, window_index], targets["train"][task].labels,
                        targets["train"][task].eligible, list(range(count)), seed, protocol["epochs"])
                    checkpoint.update({
                        "input_feature_schema": "selected_multiscale_calendar_v1", "window_feature_names": names,
                        "method": method, "task": task, "window_days": window, "model_seed": seed,
                        "epochs": protocol["epochs"], "training_dataset_sha256": dataset_hashes["train"],
                        "feature_transform_version": 2, "threshold": None, "threshold_source": "frozen_plan.json",
                    })
                    torch.save(checkpoint, path)
                    history.assign(task=task, method=method, window_days=window, model_seed=seed).to_csv(
                        output / "training" / f"{task}_{method}_w{window}_s{seed}.csv", index=False)
                    cal_score = score_window_matrix(calibration.matrix[:, window_index], path)
                    val_score = score_window_matrix(validation.matrix[:, window_index], path)
                    cal_members.append(cal_score)
                    val_members.append(val_score)
                    cutoff, table = calibrate_confident(cal_score, targets["validation_calibration"][task], calibration.families,
                        protocol["alpha"], protocol["confidence_failure_probability"])
                    quality = validation_metrics(val_score, np.ones_like(val_score, dtype=bool), targets["validation"][task], validation.families, cutoff, task)
                    rows.append({"task": task, "method": method, "window_days": window, "window_index": window_index,
                        "member": str(seed), "feature_count": count, "threshold": cutoff, **quality})
                    calibration_rows.append(table.assign(task=task, method=method, window_days=window, member=str(seed)))
                cal_score, val_score = np.mean(cal_members, axis=0), np.mean(val_members, axis=0)
                cutoff, table = calibrate_confident(cal_score, targets["validation_calibration"][task], calibration.families,
                    protocol["alpha"], protocol["confidence_failure_probability"])
                quality = validation_metrics(val_score, np.ones_like(val_score, dtype=bool), targets["validation"][task], validation.families, cutoff, task)
                rows.append({"task": task, "method": method, "window_days": window, "window_index": window_index,
                    "member": "ensemble", "feature_count": count, "threshold": cutoff, **quality})
                calibration_rows.append(table.assign(task=task, method=method, window_days=window, member="ensemble"))
                print(f"Validation {task}/{method}/window={window}: utility={quality['utility']:.3f}, max false alarm={quality['maximum_family_false_alarm_probability']:.3f}", flush=True)
        for method in ("stat_constant", "stat_calendar"):
            for window_index, window in enumerate(protocol["windows"]):
                cal_score, _ = statistical_scores(calibration, window_index, method, task)
                val_score, val_valid = statistical_scores(validation, window_index, method, task)
                cutoff, table = calibrate_confident(cal_score, targets["validation_calibration"][task], calibration.families,
                    protocol["alpha"], protocol["confidence_failure_probability"])
                quality = validation_metrics(val_score, val_valid, targets["validation"][task], validation.families, cutoff, task)
                rows.append({"task": task, "method": method, "window_days": window, "window_index": window_index,
                    "member": "statistic", "feature_count": 5, "threshold": cutoff, **quality})
                calibration_rows.append(table.assign(task=task, method=method, window_days=window, member="statistic"))
    pd.concat(calibration_rows, ignore_index=True).to_csv(output / "validation_calibration.csv", index=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "validation_candidates.csv", index=False)
    return frame


def _freeze_selection(output, batches, targets, protocol, dataset_hashes, candidates):
    selected, primary = [], {}
    for task in ("state", "warning"):
        for method in METHOD_NAMES:
            frame = candidates[(candidates.task == task) & (candidates.method == method) & candidates.member.isin(["ensemble", "statistic"])]
            selected.append(choose_validation_candidate(frame, protocol["alpha"]))
        primary[task] = choose_validation_candidate(pd.DataFrame([row for row in selected if row["task"] == task]), protocol["alpha"])["method"]
    selection = {"selected": selected, "primary_method_by_task": primary,
                 "selection_source": "validation only; ensemble rows for neural candidates", "test_used": False}
    write_json(output / "selection.json", selection)
    streams = []
    for row in selected:
        task, method, window = row["task"], row["method"], int(row["window_days"])
        seeds = [str(s) for s in protocol["model_seeds"]] + ["ensemble"] if method in FEATURE_GROUPS else ["statistic"]
        for member in seeds:
            model_seeds = protocol["model_seeds"] if member == "ensemble" else ([int(member)] if member != "statistic" else [])
            streams.append({"stream": f"{task}__{method}__{member}", "task": task, "method": method,
                "member": member, "window_days": window, "window_index": int(row["window_index"]),
                "primary": bool(primary[task] == method and member in ("ensemble", "statistic")),
                "model_paths": [f"models/{task}_{method}_w{window}_s{s}.pt" for s in model_seeds]})
    rows, saved_scores = [], {}
    calibration = batches["calibration"]
    for entry in streams:
        scores, valid = score_selected_stream(calibration, entry, output)
        cutoff, table = calibrate_confident(scores, targets["calibration"][entry["task"]], calibration.families,
            protocol["alpha"], protocol["confidence_failure_probability"] / len(streams))
        entry["threshold"] = cutoff
        saved_scores[entry["stream"]], saved_scores[entry["stream"] + "__valid"] = scores, valid
        rows.append(table.assign(stream=entry["stream"], task=entry["task"], method=entry["method"], member=entry["member"]))
    pd.concat(rows, ignore_index=True).to_csv(output / "calibration.csv", index=False)
    np.savez_compressed(output / "data" / "calibration_scores.npz", **saved_scores)
    models = {path: sha(output / path) for entry in streams for path in entry["model_paths"]}
    frozen = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": sha(output / "protocol.json"),
        "selection_sha256": sha(output / "selection.json"), "validation_candidates_sha256": sha(output / "validation_candidates.csv"),
        "dataset_sha256": dataset_hashes, "model_sha256": models, "calibration_scores_sha256": sha(output / "data" / "calibration_scores.npz"),
        "streams": streams, "primary_method_by_task": primary, "test_generated": False,
        "simultaneous_calibration_confidence": 1 - protocol["confidence_failure_probability"],
        "protected_score_streams": len(streams), "future_test_seed_declared_in_protocol": protocol["final_test_seed"]}
    write_json(output / "frozen_plan.json", frozen)
    print(f"FROZEN: selected windows, feature groups, {len(streams)} thresholds and model hashes; final test does not exist yet", flush=True)
    return frozen


def _final_evaluation(output, frozen, protocol, reserved):
    if sha(output / "selection.json") != frozen["selection_sha256"] or sha(output / "data/calibration_scores.npz") != frozen["calibration_scores_sha256"]:
        raise ValueError("selection or calibration scores changed after freezing")
    for model_path, digest in frozen["model_sha256"].items():
        if sha(output / model_path) != digest:
            raise ValueError("selected model changed before the final test")
    frozen_hash = sha(output / "frozen_plan.json")
    start_path = output / "final_test_started.json"
    if start_path.exists():
        start = json.loads(start_path.read_text(encoding="utf-8"))
        if start["frozen_plan_sha256"] != frozen_hash:
            raise ValueError("cannot reuse a final test after changing the frozen plan")
    else:
        write_json(start_path, {"started_at_utc": datetime.now(timezone.utc).isoformat(), "frozen_plan_sha256": frozen_hash})
    path = output / "data" / "test.npz"
    if path.exists():
        test = load_selection_batch(path)
    else:
        tasks = selection_tasks(protocol["final_test_seed"], "test", protocol["sizes"]["test"], protocol, reserved)
        test = generate_selection_batch(tasks, protocol["workers"])
        save_selection_batch(test, path)
    targets = selection_targets(test, protocol)
    rows, families, trajectories, raw_scores, metrics = [], [], [], {}, {}
    for entry in frozen["streams"]:
        score, valid = score_selected_stream(test, entry, output)
        metric, family, trajectory, decisions = evaluate_task_scores(score, valid, targets[entry["task"]], test.families,
            test.ids, test.rates, entry["threshold"], entry["task"], protocol["data_seed"] + 999, protocol["bootstrap_samples"])
        metrics[entry["stream"]] = metric
        rows.append({"stream": entry["stream"], "task": entry["task"], "method": entry["method"],
                     "member": entry["member"], "window_days": entry["window_days"], "primary": entry["primary"],
                     **{key: value for key, value in metric.items() if not isinstance(value, (list, tuple))}})
        families.append(family.assign(stream=entry["stream"], task=entry["task"], method=entry["method"], member=entry["member"]))
        trajectories.append(trajectory.assign(stream=entry["stream"], task=entry["task"], method=entry["method"], member=entry["member"]))
        raw_scores[entry["stream"]], raw_scores[entry["stream"] + "__valid"] = score, valid
        if entry["primary"]:
            decisions.assign(observation_end_day=decisions.decision_end_day + protocol["prefix_days"]).to_csv(output / f"primary_{entry['task']}_decisions.csv", index=False)
        print(f"FINAL {entry['stream']}: TPR={metric['window_sensitivity']:.3f}, event recall={metric['event_detection_probability']:.3f}, max false={metric['maximum_family_false_alarm_probability']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(output / "test_metrics.csv", index=False)
    pd.concat(families, ignore_index=True).to_csv(output / "family_test_metrics.csv", index=False)
    pd.concat(trajectories, ignore_index=True).to_csv(output / "trajectory_test_metrics.csv", index=False)
    np.savez_compressed(output / "data" / "test_scores.npz", **raw_scores)
    write_json(output / "metrics.json", metrics)
    for model_path, digest in frozen["model_sha256"].items():
        if sha(output / model_path) != digest:
            raise RuntimeError("selected model changed after freezing")
    if sha(output / "frozen_plan.json") != frozen_hash:
        raise RuntimeError("frozen plan changed during the final test")
    write_json(output / "final_test_complete.json", {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_plan_sha256": frozen_hash, "test_sha256": sha(path), "test_scores_sha256": sha(output / "data" / "test_scores.npz"),
        "metrics_sha256": sha(output / "metrics.json"), "test_metrics_sha256": sha(output / "test_metrics.csv")})
    return metrics


def run_selected_training(
    out="results/selected_detectors", train_per_family=150, validation_calibration_per_family=150,
    validation_per_family=150, calibration_per_family=300, test_per_family=300,
    model_seeds=(17, 29, 41), windows=(30, 90, 180, 360), epochs=100,
    data_seed=20261117, final_test_seed=20261118, workers=2, bootstrap_samples=500,
    alpha=.05, confidence_failure_probability=.05, prefix_days=730, stage_days=90, stages=12,
    lookback=3, forecast_stages=2, critical_level=.9, rate_min=.2, rate_max=50.,
):
    import torch

    torch.set_num_threads(min(4, torch.get_num_threads()))
    sizes = dict(train=train_per_family, validation_calibration=validation_calibration_per_family,
                 validation=validation_per_family, calibration=calibration_per_family, test=test_per_family)
    if min(*sizes.values(), epochs, workers, bootstrap_samples) < 1 or min(data_seed, final_test_seed, *model_seeds) < 0:
        raise ValueError("need positive dataset sizes, epochs, workers and nonnegative seeds")
    if len(model_seeds) < 2 or len(set(model_seeds)) != len(model_seeds) or data_seed == final_test_seed:
        raise ValueError("need distinct model seeds and a separate final-test seed")
    if lookback != 3 or prefix_days < 30 or stage_days < 3 or stages < lookback + forecast_stages + 3:
        raise ValueError("need three lag blocks and sufficient observation history")
    if not 0 < alpha < 1 or not 0 < confidence_failure_probability < 1 or not .8 <= critical_level <= .95 or not 0 < rate_min <= rate_max:
        raise ValueError("invalid probabilities, high-n threshold or rate range")
    windows = sorted(int(w) for w in windows)
    if not windows or len(set(windows)) != len(windows) or min(windows) < 3 or max(windows) > prefix_days + stage_days:
        raise ValueError("need distinct positive windows fitting the observed prefix")
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    for subdir in ("data", "models", "training"):
        (output / subdir).mkdir(exist_ok=True)
    sources = ("selected_training.py", "selected_features.py", "fast_window_mle.py", "confident_calibration.py",
               "task_protocol.py", "window_benchmark.py", "window_features.py", "simulation.py", "background_models.py")
    protocol = {"stage": "4_validation_selection_multiseed_frozen_test", "sizes": sizes, "windows": windows,
        "model_seeds": list(model_seeds), "epochs": epochs, "data_seed": data_seed, "final_test_seed": final_test_seed,
        "workers": workers, "bootstrap_samples": bootstrap_samples, "alpha": alpha,
        "confidence_failure_probability": confidence_failure_probability, "prefix_days": prefix_days,
        "stage_days": stage_days, "stages": stages, "lookback": lookback, "forecast_stages": forecast_stages,
        "critical_level": critical_level, "rate_min": rate_min, "rate_max": rate_max,
        "feature_names": selection_feature_names(), "feature_groups": FEATURE_GROUPS, "families": list(AUDIT_FAMILIES),
        "selection_rule": "per task/method select ensemble window: feasible validation max-family FPR<=alpha, then maximum utility (state TPR or warning clean recall), AUC, shorter window; if none feasible, minimum violation first",
        "primary_rule": "same validation rule across selected ensemble/statistical methods; never choose the best test seed",
        "final_calibration": "order-statistic tolerance bound; union bound over all frozen score streams and active negative families",
        "known_beta": 1., "no_real_data_used": True, "neural_initialization_only_varies_between_model_seeds": True,
        "future_horizon_fully_observed": True, "prefix_forecast_uses_counts_only": True,
        "optimizer": "full-batch Adam lr=0.003, class-weighted BCE; fixed epochs, no test early stopping",
        "source_sha256": {name: sha(Path(__file__).parent / name) for name in sources},
        "numpy_version": np.__version__, "torch_version": torch.__version__}
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("output already belongs to a different protocol; use a new output directory")
    else:
        write_json(protocol_path, protocol)
    if (output / "final_test_complete.json").exists():
        complete = json.loads((output / "final_test_complete.json").read_text(encoding="utf-8"))
        for relative, key in (("frozen_plan.json", "frozen_plan_sha256"), ("data/test.npz", "test_sha256"),
                              ("metrics.json", "metrics_sha256"), ("data/test_scores.npz", "test_scores_sha256")):
            if sha(output / relative) != complete[key]:
                raise ValueError("completed experiment artifacts changed; verify before reuse")
        print("Final experiment already complete; returning saved metrics without a new test", flush=True)
        return json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    if not (output / "frozen_plan.json").exists() and (output / "data/test.npz").exists():
        raise ValueError("final-test data must not exist before the selection is frozen")
    reserved, batches, hashes = [], {}, {}
    for split in ("train", "validation_calibration", "validation", "calibration"):
        path = output / "data" / f"{split}.npz"
        if path.exists():
            batch = load_selection_batch(path)
        else:
            seed = int(np.random.SeedSequence([data_seed, list(sizes).index(split)]).generate_state(1)[0])
            tasks = selection_tasks(seed, split, sizes[split], protocol, reserved)
            batch = generate_selection_batch(tasks, workers)
            save_selection_batch(batch, path)
        reserved.extend(batch.parameter_seeds.tolist() + batch.simulator_seeds.tolist())
        batches[split], hashes[split] = batch, sha(path)
    if (output / "frozen_plan.json").exists():
        frozen = json.loads((output / "frozen_plan.json").read_text(encoding="utf-8"))
        if frozen["dataset_sha256"] != hashes or frozen["protocol_sha256"] != sha(protocol_path):
            raise ValueError("frozen plan no longer matches the data or protocol")
    else:
        targets = {split: selection_targets(batch, protocol) for split, batch in batches.items()}
        candidates = _candidate_models(output, batches, targets, protocol, hashes)
        frozen = _freeze_selection(output, batches, targets, protocol, hashes, candidates)
    metrics = _final_evaluation(output, frozen, protocol, reserved)
    from .selected_report import write_selected_report

    write_selected_report(output)
    return json.loads(json.dumps(metrics, allow_nan=False))
