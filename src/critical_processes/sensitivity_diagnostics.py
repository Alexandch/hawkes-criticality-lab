from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .confident_calibration import calibrate_confident
from .n_accuracy import empirical_auc
from .selected_features import generate_selection_batch, load_selection_batch, save_selection_batch, selection_tasks
from .selected_training import score_selected_stream, selection_targets, sha, statistical_scores, write_json
from .task_protocol import calibrate_task_threshold, evaluate_task_scores
from .window_benchmark import fit_window_network, score_window_matrix


POLICIES = ("fixed_score", "pooled_trajectory", "family_nominal", "family_confident",
            "family_confident_source_scope", "family_nominal_without_near")
POLICY_NAMES = {
    "frozen_stage4": "Порог этапа 4", "fixed_score": "Фиксированный исходный порог",
    "pooled_trajectory": "5% потоков, общая смесь", "family_nominal": "5% потоков, каждое семейство",
    "family_confident": "95% доверие, одна оценка", "family_confident_source_scope": "95% доверие, весь охват этапа 4",
    "family_nominal_without_near": "Калибровка без близкого к порогу контроля",
}
METHOD_NAMES = {
    "frozen_primary": "Основная модель этапа 4", "mle_constant": "MLE последней стадии",
    "mle_calendar": "Календарный MLE последней стадии", "oracle_current": "Истинный текущий n",
    "oracle_linear": "Проекция истинной истории n", "paired_known": "Парная MLP: истинная история n",
    "paired_constant": "Парная MLP: история обычного MLE", "paired_calendar": "Парная MLP: история календарного MLE",
}


def n_history_features(batch, source, horizon=2):
    """Same eight-dimensional representation; only the source of n differs."""
    if source == "known":
        values = np.lib.stride_tricks.sliding_window_view(batch.ratios, 3, axis=1).copy()
        valid = np.ones_like(values)
    else:
        offset = 16 if source == "constant" else 39 if source == "calendar" else None
        if offset is None:
            raise ValueError("unknown n-history source")
        base = batch.matrix[:, 0]
        values = np.stack([base[:, :, offset + 5 * lag] for lag in range(3)], axis=-1)
        valid = np.stack([base[:, :, offset + 5 * lag + 4] for lag in range(3)], axis=-1)
    slope = (values[:, :, 2] - values[:, :, 0]) / 2
    projected = values[:, :, 2] + horizon * slope
    return np.concatenate([values, slope[:, :, None], projected[:, :, None], valid], axis=-1)


def score_diagnostic_batch(batch, output, config):
    result = {}
    for task in ("state", "warning"):
        entry = config["primary_entries"][task]
        score, valid = score_selected_stream(batch, entry, output)
        result[f"{task}__frozen_primary"] = (score, valid)
        for variant in ("constant", "calendar"):
            score, valid = statistical_scores(batch, config["stage_window_index"], f"stat_{variant}", task)
            result[f"{task}__mle_{variant}"] = (score, valid)
        current = batch.ratios[:, config["generator"]["lookback"] - 1:]
        result[f"{task}__oracle_current"] = (current.copy(), np.ones_like(current, dtype=bool))
        if task == "warning":
            history = n_history_features(batch, "known", config["generator"]["forecast_stages"])
            projection = np.maximum(history[:, :, 4], 0.)
            score = projection / (1 + projection)
            result[f"{task}__oracle_linear"] = (score, np.ones_like(score, dtype=bool))
        for source in ("known", "constant", "calendar"):
            features = n_history_features(batch, source, config["generator"]["forecast_stages"])
            score = score_window_matrix(features, Path(output) / "paired_models" / f"{task}_{source}.pt")
            result[f"{task}__paired_{source}"] = (score, np.ones_like(score, dtype=bool))
    return result


def calibration_policy(scores, target, families, task, method, policy, alpha=.05, source_scope=28, critical_level=.9):
    families = np.asarray(families)
    if policy not in POLICIES:
        raise ValueError("unknown calibration policy")
    if policy == "fixed_score":
        if method.startswith("mle_") or method == "oracle_linear":
            cutoff = critical_level if task == "state" else critical_level / (1 + critical_level)
        else:
            cutoff = critical_level if method == "oracle_current" else .5
        return cutoff, pd.DataFrame()
    if policy in ("family_confident", "family_confident_source_scope"):
        divisor = source_scope if policy == "family_confident_source_scope" else 1
        cutoff, table = calibrate_confident(scores, target, families, alpha, .05 / divisor)
        table["binding"] = table.family_cutoff == cutoff
        return cutoff, table
    selected = families != "hawkes_near_threshold" if policy == "family_nominal_without_near" else np.ones(len(families), dtype=bool)
    if policy == "pooled_trajectory":
        groups = np.repeat("pooled", selected.sum())
    else:
        groups = families[selected]
    from .task_protocol import TaskTargets
    subset = TaskTargets(target.labels[selected], target.eligible[selected], target.decision_days,
                         target.current_n[selected], target.first_crossing[selected])
    cutoff, table = calibrate_task_threshold(scores[selected], subset, groups, alpha)
    table["family_cutoff"] = np.nextafter(table.negative_max_quantile, np.inf)
    table["binding"] = table.family_cutoff == cutoff
    return cutoff, table


def diagnostic_threshold_grid(scores, target, operating_thresholds):
    """Grid fixed by calibration scores, never by evaluation labels or scores."""
    values = scores[target.eligible]
    quantiles = np.unique(np.r_[np.linspace(0, .95, 61), np.linspace(.95, 1, 81)])
    grid = np.quantile(values, quantiles) if len(values) else np.array([.5])
    grid = np.r_[grid, operating_thresholds, np.nextafter(0., np.inf), np.nextafter(1., np.inf)]
    return np.unique(np.clip(grid, np.nextafter(0., np.inf), np.nextafter(1., np.inf)))


def operating_curve(scores, valid, target, families, thresholds):
    positive, negative = target.labels & target.eligible, ~target.labels & target.eligible
    has_positive, has_negative = positive.any(axis=1), negative.any(axis=1)
    rows = []
    for cutoff in thresholds:
        alarm = valid & (scores >= cutoff)
        false = (alarm & negative).any(axis=1)
        hit = (alarm & positive).any(axis=1)
        eligible_alarm = alarm & target.eligible
        first = np.argmax(eligible_alarm, axis=1)
        clean = eligible_alarm.any(axis=1) & target.labels[np.arange(len(scores)), first]
        maximum_false = max(float(false[(families == f) & has_negative].mean()) for f in np.unique(families)
                            if ((families == f) & has_negative).any())
        rows.append({"threshold": cutoff,
            "window_sensitivity": float((alarm & positive).sum() / positive.sum()) if positive.any() else np.nan,
            "event_detection_probability": float(hit[has_positive].mean()) if has_positive.any() else np.nan,
            "clean_recall": float(clean[has_positive].mean()) if has_positive.any() else np.nan,
            "maximum_family_false_alarm_probability": maximum_false})
    return pd.DataFrame(rows)


def estimation_rows(batch, config):
    generator = config["generator"]
    prefix, width, lookback = [generator[key] for key in ("prefix_days", "stage_days", "lookback")]
    days = np.arange(lookback, generator["stages"] + 1) * width
    daily_truth = np.concatenate([np.repeat(batch.ratios[:, :1], prefix, axis=1), np.repeat(batch.ratios, width, axis=1)], axis=1)
    rates = np.repeat(batch.rates, len(days))
    current = batch.ratios[:, lookback - 1:]
    rate_bin = np.where(rates < 2, "0.2..2", np.where(rates < 20, "2..20", "20..50"))
    true_history = n_history_features(batch, "known", generator["forecast_stages"])
    records = []
    for window_index, window in enumerate(generator["windows"]):
        average = np.stack([daily_truth[:, prefix+day-window:prefix+day].mean(axis=1) for day in days], axis=1)
        for variant_index, variant in enumerate(("constant", "calendar")):
            fit = batch.fits[:, window_index, :, variant_index]
            valid = fit[:, :, 4] > 0
            observed_history = n_history_features(batch, variant, generator["forecast_stages"])
            records.append(pd.DataFrame({
                "trajectory_id": np.repeat(batch.ids, len(days)), "family": np.repeat(batch.families, len(days)),
                "target_rate": rates, "rate_bin": rate_bin, "window_days": window, "variant": variant,
                "decision_end_day": np.tile(days, len(batch.ids)), "true_current_n": current.ravel(),
                "true_window_mean_n": average.ravel(), "fit_valid": valid.ravel(),
                "n_hat": np.where(valid, fit[:, :, 0], np.nan).ravel(),
                "true_history_slope": true_history[:, :, 3].ravel(),
                "estimated_history_slope": observed_history[:, :, 3].ravel(),
                "history_fits_all_valid": (observed_history[:, :, 5:].min(axis=-1) > 0).ravel(),
            }))
    return pd.concat(records, ignore_index=True)


def summarize_estimation(raw):
    records = []
    for keys, frame in raw.groupby(["family", "rate_bin", "window_days", "variant"], sort=True):
        valid = frame[frame.fit_valid]
        error = valid.n_hat - valid.true_current_n
        window_error = valid.n_hat - valid.true_window_mean_n
        slope = frame[frame.history_fits_all_valid]
        records.append({**dict(zip(("family", "rate_bin", "window_days", "variant"), keys)),
            "windows": len(frame), "valid_fraction": len(valid) / len(frame),
            "bias_to_current": error.mean(), "mae_to_current": error.abs().mean(),
            "rmse_to_current": np.sqrt((error ** 2).mean()), "mae_to_window_mean": window_error.abs().mean(),
            "mean_absolute_window_smoothing": (frame.true_window_mean_n - frame.true_current_n).abs().mean(),
            "upper_bound_fraction_valid": (valid.n_hat >= .995 - 1e-8).mean() if len(valid) else np.nan,
            "slope_mae_valid_history": (slope.estimated_history_slope - slope.true_history_slope).abs().mean()})
    return pd.DataFrame(records)


def conditional_auc_rows(scores_by_method, targets, families):
    rows = []
    for key, (score, valid) in scores_by_method.items():
        task, method = key.split("__", 1)
        target = targets[task]
        for positive_family in np.unique(families):
            positive = target.labels & target.eligible & (families == positive_family)[:, None]
            if not positive.any():
                continue
            for negative_family in np.unique(families):
                negative = ~target.labels & target.eligible & (families == negative_family)[:, None]
                if not negative.any():
                    continue
                auc = empirical_auc(score[negative & valid], score[positive & valid])
                rows.append({"task": task, "method": method, "positive_family": positive_family, "negative_family": negative_family,
                    "auc_valid_only": auc, "positive_decisions": int(positive.sum()), "negative_decisions": int(negative.sum()),
                    "positive_valid_fraction": float(valid[positive].mean()), "negative_valid_fraction": float(valid[negative].mean())})
    return pd.DataFrame(rows)


def _source_snapshot(source, frozen):
    files = ["protocol.json", "selection.json", "frozen_plan.json", "final_test_complete.json", "metrics.json",
             "test_metrics.csv", "data/test.npz", "data/test_scores.npz", *frozen["model_sha256"]]
    return {name: sha(source / name) for name in files}


def summarize_effects(vectors, metrics, seed, bootstrap_samples):
    """Paired contrasts of policies or n-information sources, on identical trajectories."""
    rng, rows = np.random.default_rng(seed), []
    for task in ("state", "warning"):
        contrasts = [
            ("nominal_vs_frozen", "frozen_primary", "family_nominal", "frozen_primary", "frozen_stage4"),
            ("confidence_cost", "frozen_primary", "family_nominal", "frozen_primary", "family_confident"),
            ("multiplicity_cost", "frozen_primary", "family_confident", "frozen_primary", "family_confident_source_scope"),
            ("near_control_cost", "frozen_primary", "family_nominal_without_near", "frozen_primary", "family_nominal"),
            ("family_control_cost", "frozen_primary", "pooled_trajectory", "frozen_primary", "family_nominal"),
        ]
        for policy in ("family_nominal", "family_confident_source_scope"):
            contrasts += [(f"known_vs_{source}", "paired_known", policy, f"paired_{source}", policy) for source in ("constant", "calendar")]
            contrasts.append(("calendar_vs_constant", "paired_calendar", policy, "paired_constant", policy))
        for contrast, proposed, proposed_policy, reference, reference_policy in contrasts:
            a = f"{task}__{proposed}__{proposed_policy}"
            b = f"{task}__{reference}__{reference_policy}"
            numerator, denominator = vectors[a]
            other_numerator, other_denominator = vectors[b]
            np.testing.assert_array_equal(denominator, other_denominator)
            difference = numerator - other_numerator
            draws = []
            for _ in range(bootstrap_samples):
                indices = rng.integers(0, len(difference), len(difference))
                if denominator[indices].sum():
                    draws.append(difference[indices].sum() / denominator[indices].sum())
            lower, upper = np.quantile(draws, [.025, .975]) if draws else (np.nan, np.nan)
            rows.append({"task": task, "contrast": contrast, "proposed_method": proposed, "proposed_policy": proposed_policy,
                "reference_method": reference, "reference_policy": reference_policy,
                "utility_difference": difference.sum() / denominator.sum() if denominator.sum() else np.nan,
                "paired_ci95_lower": lower, "paired_ci95_upper": upper,
                "proposed_maximum_family_false_alarm": metrics[a]["maximum_family_false_alarm_probability"],
                "reference_maximum_family_false_alarm": metrics[b]["maximum_family_false_alarm_probability"]})
    return pd.DataFrame(rows)


def run_sensitivity_diagnostics(source="results/selected_detectors", out="results/sensitivity_diagnostics",
    train_per_family=150, calibration_per_family=300, evaluation_per_family=300,
    data_seed=20261119, evaluation_seed=20261120, model_seed=53, epochs=100, workers=2, bootstrap_samples=500):
    import torch

    torch.set_num_threads(min(4, torch.get_num_threads()))
    source, output = Path(source), Path(out)
    if source.resolve() == output.resolve():
        raise ValueError("diagnostics require a separate output directory")
    if min(train_per_family, calibration_per_family, evaluation_per_family, epochs, workers, bootstrap_samples) < 1 or min(data_seed, evaluation_seed, model_seed) < 0:
        raise ValueError("need positive sizes and nonnegative seeds")
    if calibration_per_family < 19 or evaluation_seed == data_seed:
        raise ValueError("need at least 19 calibration trajectories per family and an independent evaluation seed")
    original_protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    frozen = json.loads((source / "frozen_plan.json").read_text(encoding="utf-8"))
    snapshot = _source_snapshot(source, frozen)
    completed = json.loads((source / "final_test_complete.json").read_text(encoding="utf-8"))
    for name, field in (("frozen_plan.json", "frozen_plan_sha256"), ("metrics.json", "metrics_sha256"),
                         ("data/test.npz", "test_sha256"), ("data/test_scores.npz", "test_scores_sha256")):
        if snapshot[name] != completed[field]:
            raise ValueError("source experiment integrity check failed")
    for name, digest in frozen["model_sha256"].items():
        if snapshot[name] != digest:
            raise ValueError("source weights no longer match the frozen experiment")
    generator = {key: original_protocol[key] for key in ("prefix_days", "stage_days", "stages", "lookback", "forecast_stages", "critical_level", "rate_min", "rate_max", "windows")}
    primary = {entry["task"]: entry for entry in frozen["streams"] if entry["primary"]}
    sizes = dict(train=train_per_family, calibration=calibration_per_family, evaluation=evaluation_per_family)
    config = {"stage": "5_estimation_vs_calibration_diagnostics", "generator": generator, "sizes": sizes,
        "data_seed": data_seed, "evaluation_seed": evaluation_seed, "model_seed": model_seed,
        "epochs": epochs, "workers": workers, "bootstrap_samples": bootstrap_samples,
        "source_directory": str(source.resolve()), "source_snapshot_sha256": snapshot,
        "primary_entries": primary, "stage_window_index": generator["windows"].index(generator["stage_days"]),
        "source_protected_streams": frozen["protected_score_streams"], "alpha": .05,
        "policies": POLICIES, "curve_grid": "calibration quantiles only: 61 points 0..0.95, 81 points 0.95..1, named cutoffs and endpoints",
        "new_matched_models": "same 8 features, architecture and initialization; n history source is known/constant/calendar; labels used only for training and evaluation",
        "matched_feature_names": ["n_lag2", "n_lag1", "n_lag0", "slope", "linear_projection", "valid_lag2", "valid_lag1", "valid_lag0"],
        "paired_models_are_diagnostic_not_deployable": True, "old_holdout_values_not_used_for_selection_or_fitting": True,
        "calibration_without_near_control_still_evaluates_all_families": True,
        "diagnostic_curves_do_not_choose_a_new_deployment_threshold": True}
    config["policies"] = list(config["policies"])
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("data", "models", "paired_models"):
        (output / directory).mkdir(exist_ok=True)
    if (output / "protocol.json").exists():
        if json.loads((output / "protocol.json").read_text(encoding="utf-8")) != config:
            raise ValueError("diagnostic output belongs to a different fixed protocol")
    else:
        write_json(output / "protocol.json", config)
    for entry in primary.values():
        for relative in entry["model_paths"]:
            (output / relative).write_bytes((source / relative).read_bytes())
    reserved = []
    for split in original_protocol["sizes"]:
        with np.load(source / "data" / f"{split}.npz") as data:
            reserved.extend(data["parameter_seeds"].tolist() + data["simulator_seeds"].tolist())
    batches, hashes = {}, {}
    for index, (split, size) in enumerate(sizes.items()):
        path = output / "data" / f"{split}.npz"
        if path.exists():
            batch = load_selection_batch(path)
        else:
            seed = evaluation_seed if split == "evaluation" else int(np.random.SeedSequence([data_seed, index]).generate_state(1)[0])
            batch = generate_selection_batch(selection_tasks(seed, f"diagnostic_{split}", size, generator, reserved), workers)
            save_selection_batch(batch, path)
        batches[split], hashes[split] = batch, sha(path)
        reserved.extend(batch.parameter_seeds.tolist() + batch.simulator_seeds.tolist())
    targets = {split: selection_targets(batch, generator) for split, batch in batches.items()}
    for task in ("state", "warning"):
        for variant in ("known", "constant", "calendar"):
            x = n_history_features(batches["train"], variant, generator["forecast_stages"])
            checkpoint, history = fit_window_network(x, targets["train"][task].labels, targets["train"][task].eligible,
                list(range(8)), model_seed, epochs)
            checkpoint.update({"input_feature_schema": "n_history_diagnostic_v1", "window_feature_names": config["matched_feature_names"],
                "source": variant, "task": task, "model_seed": model_seed, "threshold": None, "diagnostic_only": True,
                "training_dataset_sha256": hashes["train"]})
            torch.save(checkpoint, output / "paired_models" / f"{task}_{variant}.pt")
            history.to_csv(output / "paired_models" / f"{task}_{variant}_loss.csv", index=False)
    scores = {split: score_diagnostic_batch(batches[split], output, config) for split in ("calibration", "evaluation")}
    for split in scores:
        np.savez_compressed(output / "data" / f"{split}_scores.npz", **{
            name: value for key, pair in scores[split].items() for name, value in ((key, pair[0]), (key + "__valid", pair[1]))})
    metrics, metric_rows, family_rows, calibration_rows, curves, vectors = {}, [], [], [], [], {}
    for key, (score, valid) in scores["evaluation"].items():
        task, method = key.split("__", 1)
        cal_score = scores["calibration"][key][0]
        policies = (("frozen_stage4",) if method == "frozen_primary" else ()) + POLICIES
        cutoffs = []
        for policy in policies:
            if policy == "frozen_stage4":
                cutoff, table = primary[task]["threshold"], pd.DataFrame()
            else:
                policy_method = method
                if method == "frozen_primary" and primary[task]["method"].startswith("stat_"):
                    policy_method = "mle_" + primary[task]["method"].removeprefix("stat_")
                cutoff, table = calibration_policy(cal_score, targets["calibration"][task], batches["calibration"].families,
                    task, policy_method, policy, source_scope=config["source_protected_streams"], critical_level=generator["critical_level"])
            cutoffs.append(cutoff)
            if not table.empty:
                calibration_rows.append(table.assign(task=task, method=method, policy=policy))
            metric, family, trajectory, _ = evaluate_task_scores(score, valid, targets["evaluation"][task], batches["evaluation"].families,
                batches["evaluation"].ids, batches["evaluation"].rates, cutoff, task,
                bootstrap_seed=data_seed + 999, bootstrap_samples=bootstrap_samples)
            name = f"{key}__{policy}"
            metrics[name] = metric
            vectors[name] = (trajectory.true_positive.to_numpy(), trajectory.positive_windows.to_numpy()) if task == "state" else (
                trajectory.clean_detection.eq(True).to_numpy(dtype=int), trajectory.has_positive.to_numpy(dtype=int))
            metric_rows.append({"task": task, "method": method, "policy": policy,
                **{column: value for column, value in metric.items() if not isinstance(value, (list, tuple))}})
            family_rows.append(family.assign(task=task, method=method, policy=policy))
        grid = diagnostic_threshold_grid(cal_score, targets["calibration"][task], cutoffs)
        curves.append(operating_curve(score, valid, targets["evaluation"][task], batches["evaluation"].families, grid).assign(task=task, method=method))
        print(f"Sensitivity diagnosis: {key}, {len(policies)} calibration policies checked", flush=True)
    write_json(output / "metrics.json", metrics)
    summarize_effects(vectors, metrics, data_seed + 1234, bootstrap_samples).to_csv(output / "paired_effects.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output / "operating_metrics.csv", index=False)
    pd.concat(family_rows, ignore_index=True).to_csv(output / "family_metrics.csv", index=False)
    pd.concat(calibration_rows, ignore_index=True).to_csv(output / "calibration_policies.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(output / "threshold_curves.csv", index=False)
    estimation = estimation_rows(batches["evaluation"], config)
    estimation.to_csv(output / "estimation_windows.csv", index=False)
    summarize_estimation(estimation).to_csv(output / "estimation_summary.csv", index=False)
    conditional_auc_rows(scores["evaluation"], targets["evaluation"], batches["evaluation"].families).to_csv(output / "conditional_auc.csv", index=False)
    original_calibration = pd.read_csv(source / "calibration.csv")
    original_calibration[original_calibration.stream.isin([entry["stream"] for entry in primary.values()])].to_csv(output / "source_calibration.csv", index=False)
    if _source_snapshot(source, frozen) != snapshot:
        raise RuntimeError("the original final experiment changed during diagnostics")
    artifact_names = ["operating_metrics.csv", "family_metrics.csv", "calibration_policies.csv", "threshold_curves.csv",
                      "estimation_windows.csv", "estimation_summary.csv", "conditional_auc.csv", "paired_effects.csv", "metrics.json"]
    write_json(output / "run_config.json", {"dataset_sha256": hashes, "old_experiment_unchanged": True,
        "artifact_sha256": {name: sha(output / name) for name in artifact_names},
        "model_sha256": {str(path.relative_to(output)).replace("\\", "/"): sha(path) for path in output.rglob("*.pt")},
        "scores_sha256": {split: sha(output / "data" / f"{split}_scores.npz") for split in scores},
        "numpy_version": np.__version__, "torch_version": torch.__version__})
    from .sensitivity_report import write_sensitivity_report

    write_sensitivity_report(output)
    return json.loads(json.dumps(metrics, allow_nan=False))
