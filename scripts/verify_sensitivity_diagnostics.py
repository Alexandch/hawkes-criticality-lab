from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from critical_processes.selected_features import (
    build_selection_features, load_selection_batch, selection_tasks, simulate_selection_trajectory,
)
from critical_processes.selected_training import selection_targets, sha
from critical_processes.sensitivity_diagnostics import (
    POLICIES, calibration_policy, conditional_auc_rows, diagnostic_threshold_grid, estimation_rows,
    n_history_features, operating_curve, score_diagnostic_batch, summarize_effects, summarize_estimation,
)
from critical_processes.task_protocol import evaluate_task_scores


def assert_csv(expected, path):
    saved = pd.read_csv(path, float_precision="round_trip")
    expected = expected.reset_index(drop=True)
    expected = expected.where(expected.notna(), np.nan)
    pd.testing.assert_frame_equal(expected, saved, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10)


def main():
    import torch

    torch.set_num_threads(min(4, torch.get_num_threads()))
    parser = argparse.ArgumentParser(description="Verify the separation of estimation and calibration effects on new diagnostic data.")
    parser.add_argument("--out", type=Path, default=Path("results/sensitivity_diagnostics"))
    output = parser.parse_args().out
    config = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    run = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    source = Path(config["source_directory"])
    for name, digest in config["source_snapshot_sha256"].items():
        assert sha(source / name) == digest, f"original experiment changed: {name}"
    for name, digest in run["artifact_sha256"].items():
        assert sha(output / name) == digest
    for name, digest in run["model_sha256"].items():
        assert sha(output / name) == digest
    for split, digest in run["scores_sha256"].items():
        assert sha(output / "data" / f"{split}_scores.npz") == digest
    source_protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    reserved, all_ids, batches, targets, tasks_by_split = [], [], {}, {}, {}
    for split in source_protocol["sizes"]:
        with np.load(source / "data" / f"{split}.npz") as data:
            reserved.extend(data["parameter_seeds"].tolist() + data["simulator_seeds"].tolist())
    for index, (split, size) in enumerate(config["sizes"].items()):
        path = output / "data" / f"{split}.npz"
        assert sha(path) == run["dataset_sha256"][split]
        batch = load_selection_batch(path)
        batches[split], targets[split] = batch, selection_targets(batch, config["generator"])
        seed = config["evaluation_seed"] if split == "evaluation" else int(np.random.SeedSequence([config["data_seed"], index]).generate_state(1)[0])
        tasks = selection_tasks(seed, f"diagnostic_{split}", size, config["generator"], reserved)
        tasks_by_split[split] = tasks
        np.testing.assert_array_equal(batch.ids, [task[0] for task in tasks])
        np.testing.assert_array_equal(batch.parameter_seeds, [task[2] for task in tasks])
        np.testing.assert_array_equal(batch.simulator_seeds, [task[3] for task in tasks])
        all_ids.extend(batch.ids.tolist())
        reserved.extend(batch.parameter_seeds.tolist() + batch.simulator_seeds.tolist())
    assert len(all_ids) == len(set(all_ids))
    assert len(reserved) == len(set(reserved))
    for task in ("state", "warning"):
        for variant in ("known", "constant", "calendar"):
            checkpoint = torch.load(output / "paired_models" / f"{task}_{variant}.pt", map_location="cpu", weights_only=False)
            assert checkpoint["diagnostic_only"]
            assert checkpoint["training_dataset_sha256"] == run["dataset_sha256"]["train"]
            x = n_history_features(batches["train"], variant, config["generator"]["forecast_stages"])[targets["train"][task].eligible]
            np.testing.assert_allclose(checkpoint["mean"], x.mean(axis=0), rtol=1e-12, atol=1e-12)
            scale = x.std(axis=0)
            scale[scale < 1e-8] = 1.
            np.testing.assert_allclose(checkpoint["scale"], scale, rtol=1e-12, atol=1e-12)
    scores = {split: score_diagnostic_batch(batches[split], output, config) for split in ("calibration", "evaluation")}
    for split in scores:
        with np.load(output / "data" / f"{split}_scores.npz") as saved:
            for key, (score, valid) in scores[split].items():
                np.testing.assert_allclose(score, saved[key], rtol=1e-12, atol=1e-12)
                np.testing.assert_array_equal(valid, saved[key + "__valid"])
    metric_rows, family_rows, calibration_tables, curves, vectors = [], [], [], [], {}
    for key, (score, valid) in scores["evaluation"].items():
        task, method = key.split("__", 1)
        cal_score = scores["calibration"][key][0]
        cutoffs = []
        for policy in (("frozen_stage4",) if method == "frozen_primary" else ()) + POLICIES:
            if policy == "frozen_stage4":
                cutoff, table = config["primary_entries"][task]["threshold"], pd.DataFrame()
            else:
                policy_method = method
                if method == "frozen_primary" and config["primary_entries"][task]["method"].startswith("stat_"):
                    policy_method = "mle_" + config["primary_entries"][task]["method"].removeprefix("stat_")
                cutoff, table = calibration_policy(cal_score, targets["calibration"][task], batches["calibration"].families,
                    task, policy_method, policy, source_scope=config["source_protected_streams"],
                    critical_level=config["generator"]["critical_level"])
            cutoffs.append(cutoff)
            if not table.empty:
                calibration_tables.append(table.assign(task=task, method=method, policy=policy))
            metric, family, trajectory, _ = evaluate_task_scores(score, valid, targets["evaluation"][task], batches["evaluation"].families,
                batches["evaluation"].ids, batches["evaluation"].rates, cutoff, task,
                config["data_seed"] + 999, config["bootstrap_samples"])
            name = f"{key}__{policy}"
            assert json.loads(json.dumps(metric)) == metrics[name]
            metric_rows.append({"task": task, "method": method, "policy": policy,
                **{column: value for column, value in metric.items() if not isinstance(value, (list, tuple))}})
            family_rows.append(family.assign(task=task, method=method, policy=policy))
            vectors[name] = (trajectory.true_positive.to_numpy(), trajectory.positive_windows.to_numpy()) if task == "state" else (
                trajectory.clean_detection.eq(True).to_numpy(dtype=int), trajectory.has_positive.to_numpy(dtype=int))
        grid = diagnostic_threshold_grid(cal_score, targets["calibration"][task], cutoffs)
        curves.append(operating_curve(score, valid, targets["evaluation"][task], batches["evaluation"].families, grid).assign(task=task, method=method))
    assert_csv(pd.DataFrame(metric_rows), output / "operating_metrics.csv")
    assert_csv(pd.concat(family_rows, ignore_index=True), output / "family_metrics.csv")
    assert_csv(pd.concat(calibration_tables, ignore_index=True), output / "calibration_policies.csv")
    assert_csv(pd.concat(curves, ignore_index=True), output / "threshold_curves.csv")
    assert_csv(summarize_effects(vectors, metrics, config["data_seed"] + 1234, config["bootstrap_samples"]), output / "paired_effects.csv")
    raw = estimation_rows(batches["evaluation"], config)
    assert_csv(raw, output / "estimation_windows.csv")
    assert_csv(summarize_estimation(raw), output / "estimation_summary.csv")
    assert_csv(conditional_auc_rows(scores["evaluation"], targets["evaluation"], batches["evaluation"].families), output / "conditional_auc.csv")
    replayed = 0
    for split, tasks in tasks_by_split.items():
        for task in tasks:
            if int(task[0].rsplit("/", 1)[1]) not in (0, config["sizes"][split] - 1):
                continue
            counts, events, ratios, rate = simulate_selection_trajectory(task)
            generator = config["generator"]
            matrix, fits = build_selection_features(counts, events, generator["windows"], generator["prefix_days"],
                generator["stage_days"], generator["lookback"], generator["forecast_stages"])
            batch = batches[split]
            index = int(np.flatnonzero(batch.ids == task[0])[0])
            np.testing.assert_array_equal(counts, batch.counts[index])
            np.testing.assert_array_equal(ratios, batch.ratios[index])
            assert rate == batch.rates[index]
            np.testing.assert_allclose(matrix, batch.matrix[index], rtol=1e-8, atol=1e-8)
            np.testing.assert_allclose(fits, batch.fits[index], rtol=1e-8, atol=1e-8)
            replayed += 1
    print(f"Verified {len(metrics)} operating rules, calibration-only grids, paired effects, estimation statistics, unchanged stage 4 and {replayed} replayed new trajectories.")


if __name__ == "__main__":
    main()
