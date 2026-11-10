from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from critical_processes.confident_calibration import calibrate_confident
from critical_processes.dynamic_estimation import (
    DAMPING_GRID, Q_GRID, calibrate_trajectory_bands, choose_filter, coverage_rows, delay_rows,
    extension_gate, filter_n_history, fit_measurement_noise, raw_estimate_path,
    stage_measurements, trajectory_error_summary, trim_path, validation_quality,
)
from critical_processes.dynamic_experiment import (
    aggregate_coverage, aggregate_errors, estimate_paths, matched_features, paired_error_effects,
    reserved_seeds, score_batch,
)
from critical_processes.selected_features import build_selection_features, load_selection_batch, selection_tasks, simulate_selection_trajectory
from critical_processes.selected_training import selection_targets, sha
from critical_processes.task_protocol import calibrate_task_threshold, evaluate_task_scores


def assert_csv(frame, path):
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), pd.read_csv(path, float_precision="round_trip"),
                                 check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10)


def main():
    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/dynamic_estimation"))
    parser.add_argument("--replays-per-family", type=int, default=1)
    args = parser.parse_args()
    out = args.out
    protocol = json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    complete = json.loads((out/"complete.json").read_text(encoding="utf-8"))
    frozen = json.loads((out/"frozen_plan.json").read_text(encoding="utf-8"))
    source_files = protocol["old_snapshot_sha256"]
    for name, digest in source_files.items():
        assert sha(name) == digest, f"earlier experiment changed: {name}"
    for name, digest in complete["artifact_sha256"].items():
        assert sha(out/name) == digest, f"artifact changed: {name}"
    module = Path(__file__).resolve().parents[1]/"src/critical_processes"
    for name, digest in protocol["code_sha256"].items():
        assert sha(module/name) == digest, f"code changed: {name}"
    assert frozen["protocol_sha256"] == sha(out/"protocol.json")
    assert json.loads((out/"evaluation_started.json").read_text(encoding="utf-8"))["frozen_plan_sha256"] == sha(out/"frozen_plan.json")
    assert not frozen["holdout_used_for_selection"]
    sources = sorted({Path(name).parent.parent for name in source_files if Path(name).parent.name == "data"})
    old_seeds = reserved_seeds(sources)
    reserved, all_ids, batches, tasks_by_split = list(old_seeds), [], {}, {}
    for index, (split, size) in enumerate(protocol["sizes"].items()):
        seed = protocol["evaluation_seed"] if split == "evaluation" else int(np.random.SeedSequence([protocol["data_seed"], index]).generate_state(1)[0])
        tasks = selection_tasks(seed, f"dynamic_{split}", size, protocol["generator"], reserved)
        batch = load_selection_batch(out/"data"/f"{split}.npz")
        np.testing.assert_array_equal(batch.ids, [t[0] for t in tasks])
        np.testing.assert_array_equal(batch.parameter_seeds, [t[2] for t in tasks])
        np.testing.assert_array_equal(batch.simulator_seeds, [t[3] for t in tasks])
        assert sha(out/"data"/f"{split}.npz") == complete["dataset_sha256"][split]
        reserved.extend(batch.parameter_seeds.tolist()+batch.simulator_seeds.tolist())
        all_ids.extend(batch.ids.tolist())
        batches[split], tasks_by_split[split] = batch, tasks
    assert len(all_ids) == len(set(all_ids))
    new_seeds = reserved[len(old_seeds):]
    assert len(new_seeds) == len(set(new_seeds)) and not set(new_seeds).intersection(old_seeds)
    noise = fit_measurement_noise(stage_measurements(batches["train"]), batches["train"].ratios)
    assert noise == json.loads((out/"measurement_noise.json").read_text(encoding="utf-8"))
    candidates = []
    for q in Q_GRID:
        for damping in DAMPING_GRID:
            path = trim_path(filter_n_history(stage_measurements(batches["validation"]), noise, q, damping))
            candidates.append({**validation_quality(path, batches["validation"], "dynamic_calendar", protocol["generator"]["stage_days"]), "q": q, "damping": damping})
    candidate_table = pd.DataFrame(candidates)
    assert_csv(candidate_table, out/"validation_candidates.csv")
    selected = choose_filter(candidate_table)
    assert selected == frozen["selected_filter"]
    baseline = validation_quality(raw_estimate_path(batches["validation"]), batches["validation"], "calendar90", protocol["generator"]["stage_days"])
    assert extension_gate(selected, baseline, protocol["generator"]["stage_days"]) == frozen["validation_gate"]
    for task in ("state", "warning"):
        target = selection_targets(batches["train"], protocol["generator"])[task]
        for method in ("calendar90", "dynamic_calendar"):
            matrix = matched_features(batches["train"], method, noise, selected)
            x = matrix[target.eligible]
            for seed in protocol["model_seeds"]:
                checkpoint = torch.load(out/"models"/f"{task}_{method}_s{seed}.pt", map_location="cpu", weights_only=False)
                np.testing.assert_allclose(checkpoint["mean"], x.mean(axis=0), rtol=1e-12, atol=1e-12)
                scale = x.std(axis=0)
                scale[scale < 1e-8] = 1.
                np.testing.assert_allclose(checkpoint["scale"], scale, rtol=1e-12, atol=1e-12)
                assert checkpoint["training_sha256"] == complete["dataset_sha256"]["train"]
    paths, scores, targets = {}, {}, {}
    for split in ("calibration", "evaluation"):
        batch = batches[split]
        paths[split] = estimate_paths(batch, noise, selected, protocol["generator"]["windows"])
        scores[split] = score_batch(batch, paths[split], out, noise, selected, protocol["model_seeds"])
        targets[split] = selection_targets(batch, protocol["generator"])
        with np.load(out/"data"/f"{split}_scores.npz") as saved:
            for key, (score, valid) in scores[split].items():
                np.testing.assert_array_equal(saved[key], score)
                np.testing.assert_array_equal(saved[key+"__valid"], valid)
    batch = batches["evaluation"]
    metrics = json.loads((out/"metrics.json").read_text(encoding="utf-8"))
    for key, (score, valid) in scores["evaluation"].items():
        task = key.split("__")[0]
        cal_score = scores["calibration"][key][0]
        for policy in protocol["policies"]:
            if policy == "family_confident":
                cutoff, _ = calibrate_confident(cal_score, targets["calibration"][task], batches["calibration"].families, .05, .05/protocol["protected_score_streams"])
            else:
                cutoff, _ = calibrate_task_threshold(cal_score, targets["calibration"][task], batches["calibration"].families, .05)
            name = f"{key}__{policy}"
            assert cutoff == frozen["thresholds"][name]
            metric, _, _, _ = evaluate_task_scores(score, valid, targets["evaluation"][task], batch.families, batch.ids,
                batch.rates, cutoff, task, bootstrap_seed=protocol["data_seed"]+500, bootstrap_samples=protocol["bootstrap_samples"])
            assert json.loads(json.dumps(metric)) == metrics[name], name
    errors = pd.concat([trajectory_error_summary(path, batch, method) for method, path in paths["evaluation"].items()], ignore_index=True)
    assert_csv(errors, out/"estimation_trajectories.csv")
    assert_csv(aggregate_errors(errors), out/"estimation_summary.csv")
    assert_csv(paired_error_effects(errors, protocol["data_seed"]+501, protocol["bootstrap_samples"]), out/"paired_errors.csv")
    width = protocol["generator"]["stage_days"]
    assert_csv(pd.concat([delay_rows(path, batch, method, width) for method, path in paths["evaluation"].items()], ignore_index=True), out/"abrupt_delay.csv")
    coverages = []
    for method, path in paths["calibration"].items():
        bands, _ = calibrate_trajectory_bands(path, batches["calibration"])
        assert bands == frozen["interval_multipliers"][method]
        coverages.append(coverage_rows(paths["evaluation"][method], batch, method, bands))
    assert_csv(aggregate_coverage(pd.concat(coverages, ignore_index=True)), out/"interval_coverage.csv")
    replay_count = 0
    for split in ("validation", "evaluation"):
        batch, tasks = batches[split], tasks_by_split[split]
        for family in np.unique(batch.families):
            for index in np.flatnonzero(batch.families == family)[:args.replays_per_family]:
                counts, events, ratios, rate = simulate_selection_trajectory(tasks[index])
                matrix, fits = build_selection_features(counts, events, protocol["generator"]["windows"],
                    protocol["generator"]["prefix_days"], width)
                np.testing.assert_array_equal(counts, batch.counts[index])
                np.testing.assert_array_equal(ratios, batch.ratios[index])
                assert rate == batch.rates[index]
                np.testing.assert_allclose(matrix, batch.matrix[index], rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(fits, batch.fits[index], rtol=1e-10, atol=1e-10)
                replay_count += 1
    print(f"Verified {len(all_ids)} disjoint trajectories, 8 validation settings, {len(metrics)} frozen detector rules, bands, delays and {replay_count} exact replays; R4/R5 unchanged.")


if __name__ == "__main__":
    main()
