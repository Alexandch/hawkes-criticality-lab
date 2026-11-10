from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from critical_processes.confident_calibration import calibrate_confident
from critical_processes.selected_features import (
    FEATURE_GROUPS, build_selection_features, load_selection_batch, selection_tasks, simulate_selection_trajectory,
)
from critical_processes.selected_training import (
    METHOD_NAMES, choose_validation_candidate, score_selected_stream, selection_targets, sha,
    statistical_scores, validation_metrics,
)
from critical_processes.task_protocol import evaluate_task_scores
from critical_processes.window_benchmark import score_window_matrix


def assert_csv(expected, path):
    saved = pd.read_csv(path, float_precision="round_trip")
    expected = expected.reset_index(drop=True)
    expected = expected.where(expected.notna(), np.nan)
    pd.testing.assert_frame_equal(expected, saved, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10)


def main():
    import torch
    import critical_processes.selected_training as selected_module

    torch.set_num_threads(min(4, torch.get_num_threads()))
    parser = argparse.ArgumentParser(description="Verify validation-only selection, multiseed weights and the single frozen final test.")
    parser.add_argument("--out", type=Path, default=Path("results/selected_detectors"))
    output = parser.parse_args().out
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    frozen = json.loads((output / "frozen_plan.json").read_text(encoding="utf-8"))
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    complete = json.loads((output / "final_test_complete.json").read_text(encoding="utf-8"))
    start = json.loads((output / "final_test_started.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert frozen["frozen_at_utc"] <= start["started_at_utc"] <= complete["completed_at_utc"]
    assert start["frozen_plan_sha256"] == complete["frozen_plan_sha256"] == sha(output / "frozen_plan.json")
    assert not frozen["test_generated"] and not selection["test_used"]
    assert frozen["protocol_sha256"] == sha(output / "protocol.json")
    assert frozen["selection_sha256"] == sha(output / "selection.json")
    assert frozen["validation_candidates_sha256"] == sha(output / "validation_candidates.csv")
    for relative, key in (("data/test.npz", "test_sha256"), ("data/test_scores.npz", "test_scores_sha256"),
                          ("metrics.json", "metrics_sha256"), ("test_metrics.csv", "test_metrics_sha256")):
        assert sha(output / relative) == complete[key]
    for name, digest in protocol["source_sha256"].items():
        assert sha(Path(selected_module.__file__).parent / name) == digest, f"source changed: {name}"
    for name, digest in frozen["model_sha256"].items():
        assert sha(output / name) == digest
    batches, targets, tasks_by_split, all_ids, reserved = {}, {}, {}, [], []
    for split_index, (split, size) in enumerate(protocol["sizes"].items()):
        batch = load_selection_batch(output / "data" / f"{split}.npz")
        batches[split], targets[split] = batch, selection_targets(batch, protocol)
        assert len(batch.ids) == size * len(protocol["families"])
        assert batch.matrix.shape[1:] == (len(protocol["windows"]), protocol["stages"] - protocol["lookback"] + 1, len(protocol["feature_names"]))
        if split != "test":
            assert frozen["dataset_sha256"][split] == sha(output / "data" / f"{split}.npz")
            seed = int(np.random.SeedSequence([protocol["data_seed"], split_index]).generate_state(1)[0])
        else:
            seed = protocol["final_test_seed"]
        tasks = selection_tasks(seed, split, size, protocol, reserved)
        tasks_by_split[split] = tasks
        np.testing.assert_array_equal(batch.ids, [task[0] for task in tasks])
        np.testing.assert_array_equal(batch.parameter_seeds, [task[2] for task in tasks])
        np.testing.assert_array_equal(batch.simulator_seeds, [task[3] for task in tasks])
        all_ids.extend(batch.ids.tolist())
        reserved.extend(batch.parameter_seeds.tolist() + batch.simulator_seeds.tolist())
    assert len(all_ids) == len(set(all_ids)) and len(reserved) == len(set(reserved))
    validation_rows, preliminary_tables = [], []
    for task in ("state", "warning"):
        for method in METHOD_NAMES:
            for window_index, window in enumerate(protocol["windows"]):
                members = [str(s) for s in protocol["model_seeds"]] if method in FEATURE_GROUPS else ["statistic"]
                cal_members, val_members = [], []
                for member in members + (["ensemble"] if method in FEATURE_GROUPS else []):
                    if member == "ensemble":
                        cal_score, val_score = np.mean(cal_members, axis=0), np.mean(val_members, axis=0)
                        valid = np.ones_like(val_score, dtype=bool)
                    elif method in FEATURE_GROUPS:
                        path = output / "models" / f"{task}_{method}_w{window}_s{member}.pt"
                        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                        assert checkpoint["training_dataset_sha256"] == frozen["dataset_sha256"]["train"]
                        assert checkpoint["task"] == task and checkpoint["model_seed"] == int(member)
                        train_x = batches["train"].matrix[:, window_index][targets["train"][task].eligible][:, checkpoint["selected_feature_indices"]]
                        np.testing.assert_allclose(checkpoint["mean"], train_x.mean(axis=0), rtol=1e-12, atol=1e-12)
                        scale = train_x.std(axis=0)
                        scale[scale < 1e-8] = 1.
                        np.testing.assert_allclose(checkpoint["scale"], scale, rtol=1e-12, atol=1e-12)
                        cal_score = score_window_matrix(batches["validation_calibration"].matrix[:, window_index], path)
                        val_score = score_window_matrix(batches["validation"].matrix[:, window_index], path)
                        valid = np.ones_like(val_score, dtype=bool)
                        cal_members.append(cal_score)
                        val_members.append(val_score)
                    else:
                        cal_score, _ = statistical_scores(batches["validation_calibration"], window_index, method, task)
                        val_score, valid = statistical_scores(batches["validation"], window_index, method, task)
                    cutoff, table = calibrate_confident(cal_score, targets["validation_calibration"][task],
                        batches["validation_calibration"].families, protocol["alpha"], protocol["confidence_failure_probability"])
                    quality = validation_metrics(val_score, valid, targets["validation"][task], batches["validation"].families, cutoff, task)
                    validation_rows.append({"task": task, "method": method, "window_days": window, "window_index": window_index,
                        "member": member, "feature_count": FEATURE_GROUPS.get(method, 5), "threshold": cutoff, **quality})
                    preliminary_tables.append(table.assign(task=task, method=method, window_days=window, member=member))
    candidates = pd.DataFrame(validation_rows)
    saved_candidates = pd.read_csv(output / "validation_candidates.csv", float_precision="round_trip")
    sort = ["task", "method", "window_days", "member"]
    pd.testing.assert_frame_equal(candidates.sort_values(sort).reset_index(drop=True), saved_candidates.sort_values(sort).reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10)
    saved_preliminary = pd.read_csv(output / "validation_calibration.csv", float_precision="round_trip")
    preliminary = pd.concat(preliminary_tables, ignore_index=True)
    for frame in (preliminary, saved_preliminary):
        frame["member"] = frame.member.astype(str)
    preliminary = preliminary.where(preliminary.notna(), np.nan)
    pd.testing.assert_frame_equal(preliminary.sort_values(sort + ["family"]).reset_index(drop=True), saved_preliminary.sort_values(sort + ["family"]).reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10)
    reconstructed = []
    for task in ("state", "warning"):
        for method in METHOD_NAMES:
            frame = candidates[(candidates.task == task) & (candidates.method == method) & candidates.member.isin(["ensemble", "statistic"])]
            reconstructed.append(choose_validation_candidate(frame, protocol["alpha"]))
        primary = choose_validation_candidate(pd.DataFrame([row for row in reconstructed if row["task"] == task]), protocol["alpha"])["method"]
        assert primary == frozen["primary_method_by_task"][task]
    assert reconstructed == selection["selected"]
    calibration_tables, family_tables, trajectory_tables, metric_rows = [], [], [], []
    assert sha(output / "data/calibration_scores.npz") == frozen["calibration_scores_sha256"]
    with np.load(output / "data/calibration_scores.npz") as cal_saved, np.load(output / "data/test_scores.npz") as test_saved:
        for entry in frozen["streams"]:
            name, task = entry["stream"], entry["task"]
            cal_score, cal_valid = score_selected_stream(batches["calibration"], entry, output)
            np.testing.assert_allclose(cal_score, cal_saved[name], rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(cal_valid, cal_saved[name + "__valid"])
            cutoff, table = calibrate_confident(cal_score, targets["calibration"][task], batches["calibration"].families,
                protocol["alpha"], protocol["confidence_failure_probability"] / len(frozen["streams"]))
            assert cutoff == entry["threshold"]
            calibration_tables.append(table.assign(stream=name, task=task, method=entry["method"], member=entry["member"]))
            score, valid = score_selected_stream(batches["test"], entry, output)
            np.testing.assert_allclose(score, test_saved[name], rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(valid, test_saved[name + "__valid"])
            batch = batches["test"]
            metric, family, trajectory, decisions = evaluate_task_scores(score, valid, targets["test"][task], batch.families, batch.ids, batch.rates,
                cutoff, task, protocol["data_seed"] + 999, protocol["bootstrap_samples"])
            assert json.loads(json.dumps(metric)) == metrics[name]
            family_tables.append(family.assign(stream=name, task=task, method=entry["method"], member=entry["member"]))
            trajectory_tables.append(trajectory.assign(stream=name, task=task, method=entry["method"], member=entry["member"]))
            metric_rows.append({"stream": name, "task": task, "method": entry["method"], "member": entry["member"],
                "window_days": entry["window_days"], "primary": entry["primary"],
                **{key: value for key, value in metric.items() if not isinstance(value, (list, tuple))}})
            if entry["primary"]:
                assert_csv(decisions.assign(observation_end_day=decisions.decision_end_day + protocol["prefix_days"]),
                           output / f"primary_{task}_decisions.csv")
    assert_csv(pd.concat(calibration_tables, ignore_index=True), output / "calibration.csv")
    assert_csv(pd.concat(family_tables, ignore_index=True), output / "family_test_metrics.csv")
    assert_csv(pd.concat(trajectory_tables, ignore_index=True), output / "trajectory_test_metrics.csv")
    assert_csv(pd.DataFrame(metric_rows), output / "test_metrics.csv")
    replayed = 0
    for split, tasks in tasks_by_split.items():
        for task in tasks:
            if int(task[0].rsplit("/", 1)[1]) not in (0, protocol["sizes"][split] - 1):
                continue
            counts, events, ratios, rate = simulate_selection_trajectory(task)
            matrix, fits = build_selection_features(counts, events, protocol["windows"], protocol["prefix_days"],
                protocol["stage_days"], protocol["lookback"], protocol["forecast_stages"])
            batch = batches[split]
            i = int(np.flatnonzero(batch.ids == task[0])[0])
            np.testing.assert_array_equal(counts, batch.counts[i])
            np.testing.assert_array_equal(ratios, batch.ratios[i])
            assert rate == batch.rates[i]
            np.testing.assert_allclose(matrix, batch.matrix[i], rtol=1e-8, atol=1e-8)
            np.testing.assert_allclose(fits, batch.fits[i], rtol=1e-8, atol=1e-8)
            replayed += 1
    print(f"Verified five disjoint datasets, all validation choices, {len(frozen['streams'])} frozen thresholds/score streams, all metrics and {replayed} replayed trajectories.")


if __name__ == "__main__":
    main()
