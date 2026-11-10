from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from critical_processes.negative_controls import stage_features_from_counts
from critical_processes.task_audit import (
    AUDIT_FAMILIES, SCORE_NAMES, audit_scores, audit_tasks, load_audit_batch,
    simulate_audit_trajectory, summarize_shared_prefix,
)
from critical_processes.task_protocol import calibrate_task_threshold, evaluate_task_scores, state_warning_targets
from critical_processes.window_features import build_window_features


def assert_csv(expected, path):
    saved = pd.read_csv(path, float_precision="round_trip")
    expected = expected.reset_index(drop=True)
    expected = expected.where(expected.notna(), np.nan)
    pd.testing.assert_frame_equal(expected, saved, check_dtype=False,
                                  check_exact=False, rtol=1e-10, atol=1e-10)


def main():
    import torch

    torch.set_num_threads(min(4, torch.get_num_threads()))
    parser = argparse.ArgumentParser(description="Verify separate state/warning targets, calibration, metrics and causal scores.")
    parser.add_argument("--out", type=Path, default=Path("results/state_warning_audit"))
    output = parser.parse_args().out
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    saved_metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    saved_thresholds = json.loads((output / "thresholds.json").read_text(encoding="utf-8"))
    for name, digest in config["dataset_sha256"].items():
        assert hashlib.sha256((output / "data" / f"{name}.npz").read_bytes()).hexdigest() == digest
    for method, digest in config["source_model_sha256"].items():
        assert hashlib.sha256((output / "models" / f"{method}.pt").read_bytes()).hexdigest() == digest
        source = Path(config["source_model_directory"]) / f"{method}.pt"
        if source.exists():
            assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, "source model changed since audit"
    batches, scores, validity, targets = {}, {}, {}, {}
    protocol = {key: config[key] for key in ("stage_days", "lookback", "forecast_stages", "critical_level")}
    all_ids, all_seeds = [], []
    for split, size in config["sizes_per_family"].items():
        batch = load_audit_batch(output / "data" / f"{split}.npz")
        batches[split] = batch
        assert len(batch.base.ids) == size * (1 if split == "counterfactual" else len(AUDIT_FAMILIES))
        all_ids.extend(batch.base.ids.tolist())
        all_seeds.extend(batch.parameter_seeds.tolist() + batch.simulator_seeds.tolist())
        scores[split], validity[split] = audit_scores(batch, output / "models", config["lookback"])
        with np.load(output / "data" / f"{split}_scores.npz") as saved:
            for method in SCORE_NAMES:
                np.testing.assert_allclose(scores[split][method], saved[method], rtol=1e-12, atol=1e-12)
                np.testing.assert_array_equal(validity[split][method], saved[method + "_valid"])
        if split != "counterfactual":
            targets[split] = state_warning_targets(batch.base.ratios, **protocol)
            state, warning = targets[split]["state"], targets[split]["warning"]
            np.testing.assert_array_equal(state.labels, batch.base.ratios[:, config["lookback"] - 1:] >= config["critical_level"])
            assert not warning.eligible[batch.base.ratios[:, 0] >= config["critical_level"]].any()
            unavailable_future = state.decision_days + config["stage_days"] * config["forecast_stages"] > config["stages"] * config["stage_days"]
            assert not warning.eligible[:, unavailable_future].any()
            for i in range(len(batch.base.ids)):
                assert (state.decision_days[warning.eligible[i]] < warning.first_crossing[i]).all()
                assert (state.decision_days[warning.eligible[i]] + config["stage_days"] * config["forecast_stages"] <= config["stages"] * config["stage_days"]).all()
    assert len(all_ids) == len(set(all_ids))
    assert len(all_seeds) == len(set(all_seeds))
    calibration_tables, family_tables, trajectory_tables = [], [], []
    for method in SCORE_NAMES:
        for task in ("state", "warning"):
            threshold, table = calibrate_task_threshold(scores["calibration"][method], targets["calibration"][task],
                batches["calibration"].base.families, config["target_false_alarm_probability"])
            assert threshold == saved_thresholds[method][task]
            calibration_tables.append(table.assign(method=method, task=task))
            batch = batches["test"].base
            metrics, family, trajectories, decisions = evaluate_task_scores(
                scores["test"][method], validity["test"][method], targets["test"][task], batch.families, batch.ids, batch.rates,
                threshold, task, config["bootstrap_seed"], config["bootstrap_samples"],
            )
            assert json.loads(json.dumps(metrics)) == saved_metrics[method][task]
            family_tables.append(family.assign(method=method, task=task))
            trajectory_tables.append(trajectories.assign(method=method, task=task))
            assert_csv(decisions, output / f"{method}_{task}_test_scores.csv")
    assert_csv(pd.concat(calibration_tables, ignore_index=True), output / "calibration.csv")
    assert_csv(pd.concat(family_tables, ignore_index=True), output / "family_test_metrics.csv")
    assert_csv(pd.concat(trajectory_tables, ignore_index=True), output / "trajectory_test_metrics.csv")
    shared, shared_summary = summarize_shared_prefix(scores["counterfactual"], validity["counterfactual"],
                                                    saved_thresholds, batches["counterfactual"], config)
    assert_csv(shared, output / "shared_prefix_scores.csv")
    assert_csv(shared_summary, output / "shared_prefix_summary.csv")
    prefix = batches["counterfactual"].base.ratios
    full = np.repeat(prefix[:, -1:], config["stages"], axis=1)
    full[:, :prefix.shape[1]] = prefix
    alternative = full.copy()
    alternative[:, prefix.shape[1] + 1:] = min(config["critical_level"] + .04, .985)
    index = prefix.shape[1] - config["lookback"]
    no_crossing = state_warning_targets(full, **protocol)["warning"]
    crossing = state_warning_targets(alternative, **protocol)["warning"]
    assert no_crossing.eligible[:, index].all() and crossing.eligible[:, index].all()
    assert not no_crossing.labels[:, index].any() and crossing.labels[:, index].all()
    generation = {key: config[key] for key in ("stages", "stage_days", "lookback", "forecast_stages", "critical_level", "rate_min", "rate_max")}
    tasks = audit_tasks(config["seed"], config["sizes_per_family"], generation)
    replay_count = 0
    for split, group in tasks.items():
        size = config["sizes_per_family"][split]
        for task in group:
            if int(task[0].rsplit("/", 1)[1]) not in (0, size - 1):
                continue
            counts, ratios, rate, events = simulate_audit_trajectory(task)
            batch = batches[split]
            i = int(np.flatnonzero(batch.base.ids == task[0])[0])
            np.testing.assert_array_equal(counts, batch.base.counts[i])
            np.testing.assert_array_equal(ratios, batch.base.ratios[i])
            assert rate == batch.base.rates[i]
            np.testing.assert_array_equal(np.bincount(np.floor(events).astype(int), minlength=len(counts)), counts)
            features, short, diagnostics = build_window_features(counts, events, config["stage_days"], config["lookback"], config["forecast_stages"])
            np.testing.assert_allclose(features, batch.windows[i], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(short, batch.short_fits[i], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(diagnostics, batch.diagnostics[i], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(stage_features_from_counts(counts, config["stage_days"]), batch.base.features[i], rtol=1e-12, atol=1e-12)
            replay_count += 1
    print(f"Verified both task definitions, all thresholds/scores/metrics, immutable models and {replay_count} replayed trajectories including shared prefixes.")


if __name__ == "__main__":
    main()
