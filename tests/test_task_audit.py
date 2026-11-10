from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from critical_processes.detector import build_detector
from critical_processes.task_audit import (
    AUDIT_FAMILIES, audit_tasks, load_audit_batch, run_task_audit, simulate_audit_trajectory,
)
from critical_processes.task_protocol import calibrate_task_threshold, evaluate_task_scores, state_warning_targets
from critical_processes.window_features import window_feature_names


class TaskProtocolTests(unittest.TestCase):
    def test_current_state_uses_completed_stage_and_warning_requires_strict_positive_lead(self):
        ratios = np.array([[0.2] * 6 + [0.92] * 4, [0.95] * 10, [0.95] * 4 + [0.4] * 6])
        targets = state_warning_targets(ratios, 10, 3, 2, 0.9)
        state, warning = targets["state"], targets["warning"]
        days = state.decision_days
        self.assertFalse(state.labels[0, days == 60].any())
        self.assertTrue(state.labels[0, days == 70].all())
        np.testing.assert_array_equal(days[warning.labels[0]], [40, 50])
        self.assertFalse(warning.eligible[0, days >= 60].any())
        self.assertTrue(state.labels[1].all())
        self.assertFalse(warning.eligible[1:].any())
        np.testing.assert_array_equal(days[state.labels[2]], [30, 40])

    def test_right_censoring_does_not_create_negative_future_labels(self):
        target = state_warning_targets(np.full((1, 10), 0.4), 10, 3, 2, 0.9)["warning"]
        np.testing.assert_array_equal(target.decision_days[target.eligible[0]], [30, 40, 50, 60, 70, 80])
        self.assertFalse(target.labels.any())

    def test_post_transition_state_detection_is_not_an_early_warning(self):
        targets = state_warning_targets(np.array([[0.2] * 6 + [0.92] * 4, [0.88] * 10]), 10, 3, 2, 0.9)
        scores = np.vstack([np.r_[np.full(4, 0.2), np.full(4, 0.92)], np.full(8, 0.88)])
        for task, expected in (("state", 1.), ("warning", 0.)):
            metric, _, _, _ = evaluate_task_scores(scores, np.ones_like(scores, dtype=bool), targets[task],
                ["rising", "near"], ["a", "b"], [2., 2.], .9, task, bootstrap_samples=20)
            self.assertEqual(metric["event_detection_probability"], expected)
        state, _, _, _ = evaluate_task_scores(scores, np.ones_like(scores, dtype=bool), targets["state"],
            ["rising", "near"], ["a", "b"], [2., 2.], .9, "state", bootstrap_samples=20)
        self.assertEqual(state["auc_valid_only"], 1.)

    def test_earlier_false_alarm_persists_and_no_negative_family_is_not_zero_fpr(self):
        targets = state_warning_targets(np.array([[0.2] * 6 + [0.92] * 4, [0.95] * 10]), 10, 3, 2, 0.9)
        scores = np.full((2, 8), .1)
        scores[0, [0, 2]] = .95
        metric, family, _, _ = evaluate_task_scores(scores, np.ones_like(scores, dtype=bool), targets["warning"],
            ["rising", "initial_high"], ["a", "b"], [2., 2.], .5, "warning", bootstrap_samples=20)
        self.assertEqual(metric["event_detection_probability"], 1.)
        self.assertEqual(metric["clean_event_detection_probability"], 0.)
        row = family[family.family == "initial_high"].iloc[0]
        self.assertEqual(row.negative_trajectories, 0)
        self.assertTrue(np.isnan(row.probability_false_alarm))

    def test_saturated_calibration_can_abstain_and_excludes_no_negative_families(self):
        ratios = np.r_[np.full((19, 8), .4), np.full((19, 8), .95)]
        target = state_warning_targets(ratios, 4, 3, 2, .9)["state"]
        scores = np.ones_like(target.labels, dtype=float)
        families = np.repeat(["low", "high"], 19)
        threshold, table = calibrate_task_threshold(scores, target, families)
        self.assertGreater(threshold, 1.)
        self.assertTrue(table.all_abstain_threshold.all())
        self.assertEqual(table[table.family == "high"].negative_trajectories.iloc[0], 0)
        metric, _, _, _ = evaluate_task_scores(scores, np.ones_like(scores, dtype=bool), target,
            families, np.arange(38).astype(str), np.ones(38), threshold, "state", bootstrap_samples=10)
        self.assertEqual(metric["true_positive"], 0)
        self.assertEqual(metric["false_positive"], 0)

    def test_abstention_is_missed_positive_and_auc_reports_valid_coverage(self):
        target = state_warning_targets(np.array([[.95] * 5, [.4] * 5]), 4, 3, 1, .9)["state"]
        scores = np.array([[0., .95, .95], [.1, .1, .1]])
        valid = np.ones_like(scores, dtype=bool)
        valid[0, 0] = False
        metric, _, _, _ = evaluate_task_scores(scores, valid, target, ["a", "b"], ["x", "y"], [2., 2.],
            .9, "state", bootstrap_samples=20)
        self.assertEqual(metric["false_negative"], 1)
        self.assertAlmostEqual(metric["window_sensitivity"], 2/3)
        self.assertAlmostEqual(metric["score_valid_fraction_eligible"], 5/6)
        self.assertEqual(metric["auc_valid_only"], 1.)


class TaskAuditIntegrationTests(unittest.TestCase):
    def test_seeds_are_unique_across_splits_and_new_regimes_replay(self):
        config = dict(stages=8, stage_days=4, lookback=3, forecast_stages=2, critical_level=.9, rate_min=.5, rate_max=2.)
        tasks = audit_tasks(719, {"calibration": 2, "test": 2, "counterfactual": 2}, config)
        all_tasks = [task for group in tasks.values() for task in group]
        seeds = [value for task in all_tasks for value in task[2:4]]
        self.assertEqual(len(seeds), len(set(seeds)))
        for task in [t for t in tasks["test"] if t[1] in ("hawkes_high", "hawkes_falling", "hawkes_abrupt")]:
            counts, ratios, rate, events = simulate_audit_trajectory(task)
            repeated = simulate_audit_trajectory(task)
            np.testing.assert_array_equal(counts, repeated[0])
            np.testing.assert_array_equal(np.bincount(np.floor(events).astype(int), minlength=32), counts)
            target = state_warning_targets(ratios[None], 4, 3, 2, .9)
            if task[1] == "hawkes_high":
                self.assertTrue(target["state"].labels.all())
                self.assertFalse(target["warning"].eligible.any())
            if task[1] == "hawkes_falling":
                self.assertTrue(target["state"].labels.any())
                self.assertFalse(target["state"].labels.all())
                self.assertFalse(target["warning"].eligible.any())

    def test_full_audit_preserves_models_and_saves_separate_targets_and_shared_prefixes(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source", Path(directory) / "audit"
            source.mkdir()
            torch.manual_seed(92)
            for method in ("count_mlp", "hawkes_mlp"):
                indices = list(range(12 if method == "count_mlp" else len(window_feature_names())))
                torch.save({"state_dict": build_detector(len(indices)).state_dict(), "mean": np.zeros(len(indices)),
                    "scale": np.ones(len(indices)), "selected_feature_indices": indices,
                    "window_feature_names": window_feature_names(), "lookback": 3, "stage_horizon": 4.,
                    "forecast_stages": 2, "critical_level": .9, "fixed_decay_per_day": 1.}, source / f"{method}.pt")
            original = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
            with contextlib.redirect_stdout(io.StringIO()):
                metrics = run_task_audit(output, source, calibration_per_family=19, test_per_family=2,
                    counterfactual_pairs=2, workers=1, seed=821, bootstrap_samples=10,
                    stage_days=4, stages=8, rate_min=.5, rate_max=2.)
            for path in source.iterdir():
                self.assertEqual(original[path.name], hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(path.read_bytes(), (output / "models" / path.name).read_bytes())
            self.assertEqual(len(metrics), 5)
            self.assertEqual(set(metrics["current_mle"]), {"state", "warning"})
            calibration, test = [load_audit_batch(output / "data" / f"{split}.npz") for split in ("calibration", "test")]
            self.assertFalse(set(calibration.base.ids) & set(test.base.ids))
            self.assertEqual(len(test.base.ids), 2 * len(AUDIT_FAMILIES))
            shared = pd.read_csv(output / "shared_prefix_summary.csv")
            np.testing.assert_allclose(shared.auc_paired_possible_futures, .5)
            np.testing.assert_allclose(shared.sensitivity_crossing_branch, shared.false_alarm_probability_no_crossing_branch)
            manifest = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["no_new_neural_training"])
            self.assertTrue((output / "analysis.md").is_file())
            self.assertTrue((output / "target_timeline.png").is_file())


if __name__ == "__main__":
    unittest.main()
