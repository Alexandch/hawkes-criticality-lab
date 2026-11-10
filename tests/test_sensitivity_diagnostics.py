from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from critical_processes.selected_training import run_selected_training, sha
from critical_processes.sensitivity_diagnostics import (
    calibration_policy, diagnostic_threshold_grid, estimation_rows, n_history_features,
    operating_curve, run_sensitivity_diagnostics, summarize_estimation,
)
from critical_processes.task_protocol import evaluate_task_scores, state_warning_targets


class SensitivityDiagnosticTests(unittest.TestCase):
    def test_paired_features_match_when_estimates_are_exact_and_do_not_use_future(self):
        ratios = np.array([[.2, .3, .4, .5, .6, .7, .8, .95]])
        batch = SimpleNamespace(ratios=ratios.copy(), matrix=np.zeros((1, 2, 6, 64)))
        exact = n_history_features(batch, "known")
        for offset in (16, 39):
            for lag in range(3):
                batch.matrix[:, :, :, offset + 5*lag] = exact[:, None, :, lag]
                batch.matrix[:, :, :, offset + 5*lag + 4] = 1.
        np.testing.assert_array_equal(exact, n_history_features(batch, "constant"))
        np.testing.assert_array_equal(exact, n_history_features(batch, "calendar"))
        batch.ratios[:, 5:] = .1
        np.testing.assert_array_equal(exact[:, :3], n_history_features(batch, "known")[:, :3])

    def test_stricter_confidence_policies_raise_thresholds_on_same_scores(self):
        rng = np.random.default_rng(9)
        targets = state_warning_targets(np.full((400, 8), .4), 4, 3, 2, .9)["state"]
        scores = np.r_[rng.uniform(0, .4, (200, 6)), rng.uniform(.4, .95, (200, 6))]
        families = np.repeat(["homogeneous", "hawkes_near_threshold"], 200)
        cutoffs = [calibration_policy(scores, targets, families, "state", "frozen_primary", policy)[0]
                   for policy in ("pooled_trajectory", "family_nominal", "family_confident", "family_confident_source_scope")]
        self.assertEqual(cutoffs, sorted(cutoffs))

    def test_removing_control_only_from_calibration_retains_its_false_alarms(self):
        target = state_warning_targets(np.full((40, 8), .4), 4, 3, 2, .9)["state"]
        scores = np.r_[np.full((20, 6), .2), np.full((20, 6), .95)]
        families = np.repeat(["homogeneous", "hawkes_near_threshold"], 20)
        full, _ = calibration_policy(scores, target, families, "state", "frozen_primary", "family_nominal")
        removed, _ = calibration_policy(scores, target, families, "state", "frozen_primary", "family_nominal_without_near")
        self.assertLess(removed, full)
        curve = operating_curve(scores, np.ones_like(scores, dtype=bool), target, families, [removed])
        self.assertEqual(curve.maximum_family_false_alarm_probability.iloc[0], 1.)

    def test_curves_match_full_evaluation_and_keep_clean_recall_nonmonotonic(self):
        ratios = np.array([[.2]*6 + [.95]*4, [.4]*10])
        targets = state_warning_targets(ratios, 10, 3, 2, .9)["warning"]
        scores = np.full((2, 8), .1)
        scores[0, 0], scores[0, 2] = .3, .8
        valid = np.ones_like(scores, dtype=bool)
        families = np.array(["rising", "low"])
        curve = operating_curve(scores, valid, targets, families, [.2, .5, .9])
        np.testing.assert_array_equal(curve.clean_recall, [0., 1., 0.])
        for row in curve.itertuples():
            metric, _, _, _ = evaluate_task_scores(scores, valid, targets, families, ["a", "b"], [2., 2.],
                row.threshold, "warning", bootstrap_samples=5)
            self.assertEqual(row.clean_recall, metric["clean_event_detection_probability"])
            self.assertEqual(row.maximum_family_false_alarm_probability, metric["maximum_family_false_alarm_probability"])

    def test_error_summary_excludes_failed_parameter_estimates(self):
        batch = SimpleNamespace(ratios=np.full((1, 8), .9), matrix=np.zeros((1, 2, 6, 64)),
            fits=np.zeros((1, 2, 6, 2, 7)), rates=np.array([2.]), ids=np.array(["a"]), families=np.array(["hawkes_high"]))
        batch.fits[:, :, :, :, 0] = .8
        batch.fits[:, :, :, :, 4] = 1.
        batch.fits[0, 0, 0, 0, 4] = 0.
        config = {"generator": dict(prefix_days=60, stage_days=4, stages=8, lookback=3, forecast_stages=2, windows=[4,8])}
        raw = estimation_rows(batch, config)
        self.assertTrue(raw.loc[~raw.fit_valid, "n_hat"].isna().all())
        row = summarize_estimation(raw).query("window_days == 4 and variant == 'constant'").iloc[0]
        self.assertAlmostEqual(row.mae_to_current, .1)
        self.assertAlmostEqual(row.valid_fraction, 5/6)

    def test_curve_grid_is_fixed_by_calibration_and_includes_named_thresholds(self):
        target = state_warning_targets(np.full((2, 8), .4), 4, 3, 2, .9)["state"]
        scores = np.linspace(0, 1, 12).reshape(2, 6)
        grid = diagnostic_threshold_grid(scores, target, [.97])
        self.assertIn(.97, grid)
        self.assertGreater(grid.min(), 0)
        self.assertGreater(grid.max(), 1)

    def test_complete_diagnostics_preserve_old_final_test(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source", Path(directory) / "diagnosis"
            with contextlib.redirect_stdout(io.StringIO()):
                run_selected_training(source, train_per_family=2, validation_calibration_per_family=3,
                    validation_per_family=3, calibration_per_family=3, test_per_family=3,
                    model_seeds=(3,7), windows=(4,8), epochs=2, data_seed=7921, final_test_seed=7922,
                    workers=1, bootstrap_samples=5, prefix_days=60, stage_days=4, stages=8, rate_min=.5, rate_max=2.)
            before = {name: sha(source / name) for name in ("metrics.json", "frozen_plan.json", "data/test.npz")}
            with contextlib.redirect_stdout(io.StringIO()):
                metrics = run_sensitivity_diagnostics(source, output, train_per_family=2, calibration_per_family=19,
                    evaluation_per_family=2, data_seed=7923, evaluation_seed=7924, epochs=2, workers=1, bootstrap_samples=5)
            self.assertEqual(len(metrics), 92)
            self.assertEqual(before, {name: sha(source / name) for name in before})
            table = pd.read_csv(output / "operating_metrics.csv")
            oracle = table.query("task == 'state' and method == 'oracle_current' and policy == 'fixed_score'").iloc[0]
            self.assertEqual(oracle.window_sensitivity, 1.)
            self.assertEqual(oracle.maximum_family_false_alarm_probability, 0.)
            families = pd.read_csv(output / "family_metrics.csv")
            self.assertIn("hawkes_near_threshold", set(families.query("policy == 'family_nominal_without_near'").family))
            self.assertTrue(json.loads((output / "run_config.json").read_text(encoding="utf-8"))["old_experiment_unchanged"])
            self.assertTrue((output / "analysis.md").is_file())
            self.assertTrue((output / "threshold_tradeoff.png").is_file())


if __name__ == "__main__":
    unittest.main()
