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

from critical_processes.dynamic_estimation import (
    calibrate_trajectory_bands, choose_filter, coverage_rows, extension_gate,
    filter_n_history, fit_measurement_noise, raw_estimate_path, stage_measurements,
    trim_path, validation_quality,
)
from critical_processes.dynamic_experiment import run_dynamic_experiment
from critical_processes.selected_training import sha


def measurements(values, se=.1):
    values = np.atleast_2d(values).astype(float)
    result = np.zeros((*values.shape, 5))
    result[:, :, 0], result[:, :, 1], result[:, :, 4] = values, se, 1.
    return result


class DynamicEstimationTests(unittest.TestCase):
    def test_future_measurements_do_not_change_past_filter_or_features(self):
        truth = np.full((2, 12), .7)
        z = measurements(truth + np.linspace(-.1, .1, 12))
        noise = fit_measurement_noise(z, truth)
        before = filter_n_history(z, noise)
        z[:, 6:, 0], z[:, 6:, 1], z[:, 6:, 4] = .1, 2., 0.
        after = filter_n_history(z, noise)
        for key in before.__dict__:
            np.testing.assert_array_equal(getattr(before, key)[:, :6], getattr(after, key)[:, :6])

    def test_missing_blocks_propagate_but_never_initialize_from_placeholders(self):
        z = measurements([.8]*9)
        noise = fit_measurement_noise(z, np.full((1, 9), .8))
        z[:, :2, 4], z[:, 4:7, 4] = 0., 0.
        path = filter_n_history(z, noise)
        np.testing.assert_array_equal(path.valid[0], [False, False, True, True, True, True, False, True, True])
        self.assertGreater(path.n_scale[0, 6], path.n_scale[0, 3])
        self.assertTrue(np.isfinite(path.n).all())

    def test_noise_model_learns_actual_error_even_for_overconfident_curvature(self):
        truth = np.full((100, 12), .9)
        z = measurements(np.full_like(truth, .6), .001)
        noise = fit_measurement_noise(z, truth)
        self.assertAlmostEqual(noise["global_mse"], .09)
        self.assertGreater(min(row["working_variance"] for row in noise["cells"]), .08)

    def test_recovery_of_nonoverlapping_blocks_and_invalid_trend(self):
        short = measurements([.1, .2, .3, .4, .5, .6, .7, .8])
        matrix = np.zeros((1, 2, 6, 64))
        for j in range(6):
            matrix[:, :, j, 39:54] = short[:, j:j+3].reshape(1, 1, 15)
        batch = SimpleNamespace(matrix=matrix, fits=np.zeros((1, 2, 6, 2, 7)))
        np.testing.assert_array_equal(stage_measurements(batch), short)
        matrix[:, :, 0, 43] = 0.
        path = raw_estimate_path(batch)
        self.assertTrue(path.valid[0, 0])
        self.assertFalse(path.slope_valid[0, 0])

    def test_constant_truth_with_noise_is_recovered_and_scales_remain_positive(self):
        rng = np.random.default_rng(47)
        truth = np.full((500, 12), .7)
        z = measurements(truth + rng.normal(0, .08, truth.shape))
        noise = fit_measurement_noise(z, truth)
        path = filter_n_history(z, noise, .01, .5)
        self.assertLess(np.abs(path.n[:, 4:] - .7).mean(), .7*np.abs(z[:, 4:, 0] - .7).mean())
        self.assertTrue((path.n_scale > 0).all())
        self.assertTrue((path.slope_scale > 0).all())

    def test_bands_cover_whole_trajectories_and_do_not_use_holdout_truth(self):
        rng = np.random.default_rng(18)
        truth = np.full((200, 12), .8)
        z = measurements(truth + rng.normal(0, .06, truth.shape))
        noise = fit_measurement_noise(z[:100], truth[:100])
        path = trim_path(filter_n_history(z, noise))
        batch = SimpleNamespace(ratios=truth, families=np.repeat(["a", "b"], 100), ids=np.arange(200))
        cutoff, table = calibrate_trajectory_bands(path, batch)
        self.assertEqual(set(table["target"]), {"n", "slope"})
        coverage = coverage_rows(path, batch, "test", cutoff)
        self.assertGreaterEqual(coverage.trajectory_covered.mean(), .95)
        self.assertTrue((coverage.mean_width > 0).all())

    def test_selection_and_gate_penalize_delay_and_worst_family(self):
        base = dict(macro_mae=.1, near_high_mae=.1, maximum_family_mae=.2, mean_capped_abrupt_delay_days=180.)
        better = dict(macro_mae=.09, near_high_mae=.08, maximum_family_mae=.2, mean_capped_abrupt_delay_days=180.)
        self.assertTrue(extension_gate(better, base)["passed"])
        self.assertFalse(extension_gate({**better, "mean_capped_abrupt_delay_days": 360.}, base)["passed"])
        self.assertFalse(extension_gate({**better, "maximum_family_mae": .3}, base)["passed"])
        table = pd.DataFrame([{**better, "q": .01, "damping": .5}, {**better, "q": .1, "damping": .9}])
        self.assertEqual(choose_filter(table)["q"], .1)

    def test_full_protocol_freezes_before_holdout_and_refuses_changed_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            kwargs = dict(out=out, train_per_family=2, validation_per_family=2, calibration_per_family=19,
                evaluation_per_family=2, data_seed=8711, evaluation_seed=8712, model_seeds=(5,7), epochs=2,
                workers=1, bootstrap_samples=5, prefix_days=60, stage_days=4, stages=8, rate_min=.5, rate_max=2., sources=[])
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_dynamic_experiment(**kwargs)
            self.assertEqual(len(result), 40)
            frozen = json.loads((out/"frozen_plan.json").read_text(encoding="utf-8"))
            started = json.loads((out/"evaluation_started.json").read_text(encoding="utf-8"))
            self.assertEqual(started["frozen_plan_sha256"], sha(out/"frozen_plan.json"))
            self.assertFalse(frozen["holdout_used_for_selection"])
            before = sha(out/"data/evaluation.npz")
            self.assertEqual(run_dynamic_experiment(**kwargs), result)
            self.assertEqual(sha(out/"data/evaluation.npz"), before)
            with self.assertRaises(ValueError):
                run_dynamic_experiment(**{**kwargs, "epochs": 3})
            self.assertTrue((out/"causal_examples.png").is_file())


if __name__ == "__main__":
    unittest.main()
