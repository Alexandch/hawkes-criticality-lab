from __future__ import annotations

import unittest

import numpy as np
from scipy.stats import binom

from critical_processes.background_models import fit_background_window
from critical_processes.confident_calibration import calibrate_confident, tolerance_rank
from critical_processes.fast_window_mle import fit_fast_window
from critical_processes.selected_features import (
    build_selection_features, selection_feature_names, selection_tasks, simulate_selection_trajectory,
)
from critical_processes.task_protocol import state_warning_targets
from critical_processes.window_features import prepare_hawkes_history
from critical_processes.simulation import simulate_staged_hawkes_counts


class SelectedFeatureTests(unittest.TestCase):
    def test_fast_solver_matches_scipy_on_varied_profiles_ties_and_bounds(self):
        rng = np.random.default_rng(845)
        for i in range(15):
            events = rng.uniform(0, 100, 80 + i * 80)
            if i % 3 == 0:
                events = np.round(events, 1)
            history = prepare_hawkes_history(events)
            shape = np.exp(.6 * np.sin(np.arange(100) * 2 * np.pi / 7)) if i % 2 else np.ones(100)
            expected = fit_background_window(history, 20, 100, shape, False)
            actual = fit_fast_window(history, 20, 100, shape)
            self.assertTrue(expected["fit_valid"] and actual[4])
            self.assertAlmostEqual(expected["n_hat"], actual[0], delta=2e-5)
            self.assertTrue(np.isfinite(actual).all())
        self.assertFalse(fit_fast_window(prepare_hawkes_history([]), 0, 10)[4])
        for n in (.5, .9, .98):
            _, events = simulate_staged_hawkes_counts(np.array([20 * (1-n)]), np.array([n]), 400, 1.,
                int(1190 + n * 100), burn_in=8/(1-n), record_events=True)
            history = prepare_hawkes_history(events)
            expected = fit_background_window(history, 100, 400, np.ones(400), False)
            actual = fit_fast_window(history, 100, 400)
            self.assertTrue(expected["fit_valid"] and actual[4])
            self.assertAlmostEqual(expected["n_hat"], actual[0], delta=2e-5)

    def test_prefix_forecast_and_first_features_do_not_use_future_events(self):
        rng = np.random.default_rng(891)
        events = np.sort(rng.uniform(0, 120, 450))
        counts = np.bincount(np.floor(events).astype(int), minlength=120)
        original, original_fits = build_selection_features(counts, events, [10, 30], 60, 10)
        changed = np.sort(np.r_[events[events < 90], rng.uniform(90, 120, 900)])
        changed_counts = np.bincount(np.floor(changed).astype(int), minlength=120)
        matrix, fits = build_selection_features(changed_counts, changed, [10, 30], 60, 10)
        np.testing.assert_array_equal(original[:, 0], matrix[:, 0])
        np.testing.assert_array_equal(original_fits[:, 0], fits[:, 0])
        self.assertEqual(matrix.shape, (2, 4, len(selection_feature_names())))
        self.assertEqual(len(selection_feature_names()), 64)

    def test_new_simulation_keeps_prefix_and_staged_count_consistency(self):
        config = dict(prefix_days=60, stages=8, stage_days=4, lookback=3, forecast_stages=2,
                      critical_level=.9, rate_min=.5, rate_max=2., windows=[4, 8])
        tasks = selection_tasks(71, "train", 1, config)
        reserved = [v for task in tasks for v in task[2:4]]
        second = selection_tasks(71, "test", 1, config, reserved)
        self.assertFalse(set(reserved) & {v for task in second for v in task[2:4]})
        for task in (tasks[0], tasks[4], tasks[-1]):
            counts, events, ratios, rate = simulate_selection_trajectory(task)
            np.testing.assert_array_equal(np.bincount(np.floor(events).astype(int), minlength=92), counts)
            repeated = simulate_selection_trajectory(task)
            np.testing.assert_array_equal(counts, repeated[0])
            self.assertEqual(len(ratios), 8)

    def test_tolerance_rank_matches_exact_binomial_coverage_and_is_monotone(self):
        rank, bound = tolerance_rank(300, .05, .001)
        self.assertLessEqual(bound, .05)
        self.assertGreaterEqual(binom.cdf(rank - 1, 300, .95), .999)
        if rank > 1:
            self.assertLess(binom.cdf(rank - 2, 300, .95), .999)
        stricter, _ = tolerance_rank(300, .05, .0001)
        self.assertGreaterEqual(stricter, rank)
        self.assertIsNone(tolerance_rank(10, .05, .001)[0])

    def test_calibration_handles_ties_and_small_samples_by_abstaining(self):
        target = state_warning_targets(np.full((300, 8), .4), 4, 3, 2, .9)["state"]
        scores = np.full(target.labels.shape, .4)
        threshold, table = calibrate_confident(scores, target, np.repeat("a", 300), .05, .01)
        self.assertGreater(threshold, .4)
        self.assertFalse((scores >= threshold).any())
        self.assertLessEqual(table.tail_probability_upper_bound.max(), .05)
        short_target = state_warning_targets(np.full((2, 8), .4), 4, 3, 2, .9)["state"]
        threshold, table = calibrate_confident(scores[:2], short_target, ["a", "a"], .05, .001)
        self.assertGreater(threshold, 1)
        self.assertTrue(table.forced_abstention_insufficient_calibration.all())


if __name__ == "__main__":
    unittest.main()
