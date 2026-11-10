from __future__ import annotations

import unittest
import contextlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np

from critical_processes.estimation import hawkes_log_likelihood, poisson_mle
from critical_processes.robust_training import simulate_robust_trajectory
from critical_processes.window_benchmark import run_window_benchmark, score_window_matrix
from critical_processes.detector import score_feature_sequences
from critical_processes.window_features import (
    build_window_features, conditional_window_data, fit_hawkes_window,
    fixed_decay_loss, prepare_hawkes_history, window_feature_names,
)


class WindowFeatureTests(unittest.TestCase):
    def test_recording_events_preserves_counts_and_simulator_stream(self) -> None:
        for family in ("hawkes_rising", "seasonal"):
            counts, n, rate, _ = simulate_robust_trajectory(family, np.random.default_rng(681), stages=8, stage_days=5, rate_min=1, rate_max=2)
            observed, actual_n, actual_rate, events = simulate_robust_trajectory(family, np.random.default_rng(681), stages=8, stage_days=5, rate_min=1, rate_max=2, record_events=True)
            np.testing.assert_array_equal(counts, observed)
            np.testing.assert_array_equal(n, actual_n)
            self.assertEqual(rate, actual_rate)
            np.testing.assert_array_equal(np.bincount(np.floor(events).astype(int), minlength=40), counts)

    def test_likelihood_matches_existing_grouped_event_likelihood(self) -> None:
        events = np.array([0.1, 0.1, 0.4])
        history = prepare_hawkes_history(events)
        memory, counts, total, integral, rate = conditional_window_data(history, 0, 1)
        loss, _ = fixed_decay_loss(np.array([2 / rate, 0.5]), memory, counts, total, integral)
        expected = poisson_mle(events, 1)["log_likelihood"] - hawkes_log_likelihood(events, 1, 2, 0.5, 1)
        self.assertAlmostEqual(loss, expected, places=10)

    def test_conditional_compensator_and_gradient_include_observed_past(self) -> None:
        events = np.array([0.2, 0.8, 1.2, 1.2, 1.6])
        history = prepare_hawkes_history(events, 1.3)
        memory, counts, total, integral, rate = conditional_window_data(history, 1, 2)
        expected_integral = sum(np.exp(-1.3 * max(1 - time, 0)) - np.exp(-1.3 * (2 - time)) for time in events)
        self.assertAlmostEqual(integral, expected_integral, places=10)
        parameters = np.array([0.6, 0.4])
        _, gradient = fixed_decay_loss(parameters, memory, counts, total, integral)
        for i in range(2):
            plus, minus = parameters.copy(), parameters.copy()
            plus[i] += 1e-6
            minus[i] -= 1e-6
            numerical = (fixed_decay_loss(plus, memory, counts, total, integral)[0] - fixed_decay_loss(minus, memory, counts, total, integral)[0]) / 2e-6
            self.assertAlmostEqual(gradient[i], numerical, places=6)

    def test_future_events_cannot_change_an_earlier_feature_vector(self) -> None:
        rng = np.random.default_rng(11)
        events = np.sort(rng.uniform(0, 50, 200))
        counts = np.bincount(np.floor(events).astype(int), minlength=50)
        original, _, _ = build_window_features(counts, events, stage_days=10)
        changed_events = np.sort(np.concatenate([events[events < 30], rng.uniform(30, 50, 400)]))
        changed_counts = np.bincount(np.floor(changed_events).astype(int), minlength=50)
        changed, _, _ = build_window_features(changed_counts, changed_events, stage_days=10)
        np.testing.assert_array_equal(original[0], changed[0])
        self.assertEqual(original.shape, (3, len(window_feature_names())))

    def test_empty_windows_have_finite_explicit_missing_fit_flags(self) -> None:
        feature, short, diagnostic = build_window_features(np.zeros(30, dtype=int), np.empty(0), stage_days=10)
        self.assertTrue(np.isfinite(feature).all())
        self.assertFalse(short[:, 4].any())
        self.assertFalse(diagnostic[:, 5:].any())
        self.assertEqual(fit_hawkes_window(prepare_hawkes_history(np.empty(0)), 0, 10)[0], 0)


class WindowBenchmarkIntegrationTests(unittest.TestCase):
    def test_three_methods_share_the_same_holdout_and_serialize_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison"
            with contextlib.redirect_stdout(io.StringIO()):
                results = run_window_benchmark(
                    output, seed=7113, train_per_family=2, calibration_per_family=19,
                    test_per_family=2, epochs=2, workers=1,
                    stage_days=4, stages=8, rate_min=0.5, rate_max=2,
                )
            self.assertEqual(set(results), {"count_mlp", "hawkes_mlp", "hawkes_trend"})
            self.assertEqual({row["crossing_trajectories"] for row in results.values()}, {2})
            config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(len(config["feature_names"]), 35)
            self.assertTrue(config["no_model_or_threshold_selection_on_test"])
            with np.load(output / "data/test.npz") as dataset:
                windows = dataset["windows"].copy()
                features = dataset["features"].copy()
            scores = score_window_matrix(windows, output / "models/hawkes_mlp.pt")
            self.assertEqual(scores.shape, (18, 6))
            with self.assertRaisesRegex(ValueError, "event-time window features"):
                score_feature_sequences(features, output / "models/hawkes_mlp.pt")
            self.assertTrue((output / "models/hawkes_trend.json").is_file())


if __name__ == "__main__":
    unittest.main()
