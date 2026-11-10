from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from critical_processes.detector import build_detector
from critical_processes.robust_training import (
    FAMILIES, TrajectoryBatch, calibrate_family_threshold, evaluate_warning_scores,
    generate_robust_batch, train_robust_detector, warning_targets,
)
from critical_processes.simulation import _staged_hawkes_counts, simulate_staged_hawkes_counts

try:
    import torch
except ImportError:
    torch = None


class WarningProtocolTests(unittest.TestCase):
    def test_zero_lead_and_post_crossing_decisions_are_excluded(self) -> None:
        ratios = np.array([[0.2] * 6 + [0.92] * 4, [0.4] * 10])
        labels, eligible, crossing, days = warning_targets(ratios, 10, 3, 2, 0.9)
        np.testing.assert_array_equal(days[labels[0]], [40, 50])
        self.assertEqual(crossing[0], 60)
        self.assertFalse(eligible[0, days == 60].any())
        self.assertTrue(eligible[1].all())
        self.assertFalse(labels[1].any())

    def test_calibration_excludes_positive_and_post_crossing_scores_and_handles_ties(self) -> None:
        scores = np.zeros((200, 3))
        scores[:100, 0] = np.linspace(0.0, 0.2, 100)
        scores[100:, 0] = np.linspace(0.0, 0.6, 100)
        scores[:, 1:] = [0.99, 1.0]
        labels = np.zeros_like(scores, dtype=bool)
        labels[:, 1] = True
        eligible = np.ones_like(labels)
        eligible[:, 2] = False
        threshold, table = calibrate_family_threshold(scores, labels, eligible, np.repeat(["a", "b"], 100), 0.1)
        self.assertLess(threshold, 0.6)
        self.assertGreater(threshold, table.negative_max_quantile.max())
        self.assertLessEqual(table.calibration_probability_false_alarm.max(), 0.1)
        tied = np.full((20, 2), 0.4)
        tie_threshold, _ = calibrate_family_threshold(tied, np.zeros_like(tied, dtype=bool), np.ones_like(tied, dtype=bool), np.repeat("a", 20), 0.05)
        self.assertFalse((tied >= tie_threshold).any())

    def test_late_detection_does_not_erase_a_premature_false_alarm(self) -> None:
        ratios = np.array([[0.2] * 6 + [0.92] * 4, [0.4] * 10])
        batch = TrajectoryBatch(np.zeros((2, 10, 4)), ratios, np.ones(2), np.array(["hawkes_rising", "hawkes_stable"]), np.array(["test/0", "test/1"]), np.zeros((2, 100)))
        scores = np.full((2, 8), 0.1)
        scores[0, [0, 2, 3]] = 0.9
        scores[1, 3] = 0.9
        metrics, _, trajectories, _ = evaluate_warning_scores(scores, batch, 0.5, 10, 3, 2, 0.9)
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_positive"], 2)
        self.assertEqual(metrics["event_detection_probability"], 1.0)
        self.assertEqual(metrics["clean_event_detection_probability"], 0.0)
        self.assertEqual(metrics["median_lead_days_among_detected"], 10.0)
        self.assertEqual(trajectories.iloc[0].first_alarm_day, 30)


@unittest.skipIf(_staged_hawkes_counts is None, "requires the optional fast dependency")
class StagedHawkesTests(unittest.TestCase):
    def test_excitation_survives_parameter_boundary(self) -> None:
        counts = simulate_staged_hawkes_counts(np.array([60.0, 0.01]), np.array([0.8, 0.0]), 5, 0.2, seed=52, burn_in=100.0)
        self.assertGreater(counts[5:].sum(), 100)
        np.testing.assert_array_equal(counts, simulate_staged_hawkes_counts(np.array([60.0, 0.01]), np.array([0.8, 0.0]), 5, 0.2, seed=52, burn_in=100.0))

    def test_poisson_limit_has_correct_count_mean(self) -> None:
        totals = [simulate_staged_hawkes_counts(np.array([4.0]), np.array([0.0]), 20, 1.0, seed=i).sum() for i in range(150)]
        self.assertLess(abs(np.mean(totals) - 80.0), 3.5)

    def test_partitions_are_reproducible_and_separate(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            a = generate_robust_batch(1, "train", 0, 818, stages=8, stage_days=4, rate_min=0.5, rate_max=2.0)
            repeat = generate_robust_batch(1, "train", 0, 818, stages=8, stage_days=4, rate_min=0.5, rate_max=2.0)
            b = generate_robust_batch(1, "test", 2, 818, stages=8, stage_days=4, rate_min=0.5, rate_max=2.0)
        np.testing.assert_array_equal(a.counts, repeat.counts)
        self.assertFalse(set(a.ids) & set(b.ids))
        self.assertFalse(np.array_equal(a.counts, b.counts))
        labels, _, _, _ = warning_targets(a.ratios, 4, 3, 2, 0.9)
        self.assertEqual(labels.sum(), 2)
        self.assertEqual(a.features.shape, (len(FAMILIES), 8, 4))


@unittest.skipIf(torch is None or _staged_hawkes_counts is None, "requires neural and fast dependencies")
class RobustTrainingIntegrationTests(unittest.TestCase):
    def test_training_calibration_checkpoint_and_test_are_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.pt"
            torch.manual_seed(4)
            torch.save({
                "state_dict": build_detector(12).state_dict(), "mean": np.zeros(12), "scale": np.ones(12),
                "lookback": 3, "feature_count": 4, "threshold": 0.5, "stage_horizon": 4.0,
                "forecast_stages": 2, "critical_level": 0.9, "rate_min": 0.5, "rate_max": 2.0,
            }, baseline)
            output = Path(directory) / "training"
            with contextlib.redirect_stdout(io.StringIO()):
                result = train_robust_detector(
                    output, baseline, train_per_family=2, calibration_per_family=19, test_per_family=2,
                    epochs=2, stage_days=4, stages=8, rate_min=0.5, rate_max=2.0, seed=89,
                )
            checkpoint = torch.load(output / "model.pt", weights_only=False)
            manifest = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
            with np.load(output / "data/train.npz") as train:
                train_ids = train["ids"].copy()
            with np.load(output / "data/calibration.npz") as calibration:
                calibration_ids = calibration["ids"].copy()
            with np.load(output / "data/test.npz") as test:
                test_ids = test["ids"].copy()
            self.assertEqual(checkpoint["feature_transform_version"], 2)
            self.assertEqual(checkpoint["threshold"], result["robust"]["threshold"])
            self.assertFalse(manifest["calibration_and_test_used_for_gradient_training"])
            self.assertFalse(set(train_ids) & set(calibration_ids))
            self.assertFalse(set(train_ids) & set(test_ids))
            self.assertEqual(result["robust"]["crossing_trajectories"], 2)
            self.assertTrue((output / "analysis.md").is_file())


if __name__ == "__main__":
    unittest.main()
