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

from critical_processes.detector import score_feature_sequences, transform_detector_features
from critical_processes.indicators import summarize_counts
from critical_processes.negative_controls import (
    SCENARIOS, evaluate_negative_controls, poisson_control_rates,
    stage_features_from_counts, summarize_negative_scores, wilson_interval,
)

try:
    import torch
    from torch import nn
except ImportError:
    torch = None


class NegativeControlTests(unittest.TestCase):
    def test_profiles_have_equal_time_average_without_excitation(self) -> None:
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                rates = poisson_control_rates(scenario, 20.0, 1080, np.random.default_rng(56))
                self.assertTrue((rates > 0).all())
                self.assertAlmostEqual(float(rates.mean()), 20.0, places=12)
        up = poisson_control_rates("step_up", 20.0, 1080, np.random.default_rng(56))
        down = poisson_control_rates("step_down", 20.0, 1080, np.random.default_rng(56))
        self.assertAlmostEqual(up[-1] / up[0], 4.0)
        self.assertAlmostEqual(down[-1] / down[0], 0.25)

    def test_intraday_profile_preserves_the_daily_count_distribution(self) -> None:
        samples = []
        for scenario in ("homogeneous", "intraday"):
            rng = np.random.default_rng(89)
            samples.append(rng.poisson(poisson_control_rates(scenario, 10.0, 90, rng)))
        np.testing.assert_array_equal(samples[0], samples[1])

    def test_features_do_not_cross_stage_boundaries(self) -> None:
        counts = np.array([0, 1, 3, 12, 18, 20])
        features = stage_features_from_counts(counts, 3)
        np.testing.assert_allclose(features, [summarize_counts(counts[:3], 1.0), summarize_counts(counts[3:], 1.0)])

    def test_window_alarms_and_episodes_are_distinct(self) -> None:
        scores = np.array([[0.1, 0.9, 0.9, 0.1], [0.8, 0.1, 0.8, 0.9]])
        result = summarize_negative_scores(scores, 0.5, np.random.default_rng(45))
        self.assertEqual(result["false_alarm_windows"], 5)
        self.assertEqual(result["windows"], 8)
        self.assertEqual(result["window_false_alarm_rate"], 5 / 8)
        self.assertEqual(result["probability_any_alarm"], 1.0)
        self.assertEqual(result["mean_alarm_episodes_per_trajectory"], 1.5)

    def test_zero_observed_alarms_do_not_give_zero_uncertainty(self) -> None:
        result = summarize_negative_scores(np.zeros((30, 10)), 0.5, np.random.default_rng(45))
        self.assertEqual(result["window_false_alarm_rate"], 0.0)
        self.assertGreater(result["window_fpr_ci95_upper"], 0.0)
        self.assertGreater(result["any_alarm_ci95_upper"], 0.0)
        low, high = wilson_interval(15, 30)
        self.assertAlmostEqual(low + high, 1.0)

    def test_all_lags_are_normalized_and_legacy_transform_is_preserved(self) -> None:
        features = np.array([[2.0, 4.0, 0.3, 6.0, 10.0, 20.0, 0.1, 30.0]])
        expected = [[np.log(3), 2, 0.3, 3, np.log(11), 2, 0.1, 3]]
        np.testing.assert_allclose(transform_detector_features(features), expected)
        legacy = transform_detector_features(features, version=1)
        np.testing.assert_allclose(legacy[:, :4], np.asarray(expected)[:, :4])
        np.testing.assert_array_equal(legacy[:, 4:], features[:, 4:])
        self.assertEqual(features[0, 0], 2.0)


@unittest.skipIf(torch is None, "requires the optional neural dependency")
class NeuralControlIntegrationTests(unittest.TestCase):
    def make_checkpoint(self, directory: str) -> tuple[Path, object]:
        torch.manual_seed(51)
        model = nn.Module()
        model.layers = nn.Sequential(nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        path = Path(directory) / "legacy_detector.pt"
        torch.save({
            "state_dict": model.state_dict(), "mean": np.zeros(12), "scale": np.ones(12),
            "lookback": 3, "feature_count": 4, "threshold": 0.505,
            "stage_horizon": 4.0, "forecast_stages": 2, "critical_level": 0.9,
            "rate_min": 0.2, "rate_max": 50.0,
        }, path)
        return path, model

    def test_batched_scores_match_original_inference_and_never_join_trajectories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, model = self.make_checkpoint(directory)
            rng = np.random.default_rng(61)
            features = rng.uniform(0.1, 2.0, size=(2, 5, 4))
            for version in (1, 2):
                with self.subTest(version=version):
                    if version == 2:
                        checkpoint = torch.load(path, weights_only=False)
                        checkpoint["feature_transform_version"] = 2
                        torch.save(checkpoint, path)
                    actual = score_feature_sequences(features, path, batch_size=2)
                    expected = []
                    for sequence in features:
                        flat = np.vstack([sequence[i:i + 3].reshape(-1) for i in range(3)])
                        for offset in (range(0, 12, 4) if version == 2 else [0]):
                            mean_count = flat[:, offset].copy()
                            flat[:, offset] = np.log1p(flat[:, offset])
                            flat[:, offset + 1] /= mean_count
                            flat[:, offset + 3] /= mean_count
                        with torch.no_grad():
                            expected.append(torch.sigmoid(model.layers(torch.tensor(flat, dtype=torch.float32))).numpy().ravel())
                    self.assertEqual(actual.shape, (2, 3))
                    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_audit_preserves_checkpoint_and_summary_matches_every_saved_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.make_checkpoint(directory)
            original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            out = Path(directory) / "audit"
            with contextlib.redirect_stdout(io.StringIO()):
                summary = evaluate_negative_controls(path, out, rates=[2.0], replicates=3, stages=4, bootstrap_samples=50)
            config = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
            decisions = pd.read_csv(out / "window_scores.csv")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), original_hash)
            self.assertEqual(config["model_sha256"], original_hash)
            self.assertFalse(config["threshold_recalibrated"])
            self.assertEqual(config["model_config"]["feature_transform_version"], 1)
            self.assertTrue((decisions["true_branching_ratio"] == 0).all())
            self.assertEqual(len(decisions), 7 * 3 * 2)
            for row in summary.to_dict(orient="records"):
                subset = decisions[decisions["scenario"] == row["scenario"]]
                self.assertEqual(int(subset["warning"].sum()), row["false_alarm_windows"])
                self.assertAlmostEqual(float(subset["warning"].mean()), row["window_false_alarm_rate"])
            a = decisions[decisions["scenario"] == "homogeneous"]["score"].to_numpy()
            b = decisions[decisions["scenario"] == "intraday"]["score"].to_numpy()
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
