from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from critical_processes.n_accuracy import (
    accuracy_tasks, empirical_auc, run_n_accuracy, simulate_accuracy_trial,
    summarize_accuracy, validate_accuracy_grid,
)


class NAccuracyTests(unittest.TestCase):
    def test_invalid_estimates_are_not_counted_as_zero_or_removed_from_success_denominator(self) -> None:
        raw = pd.DataFrame({
            "true_n": [0.9, 0.9], "target_rate": [0.2, 0.2], "window_days": [30, 30],
            "fit_valid": [True, False], "n_hat": [0.82, np.nan],
            "event_count": [10, 0], "observed_rate": [1 / 3, 0],
            "failure_reason": ["", "insufficient_events"],
        })
        row = summarize_accuracy(raw).iloc[0]
        self.assertAlmostEqual(row.mae_valid, 0.08)
        self.assertEqual(row.valid_fraction, 0.5)
        self.assertEqual(row.accuracy_within_0p10_all_trials, 0.5)
        self.assertEqual(row.insufficient_events_fraction, 0.5)

    def test_auc_direction_and_ties(self) -> None:
        self.assertEqual(empirical_auc([0.1, 0.2], [0.8, 0.9]), 1.0)
        self.assertEqual(empirical_auc([0.5, 0.5], [0.5, 0.5]), 0.5)
        self.assertEqual(empirical_auc([0.8, 0.9], [0.1, 0.2]), 0.0)
        self.assertTrue(np.isnan(empirical_auc([], [0.5])))

    def test_seeds_are_unique_and_baseline_preserves_expected_rate(self) -> None:
        tasks = accuracy_tasks([0.0, 0.9], [2.0], [10, 30], 4, 132, 1.0, 30, 8.0)
        self.assertEqual(len({task[5] for task in tasks}), len(tasks))
        rows = simulate_accuracy_trial(tasks[-1])
        for row in rows:
            self.assertAlmostEqual(row["true_baseline"] / (1 - row["true_n"]), row["target_rate"])
        np.testing.assert_equal(rows, simulate_accuracy_trial(tasks[-1]))
        shorter = simulate_accuracy_trial(tuple([*tasks[-1][:3], [10], *tasks[-1][4:]]))[0]
        for name in ("n_hat", "event_count", "fit_valid"):
            np.testing.assert_equal(rows[0][name], shorter[name])

    def test_rejects_duplicate_and_out_of_range_grid(self) -> None:
        with self.assertRaises(ValueError):
            validate_accuracy_grid([0.9, 0.9], [2], [30], 10, 1, 30, 8)
        with self.assertRaises(ValueError):
            validate_accuracy_grid([1], [2], [30], 10, 1, 30, 8)

    def test_small_complete_experiment_saves_raw_data_protocol_and_maps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stdout(io.StringIO()):
                summary = run_n_accuracy(
                    directory, ratios=[0.0, 0.9], rates=[1.0], windows=[10, 30],
                    replicates=4, seed=8182, workers=1, auc_bootstrap_samples=20,
                )
            output = Path(directory)
            raw = pd.read_csv(output / "replicates.csv")
            config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(len(raw), 16)
            self.assertEqual(len(summary), 4)
            self.assertEqual(config["independent_trajectories"], 8)
            self.assertTrue(raw.drop_duplicates("trajectory_id").simulator_seed.is_unique)
            self.assertTrue(raw.loc[~raw.fit_valid, "n_hat"].isna().all())
            self.assertTrue((output / "analysis.md").is_file())
            self.assertTrue((output / "mae_map.png").is_file())
            self.assertTrue((output / "auc_map.png").is_file())


if __name__ == "__main__":
    unittest.main()
