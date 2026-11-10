from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from critical_processes.selected_features import load_selection_batch
from critical_processes.selected_training import choose_validation_candidate, run_selected_training, selection_targets, sha


class SelectedTrainingTests(unittest.TestCase):
    def test_selection_enforces_validation_constraint_and_fixed_ties(self):
        frame = pd.DataFrame({"method": ["a", "a", "a"], "window_days": [30, 90, 180], "feature_count": [16]*3,
            "utility": [.8, .3, .3], "auc_valid_only": [.9, .9, .9],
            "maximum_family_false_alarm_probability": [.08, .03, .02]})
        selected = choose_validation_candidate(frame)
        self.assertEqual(selected["window_days"], 90)
        self.assertTrue(selected["validation_feasible"])
        frame.maximum_family_false_alarm_probability = [.2, .1, .08]
        selected = choose_validation_candidate(frame)
        self.assertEqual(selected["window_days"], 180)
        self.assertFalse(selected["validation_feasible"])

    def test_training_freezes_before_test_and_normalizes_only_training_decisions(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            options = dict(out=output, train_per_family=2, validation_calibration_per_family=3,
                validation_per_family=3, calibration_per_family=3, test_per_family=3,
                model_seeds=(3, 7), windows=(4, 8), epochs=2, data_seed=7321, final_test_seed=7322,
                workers=1, bootstrap_samples=10, prefix_days=60, stage_days=4, stages=8, rate_min=.5, rate_max=2.)
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_selected_training(**options)
            protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
            frozen = json.loads((output / "frozen_plan.json").read_text(encoding="utf-8"))
            start = json.loads((output / "final_test_started.json").read_text(encoding="utf-8"))
            complete = json.loads((output / "final_test_complete.json").read_text(encoding="utf-8"))
            self.assertEqual(len(result), 22)
            self.assertFalse(frozen["test_generated"])
            self.assertEqual(start["frozen_plan_sha256"], sha(output / "frozen_plan.json"))
            self.assertLessEqual(frozen["frozen_at_utc"], start["started_at_utc"])
            self.assertLessEqual(start["started_at_utc"], complete["completed_at_utc"])
            self.assertEqual(frozen["selection_sha256"], sha(output / "selection.json"))
            batches = {split: load_selection_batch(output / "data" / f"{split}.npz") for split in protocol["sizes"]}
            ids = [identity for batch in batches.values() for identity in batch.ids]
            seeds = [int(s) for batch in batches.values() for s in np.r_[batch.parameter_seeds, batch.simulator_seeds]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(len(seeds), len(set(seeds)))
            path = output / "models/state_counts_w4_s3.pt"
            checkpoint = torch.load(path, weights_only=False)
            train = batches["train"]
            eligible = selection_targets(train, protocol)["state"].eligible
            expected = train.matrix[:, 0][eligible][:, checkpoint["selected_feature_indices"]].mean(axis=0)
            np.testing.assert_allclose(checkpoint["mean"], expected, rtol=1e-12, atol=1e-12)
            self.assertEqual(checkpoint["training_dataset_sha256"], sha(output / "data/train.npz"))
            before = sha(output / "data/test.npz")
            with contextlib.redirect_stdout(io.StringIO()):
                repeated = run_selected_training(**options)
            self.assertEqual(result, repeated)
            self.assertEqual(before, sha(output / "data/test.npz"))
            self.assertTrue((output / "analysis.md").is_file())
            self.assertTrue((output / "final_comparison.png").is_file())
            with self.assertRaises(ValueError):
                with contextlib.redirect_stdout(io.StringIO()):
                    run_selected_training(**{**options, "epochs": 3})


if __name__ == "__main__":
    unittest.main()
