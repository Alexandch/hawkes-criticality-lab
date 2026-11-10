from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from critical_processes.robust_training import (
    FAMILIES, TrajectoryBatch, calibrate_family_threshold, evaluate_warning_scores,
    simulate_robust_trajectory, warning_targets,
)
from critical_processes.window_benchmark import MODEL_NAMES, score_window_matrix
from critical_processes.window_features import build_window_features, hawkes_trend_scores


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description="Verify window-feature comparison from immutable datasets and saved scores.")
    parser.add_argument("--out", type=Path, default=Path("results/window_detector_comparison"))
    output = parser.parse_args().out
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    results = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    protocol = [config[key] for key in ("stage_days", "lookback", "forecast_stages", "critical_level")]
    generation = {key: config[key] for key in ("stages", "stage_days", "lookback", "forecast_stages", "critical_level", "rate_min", "rate_max")}
    batches, window_arrays, diagnostic_arrays = {}, {}, {}
    for split, expected_hash in config["dataset_sha256"].items():
        path = output / "data" / f"{split}.npz"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
        with np.load(path) as data:
            batches[split] = TrajectoryBatch(**{key: data[key] for key in TrajectoryBatch.__dataclass_fields__})
            window_arrays[split] = data["windows"]
            diagnostic_arrays[split] = data["diagnostics"]
        # Regenerate event streams for examples from both Hawkes and Poisson.
        per_family = config[f"{split}_per_family"]
        for family_index, replicate in ((0, 0), (3, 0), (8, per_family - 1)):
            row = family_index * per_family + replicate
            rng = np.random.default_rng(np.random.SeedSequence([config["seed"], config["partition_seed_indices"][split], family_index, replicate]))
            counts, ratios, rate, events = simulate_robust_trajectory(FAMILIES[family_index], rng, record_events=True, **generation)
            np.testing.assert_array_equal(counts, batches[split].counts[row])
            np.testing.assert_array_equal(ratios, batches[split].ratios[row])
            np.testing.assert_allclose(rate, batches[split].rates[row])
            actual, _, _ = build_window_features(counts, events, config["stage_days"], config["lookback"], config["forecast_stages"])
            np.testing.assert_allclose(actual, window_arrays[split][row], rtol=1e-6, atol=1e-7)
    for left, right in (("train", "calibration"), ("train", "test"), ("calibration", "test")):
        assert not set(batches[left].ids) & set(batches[right].ids)
    cal_labels, cal_eligible, _, _ = warning_targets(batches["calibration"].ratios, *protocol)
    _, train_eligible, _, _ = warning_targets(batches["train"].ratios, *protocol)

    def saved_scores(model, split):
        frame = pd.read_csv(output / f"{model}_{split}_scores.csv", float_precision="round_trip")
        _, _, _, days = warning_targets(batches[split].ratios, *protocol)
        return frame.pivot(index="trajectory_id", columns="decision_end_day", values="score").reindex(index=batches[split].ids, columns=days).to_numpy()

    for model in MODEL_NAMES:
        suffix = ".json" if model == "hawkes_trend" else ".pt"
        path = output / "models" / f"{model}{suffix}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == config["model_sha256"][model]
        threshold, _ = calibrate_family_threshold(
            saved_scores(model, "calibration"), cal_labels, cal_eligible,
            batches["calibration"].families, config["target_false_alarm_probability"],
        )
        assert threshold == results[model]["threshold"]
        if model != "hawkes_trend":
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            train_x = window_arrays["train"][train_eligible][:, checkpoint["selected_feature_indices"]]
            np.testing.assert_allclose(train_x.mean(axis=0), checkpoint["mean"], atol=1e-12)
            scale = train_x.std(axis=0)
            scale[scale < 1e-8] = 1
            np.testing.assert_allclose(scale, checkpoint["scale"], atol=1e-12)
            assert checkpoint["threshold"] == threshold
            rescored = score_window_matrix(window_arrays["test"], path)
        else:
            assert json.loads(path.read_text(encoding="utf-8"))["threshold"] == threshold
            rescored = hawkes_trend_scores(diagnostic_arrays["test"])
        np.testing.assert_allclose(rescored, saved_scores(model, "test"), rtol=1e-6, atol=1e-7)
        metrics, _, _, _ = evaluate_warning_scores(saved_scores(model, "test"), batches["test"], threshold, *protocol)
        for key, value in metrics.items():
            if value is None:
                assert results[model][key] is None
            else:
                np.testing.assert_allclose(value, results[model][key], atol=1e-12)
    print("Verified hashes, disjoint partitions, regenerated event features, train-only scalers, calibration and all three held-out evaluations.")


if __name__ == "__main__":
    main()
