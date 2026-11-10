from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from critical_processes.detector import feature_sequence_windows, transform_detector_features
from critical_processes.robust_training import (
    TrajectoryBatch, calibrate_family_threshold, evaluate_warning_scores, warning_targets,
)


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description="Independently reconstruct the saved experiment metrics and calibration.")
    parser.add_argument("--out", type=Path, default=Path("results/neural_detector_robust_90d"))
    output = parser.parse_args().out
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    results = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=False)
    batches = {}
    for split, expected_hash in config["dataset_sha256"].items():
        path = output / "data" / f"{split}.npz"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
        with np.load(path) as data:
            batches[split] = TrajectoryBatch(**{key: data[key] for key in data.files})
    for left, right in (("train", "calibration"), ("train", "test"), ("calibration", "test")):
        assert not set(batches[left].ids) & set(batches[right].ids)
    assert hashlib.sha256((output / "model.pt").read_bytes()).hexdigest() == config["model_sha256"]
    assert hashlib.sha256(Path(config["baseline_model"]).read_bytes()).hexdigest() == config["baseline_sha256"]
    protocol = [config[key] for key in ("stage_days", "lookback", "forecast_stages", "critical_level")]
    train = batches["train"]
    _, eligible, _, _ = warning_targets(train.ratios, *protocol)
    features = transform_detector_features(feature_sequence_windows(train.features, config["lookback"])[eligible], version=2)
    np.testing.assert_allclose(features.mean(axis=0), checkpoint["mean"], atol=1e-12)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1
    np.testing.assert_allclose(scale, checkpoint["scale"], atol=1e-12)

    def read_scores(filename: str, batch: TrajectoryBatch) -> np.ndarray:
        frame = pd.read_csv(output / filename, float_precision="round_trip")
        _, _, _, days = warning_targets(batch.ratios, *protocol)
        return frame.pivot(index="trajectory_id", columns="decision_end_day", values="score").reindex(index=batch.ids, columns=days).to_numpy()

    calibration = batches["calibration"]
    labels, eligible, _, _ = warning_targets(calibration.ratios, *protocol)
    threshold, _ = calibrate_family_threshold(
        read_scores("calibration_scores.csv", calibration), labels, eligible, calibration.families,
        config["target_false_alarm_probability_per_trajectory_per_family"],
    )
    assert threshold == checkpoint["threshold"]
    for model in ("baseline", "robust"):
        actual, _, _, _ = evaluate_warning_scores(
            read_scores(f"{model}_test_scores.csv", batches["test"]), batches["test"],
            results[model]["threshold"], *protocol,
        )
        for key, value in actual.items():
            if value is None:
                assert results[model][key] is None
            else:
                np.testing.assert_allclose(value, results[model][key], atol=1e-12)
    print("Verified dataset/model hashes, disjoint partitions, train-only scaler, calibration threshold and every test metric.")


if __name__ == "__main__":
    main()
