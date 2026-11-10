from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .indicators import summarize_counts
from .simulation import simulate_hawkes


def build_detector(input_size: int):
    from torch import nn

    class Detector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.Sequential(nn.Linear(input_size, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

        def forward(self, values):
            return self.layers(values).squeeze(-1)

    return Detector()


def transform_detector_features(
    features: np.ndarray, bin_width: float = 1.0, version: int = 2
) -> np.ndarray:
    """Normalize each lag block; version 1 preserves the saved legacy models."""
    transformed = np.array(features, dtype=float, copy=True, order="C")
    if transformed.ndim != 2 or transformed.shape[1] == 0 or transformed.shape[1] % 4:
        raise ValueError("features must have four columns per lag block")
    if not np.isfinite(transformed).all() or not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("features must be finite and bin_width positive")
    if version not in (1, 2):
        raise ValueError("unsupported feature transform version")
    blocks = transformed[:, :4] if version == 1 else transformed.reshape(-1, 4)
    mean_count = np.maximum(blocks[:, 0] * bin_width, 1e-8)
    blocks[:, 0] = np.log1p(np.maximum(blocks[:, 0], 0.0))
    blocks[:, 1] /= mean_count
    blocks[:, 3] /= mean_count
    return transformed


def make_trajectory_data(
    trajectories: int = 120,
    stages: int = 12,
    stage_horizon: float = 30.0,
    lookback: int = 3,
    forecast_stages: int = 2,
    critical_level: float = 0.9,
    rate_min: float = 0.2,
    rate_max: float = 50.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build feature windows, future-critical labels, and trajectory identifiers."""
    rng = np.random.default_rng(seed)
    if rate_min <= 0 or rate_max < rate_min:
        raise ValueError("rate bounds must satisfy 0 < rate_min <= rate_max")
    sequences: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    for trajectory in range(trajectories):
        start_n = rng.uniform(0.25, 0.62)
        end_n = rng.uniform(max(critical_level, 0.92), 0.985)
        target_rate = float(np.exp(rng.uniform(np.log(rate_min), np.log(rate_max))))
        ratios = np.linspace(start_n, end_n, stages)
        ratios += rng.normal(0.0, 0.008, size=stages)
        ratios = np.maximum.accumulate(np.clip(ratios, 0.05, 0.995))
        stage_features = []
        for n in ratios:
            events = simulate_hawkes(target_rate * (1.0 - float(n)), float(n), 1.0, stage_horizon, rng, burn_in=8.0 / (1.0 - float(n)))
            counts = np.histogram(events, bins=np.arange(0, stage_horizon + 1))[0]
            stage_features.append(summarize_counts(counts, 1.0))
        stage_features = np.asarray(stage_features)
        trajectory_x, trajectory_y = [], []
        for index in range(lookback - 1, stages):
            future = ratios[index + 1:index + 1 + forecast_stages]
            is_early = ratios[index] < critical_level and np.any(future >= critical_level)
            trajectory_x.append(stage_features[index - lookback + 1:index + 1].reshape(-1))
            trajectory_y.append(float(is_early))
        sequences.append(np.asarray(trajectory_x))
        labels.append(np.asarray(trajectory_y))
        groups.append(np.full(len(trajectory_y), trajectory, dtype=int))
    return np.concatenate(sequences), np.concatenate(labels), np.concatenate(groups)


def train_neural_detector(
    out: str | Path,
    trajectories: int = 120,
    stages: int = 12,
    lookback: int = 3,
    seed: int = 42,
    epochs: int = 80,
    stage_horizon: float = 30.0,
    forecast_stages: int = 2,
    critical_level: float = 0.9,
    rate_min: float = 0.2,
    rate_max: float = 50.0,
) -> dict[str, float]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("install the optional dependency with: pip install -e '.[neural]'") from exc

    torch.manual_seed(seed)
    x, y, groups = make_trajectory_data(
        trajectories=trajectories,
        stages=stages,
        stage_horizon=stage_horizon,
        lookback=lookback,
        forecast_stages=forecast_stages,
        critical_level=critical_level,
        rate_min=rate_min,
        rate_max=rate_max,
        seed=seed,
    )
    x = transform_detector_features(x)
    rng = np.random.default_rng(seed)
    trajectory_ids = np.arange(trajectories)
    rng.shuffle(trajectory_ids)
    train_end = max(1, int(0.7 * trajectories))
    validation_end = max(train_end + 1, int(0.85 * trajectories))
    train_ids = set(trajectory_ids[:train_end])
    validation_ids = set(trajectory_ids[train_end:validation_end])
    test_ids = set(trajectory_ids[validation_end:])
    train_mask = np.array([group in train_ids for group in groups])
    validation_mask = np.array([group in validation_ids for group in groups])
    test_mask = np.array([group in test_ids for group in groups])
    if not validation_mask.any() or not test_mask.any():
        raise ValueError("at least three trajectories are required for train, validation, and test splits")
    mean, scale = x[train_mask].mean(axis=0), x[train_mask].std(axis=0)
    scale[scale < 1e-8] = 1.0
    x_scaled = (x - mean) / scale

    model = build_detector(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    train_x = torch.tensor(x_scaled[train_mask], dtype=torch.float32)
    train_y = torch.tensor(y[train_mask], dtype=torch.float32)
    positive = max(float(train_y.sum()), 1.0)
    negative = max(float(len(train_y) - train_y.sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive))
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(train_x), train_y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        validation_probabilities = torch.sigmoid(model(torch.tensor(x_scaled[validation_mask], dtype=torch.float32))).numpy()
        probabilities = torch.sigmoid(model(torch.tensor(x_scaled[test_mask], dtype=torch.float32))).numpy()
    validation_truth = y[validation_mask].astype(bool)
    truth = y[test_mask].astype(bool)

    def metrics(probability: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
        predicted = probability >= threshold
        tp = int(np.sum(predicted & labels))
        fp = int(np.sum(predicted & ~labels))
        fn = int(np.sum(~predicted & labels))
        tn = int(np.sum(~predicted & ~labels))
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        return {
            "threshold": float(threshold),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "false_alarm_rate": 1.0 - specificity,
            "precision": tp / max(tp + fp, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        }

    thresholds = np.linspace(0.05, 0.95, 181)
    calibrated_threshold = max(
        thresholds,
        key=lambda threshold: (
            metrics(validation_probabilities, validation_truth, float(threshold))["sensitivity"]
            + metrics(validation_probabilities, validation_truth, float(threshold))["specificity"]
        ),
    )
    result = {
        "test_fixed_0_5": metrics(probabilities, truth, 0.5),
        "test_calibrated": metrics(probabilities, truth, float(calibrated_threshold)),
        "validation_calibrated_threshold": float(calibrated_threshold),
        "test_windows": float(len(truth)),
        "test_trajectories": float(len(test_ids)),
        "test_positive_fraction": float(np.mean(truth)),
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "mean": mean,
        "scale": scale,
        "lookback": lookback,
        "feature_count": 4,
        "threshold": float(calibrated_threshold),
        "stage_horizon": stage_horizon,
        "forecast_stages": forecast_stages,
        "critical_level": critical_level,
        "rate_min": rate_min,
        "rate_max": rate_max,
        "feature_transform_version": 2,
    }, out)
    metadata = {
        "seed": seed,
        "trajectories": trajectories,
        "stages": stages,
        "lookback": lookback,
        "epochs": epochs,
        "stage_horizon": stage_horizon,
        "forecast_stages": forecast_stages,
        "critical_level": critical_level,
        "feature_transform_version": 2,
        "training_windows": int(train_mask.sum()),
        "validation_windows": int(validation_mask.sum()),
        "test_windows": int(test_mask.sum()),
        **result,
    }
    out.with_suffix(".metrics.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def score_feature_matrix(features: np.ndarray, model_path: str | Path) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional stage feature matrix")
    return score_feature_sequences(features[np.newaxis, :, :], model_path)[0]


def feature_sequence_windows(features: np.ndarray, lookback: int) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    if features.ndim != 3 or features.shape[0] == 0 or not np.isfinite(features).all():
        raise ValueError("features must be finite independent trajectory sequences")
    if lookback < 1 or features.shape[1] < lookback:
        raise ValueError("each sequence must contain at least lookback stages")
    return np.stack([
        features[:, i - lookback + 1:i + 1, :].reshape(len(features), -1)
        for i in range(lookback - 1, features.shape[1])
    ], axis=1)


def score_feature_sequences(
    features: np.ndarray, model_path: str | Path, batch_size: int = 8192
) -> np.ndarray:
    """Score independent stage sequences without forming windows across trajectories."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("install the optional dependency with: pip install -e '.[neural]'") from exc

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if checkpoint.get("input_feature_schema", "count_features_v1") != "count_features_v1":
        raise ValueError("this model requires event-time window features; use build_window_features and score_window_matrix")
    lookback = int(checkpoint["lookback"])
    feature_count = int(checkpoint["feature_count"])
    features = np.asarray(features, dtype=float)
    if features.ndim != 3 or features.shape[0] == 0 or features.shape[2] != feature_count:
        raise ValueError("features must have shape (trajectories, stages, feature_count)")
    if lookback < 1 or batch_size < 1 or not np.isfinite(features).all():
        raise ValueError("lookback and batch_size must be positive, and features finite")
    if features.shape[1] < lookback:
        raise ValueError(f"at least {lookback} feature windows are required")
    windows = feature_sequence_windows(features, lookback)
    matrix = windows.reshape(-1, lookback * feature_count)
    matrix = transform_detector_features(matrix, version=int(checkpoint.get("feature_transform_version", 1)))
    matrix = (matrix - checkpoint["mean"]) / checkpoint["scale"]

    model = build_detector(lookback * feature_count)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        scores = [
            torch.sigmoid(model(torch.tensor(matrix[start:start + batch_size], dtype=torch.float32))).numpy()
            for start in range(0, len(matrix), batch_size)
        ]
    return np.concatenate(scores).reshape(windows.shape[:2]).astype(float)


def detector_config(model_path: str | Path) -> dict[str, float | int]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("install the optional dependency with: pip install -e '.[neural]'") from exc
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    return {
        "lookback": int(checkpoint["lookback"]),
        "stage_horizon": float(checkpoint.get("stage_horizon", 30.0)),
        "threshold": float(checkpoint.get("threshold", 0.5)),
        "critical_level": float(checkpoint.get("critical_level", 0.9)),
        "forecast_stages": int(checkpoint.get("forecast_stages", 2)),
        "rate_min": float(checkpoint.get("rate_min", 2.0)),
        "rate_max": float(checkpoint.get("rate_max", 2.0)),
        "feature_transform_version": int(checkpoint.get("feature_transform_version", 1)),
    }
