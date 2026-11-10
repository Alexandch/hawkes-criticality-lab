from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .detector import (
    build_detector, detector_config, feature_sequence_windows,
    score_feature_sequences, transform_detector_features,
)
from .negative_controls import SCENARIOS, poisson_control_rates, stage_features_from_counts, wilson_interval
from .simulation import simulate_staged_hawkes_counts


FAMILIES = ("hawkes_rising", "hawkes_stable", "hawkes_subthreshold") + tuple(
    name for name in SCENARIOS if name != "intraday"
)
FAMILY_NAMES = {
    **SCENARIOS,
    "hawkes_rising": "Хоукс с пересечением порога",
    "hawkes_stable": "Хоукс с постоянным n",
    "hawkes_subthreshold": "Рост n без пересечения",
}


@dataclass
class TrajectoryBatch:
    features: np.ndarray
    ratios: np.ndarray
    rates: np.ndarray
    families: np.ndarray
    ids: np.ndarray
    counts: np.ndarray


def simulate_robust_trajectory(
    family: str, rng: np.random.Generator, stages: int = 12, stage_days: int = 90,
    lookback: int = 3, forecast_stages: int = 2, critical_level: float = 0.9,
    rate_min: float = 0.2, rate_max: float = 50.0, record_events: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    if family not in FAMILIES:
        raise ValueError("unknown robust simulation family")
    rate = float(np.exp(rng.uniform(np.log(rate_min), np.log(rate_max))))
    anchors = [value for value in (2.0, 20.0, 50.0) if rate_min <= value <= rate_max]
    if anchors and rng.random() < 0.3:
        rate = float(rng.choice(anchors))
    if family == "hawkes_rising":
        crossing_stage = int(rng.integers(lookback + forecast_stages + 1, stages - 1))
        initial = float(rng.uniform(0.15, 0.5))
        before = np.linspace(initial, rng.uniform(critical_level - 0.10, critical_level - 0.005), crossing_stage)
        before += rng.normal(0, 0.005, size=len(before))
        before = np.maximum.accumulate(np.clip(before, 0.05, critical_level - 0.002))
        after = np.linspace(rng.uniform(critical_level + 0.005, min(critical_level + 0.04, 0.985)), 0.985, stages - crossing_stage)
        n = np.concatenate([before, after])
    elif family == "hawkes_stable":
        n = np.full(stages, rng.uniform(0.05, critical_level - 0.01))
    elif family == "hawkes_subthreshold":
        n = np.linspace(rng.uniform(0.15, 0.5), rng.uniform(critical_level - 0.15, critical_level - 0.005), stages)
    else:
        n = np.zeros(stages)
    events = np.empty(0)
    if family.startswith("hawkes_"):
        result = simulate_staged_hawkes_counts(
            rate * (1 - n), n, stage_days, 1.0,
            int(rng.integers(0, 2**32, dtype=np.uint64)), burn_in=8.0 / (1 - n[0]), record_events=record_events,
        )
        if record_events:
            counts, events = result
        else:
            counts = result
    else:
        counts = rng.poisson(poisson_control_rates(family, rate, stages * stage_days, rng))
        if record_events:
            # This is exact for a Poisson process constant within each day,
            # unlike assigning artificial times to already observed real data.
            days = np.repeat(np.arange(len(counts)), counts)
            events = np.sort(days + rng.uniform(size=len(days)))
    return counts, n, rate, events


def warning_targets(
    ratios: np.ndarray, stage_days: int, lookback: int, forecast_stages: int, critical_level: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ratios = np.asarray(ratios, dtype=float)
    if ratios.ndim != 2 or not np.isfinite(ratios).all() or ((ratios < 0) | (ratios >= 1)).any():
        raise ValueError("ratios must be finite trajectory-by-stage values in [0, 1)")
    if not 1 <= lookback <= ratios.shape[1] or stage_days < 1 or forecast_stages < 1 or not 0 < critical_level < 1:
        raise ValueError("invalid lookback, forecast horizon, stage duration, or critical level")
    critical = ratios >= critical_level
    crossing = np.where(critical.any(axis=1), critical.argmax(axis=1) * stage_days, np.inf)
    decision_days = np.arange(lookback, ratios.shape[1] + 1) * stage_days
    lead = crossing[:, None] - decision_days[None, :]
    eligible = decision_days[None, :] < crossing[:, None]
    labels = eligible & (lead <= forecast_stages * stage_days)
    return labels, eligible, crossing, decision_days


def generate_robust_batch(
    per_family: int, split: str, split_id: int, seed: int, stages: int = 12,
    stage_days: int = 90, lookback: int = 3, forecast_stages: int = 2,
    critical_level: float = 0.9, rate_min: float = 0.2, rate_max: float = 50.0,
) -> TrajectoryBatch:
    if per_family < 1 or stages < lookback + forecast_stages + 3 or stage_days < 3:
        raise ValueError("need positive family size, stages >= lookback+forecast_stages+3, stage_days >= 3")
    if not 0.8 <= critical_level <= 0.95 or not 0 < rate_min <= rate_max:
        raise ValueError("critical_level must be in [0.8, 0.95], rate bounds positive and ordered")
    features, ratios, rates, families, ids, daily_counts = [], [], [], [], [], []
    for family_index, family in enumerate(FAMILIES):
        for replicate in range(per_family):
            sequence = np.random.SeedSequence([seed, split_id, family_index, replicate])
            rng = np.random.default_rng(sequence)
            counts, n, rate, _ = simulate_robust_trajectory(
                family, rng, stages, stage_days, lookback, forecast_stages, critical_level, rate_min, rate_max
            )
            features.append(stage_features_from_counts(counts, stage_days))
            ratios.append(n)
            rates.append(rate)
            families.append(family)
            ids.append(f"{split}/{family}/{replicate}")
            daily_counts.append(counts.astype(np.int32))
        print(f"Generated {split}: {family}, {per_family} trajectories", flush=True)
    return TrajectoryBatch(
        np.stack(features), np.stack(ratios), np.asarray(rates), np.asarray(families),
        np.asarray(ids), np.stack(daily_counts),
    )


def calibrate_family_threshold(
    scores: np.ndarray, labels: np.ndarray, eligible: np.ndarray,
    families: np.ndarray, target_false_alarm_probability: float = 0.05,
) -> tuple[float, pd.DataFrame]:
    """Calibrate on one maximum negative score per independent trajectory/family."""
    scores = np.asarray(scores, dtype=float)
    labels, eligible = np.asarray(labels, dtype=bool), np.asarray(eligible, dtype=bool)
    families = np.asarray(families)
    if scores.ndim != 2 or scores.shape != labels.shape or scores.shape != eligible.shape or len(families) != len(scores):
        raise ValueError("scores, targets and independent family identifiers must match")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any() or not 0 < target_false_alarm_probability < 1:
        raise ValueError("invalid scores or false alarm target")
    negative = eligible & ~labels
    maxima = np.max(np.where(negative, scores, -np.inf), axis=1)
    rows = []
    for family in np.unique(families):
        values = np.sort(maxima[(families == family) & np.isfinite(maxima)])
        rank = int(math.ceil((len(values) + 1) * (1 - target_false_alarm_probability)))
        if len(values) == 0 or rank > len(values):
            raise ValueError("calibration family too small for requested false alarm probability")
        rows.append({"family": family, "trajectories": len(values), "rank": rank, "negative_max_quantile": float(values[rank - 1])})
    threshold = float(np.nextafter(max(row["negative_max_quantile"] for row in rows), np.inf))
    if threshold >= 1:
        raise ValueError("negative calibration scores saturate at one; no usable threshold below one")
    for row in rows:
        values = maxima[(families == row["family"]) & np.isfinite(maxima)]
        row["calibration_probability_false_alarm"] = float(np.mean(values >= threshold))
    return threshold, pd.DataFrame(rows)


def evaluate_warning_scores(
    scores: np.ndarray, batch: TrajectoryBatch, threshold: float,
    stage_days: int, lookback: int, forecast_stages: int, critical_level: float,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = np.asarray(scores, dtype=float)
    labels, eligible, crossing, decision_days = warning_targets(batch.ratios, stage_days, lookback, forecast_stages, critical_level)
    if scores.shape != labels.shape or not np.isfinite(scores).all() or not 0 < threshold < 1:
        raise ValueError("scores must match eligible decisions and threshold must be in (0, 1)")
    alarm = scores >= threshold
    tp = int(np.sum(alarm & labels))
    fp = int(np.sum(alarm & eligible & ~labels))
    fn = int(np.sum(~alarm & labels))
    tn = int(np.sum(~alarm & eligible & ~labels))
    trajectories, windows = [], []
    for i in range(len(scores)):
        valid = alarm[i] & labels[i]
        false = alarm[i] & eligible[i] & ~labels[i]
        at_risk_alarm = np.flatnonzero(alarm[i] & eligible[i])
        first = int(at_risk_alarm[0]) if len(at_risk_alarm) else None
        valid_first = first is not None and bool(labels[i, first])
        first_valid = np.flatnonzero(valid)
        trajectories.append({
            "trajectory_id": batch.ids[i], "family": batch.families[i], "mean_rate": batch.rates[i],
            "crossing_day": float(crossing[i]) if np.isfinite(crossing[i]) else None,
            "false_alarm": bool(false.any()), "detected": bool(valid.any()) if np.isfinite(crossing[i]) else None,
            "clean_detection": valid_first if np.isfinite(crossing[i]) else None,
            "first_alarm_day": int(decision_days[first]) if first is not None else None,
            "lead_days": float(crossing[i] - decision_days[first_valid[0]]) if len(first_valid) else None,
        })
        for j, day in enumerate(decision_days):
            windows.append({
                "trajectory_id": batch.ids[i], "family": batch.families[i], "decision_end_day": int(day),
                "score": float(scores[i, j]), "eligible": bool(eligible[i, j]),
                "positive_target": bool(labels[i, j]), "warning": bool(alarm[i, j]),
            })
    trajectory_frame = pd.DataFrame(trajectories)
    family_rows = []
    for family, frame in trajectory_frame.groupby("family", sort=False):
        lower, upper = wilson_interval(int(frame["false_alarm"].sum()), len(frame))
        family_rows.append({
            "family": family, "trajectories": len(frame),
            "probability_false_alarm": float(frame["false_alarm"].mean()),
            "false_alarm_ci95_lower": lower, "false_alarm_ci95_upper": upper,
        })
    rising = trajectory_frame[trajectory_frame["crossing_day"].notna()]
    detected = int(rising["detected"].sum())
    clean = int(rising["clean_detection"].sum())
    detection_ci = wilson_interval(detected, len(rising)) if len(rising) else (None, None)
    clean_ci = wilson_interval(clean, len(rising)) if len(rising) else (None, None)
    leads = rising["lead_days"].dropna()
    metrics = {
        "threshold": threshold, "eligible_windows": int(eligible.sum()),
        "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
        "window_sensitivity": tp / (tp + fn) if tp + fn else None,
        "window_false_alarm_rate": fp / (fp + tn) if fp + tn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "crossing_trajectories": len(rising), "detected_trajectories": detected,
        "event_detection_probability": detected / len(rising) if len(rising) else None,
        "event_detection_ci95": detection_ci,
        "clean_event_detection_probability": clean / len(rising) if len(rising) else None,
        "clean_detection_ci95": clean_ci,
        "median_lead_days_among_detected": float(leads.median()) if len(leads) else None,
        "maximum_family_false_alarm_probability": max(row["probability_false_alarm"] for row in family_rows),
    }
    return metrics, pd.DataFrame(family_rows), trajectory_frame, pd.DataFrame(windows)


def train_robust_detector(
    out: str | Path, baseline_model: str | Path,
    train_per_family: int = 300, calibration_per_family: int = 200, test_per_family: int = 200,
    epochs: int = 200, seed: int = 20261111, stage_days: int = 90, stages: int = 12,
    lookback: int = 3, forecast_stages: int = 2, critical_level: float = 0.9,
    rate_min: float = 0.2, rate_max: float = 50.0, target_false_alarm_probability: float = 0.05,
) -> dict[str, object]:
    import torch
    from torch import nn

    if min(train_per_family, calibration_per_family, test_per_family) < 1 or epochs < 1 or seed < 0 or not 0 < target_false_alarm_probability < 1:
        raise ValueError("positive epochs, nonnegative seed and target probability in (0,1) required")
    if calibration_per_family < math.ceil(1 / target_false_alarm_probability) - 1:
        raise ValueError("calibration_per_family is too small for the requested false alarm target")
    baseline_config = detector_config(baseline_model)
    for name, requested in (("stage_horizon", stage_days), ("lookback", lookback), ("forecast_stages", forecast_stages), ("critical_level", critical_level)):
        if baseline_config[name] != requested:
            raise ValueError(f"baseline model uses a different {name}; paired comparison would not be valid")
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    data_directory = output / "data"
    data_directory.mkdir(exist_ok=True)
    generation_config = dict(stages=stages, stage_days=stage_days, lookback=lookback, forecast_stages=forecast_stages, critical_level=critical_level, rate_min=rate_min, rate_max=rate_max)
    batches = {}
    data_hashes = {}
    for split_id, (split, size) in enumerate((("train", train_per_family), ("calibration", calibration_per_family), ("test", test_per_family))):
        batch = generate_robust_batch(size, split, split_id, seed, **generation_config)
        batches[split] = batch
        path = data_directory / f"{split}.npz"
        np.savez_compressed(path, **batch.__dict__)
        data_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    if any(set(batches[left].ids) & set(batches[right].ids) for left, right in (("train", "calibration"), ("train", "test"), ("calibration", "test"))):
        raise ValueError("trajectory partitions overlap")

    train = batches["train"]
    labels, eligible, _, _ = warning_targets(train.ratios, stage_days, lookback, forecast_stages, critical_level)
    raw_windows = feature_sequence_windows(train.features, lookback)
    x = transform_detector_features(raw_windows[eligible], version=2)
    y = labels[eligible].astype(float)
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("training needs both positive and negative eligible windows")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = build_detector(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    train_x = torch.tensor((x - mean) / scale, dtype=torch.float32)
    train_y = torch.tensor(y, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y) - y.sum()) / y.sum(), dtype=torch.float32))
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch + 1, "training_loss": float(loss.detach())})
    pd.DataFrame(history).to_csv(output / "training_loss.csv", index=False)
    checkpoint = {
        "state_dict": model.state_dict(), "mean": mean, "scale": scale, "lookback": lookback,
        "feature_count": 4, "threshold": 0.5, "stage_horizon": float(stage_days),
        "forecast_stages": forecast_stages, "critical_level": critical_level,
        "rate_min": rate_min, "rate_max": rate_max, "feature_transform_version": 2,
        "training_protocol": "continuous_history_mixed_controls_v1",
    }
    model_path = output / "model.pt"
    torch.save(checkpoint, model_path)
    calibration = batches["calibration"]
    calibration_scores = score_feature_sequences(calibration.features, model_path)
    cal_labels, cal_eligible, _, _ = warning_targets(calibration.ratios, stage_days, lookback, forecast_stages, critical_level)
    threshold, calibration_table = calibrate_family_threshold(
        calibration_scores, cal_labels, cal_eligible, calibration.families, target_false_alarm_probability
    )
    checkpoint["threshold"] = threshold
    checkpoint["target_false_alarm_probability"] = target_false_alarm_probability
    torch.save(checkpoint, model_path)
    calibration_table.to_csv(output / "calibration.csv", index=False)
    _, _, _, cal_decisions = evaluate_warning_scores(calibration_scores, calibration, threshold, stage_days, lookback, forecast_stages, critical_level)
    cal_decisions.to_csv(output / "calibration_scores.csv", index=False)
    results = {}
    family_tables, trajectory_tables = [], []
    for label, path in (("baseline", Path(baseline_model)), ("robust", model_path)):
        scores = score_feature_sequences(batches["test"].features, path)
        operating_threshold = float(detector_config(path)["threshold"])
        metrics, family_table, trajectory_table, window_table = evaluate_warning_scores(
            scores, batches["test"], operating_threshold, stage_days, lookback, forecast_stages, critical_level
        )
        results[label] = metrics
        family_tables.append(family_table.assign(model=label))
        trajectory_tables.append(trajectory_table.assign(model=label))
        window_table.to_csv(output / f"{label}_test_scores.csv", index=False)
    pd.concat(family_tables, ignore_index=True).to_csv(output / "family_test_metrics.csv", index=False)
    pd.concat(trajectory_tables, ignore_index=True).to_csv(output / "trajectory_test_metrics.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    run_config = {
        **generation_config, "seed": seed, "epochs": epochs,
        "train_per_family": train_per_family, "calibration_per_family": calibration_per_family,
        "test_per_family": test_per_family, "families": FAMILIES,
        "target_false_alarm_probability_per_trajectory_per_family": target_false_alarm_probability,
        "threshold_rule": "maximum family quantile of trajectory negative maxima, rank ceil((m+1)*(1-alpha)), strict exceedance",
        "baseline_model": str(baseline_model), "baseline_sha256": hashlib.sha256(Path(baseline_model).read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(), "dataset_sha256": data_hashes,
        "partition_seed_indices": {"train": 0, "calibration": 1, "test": 2},
        "feature_transform_version": 2, "hawkes_history_reset_between_stages": False,
        "forecast_label": "0 < crossing_day - decision_end_day <= forecast_stages * stage_days",
        "calibration_and_test_used_for_gradient_training": False,
        "numpy_version": np.__version__, "torch_version": torch.__version__,
    }
    (output / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    write_robust_report(output)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    return results


def write_robust_report(output: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    results = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    family = pd.read_csv(output / "family_test_metrics.csv")
    calibration = pd.read_csv(output / "calibration.csv")
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
    positions = np.arange(len(FAMILIES))
    for model, offset, color, title in (("baseline", -0.17, "#b64f51", "Исходная модель"), ("robust", 0.17, "#17836c", "Новая модель")):
        frame = family[family["model"] == model].set_index("family").reindex(FAMILIES)
        axes[0].barh(positions + offset, frame["probability_false_alarm"] * 100, height=0.30, color=color, label=title)
        metrics = results[model]
        values = [metrics["event_detection_probability"], metrics["clean_event_detection_probability"], metrics["window_sensitivity"]]
        axes[1].bar(np.arange(3) + offset, np.asarray(values, dtype=float) * 100, width=0.30, color=color, label=title)
    axes[0].set_yticks(positions, [FAMILY_NAMES[name] for name in FAMILIES])
    axes[0].invert_yaxis()
    axes[0].axvline(config["target_false_alarm_probability_per_trajectory_per_family"] * 100, color="#333333", linestyle="--", linewidth=1)
    axes[0].set(title="Хотя бы одна ложная тревога на траекторию", xlabel="% независимых тестовых траекторий", xlim=(0, 100))
    axes[0].legend(fontsize=9, loc="lower right")
    axes[1].set_xticks(np.arange(3), ["Обнаружение\nдо порога", "Без ранней\nложной тревоги", "Доля верных\nположительных окон"])
    axes[1].set(title=f"Чувствительность к пересечению n={config['critical_level']:g}", ylabel="%", ylim=(0, 100))
    axes[1].legend(fontsize=9)
    axes[0].grid(axis="x", alpha=0.2)
    axes[1].grid(axis="y", alpha=0.2)
    figure.savefig(output / "model_comparison.png", dpi=170)
    plt.close(figure)

    def percent(value):
        return "не определено" if value is None else f"{100 * value:.1f}%"

    def duration(value):
        return "нет обнаружений" if value is None else f"{value:g} суток"

    report = [
        "# Переобучение детектора с отрицательными контролями",
        "",
        f"Результат: доля ложных тревог на отрицательных окнах снизилась с {percent(results['baseline']['window_false_alarm_rate'])} до {percent(results['robust']['window_false_alarm_rate'])}, но предупреждение получили лишь {results['robust']['detected_trajectories']} из {results['robust']['crossing_trajectories']} пересечений ({percent(results['robust']['event_detection_probability'])}). Медиана упреждения {duration(results['robust']['median_lead_days_among_detected'])} относится только к этим обнаруженным пересечениям. Это существенная потеря чувствительности; новую модель нельзя считать достаточным решением задачи раннего обнаружения.",
        "",
        f"Целевая вероятность ложной тревоги на траекторию при калибровке — {100*config['target_false_alarm_probability_per_trajectory_per_family']:g}%. На независимом тесте максимальная точечная оценка по семействам — {percent(results['robust']['maximum_family_false_alarm_probability'])}. " + ("Она превышает целевой уровень; перенос калибровочного ограничения на этот тест не подтвердился по всем семействам." if results['robust']['maximum_family_false_alarm_probability'] > config['target_false_alarm_probability_per_trajectory_per_family'] else "По точечным оценкам теста целевой уровень не превышен; интервалы неопределенности приведены ниже."),
        "",
        f"Новая модель обучена на {len(FAMILIES) * config['train_per_family']} траекториях. Калибровочная и тестовая выборки содержат по {config['calibration_per_family']} и {config['test_per_family']} новых траекторий каждого из девяти семейств. Разбиение выполнено до генерации, с независимыми SeedSequence потоками и непересекающимися ID. Средние интенсивности лежат в [{config['rate_min']:g}; {config['rate_max']:g}] событий/сутки; часть траекторий использует опорные значения 2, 20 и 50.",
        "",
        "Используется прежняя архитектура MLP 32–16–1 с исправленной нормировкой всех lookback-блоков (версия 2). Параметры нормировки вычислены только на обучении. Число эпох фиксировано заранее; калибровка и тест не используются для градиентного обучения или выбора эпохи. Исходная модель со своей нормировкой и порогом оценена на тех же новых тестовых реализациях.",
        "",
        "## Генерация и определение упреждения",
        "",
        "Положительные траектории: процесс Хоукса с растущим n, пересекающим заданный порог. Трудные отрицательные контроли: стационарный Хоукс и рост n без достижения порога. Остальные контроли: однородный, сезонный и трендовый Пуассон, скачки интенсивности вверх/вниз, сезонность вместе с ростом. Чисто внутрисуточный контроль не дублируется в обучении, поскольку его суточный ряд совпадает по распределению с однородным Пуассоном.",
        "",
        "Хоукс моделируется непрерывно методом прореживания: история возбуждения сохраняется между этапами. Событие, родившееся на этапе k, вносит вклад n[k]*beta*exp(-beta*age) и после последующих границ. Поэтому n[k] определяет ожидаемое число прямых потомков этого события. Базовая интенсивность на этапе равна rate*(1-n[k]); beta=1/сутки. Начальный режим прогревается на 8 времен релаксации. На переходах параметры не задают точной стационарности среднего потока.",
        "",
        f"Решение принимается только после завершения {config['stage_days']}-суточного блока, lookback={config['lookback']}. Успешное предупреждение: 0 < день пересечения − день решения <= {config['forecast_stages'] * config['stage_days']} суток. При таком шаге положительное упреждение дискретно. Решение ровно в момент пересечения и все последующие решения исключены из оценки раннего обнаружения. На траекториях без пересечения режим остается ниже порога и за концом наблюдаемого горизонта.",
        "",
        "«Чистое обнаружение» означает, что первая тревога на траектории попала в допустимое окно предупреждения. Если модель сначала тревожилась слишком рано, а затем предупредила правильно, обычное обнаружение засчитывается, но чистое — нет. Упреждение усредняется только по обнаруженным пересечениям; пропуски показаны отдельно.",
        "",
        "## Калибровка порога",
        "",
        f"Цель: вероятность хотя бы одной ложной тревоги <= {100 * config['target_false_alarm_probability_per_trajectory_per_family']:g}% для каждого известного семейства. Для каждой калибровочной траектории вычислен максимальный score на отрицательных допустимых окнах. Для каждого семейства взята порядковая статистика с рангом ceil((m+1)*(1-alpha)); общий порог строго выше наибольшей из этих статистик. Для растущего Хоукса отрицательными считаются только допустимые решения до окна упреждения.",
        "",
        "Это применение к максимумам траекторий квантильной калибровки с поправкой на конечную выборку, см. [Angelopoulos & Bates](https://arxiv.org/abs/2107.07511). Интерпретация ограничения требует обменности калибровочных и новых траекторий внутри того же семейства и распределения параметров. Это маргинальное ограничение по случайной калибровке, а не 95%-гарантия для одного фиксированного порога, всех интенсивностей по отдельности или реальных городских данных.",
        "",
        f"Полученный порог: {results['robust']['threshold']:.6f}. Максимальная калибровочная доля траекторий с ложной тревогой: {100 * calibration['calibration_probability_false_alarm'].max():.1f}%. Тестовые значения не использовались для изменения порога.",
        "",
        "## Общий независимый тест",
        "",
        "| Показатель | Исходная модель | Новая модель |",
        "| --- | ---: | ---: |",
    ]
    for key, title in (
        ("window_false_alarm_rate", "Доля ложных тревог на отрицательных окнах"),
        ("window_sensitivity", "Чувствительность по положительным окнам"),
        ("event_detection_probability", "Хотя бы одно предупреждение до пересечения"),
        ("clean_event_detection_probability", "Обнаружение без преждевременной тревоги"),
        ("maximum_family_false_alarm_probability", "Максимальная доля траекторий с ложной тревогой среди семейств"),
    ):
        report.append(f"| {title} | {percent(results['baseline'][key])} | {percent(results['robust'][key])} |")
    report.append(f"| Медиана упреждения среди обнаруженных | {duration(results['baseline']['median_lead_days_among_detected'])} | {duration(results['robust']['median_lead_days_among_detected'])} |")
    report.extend(["", "## Ложные тревоги по семействам", "", "| Семейство | Исходная модель | Новая модель | 95% Wilson для новой модели |", "| --- | ---: | ---: | --- |"])
    for name in FAMILIES:
        old = family[(family["family"] == name) & (family["model"] == "baseline")].iloc[0]
        new = family[(family["family"] == name) & (family["model"] == "robust")].iloc[0]
        report.append(f"| {FAMILY_NAMES[name]} | {percent(old['probability_false_alarm'])} | {percent(new['probability_false_alarm'])} | [{100*new['false_alarm_ci95_lower']:.1f}; {100*new['false_alarm_ci95_upper']:.1f}]% |")
    audit_paths = [output / name / "negative_control_summary.csv" for name in ("baseline_poisson_audit", "poisson_audit")]
    if all(path.exists() for path in audit_paths):
        audit = [pd.read_csv(path) for path in audit_paths]
        audit_config = [json.loads((path.parent / "run_config.json").read_text(encoding="utf-8")) for path in audit_paths]
        for key in ("seed", "rates", "stages", "replicates_per_scenario_and_rate"):
            if audit_config[0][key] != audit_config[1][key]:
                raise ValueError("separate Poisson audits must use matching random streams and cohorts")
        report.extend([
            "", "## Отдельные пуассоновские контроли", "",
            f"Обе модели дополнительно проверены на одинаковых новых потоках с seed={audit_config[0]['seed']}, по {audit_config[0]['replicates_per_scenario_and_rate']} реализаций для каждой пары сценарий/средняя интенсивность. Этот seed не использовался для обучения или калибровки. Ни порог, ни веса после проверки не менялись.",
            "", "| Сценарий | Событий/сутки | FPR окон: исходная | FPR окон: новая | Хотя бы одна: исходная | Хотя бы одна: новая |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for scenario, rate in (("seasonal", 20.0), ("step_down", 50.0), ("trend", 680.0), ("trend", 1560.0)):
            matched = [frame[(frame["scenario"] == scenario) & (frame["mean_rate"] == rate)].iloc[0] for frame in audit]
            report.append(f"| {SCENARIOS[scenario]} | {rate:g} | {percent(matched[0]['window_false_alarm_rate'])} | {percent(matched[1]['window_false_alarm_rate'])} | {percent(matched[0]['probability_any_alarm'])} | {percent(matched[1]['probability_any_alarm'])} |")
        report.extend([
            "", "Вне диапазона обучения перенос не получился: при высоких средних интенсивностях трендовый Пуассон вызывает множество тревог новой модели, хотя n=0. Поэтому эту модель пока нельзя переносить на агрегированные городские потоки. Подробные таблицы и интервалы: [новая модель](poisson_audit/analysis.md), [исходная модель на тех же потоках](baseline_poisson_audit/analysis.md).",
        ])
    report.extend([
        "",
        "Изменения включают исправленную нормировку, новый состав обучения, непрерывную историю и строгую разметку предупреждений. Это не абляционный эксперимент, позволяющий приписать эффект одной правке. Малое число ложных тревог нужно читать вместе с чувствительностью и долей пропущенных пересечений; подавление всех сигналов не является успешным ранним обнаружением.",
        "",
        "Синтетические положительные сценарии задают операционный порог n=0.9, а не достижение математической границы n=1. Все n остаются ниже единицы. Данные городов в обучении не участвуют; качество на реальных данных и на новых типах изменения режима этой проверкой не установлено. Интервалы по семействам точечные, не одновременные.",
        "",
        "Воспроизводимость: `model.pt`, `run_config.json`, `metrics.json`, `calibration.csv`, `calibration_scores.csv`, `family_test_metrics.csv`, `trajectory_test_metrics.csv`, оба `*_test_scores.csv` и исходные симуляции в `data/*.npz`. SHA-256 данных и моделей сохранены в конфигурации. Прежние результаты и веса сохранены отдельно.",
        "",
        "![Сравнение моделей](model_comparison.png)",
    ])
    (output / "analysis.md").write_text("\n".join(report) + "\n", encoding="utf-8")
