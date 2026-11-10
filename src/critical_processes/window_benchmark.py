from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .detector import build_detector
from .negative_controls import stage_features_from_counts
from .robust_training import (
    FAMILIES, FAMILY_NAMES, TrajectoryBatch, calibrate_family_threshold,
    evaluate_warning_scores, simulate_robust_trajectory, warning_targets,
)
from .window_features import build_window_features, hawkes_trend_scores, window_feature_names


MODEL_NAMES = {
    "count_mlp": "MLP: счётчики",
    "hawkes_mlp": "MLP: счётчики + MLE",
    "hawkes_trend": "Тренд оконного n",
}


@dataclass
class WindowBatch:
    base: TrajectoryBatch
    windows: np.ndarray
    short_fits: np.ndarray
    diagnostics: np.ndarray


def _generate_window_trajectory(task):
    seed, split_id, family_index, replicate, config = task
    rng = np.random.default_rng(np.random.SeedSequence([seed, split_id, family_index, replicate]))
    counts, ratios, rate, events = simulate_robust_trajectory(FAMILIES[family_index], rng, record_events=True, **config)
    windows, short, diagnostic = build_window_features(
        counts, events, config["stage_days"], config["lookback"], config["forecast_stages"], decay=1.0
    )
    return counts.astype(np.int32), ratios, rate, windows, short, diagnostic


def generate_window_batch(per_family: int, split: str, split_id: int, seed: int, config: dict, workers: int = 2) -> WindowBatch:
    tasks = [(seed, split_id, family, replicate, config) for family in range(len(FAMILIES)) for replicate in range(per_family)]
    records = []
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        iterator = pool.map(_generate_window_trajectory, tasks, chunksize=4) if pool else map(_generate_window_trajectory, tasks)
        for index, record in enumerate(iterator, 1):
            records.append(record)
            if index % per_family == 0:
                print(f"{split}: generated events and MLE features for {index}/{len(tasks)} trajectories", flush=True)
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
    counts, ratios, rates, windows, short, diagnostics = zip(*records)
    base = TrajectoryBatch(
        np.stack([stage_features_from_counts(row, config["stage_days"]) for row in counts]),
        np.stack(ratios), np.asarray(rates), np.repeat(FAMILIES, per_family),
        np.asarray([f"{split}/{FAMILIES[family]}/{replicate}" for _, _, family, replicate, _ in tasks]),
        np.stack(counts),
    )
    return WindowBatch(base, np.stack(windows), np.stack(short), np.stack(diagnostics))


def fit_window_network(
    matrix: np.ndarray, labels: np.ndarray, eligible: np.ndarray,
    indices: list[int], seed: int, epochs: int,
) -> tuple[dict, pd.DataFrame]:
    import torch
    from torch import nn

    x = np.asarray(matrix, dtype=float)[eligible][:, indices]
    y = np.asarray(labels, dtype=float)[eligible]
    if not np.isfinite(x).all() or y.sum() == 0 or y.sum() == len(y):
        raise ValueError("need finite features and both training classes")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = build_detector(len(indices))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y) - y.sum()) / y.sum(), dtype=torch.float32))
    train_x, train_y = torch.tensor((x - mean) / scale, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch + 1, "training_loss": float(loss.detach())})
    return {"state_dict": model.state_dict(), "mean": mean, "scale": scale, "selected_feature_indices": indices}, pd.DataFrame(history)


def score_window_matrix(matrix: np.ndarray, model_path: str | Path) -> np.ndarray:
    import torch

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 3 or matrix.shape[2] != len(checkpoint["window_feature_names"]) or not np.isfinite(matrix).all():
        raise ValueError("matrix must contain the finite causal window features matching the checkpoint schema")
    indices = checkpoint["selected_feature_indices"]
    x = (matrix[:, :, indices].reshape(-1, len(indices)) - checkpoint["mean"]) / checkpoint["scale"]
    model = build_detector(len(indices))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        scores = [torch.sigmoid(model(torch.tensor(x[start:start + 8192], dtype=torch.float32))).numpy() for start in range(0, len(x), 8192)]
    return np.concatenate(scores).reshape(matrix.shape[:2]).astype(float)


def run_window_benchmark(
    out: str | Path = "results/window_detector_comparison", seed: int = 20261113,
    train_per_family: int = 200, calibration_per_family: int = 200, test_per_family: int = 200,
    epochs: int = 200, workers: int = 2, stage_days: int = 90, stages: int = 12,
    lookback: int = 3, forecast_stages: int = 2, critical_level: float = 0.9,
    rate_min: float = 0.2, rate_max: float = 50.0, target_false_alarm_probability: float = 0.05,
) -> dict:
    import torch

    if min(train_per_family, calibration_per_family, test_per_family, epochs, workers) < 1 or seed < 0:
        raise ValueError("positive dataset sizes, epochs and workers, nonnegative seed required")
    if not 0 < target_false_alarm_probability < 1 or calibration_per_family < np.ceil(1 / target_false_alarm_probability) - 1:
        raise ValueError("calibration size must support the requested probability")
    if stage_days < 3 or lookback < 2 or stages < lookback + forecast_stages + 3 or not 0.8 <= critical_level <= 0.95 or not 0 < rate_min <= rate_max:
        raise ValueError("invalid simulation protocol")
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    config = dict(stages=stages, stage_days=stage_days, lookback=lookback, forecast_stages=forecast_stages, critical_level=critical_level, rate_min=rate_min, rate_max=rate_max)
    batches, data_hashes = {}, {}
    for split_id, (split, size) in enumerate((("train", train_per_family), ("calibration", calibration_per_family), ("test", test_per_family))):
        batch = generate_window_batch(size, split, split_id, seed, config, workers)
        batches[split] = batch
        path = output / "data" / f"{split}.npz"
        np.savez_compressed(path, **batch.base.__dict__, windows=batch.windows, short_fits=batch.short_fits, diagnostics=batch.diagnostics)
        data_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    for left, right in (("train", "calibration"), ("train", "test"), ("calibration", "test")):
        if set(batches[left].base.ids) & set(batches[right].base.ids):
            raise ValueError("simulation partitions overlap")
    protocol = [stage_days, lookback, forecast_stages, critical_level]
    train, calibration, test = [batches[key] for key in ("train", "calibration", "test")]
    train_labels, train_eligible, _, _ = warning_targets(train.base.ratios, *protocol)
    cal_labels, cal_eligible, _, _ = warning_targets(calibration.base.ratios, *protocol)
    names = window_feature_names(lookback)
    results, calibration_tables, family_tables, trajectory_tables, artifacts = {}, [], [], [], {}
    for model_index, model_name in enumerate(MODEL_NAMES):
        path = output / "models" / f"{model_name}.pt"
        if model_name != "hawkes_trend":
            indices = list(range(4 * lookback if model_name == "count_mlp" else len(names)))
            checkpoint, history = fit_window_network(train.windows, train_labels, train_eligible, indices, seed + 100 + model_index, epochs)
            checkpoint.update({
                "input_feature_schema": "conditional_hawkes_windows_v1", "window_feature_names": names,
                "lookback": lookback, "stage_horizon": float(stage_days), "forecast_stages": forecast_stages,
                "critical_level": critical_level, "rate_min": rate_min, "rate_max": rate_max,
                "feature_transform_version": 2, "threshold": 0.5, "fixed_decay_per_day": 1.0,
            })
            torch.save(checkpoint, path)
            history.to_csv(output / f"{model_name}_training_loss.csv", index=False)
            cal_scores = score_window_matrix(calibration.windows, path)
        else:
            cal_scores = hawkes_trend_scores(calibration.diagnostics)
        threshold, table = calibrate_family_threshold(cal_scores, cal_labels, cal_eligible, calibration.base.families, target_false_alarm_probability)
        calibration_tables.append(table.assign(model=model_name, threshold=threshold))
        if model_name != "hawkes_trend":
            checkpoint["threshold"] = threshold
            torch.save(checkpoint, path)
            artifacts[model_name] = hashlib.sha256(path.read_bytes()).hexdigest()
            test_scores = score_window_matrix(test.windows, path)
        else:
            path = path.with_suffix(".json")
            path.write_text(json.dumps({"threshold": threshold, "fixed_decay_per_day": 1.0, "score": "max(0,n_hat+forecast_stages*slope)/(1+max(0,n_hat+forecast_stages*slope)); zero if fits invalid"}, indent=2), encoding="utf-8")
            artifacts[model_name] = hashlib.sha256(path.read_bytes()).hexdigest()
            test_scores = hawkes_trend_scores(test.diagnostics)
        _, _, _, cal_frame = evaluate_warning_scores(cal_scores, calibration.base, threshold, *protocol)
        cal_frame.to_csv(output / f"{model_name}_calibration_scores.csv", index=False)
        metrics, family, trajectories, windows = evaluate_warning_scores(test_scores, test.base, threshold, *protocol)
        results[model_name] = metrics
        family_tables.append(family.assign(model=model_name))
        trajectory_tables.append(trajectories.assign(model=model_name))
        windows.to_csv(output / f"{model_name}_test_scores.csv", index=False)
        print(f"{model_name}: threshold={threshold:.6f}, recall={metrics['event_detection_probability']:.3f}, max trajectory false alarm={metrics['maximum_family_false_alarm_probability']:.3f}", flush=True)
    pd.concat(calibration_tables).to_csv(output / "calibration.csv", index=False)
    pd.concat(family_tables).to_csv(output / "family_test_metrics.csv", index=False)
    pd.concat(trajectory_tables).to_csv(output / "trajectory_test_metrics.csv", index=False)
    diagnostic_rows = []
    for i, identity in enumerate(test.base.ids):
        for stage in range(stages):
            n, se, baseline_fraction, gain, valid = test.short_fits[i, stage]
            diagnostic_rows.append({
                "trajectory_id": identity, "family": test.base.families[i], "stage": stage,
                "true_n_for_audit_only": test.base.ratios[i, stage], "n_hat": n, "n_curvature_se": se,
                "fit_valid": valid, "baseline_fraction": baseline_fraction, "ll_gain_per_event": gain,
            })
    pd.DataFrame(diagnostic_rows).to_csv(output / "mle_test_diagnostics.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    manifest = {
        **config, "seed": seed, "epochs": epochs, "workers": workers,
        "train_per_family": train_per_family, "calibration_per_family": calibration_per_family,
        "test_per_family": test_per_family, "families": FAMILIES,
        "target_false_alarm_probability": target_false_alarm_probability,
        "feature_names": names, "fixed_decay_per_day": 1.0,
        "feature_input": "observed event times and counts only; true ratios used only for target labels and audit",
        "test_is_new_after_previous_experiment": True,
        "no_model_or_threshold_selection_on_test": True,
        "threshold_rule": "maximum family quantile of negative trajectory maxima, rank ceil((m+1)*(1-alpha)); strict exceedance",
        "partition_seed_indices": {"train": 0, "calibration": 1, "test": 2},
        "dataset_sha256": data_hashes, "model_sha256": artifacts,
        "numpy_version": np.__version__, "torch_version": torch.__version__,
        "event_storage": "events generated and consumed per trajectory; regenerate from seed to audit, no artificial real-data jitter",
    }
    (output / "run_config.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_window_benchmark_report(output)
    return results


def write_window_benchmark_report(output: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    family = pd.read_csv(output / "family_test_metrics.csv")
    diagnostics = pd.read_csv(output / "mle_test_diagnostics.csv")
    calibration = pd.read_csv(output / "calibration.csv")
    trajectories = pd.read_csv(output / "trajectory_test_metrics.csv", dtype={"detected": "boolean"})
    rising = trajectories[trajectories.family == "hawkes_rising"]
    paired = rising.pivot(index="trajectory_id", columns="model", values="detected")
    paired = paired.astype(np.int8)
    pair_rng = np.random.default_rng(np.random.SeedSequence([config["seed"], 999]))
    resampled_indices = pair_rng.integers(0, len(paired), size=(5000, len(paired)))
    differences = []
    for name in ("hawkes_mlp", "hawkes_trend"):
        delta = paired[name].to_numpy() - paired["count_mlp"].to_numpy()
        lower, upper = np.quantile(delta[resampled_indices].mean(axis=1), [0.025, 0.975])
        differences.append({
            "model": name, "reference": "count_mlp", "trajectories": len(paired),
            "detection_probability_difference": float(delta.mean()),
            "paired_bootstrap_ci95_lower": float(lower), "paired_bootstrap_ci95_upper": float(upper),
        })
    pd.DataFrame(differences).to_csv(output / "paired_detection_difference.csv", index=False)
    diagnostic_quality = diagnostics.assign(
        error=diagnostics.n_hat - diagnostics.true_n_for_audit_only,
        absolute_error=(diagnostics.n_hat - diagnostics.true_n_for_audit_only).abs(),
    ).groupby("family", as_index=False).agg(
        fitted_windows=("fit_valid", "count"), valid_fraction=("fit_valid", "mean"),
        mean_true_n=("true_n_for_audit_only", "mean"), mean_fitted_n=("n_hat", "mean"),
        mean_error=("error", "mean"), mean_absolute_error=("absolute_error", "mean"),
    )
    diagnostic_quality.to_csv(output / "mle_quality.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
    colors = ["#477d94", "#16876d", "#bc6c3b"]
    for index, (model, title) in enumerate(MODEL_NAMES.items()):
        offset = (index - 1) * 0.24
        sample = family[family.model == model].set_index("family").reindex(FAMILIES)
        axes[0].barh(np.arange(len(FAMILIES)) + offset, sample.probability_false_alarm * 100, height=0.22, color=colors[index], label=title)
        values = [metrics[model][key] for key in ("event_detection_probability", "clean_event_detection_probability", "window_sensitivity")]
        bars = axes[1].bar(np.arange(3) + offset, np.asarray(values) * 100, width=0.22, color=colors[index], label=title)
        axes[1].bar_label(bars, labels=[f"{100*value:.1f}%" for value in values], padding=3, fontsize=8)
    axes[0].set_yticks(np.arange(len(FAMILIES)), [FAMILY_NAMES[name] for name in FAMILIES])
    axes[0].invert_yaxis()
    axes[0].axvline(config["target_false_alarm_probability"] * 100, color="#444444", linestyle="--", linewidth=1, label=f"Цель {100*config['target_false_alarm_probability']:g}%")
    false_alarm_limit = max(10, 5 * np.ceil(100 * family.probability_false_alarm.max() * 1.2 / 5))
    axes[0].set(title="Хотя бы одна ложная тревога на траекторию", xlabel="% независимых траекторий", xlim=(0, false_alarm_limit))
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].set_xticks(np.arange(3), ["Обнаружение\nдо порога", "Без ранней\nложной тревоги", "Доля верных\nположительных окон"])
    sensitivity_limit = max(10, 5 * np.ceil(100 * max(row["event_detection_probability"] for row in metrics.values()) * 1.3 / 5))
    axes[1].set(title="Раннее обнаружение на новом тесте", ylabel="%", ylim=(0, sensitivity_limit))
    axes[1].legend(fontsize=8)
    axes[0].grid(axis="x", alpha=0.2)
    axes[1].grid(axis="y", alpha=0.2)
    figure.savefig(output / "comparison.png", dpi=170)
    plt.close(figure)
    report = [
        "# Оконные оценки Хоукса: нейросеть и статистический ориентир", "",
        f"На новом тесте MLP по счётчикам обнаружила {metrics['count_mlp']['detected_trajectories']}/{metrics['count_mlp']['crossing_trajectories']} пересечений, расширенная MLP — {metrics['hawkes_mlp']['detected_trajectories']}/{metrics['hawkes_mlp']['crossing_trajectories']}, индикатор тренда — {metrics['hawkes_trend']['detected_trajectories']}/{metrics['hawkes_trend']['crossing_trajectories']}. Максимальные тестовые доли траекторий с ложной тревогой: {100*metrics['count_mlp']['maximum_family_false_alarm_probability']:.1f}%, {100*metrics['hawkes_mlp']['maximum_family_false_alarm_probability']:.1f}% и {100*metrics['hawkes_trend']['maximum_family_false_alarm_probability']:.1f}% соответственно. Большее число обнаружений расширенной MLP следует сопоставлять с превышением целевых {100*config['target_false_alarm_probability']:g}% ложных тревог.", "",
        f"Новый эксперимент, seed={config['seed']}. Предыдущий тест повлиял на выбор признаков, поэтому его повторная оценка не используется как независимое подтверждение. Для обучения, калибровки и нового теста сгенерированы раздельные потоки по {config['train_per_family']}, {config['calibration_per_family']}, {config['test_per_family']} реализаций каждого из девяти семейств. Все три метода оцениваются на одних и тех же новых тестовых траекториях.", "",
        "## Признаки и оценивание", "",
        "Моменты событий Хоукса генерируются симулятором с непрерывной историей. Для пуассоновских контролей с постоянной внутри каждого дня интенсивностью времена условно на суточном счётчике равномерны; это точная генерация модели, а не восстановление неизвестных времён из реальных данных.", "",
        f"Из каждого завершенного {config['stage_days']}-суточного блока оцениваются n и базовая интенсивность методом условного максимального правдоподобия. Наблюденные события до левой границы входят в память и компенсатор. Дополнительно выполняется подгонка по последним lookback блокам. Скорость затухания зафиксирована равной 1/сутки, как в генераторе: это благоприятное модельное предположение, не проверка переносимости на неизвестное ядро.", "",
        "При фиксированном затухании интенсивность линейна по базовой интенсивности и n, а отрицательная log-likelihood выпукла. Используется [SciPy L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html) с аналитическим градиентом. Группы совпадающих времён не возбуждают друг друга мгновенно. Нестабильные или слишком бедные событиями подгонки имеют явный флаг; такие окна не удаляются из теста. Кривизна likelihood используется как дополнительный признак неопределенности, не как доверительный интервал с гарантированным покрытием на границе n=0.", "",
        f"Базовая MLP получает {4 * config['lookback']} нормированных признаков счётчиков. Расширенная MLP получает {len(config['feature_names'])} признаков: те же счётчики, оценки n, их кривизну, долю базовой интенсивности, прирост likelihood, признаки валидности по коротким и длинному окну, наклон n, линейную проекцию и разницу короткой/длинной оценок. Полный список сохранен в конфигурации. Истинные n, семейство, время пересечения и будущее не передаются в признаки. Нормировка по среднему/стандартному отклонению вычисляется только на обучении.", "",
        "Статистический ориентир: линейная проекция текущей короткой оценки n на горизонт предупреждения по наклону последних lookback оценок. Неотрицательная проекция u преобразуется в ограниченный score u/(1+u); при невалидной подгонке score=0. Это прозрачный индикатор, не вероятность критичности и не p-value.", "",
        "## Одинаковая калибровка и оценка", "",
        f"Для каждого метода отдельно выбран порог по максимуму отрицательных scores на калибровочной траектории. Цель — не более {100*config['target_false_alarm_probability']:g}% траекторий с хотя бы одной ложной тревогой в каждом семействе при том же распределении параметров. Ранг квантили ceil((m+1)*(1-alpha)), общий порог строго выше наибольшей семейной квантили. Тест не используется для выбора эпохи, архитектуры, признаков или порога в этом эксперименте. Это маргинальная калибровка по траекториям, не гарантия для всех средних интенсивностей или фиксированного единственного порога.", "",
        f"Успешное предупреждение: 0 < день пересечения n={config['critical_level']:g} − день решения <= {config['forecast_stages']*config['stage_days']} суток. Сигналы в момент пересечения и после него не засчитываются. Чистое обнаружение дополнительно требует отсутствия предшествующей ложной тревоги. Медиана упреждения рассчитывается только среди обнаруженных случаев.", "",
        "| Метод | FPR окон | Обнаружено пересечений | Чистое обнаружение | Максимум ложных тревог по траекториям | Медиана упреждения |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for model, title in MODEL_NAMES.items():
        row = metrics[model]
        lead = row['median_lead_days_among_detected']
        report.append(f"| {title} | {100*row['window_false_alarm_rate']:.2f}% | {row['detected_trajectories']}/{row['crossing_trajectories']} ({100*row['event_detection_probability']:.1f}%) | {100*row['clean_event_detection_probability']:.1f}% | {100*row['maximum_family_false_alarm_probability']:.1f}% | {lead:g} суток |" if lead is not None else f"| {title} | {100*row['window_false_alarm_rate']:.2f}% | 0/{row['crossing_trajectories']} (0%) | 0% | {100*row['maximum_family_false_alarm_probability']:.1f}% | нет обнаружений |")
    comparison = differences[0]
    report.extend([
        "", f"Парная разница обнаружения расширенной MLP относительно MLP по счётчикам: {100*comparison['detection_probability_difference']:.1f} процентного пункта, 95%-интервал bootstrap по целым положительным траекториям [{100*comparison['paired_bootstrap_ci95_lower']:.1f}; {100*comparison['paired_bootstrap_ci95_upper']:.1f}] п.п. " + ("Интервал включает ноль: надёжное преимущество по обнаружению не установлено этим тестом." if comparison['paired_bootstrap_ci95_lower'] <= 0 <= comparison['paired_bootstrap_ci95_upper'] else "Интервал не включает ноль, но это не устраняет превышение частоты ложных тревог и неопределенность повторных обучений."),
        "",
    ])
    report.extend(["", "## По семействам", "", "| Семейство | MLP: счётчики | MLP: счётчики + MLE | Тренд n |", "| --- | ---: | ---: | ---: |"])
    for name in FAMILIES:
        values = [family[(family.family == name) & (family.model == model)].iloc[0].probability_false_alarm for model in MODEL_NAMES]
        report.append(f"| {FAMILY_NAMES[name]} | " + " | ".join(f"{100*value:.1f}%" for value in values) + " |")
    report.extend([
        "", "## Качество оценок и ограничения", "",
        f"Доля валидных коротких подгонок на тесте: {100*diagnostics.fit_valid.mean():.2f}%. Флаги валидности и оценки по каждому блоку сохранены в mle_test_diagnostics.csv; истинный n в этом файле используется только для аудита. Максимальная калибровочная доля ложных тревог среди всех методов/семейств: {100*calibration.calibration_probability_false_alarm.max():.1f}%.", "",
        "Качество самих оценок сохранено в mle_quality.csv. Аудит включает все наблюденные блоки, в том числе после пересечения, и не является отдельной метрикой раннего обнаружения. Значительная положительная оценка n на сезонном Пуассоне демонстрирует ошибку спецификации постоянной базы, хотя истинный коэффициент самовозбуждения там равен нулю.", "",
        "Оценка n в стационарной модели может реагировать на сезонность и тренды интенсивности. Она не является непосредственным измерением истинной критичности. Для окон с меняющимся n стационарная подгонка — локальное приближение. Первый наблюдаемый блок не содержит ненаблюденной истории прогрева; последующие окна используют только доступную наблюденную историю.", "",
        "Для каждой модели фиксирован один seed обучения. Интервалы по обнаружению и семействам в metrics.json и family_test_metrics.csv отражают вариацию тестовых траекторий, но не случайность повторных обучений. Новое семейство, неизвестное затухание и городской масштаб этой проверкой не покрываются. При сравнении нельзя выбирать лучший порог по показанному тесту.", "",
        "Артефакты: модели и схема признаков в models/, исходные счётчики, режимы и оконные оценки в data/*.npz, все решения, калибровочные таблицы, metrics.json, SHA-256 и параметры в run_config.json. Моменты событий можно воспроизвести по seed, индексу разбиения, семейству и номеру реализации; целиком они не сохраняются, чтобы не раздувать объем данных.", "",
        "![Сравнение](comparison.png)",
    ])
    (output / "analysis.md").write_text("\n".join(report) + "\n", encoding="utf-8")
