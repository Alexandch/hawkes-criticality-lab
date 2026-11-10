from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .detector import detector_config, score_feature_sequences
from .indicators import summarize_counts


SCENARIOS = {
    "homogeneous": "Однородный Пуассон",
    "intraday": "Только час суток",
    "seasonal": "Неделя и год",
    "trend": "Плавный рост",
    "step_up": "Скачок вверх",
    "step_down": "Скачок вниз",
    "seasonal_trend": "Сезонность и рост",
}


def poisson_control_rates(
    scenario: str, rate: float, days: int, rng: np.random.Generator
) -> np.ndarray:
    """Exact daily intensity integrals of an exogenous Poisson control.

    Random phases/change points are sampled before the counts, independently
    of event history. Every trajectory has the same requested time-average rate.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown negative control: {scenario}")
    if not np.isfinite(rate) or rate <= 0 or days < 2:
        raise ValueError("rate must be finite and positive, days must be >= 2")
    if scenario in ("homogeneous", "intraday"):
        # A normalized 24-hour profile changes event times but not daily counts.
        return np.full(days, rate, dtype=float)
    day = np.arange(days, dtype=float)
    weights = np.ones(days)
    if scenario in ("seasonal", "seasonal_trend"):
        weekly_phase, annual_phase = rng.uniform(0.0, 2 * np.pi, size=2)
        weights *= 1 + 0.45 * np.sin(2 * np.pi * day / 7 + weekly_phase)
        weights *= 1 + 0.60 * np.sin(2 * np.pi * day / 365.25 + annual_phase)
    if scenario in ("trend", "seasonal_trend"):
        weights *= np.geomspace(0.5, 2.0, days)
    if scenario in ("step_up", "step_down"):
        change_day = int(rng.uniform(0.35, 0.65) * days)
        weights = np.where(day < change_day, 0.5, 2.0)
        if scenario == "step_down":
            weights = weights[::-1]
    return rate * weights / weights.mean()


def stage_features_from_counts(counts: np.ndarray, stage_days: int) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 1 or stage_days < 3 or len(counts) == 0 or len(counts) % stage_days:
        raise ValueError("counts must contain complete stages of at least three days")
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("counts must be finite and nonnegative")
    return np.vstack([summarize_counts(stage, 1.0) for stage in counts.reshape(-1, stage_days)])


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("successes must be between zero and positive total")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def summarize_negative_scores(
    scores: np.ndarray, threshold: float, rng: np.random.Generator, bootstrap_samples: int = 1000
) -> dict[str, object]:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or not all(scores.shape) or not np.isfinite(scores).all():
        raise ValueError("scores must be a nonempty finite trajectory-by-window matrix")
    if ((scores < 0) | (scores > 1)).any() or not 0 < threshold < 1 or bootstrap_samples < 1:
        raise ValueError("scores must be in [0, 1], threshold in (0, 1), bootstrap_samples positive")
    alarms = scores >= threshold
    trajectory_fpr = alarms.mean(axis=1)
    any_alarm = alarms.any(axis=1)
    fpr = float(trajectory_fpr.mean())
    any_lower, any_upper = wilson_interval(int(any_alarm.sum()), len(any_alarm))
    if fpr == 0.0:
        fpr_lower, fpr_upper = 0.0, any_upper
        ci_method = "Wilson upper bound for any alarm across independent trajectories"
    elif fpr == 1.0:
        fpr_lower, fpr_upper = wilson_interval(len(any_alarm), len(any_alarm))
        ci_method = "Wilson lower bound for all windows alarming across independent trajectories"
    else:
        indices = rng.integers(0, len(scores), size=(bootstrap_samples, len(scores)))
        means = trajectory_fpr[indices].mean(axis=1)
        fpr_lower, fpr_upper = np.quantile(means, [0.025, 0.975])
        ci_method = "trajectory bootstrap percentile 95%"
    episodes = alarms[:, 0].astype(int) + np.sum(alarms[:, 1:] & ~alarms[:, :-1], axis=1)
    return {
        "trajectories": int(len(scores)),
        "windows_per_trajectory": int(scores.shape[1]),
        "windows": int(scores.size),
        "false_alarm_windows": int(alarms.sum()),
        "window_false_alarm_rate": fpr,
        "window_fpr_ci95_lower": float(fpr_lower),
        "window_fpr_ci95_upper": float(fpr_upper),
        "window_fpr_ci_method": ci_method,
        "trajectories_with_alarm": int(any_alarm.sum()),
        "probability_any_alarm": float(any_alarm.mean()),
        "any_alarm_ci95_lower": any_lower,
        "any_alarm_ci95_upper": any_upper,
        "mean_alarm_episodes_per_trajectory": float(episodes.mean()),
        "mean_score": float(scores.mean()),
        "score_p95": float(np.quantile(scores, 0.95)),
    }


def evaluate_negative_controls(
    model_path: str | Path,
    out: str | Path,
    rates: list[float] | None = None,
    replicates: int = 300,
    stages: int = 12,
    seed: int = 20261110,
    bootstrap_samples: int = 1000,
) -> pd.DataFrame:
    """Audit a frozen model on controls with branching ratio zero throughout."""
    rates = [2.0, 20.0, 50.0, 680.0, 1560.0] if rates is None else list(rates)
    if not rates or len(set(rates)) != len(rates) or any(not np.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("rates must be distinct, finite and positive")
    if replicates < 2 or stages < 1 or bootstrap_samples < 1 or seed < 0:
        raise ValueError("replicates >= 2, stages >= 1, bootstrap_samples >= 1 and seed >= 0 required")
    config = detector_config(model_path)
    stage_days = int(config["stage_horizon"])
    if stage_days < 3 or float(stage_days) != config["stage_horizon"] or stages < config["lookback"]:
        raise ValueError("the model requires whole-day stages >= 3 and at least lookback stages")
    horizon_days = stage_days * stages
    threshold = float(config["threshold"])
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    summary_rows, trajectory_rows, score_rows = [], [], []
    end_days = np.arange(int(config["lookback"]), stages + 1) * stage_days
    for scenario_index, scenario in enumerate(SCENARIOS):
        for rate_index, rate in enumerate(rates):
            features = []
            outside_fractions = []
            for replicate in range(replicates):
                # Pair the intraday-only and homogeneous controls as an exact
                # daily-aggregation invariance check, not as independent samples.
                stream_index = 0 if scenario == "intraday" else scenario_index
                rng = np.random.default_rng(np.random.SeedSequence([seed, stream_index, rate_index, replicate]))
                means = poisson_control_rates(scenario, rate, horizon_days, rng)
                counts = rng.poisson(means)
                features.append(stage_features_from_counts(counts, stage_days))
                outside_fractions.append(float(np.mean((means < config["rate_min"]) | (means > config["rate_max"]))))
            scores = score_feature_sequences(np.stack(features), model_path)
            bootstrap_rng = np.random.default_rng(np.random.SeedSequence([seed, 999, scenario_index, rate_index]))
            metrics = summarize_negative_scores(scores, threshold, bootstrap_rng, bootstrap_samples)
            summary_rows.append({
                "scenario": scenario,
                "mean_rate": rate,
                "average_rate_in_training_range": bool(config["rate_min"] <= rate <= config["rate_max"]),
                "mean_daily_expected_rate_outside_training_range_fraction": float(np.mean(outside_fractions)),
                "true_branching_ratio": 0.0,
                "threshold": threshold,
                **metrics,
            })
            for replicate, sequence in enumerate(scores):
                alarms = sequence >= threshold
                alarm_positions = np.flatnonzero(alarms)
                episodes = int(alarms[0]) + int(np.sum(alarms[1:] & ~alarms[:-1]))
                trajectory_rows.append({
                    "scenario": scenario, "mean_rate": rate, "replicate": replicate,
                    "false_alarm_windows": int(alarms.sum()), "windows": len(sequence),
                    "any_alarm": bool(alarms.any()), "alarm_episodes": episodes,
                    "first_alarm_day": int(end_days[alarm_positions[0]]) if len(alarm_positions) else None,
                })
                for day, score in zip(end_days, sequence):
                    score_rows.append({
                        "scenario": scenario, "mean_rate": rate, "replicate": replicate,
                        "decision_end_day": int(day), "score": float(score),
                        "warning": bool(score >= threshold), "true_branching_ratio": 0.0,
                    })
            print(f"{scenario}, rate={rate:g}: window FPR={metrics['window_false_alarm_rate']:.3f}, any alarm={metrics['probability_any_alarm']:.3f}", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "negative_control_summary.csv", index=False)
    pd.DataFrame(trajectory_rows).to_csv(output / "trajectory_metrics.csv", index=False)
    pd.DataFrame(score_rows).to_csv(output / "window_scores.csv", index=False)
    model_file = Path(model_path)
    run_config = {
        "model_path": str(model_file),
        "model_sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
        "model_config": config,
        "model_retrained": False,
        "threshold_recalibrated": False,
        "rates": rates,
        "replicates_per_scenario_and_rate": replicates,
        "stages": stages,
        "horizon_days": horizon_days,
        "seed": seed,
        "bootstrap_samples": bootstrap_samples,
        "daily_aggregation": "independent Poisson counts with exact daily intensity integrals",
        "intraday_homogeneous_pairing": "identical daily counts and scores by design; do not pool these controls",
        "rate_profiles": {
            "intraday": "q(h) = 1 + 0.8*cos(2*pi*(h+0.5)/24); daily integral unchanged",
            "seasonal": "weekly amplitude 0.45, annual amplitude 0.60, random phases",
            "trend": "geometric growth from 0.5 to 2.0, normalized to requested time-average rate",
            "step": "4-fold change at a random point between 35% and 65% of the horizon",
        },
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    (output / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(summary, output, run_config)
    return summary


def _write_report(summary: pd.DataFrame, output: Path, run_config: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rates = run_config["rates"]
    config = run_config["model_config"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    for axis, metric, title in zip(axes, ("window_false_alarm_rate", "probability_any_alarm"), ("Доля окон с ложной тревогой", "Хотя бы одна тревога за траекторию")):
        values = summary.pivot(index="scenario", columns="mean_rate", values=metric).reindex(index=SCENARIOS, columns=rates).to_numpy()
        plot = axis.imshow(values * 100, vmin=0, vmax=100, cmap="YlOrRd", aspect="auto")
        for row in range(len(SCENARIOS)):
            for column in range(len(rates)):
                axis.text(column, row, f"{100 * values[row, column]:.1f}%", ha="center", va="center", color="white" if values[row, column] > 0.55 else "black")
        axis.set_yticks(range(len(SCENARIOS)), list(SCENARIOS.values()))
        axis.set_xticks(range(len(rates)), [f"{rate:g}" + ("*" if not config["rate_min"] <= rate <= config["rate_max"] else "") for rate in rates])
        axis.set(title=title, xlabel="Среднее число событий в сутки")
    figure.colorbar(plot, ax=axes, label="%", shrink=0.8)
    figure.suptitle(f"Отрицательные контроли: n=0, порог={config['threshold']:.3f}")
    figure.supxlabel(f"* Средняя интенсивность вне диапазона обучения [{config['rate_min']:g}; {config['rate_max']:g}]", fontsize=9)
    figure.savefig(output / "false_alarms.png", dpi=170)
    plt.close(figure)

    report = [
        "# Ложные тревоги нейросетевого детектора: отрицательные контроли",
        "",
        f"Модель: `{run_config['model_path']}`. SHA-256 и параметры сохранены в `run_config.json`. Веса и порог ({config['threshold']:.3f}) зафиксированы до эксперимента; переобучения и подбора порога по этим контролям не было.",
        "",
        f"На каждую пару сценарий/средняя интенсивность создано {run_config['replicates_per_scenario_and_rate']} реализаций по {run_config['horizon_days']} суток. Каждый блок признаков использует {config['stage_horizon']:g} суток; lookback={config['lookback']}. Решения вычисляются после завершения полного блока, с тем же шагом, что и блок. Все процессы имеют n=0 на всем горизонте, поэтому каждая тревога ложная относительно задачи обнаружения критичности.",
        "",
        "Суточные количества моделируются непосредственно независимыми пуассоновскими величинами с заданными интегралами интенсивности. Для суточного детектора это точное моделирование наблюдаемого входа, без необходимости генерировать каждую временную отметку. Фазы сезонности и момент скачка выбираются независимо от потока; все профили нормированы к одному заданному среднему по времени. Скачки четырехкратные, плавный рост — от 0.5 до 2 в относительных единицах.",
        "",
        "Контроль «Только час суток» имеет тот же суточный интеграл, что и однородный Пуассон. Эти два контроля специально используют одни и те же суточные количества и потому дают одинаковые scores. Это проверка инвариантности агрегации; их нельзя объединять как независимые наблюдения. Из одного внутрисуточного ритма нельзя получить изменение суточного ряда при сохранении его интеграла.",
        "",
        f"Диапазон средних интенсивностей при обучении: [{config['rate_min']:g}; {config['rate_max']:g}] событий/сутки. Отметка * обозначает среднюю интенсивность вне этого диапазона. Колонка `mean_daily_expected_rate_outside_training_range_fraction` также показывает долю дней, чья ожидаемая интенсивность выходит за диапазон: она может быть положительной даже при допустимом среднем.",
        "",
        "## Результаты",
        "",
        "FPR окон — доля решений с ложной тревогой; P(хотя бы одна) — доля независимых траекторий с одной или несколькими тревогами. Для FPR окна 95%-интервал получен bootstrap целых траекторий, сохраняя зависимость соседних окон. При полном отсутствии/наличии тревог использована консервативная относительно окон граница Wilson по независимым траекториям. Для P(хотя бы одна) используются интервалы Wilson. Интервалы точечные для каждой строки, не одновременные по всей таблице.",
        "",
        "| Сценарий | Событий/сутки | FPR окон | 95% интервал FPR | P(хотя бы одна) | 95% интервал P |",
        "| --- | ---: | ---: | --- | ---: | --- |",
    ]
    for row in summary.to_dict(orient="records"):
        marker = "" if row["average_rate_in_training_range"] else "*"
        report.append(f"| {SCENARIOS[row['scenario']]} | {row['mean_rate']:g}{marker} | {100*row['window_false_alarm_rate']:.1f}% | [{100*row['window_fpr_ci95_lower']:.1f}; {100*row['window_fpr_ci95_upper']:.1f}]% | {100*row['probability_any_alarm']:.1f}% | [{100*row['any_alarm_ci95_lower']:.1f}; {100*row['any_alarm_ci95_upper']:.1f}]% |")
    report.extend([
        "",
        "## Интерпретация",
        "",
    ])
    in_range = summary[summary["average_rate_in_training_range"]]
    worst = in_range.loc[in_range["window_false_alarm_rate"].idxmax()] if len(in_range) else summary.loc[summary["window_false_alarm_rate"].idxmax()]
    report.extend([
        f"Наибольшая доля ложных тревог среди проверенных средних интенсивностей {'в диапазоне обучения' if len(in_range) else ''}: сценарий «{SCENARIOS[worst['scenario']]}», средняя интенсивность {worst['mean_rate']:g}, FPR окон {100 * worst['window_false_alarm_rate']:.1f}%, хотя бы одна тревога у {100 * worst['probability_any_alarm']:.1f}% траекторий. Коэффициент ветвления n=0 известен по конструкции. " + ("Таким образом, данный некритический сценарий способен вызывать срабатывания сохраненной модели." if worst["window_false_alarm_rate"] > 0 else "В этих контролях наблюдаемых ложных тревог не было."),
        "",
        "Этот аудит измеряет специфичность сохраненной модели на известных некритических потоках. Он не измеряет чувствительность, время упреждения или вероятность критичности по score: для них нужны независимые положительные сценарии с известным приближением n к порогу. Значения на средних интенсивностях вне диапазона обучения характеризуют перенос на другой масштаб, отдельно от устойчивости к нестационарности при допустимом среднем. Отсутствие наблюдаемых тревог не означает нулевую вероятность тревоги и не подтверждает чувствительность модели на городском масштабе.",
        "",
        ("Версия нормировки признаков сохраненной модели: 1. Нормировался только первый блок из lookback, оставшиеся блоки подавались в исходном масштабе. Это свойство исходных весов сохранено при аудите для воспроизводимости. Новые обучения используют версию 2 с отдельной нормировкой каждого блока; они требуют отдельного сравнения с этим результатом." if config['feature_transform_version'] == 1 else "Версия нормировки признаков сохраненной модели: 2. Каждый lookback-блок нормируется отдельно; это соответствует обучению новых весов. Порог и веса в этом аудите не изменяются."),
        "",
        "Результат отрицательных контролей следует сопоставлять с чувствительностью и упреждением на положительных сценариях с известным n. Проверенные здесь test-контроли не должны использоваться для повторной калибровки порога или переобучения с последующей оценкой на тех же реализациях.",
        "",
        "Результаты: `negative_control_summary.csv` содержит групповые оценки и интервалы, `trajectory_metrics.csv` — эпизоды и время первой ложной тревоги по реализациям, `window_scores.csv` — каждое решение. Набор `intraday` не является дополнительной независимой выборкой относительно `homogeneous`.",
        "",
        "Основание моделирования: независимые приращения неоднородного Пуассона и интегральная интенсивность, см. [лекцию University of Chicago](https://www.stat.uchicago.edu/~yibi/teaching/stat317/2021/Lectures/Lecture11.pdf). Формула интервала Wilson: [NIST](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm).",
        "",
        "![Ложные тревоги](false_alarms.png)",
    ])
    (output / "analysis.md").write_text("\n".join(report) + "\n", encoding="utf-8")
