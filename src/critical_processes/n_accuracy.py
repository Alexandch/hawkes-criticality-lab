from __future__ import annotations

import hashlib
import json
import platform
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, t

from .negative_controls import wilson_interval
from .simulation import simulate_staged_hawkes_counts
from .window_features import fit_hawkes_window, prepare_hawkes_history


GROUP_COLUMNS = ["true_n", "target_rate", "window_days"]
PAIR_COLUMNS = [
    "target_rate", "window_days", "lower_true_n", "upper_true_n", "negative_trials", "positive_trials",
    "valid_negative", "valid_positive", "valid_pair_probability", "auc_valid_only",
    "auc_bootstrap_ci95_lower", "auc_bootstrap_ci95_upper", "predefined_midpoint_threshold",
    "sensitivity_all_trials", "specificity_all_trials", "balanced_accuracy_all_trials_abstention_as_error",
]


def validate_accuracy_grid(ratios, rates, windows, replicates, decay, history_days, burn_in_times):
    ratios = list(ratios)
    rates = list(rates)
    windows = list(windows)
    if not ratios or len(set(ratios)) != len(ratios) or any(not np.isfinite(n) or not 0 <= n < 0.995 for n in ratios):
        raise ValueError("ratios must be distinct finite values in [0, 0.995)")
    if not rates or len(set(rates)) != len(rates) or any(not np.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("rates must be distinct finite positive values")
    if not windows or len(set(windows)) != len(windows) or any(not np.isfinite(window) or window < 3 or int(window) != window for window in windows):
        raise ValueError("windows must be distinct whole-day durations >= 3")
    if replicates < 2 or not np.isfinite(decay) or decay <= 0 or not np.isfinite(burn_in_times) or burn_in_times <= 0:
        raise ValueError("need replicates >= 2, positive finite decay and burn_in_times")
    if history_days is not None and (not np.isfinite(history_days) or history_days < 1 or int(history_days) != history_days):
        raise ValueError("history_days must be a positive integer")
    return sorted(ratios), sorted(rates), sorted(int(window) for window in windows)


def accuracy_tasks(ratios, rates, windows, replicates, seed, decay, history_days, burn_in_times):
    rng = np.random.default_rng(seed)
    used = set()
    tasks = []
    for n_index, n in enumerate(ratios):
        for rate_index, rate in enumerate(rates):
            for replicate in range(replicates):
                simulator_seed = int(rng.integers(0, 2**32, dtype=np.uint64))
                while simulator_seed in used:
                    simulator_seed = int(rng.integers(0, 2**32, dtype=np.uint64))
                used.add(simulator_seed)
                tasks.append((f"n{n_index}/rate{rate_index}/rep{replicate}", n, rate, windows, replicate, simulator_seed, decay, history_days, burn_in_times))
    return tasks


def simulate_accuracy_trial(task):
    identity, n, rate, windows, replicate, simulator_seed, decay, history_days, burn_in_times = task
    baseline = rate * (1 - n)
    burn_in = burn_in_times / (decay * (1 - n))
    horizon = history_days + max(windows)
    counts, events = simulate_staged_hawkes_counts(
        np.array([baseline]), np.array([n]), horizon, decay, simulator_seed,
        burn_in=burn_in, record_events=True,
    )
    history = prepare_hawkes_history(events, decay)
    records = []
    for window in windows:
        event_count = int(counts[history_days:history_days + window].sum())
        fit = fit_hawkes_window(history, history_days, history_days + window)
        valid = bool(fit[4])
        records.append({
            "trajectory_id": identity, "replicate": replicate, "simulator_seed": simulator_seed,
            "true_n": n, "target_rate": rate, "true_baseline": baseline,
            "decay_per_day": decay, "burn_in_days": burn_in,
            "history_days": history_days, "window_days": window,
            "event_count": event_count, "observed_rate": event_count / window,
            "history_event_count": int(counts[:history_days].sum()),
            "fit_valid": valid,
            "failure_reason": "" if valid else ("insufficient_events" if event_count < 3 else "optimization_failed"),
            "n_hat": float(fit[0]) if valid else np.nan,
            "n_curvature_se": float(fit[1]) if valid else np.nan,
            "baseline_hat": float(fit[2] * event_count / window) if valid else np.nan,
            "ll_gain_per_event": float(fit[3]) if valid else np.nan,
        })
    return records


def summarize_accuracy(raw: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (n, rate, window), frame in raw.groupby(GROUP_COLUMNS, sort=True):
        valid = frame.loc[frame.fit_valid]
        error = valid.n_hat.to_numpy() - n
        count, valid_count = len(frame), len(valid)
        mean_error = float(error.mean()) if valid_count else np.nan
        sd_error = float(error.std(ddof=1)) if valid_count >= 2 else np.nan
        mc_se = sd_error / np.sqrt(valid_count) if valid_count >= 2 else np.nan
        critical_value = float(t.ppf(0.975, valid_count - 1)) if valid_count >= 2 else np.nan
        lower, upper = wilson_interval(valid_count, count)
        record = {
            "true_n": n, "target_rate": rate, "window_days": int(window),
            "replicates": count, "valid_estimates": valid_count,
            "valid_fraction": valid_count / count, "valid_ci95_lower": lower, "valid_ci95_upper": upper,
            "insufficient_events_fraction": float((frame.event_count < 3).mean()),
            "optimizer_failure_fraction": float((frame.failure_reason == "optimization_failed").mean()),
            "mean_n_hat_valid": float(valid.n_hat.mean()) if valid_count else np.nan,
            "median_n_hat_valid": float(valid.n_hat.median()) if valid_count else np.nan,
            "n_hat_q05_valid": float(valid.n_hat.quantile(0.05)) if valid_count else np.nan,
            "n_hat_q95_valid": float(valid.n_hat.quantile(0.95)) if valid_count else np.nan,
            "bias_valid": mean_error,
            "mae_valid": float(np.abs(error).mean()) if valid_count else np.nan,
            "rmse_valid": float(np.sqrt(np.mean(error * error))) if valid_count else np.nan,
            "sd_n_hat_valid": sd_error, "bias_monte_carlo_se": mc_se,
            "bias_mc_ci95_lower": mean_error - critical_value * mc_se,
            "bias_mc_ci95_upper": mean_error + critical_value * mc_se,
            "lower_boundary_fraction_valid": float((valid.n_hat <= 1e-6).mean()) if valid_count else np.nan,
            "upper_boundary_fraction_valid": float((valid.n_hat >= 0.995 - 1e-6).mean()) if valid_count else np.nan,
            "mean_observed_rate": float(frame.observed_rate.mean()),
            "observed_rate_mc_se": float(frame.observed_rate.std(ddof=1) / np.sqrt(count)),
        }
        for tolerance, label in ((0.05, "0p05"), (0.10, "0p10")):
            success = int(np.sum(np.abs(error) <= tolerance))
            success_lower, success_upper = wilson_interval(success, count)
            record[f"accuracy_within_{label}_all_trials"] = success / count
            record[f"accuracy_within_{label}_ci95_lower"] = success_lower
            record[f"accuracy_within_{label}_ci95_upper"] = success_upper
        records.append(record)
    return pd.DataFrame(records)


def empirical_auc(negative: np.ndarray, positive: np.ndarray) -> float:
    negative, positive = np.asarray(negative, dtype=float), np.asarray(positive, dtype=float)
    if len(negative) == 0 or len(positive) == 0:
        return np.nan
    ranks = rankdata(np.concatenate([negative, positive]), method="average")
    statistic = ranks[len(negative):].sum() - len(positive) * (len(positive) + 1) / 2
    return float(statistic / (len(negative) * len(positive)))


def summarize_distinguishability(raw: pd.DataFrame, ratios: list[float], seed: int, bootstrap_samples: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 999]))
    records = []
    for (rate, window), cell in raw.groupby(["target_rate", "window_days"], sort=True):
        for lower_n, upper_n in zip(ratios[:-1], ratios[1:]):
            negative = cell[cell.true_n == lower_n]
            positive = cell[cell.true_n == upper_n]
            a = negative.loc[negative.fit_valid, "n_hat"].to_numpy()
            b = positive.loc[positive.fit_valid, "n_hat"].to_numpy()
            auc = empirical_auc(a, b)
            if len(a) and len(b):
                draws = [empirical_auc(rng.choice(a, len(a), replace=True), rng.choice(b, len(b), replace=True)) for _ in range(bootstrap_samples)]
                auc_lower, auc_upper = np.quantile(draws, [0.025, 0.975])
            else:
                auc_lower = auc_upper = np.nan
            threshold = (lower_n + upper_n) / 2
            specificity = float(np.sum(a < threshold) / len(negative))
            sensitivity = float(np.sum(b >= threshold) / len(positive))
            records.append({
                "target_rate": rate, "window_days": int(window), "lower_true_n": lower_n,
                "upper_true_n": upper_n, "negative_trials": len(negative), "positive_trials": len(positive),
                "valid_negative": len(a), "valid_positive": len(b),
                "valid_pair_probability": len(a) * len(b) / (len(negative) * len(positive)),
                "auc_valid_only": auc, "auc_bootstrap_ci95_lower": auc_lower, "auc_bootstrap_ci95_upper": auc_upper,
                "predefined_midpoint_threshold": threshold,
                "sensitivity_all_trials": sensitivity, "specificity_all_trials": specificity,
                "balanced_accuracy_all_trials_abstention_as_error": (sensitivity + specificity) / 2,
            })
    return pd.DataFrame(records, columns=PAIR_COLUMNS)


def run_n_accuracy(
    out: str | Path = "results/n_estimation_accuracy", ratios=None, rates=None, windows=None,
    replicates: int = 500, seed: int = 20261114, decay: float = 1.0,
    history_days: int | None = None, burn_in_times: float = 8.0,
    workers: int = 2, auc_bootstrap_samples: int = 500,
) -> pd.DataFrame:
    ratios = [0.0, 0.5, 0.8, 0.9, 0.95] if ratios is None else ratios
    rates = [0.2, 2.0, 20.0, 50.0] if rates is None else rates
    windows = [30, 90, 180, 360] if windows is None else windows
    ratios, rates, windows = validate_accuracy_grid(ratios, rates, windows, replicates, decay, history_days, burn_in_times)
    if seed < 0 or workers < 1 or auc_bootstrap_samples < 1:
        raise ValueError("need nonnegative seed, positive workers and bootstrap samples")
    history_days = int(np.ceil(30 / decay)) if history_days is None else int(history_days)
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    tasks = accuracy_tasks(ratios, rates, windows, replicates, seed, decay, history_days, burn_in_times)
    records = []
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        results = pool.map(simulate_accuracy_trial, tasks, chunksize=8) if pool else map(simulate_accuracy_trial, tasks)
        for index, group in enumerate(results, 1):
            records.extend(group)
            if index % replicates == 0:
                print(f"n accuracy: {index}/{len(tasks)} independent trajectories completed", flush=True)
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
    raw = pd.DataFrame(records)
    summary = summarize_accuracy(raw)
    pairs = summarize_distinguishability(raw, ratios, seed, auc_bootstrap_samples)
    raw_path = output / "replicates.csv"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(output / "summary.csv", index=False)
    pairs.to_csv(output / "distinguishability.csv", index=False)
    config = {
        "stage": "1_current_n_accuracy", "ratios": ratios, "target_rates": rates,
        "windows_days": windows, "replicates_per_parameter_pair": replicates,
        "independent_trajectories": len(tasks), "window_fits": len(raw),
        "seed": seed, "decay_per_day": decay, "decay_known_to_estimator": True,
        "baseline_known_to_estimator": False,
        "baseline_design": "mu=target_rate*(1-true_n), so stationary expected rate is target_rate",
        "history_days": history_days, "burn_in_relaxation_times": burn_in_times,
        "burn_in_formula": "burn_in_times/(decay*(1-n))",
        "nested_windows_share_trajectory": True, "unique_uint32_simulator_seeds": True,
        "failed_fits_are_not_zero_estimates": True,
        "auc_bootstrap_samples": auc_bootstrap_samples, "workers": workers,
        "estimator": "existing fixed-decay conditional window MLE; mu and n estimated jointly",
        "n_parameter_bounds": [0.0, 0.995], "curvature_se_is_not_a_calibrated_confidence_interval": True,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "replicates_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    }
    (output / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    write_n_accuracy_report(output)
    print(f"Saved n-estimation accuracy experiment to {output.resolve()}", flush=True)
    return summary


def write_n_accuracy_report(output: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(output / "summary.csv")
    pairs = pd.read_csv(output / "distinguishability.csv")
    rates, ratios, windows = config["target_rates"], config["ratios"], config["windows_days"]

    def draw_maps(data, metric, filename, title, row_key, row_values, row_labels, cmap, vmin, vmax, format_value):
        figure, axes = plt.subplots(1, len(rates), figsize=(3.8 * len(rates), 4.7), layout="constrained", squeeze=False)
        for axis, rate in zip(axes.flat, rates):
            matrix = data[data.target_rate == rate].pivot(index=row_key, columns="window_days", values=metric).reindex(index=row_values, columns=windows).to_numpy()
            picture = axis.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            for i in range(len(row_values)):
                for j in range(len(windows)):
                    value = matrix[i, j]
                    rgba = plt.get_cmap(cmap)((value - vmin) / (vmax - vmin)) if np.isfinite(value) else (1, 1, 1, 1)
                    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                    axis.text(j, i, format_value(value) if np.isfinite(value) else "нет", ha="center", va="center", fontsize=9, color="white" if luminance < 0.45 else "black")
            axis.set_xticks(np.arange(len(windows)), windows)
            axis.set_yticks(np.arange(len(row_values)), row_labels)
            axis.set(title=f"Средний поток {rate:g}/сутки", xlabel="Окно, суток", ylabel="Истинный n" if row_key == "true_n" else "Пара истинных n")
        figure.colorbar(picture, ax=list(axes.flat), shrink=0.8)
        figure.suptitle(title)
        figure.savefig(output / filename, dpi=170)
        plt.close(figure)

    maximum_mae = float(summary.mae_valid.max()) if summary.mae_valid.notna().any() else 1.0
    maximum_bias = max(0.05, float(summary.bias_valid.abs().max())) if summary.bias_valid.notna().any() else 1.0
    draw_maps(summary, "mae_valid", "mae_map.png", "Средняя абсолютная ошибка n: только валидные оценки", "true_n", ratios, [f"{n:g}" for n in ratios], "YlOrRd", 0, max(0.1, maximum_mae), lambda value: f"{value:.3f}")
    draw_maps(summary, "bias_valid", "bias_map.png", "Смещение оценки n: только валидные оценки", "true_n", ratios, [f"{n:g}" for n in ratios], "RdBu_r", -maximum_bias, maximum_bias, lambda value: f"{value:+.3f}")
    draw_maps(summary, "accuracy_within_0p10_all_trials", "accuracy_map.png", "Доля всех реализаций с ошибкой n не более 0.1; неудача = ошибка", "true_n", ratios, [f"{n:g}" for n in ratios], "YlGn", 0, 1, lambda value: f"{100*value:.1f}%")
    if len(pairs):
        pairs["pair"] = pairs.apply(lambda row: f"{row.lower_true_n:g}–{row.upper_true_n:g}", axis=1)
        pair_labels = [f"{a:g}–{b:g}" for a, b in zip(ratios[:-1], ratios[1:])]
        draw_maps(pairs, "auc_valid_only", "auc_map.png", "Различимость соседних уровней n: AUC среди валидных пар", "pair", pair_labels, pair_labels, "YlGnBu", 0, 1, lambda value: f"{value:.3f}")
    report = [
        "# Этап 1. Точность оценки текущего коэффициента ветвления", "",
        f"Проверены n={ratios}, средние интенсивности {rates} событий/сутки и окна {windows} суток. Для каждой пары n/интенсивность сгенерированы {config['replicates_per_parameter_pair']} независимых потоков: всего {config['independent_trajectories']} потоков и {config['window_fits']} оконных подгонок. Окна разных длин вложены в один поток; их нельзя считать независимыми повторениями.", "",
        "## Протокол", "",
        "Для каждого n задано mu=rate*(1-n). Таким образом, стационарная ожидаемая интенсивность rate=mu/(1-n) одинакова для всех n при фиксированном rate. Реализованные количества событий случайны и не нормируются к заранее заданному числу. n=0 соответствует однородному Пуассону.", "",
        f"Симулятор: прореживание Огаты, экспоненциальное ядро, beta={config['decay_per_day']:g}/сутки. Прогрев: {config['burn_in_relaxation_times']:g} времен релаксации 1/(beta*(1-n)). До левой границы всех окон наблюдаются дополнительные {config['history_days']} суток истории. Оцениватель использует только события до правой границы окна, включая доступную историю перед левой границей.", "",
        "Оцениваются mu и n совместно методом условного максимального правдоподобия; beta задано верно и известно. Это диагностика в благоприятной правильно специфицированной стационарной модели, а не проверка сезонного фона, неизвестного ядра или раннего прогноза. Истинный n и истинное mu не передаются оценивателю. Граница оптимизации для n — [0; 0.995].",
        "",
        "## Как читать ошибки", "",
        "Bias, MAE, RMSE и квантили n_hat рассчитаны только среди валидных оценок. Если событий меньше трех или оптимизация не удалась, n_hat отсутствует, а не заменяется нулем. При редких событиях условные ошибки могут выглядеть благоприятнее из-за отбора удачных подгонок, поэтому рядом обязательно учитывается valid_fraction.", "",
        "Дополнительно показана доля всех реализаций, для которых получена оценка с ошибкой <=0.05 и <=0.10: неудачная подгонка считается неуспехом. 95%-интервалы Wilson относятся к этим долям. bias_mc_ci95_* — интервал Монте-Карло для среднего смещения, а не доверительный интервал неизвестного n отдельного потока. Кривизна likelihood не используется как гарантированный интервал покрытия.", "",
        "![Карта MAE](mae_map.png)", "", "![Карта смещения](bias_map.png)", "", "![Доля точных оценок](accuracy_map.png)", "",
        "## Окно максимальной длины", "",
        "| Средний поток | Истинный n | Валидных оценок | Среднее n_hat | Bias | MAE | Ошибка <=0.1 среди всех |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary[summary.window_days == max(windows)].to_dict(orient="records"):
        report.append(f"| {row['target_rate']:g} | {row['true_n']:g} | {100*row['valid_fraction']:.1f}% | {row['mean_n_hat_valid']:.3f} | {row['bias_valid']:+.3f} | {row['mae_valid']:.3f} | {100*row['accuracy_within_0p10_all_trials']:.1f}% |")
    report.extend([
        "", "## Различимость уровней", "",
        "AUC — вероятность, что валидная оценка из потока с большим истинным n выше оценки из потока с меньшим n, с половинным весом совпадений. AUC=0.5 соответствует случайному порядку. Это диагностика различимости распределений оценок, не вероятность успешного раннего предупреждения и не доказательство оптимальной классифицируемости по всем доступным данным.", "",
        "AUC и bootstrap-интервалы условны на валидности обеих оценок. valid_pair_probability в CSV показывает вероятность получить обе оценки. Дополнительно сравнивается заранее заданный порог посередине двух истинных уровней: неудачная оценка считается ошибкой/воздержанием, а не удаляется из знаменателя. Ни один порог не выбирается по максимуму тестового качества.", "",
    ])
    if len(pairs):
        report.extend(["![Карта AUC](auc_map.png)", ""])
    near_critical = pairs[(pairs.lower_true_n == 0.9) & (pairs.upper_true_n == 0.95)]
    if len(near_critical):
        report.extend([
            "### Уровни 0.9 и 0.95", "",
            "| Средний поток | Окно | Вероятность двух валидных оценок | AUC | 95% bootstrap AUC |",
            "| --- | ---: | ---: | ---: | --- |",
        ])
        for row in near_critical[near_critical.window_days.isin([window for window in (90, max(windows)) if window in windows])].to_dict(orient="records"):
            report.append(f"| {row['target_rate']:g} | {row['window_days']} | {100*row['valid_pair_probability']:.1f}% | {row['auc_valid_only']:.3f} | [{row['auc_bootstrap_ci95_lower']:.3f}; {row['auc_bootstrap_ci95_upper']:.3f}] |")
        report.append("")
    selected = summary[(summary.true_n == 0.95) & (summary.target_rate == 20) & (summary.window_days.isin([90, 360]))]
    if len(selected) == 2 and len(near_critical):
        short = selected[selected.window_days == 90].iloc[0]
        long = selected[selected.window_days == 360].iloc[0]
        auc_example = near_critical[near_critical.target_rate == 20].set_index("window_days")
        report.extend([
            "## Основные наблюдения", "",
            f"При среднем потоке 20/сутки и истинном n=0.95 MAE среди валидных оценок уменьшается с {short.mae_valid:.3f} на 90 сутках до {long.mae_valid:.3f} на 360. AUC оценок для уровней 0.9 и 0.95 возрастает с {auc_example.loc[90, 'auc_valid_only']:.3f} до {auc_example.loc[360, 'auc_valid_only']:.3f}. Это результат стационарной правильно заданной модели с известным затуханием.", "",
        ])
    report.extend([
        "Условия редких событий нужно читать по доле точных оценок среди всех потоков, а не только по MAE удачных подгонок. Изменение условной MAE при увеличении окна может отражать изменение состава валидных потоков. AUC около 0.5 означает слабую различимость именно по исследуемой оценке n, а не доказанную невозможность различить процессы любым методом.", "",
        "При выборе окна для меняющегося процесса дополнительно потребуется измерить задержку: длинное окно может смешивать несколько режимов. Улучшение стационарной точности само по себе не подтверждает качество прогноза будущего пересечения.", "",
    ])
    report.extend([
        "## Воспроизводимость и следующий этап", "",
        "replicates.csv содержит каждую оценку и уникальный seed симулятора; summary.csv — ошибки и неудачи; distinguishability.csv — AUC и качество по заранее выбранным серединам уровней; run_config.json — полный протокол и SHA-256 исходной таблицы. Отдельные потоки имеют разные uint32 seeds. Неопределенность относится к повторениям внутри одной ячейки; интервалы не одновременные по всей сетке.", "",
        "Этот этап отвечает только на вопрос о точности оценки постоянного текущего n. Он не обучает нейросеть, не прогнозирует будущее пересечение и не использует городские данные. Следующий этап — отдельно сравнить известный и оцененный фон, включая сезонность и изменение интенсивности. Выбор окон для следующих моделей следует делать по этой диагностике и валидации, а итоговый тест генерировать заново.",
    ])
    (output / "analysis.md").write_text("\n".join(report) + "\n", encoding="utf-8")
