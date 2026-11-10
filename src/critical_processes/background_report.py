from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .background_comparison import METHODS, SCENARIOS


def write_background_report(output: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(output / "summary.csv")
    pairs = pd.read_csv(output / "paired_comparison.csv")
    examples = pd.read_csv(output / "baseline_examples.csv")
    scenarios, windows = config["scenarios"], config["windows_days"]
    rate = min(config["target_rates"], key=lambda r: abs(r - 20))
    selected = summary[summary.target_rate == rate]
    columns = min(4, len(scenarios))
    rows = int(np.ceil(len(scenarios) / columns))
    colors = dict(zip(METHODS, plt.get_cmap("tab10").colors[:len(METHODS)]))
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.1 * rows + 1.0), squeeze=False)
    for axis, scenario in zip(axes.flat, scenarios):
        cell = selected[selected.scenario == scenario]
        for method, label in METHODS.items():
            data = cell[cell.method == method].sort_values("window_days")
            axis.plot(data.window_days, data.mean_n_hat_valid, "o-", color=colors[method], label=label, markersize=4)
        axis.axhline(cell.true_n.iloc[0], color="black", linestyle=":", label="Истинный n")
        axis.set(title=SCENARIOS[scenario], xlabel="Окно, суток", ylabel="Средняя оценка n", ylim=(-0.03, 1.02))
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(scenarios):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    figure.suptitle(f"Влияние фона на оценку n; целевой средний поток {rate:g}/сутки\nСредние среди валидных подгонок; доли валидности приведены в summary.csv")
    figure.tight_layout(rect=(0, 0.12 / rows, 1, 0.92))
    figure.savefig(output / "n_background_comparison.png", dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.1 * rows + 0.8), squeeze=False)
    for axis, scenario in zip(axes.flat, scenarios):
        data = examples[(examples.target_rate == rate) & (examples.scenario == scenario)]
        axis.plot(data.day, data.true_baseline, color="black", label="Истинный фон")
        axis.plot(data.day, data.forecast_baseline, color="tab:orange", label="Фон по прошлым счётчикам")
        axis.axvspan(0, config["prefix_days"], color="tab:blue", alpha=0.07, label="Обучающий участок")
        axis.axvline(config["prefix_days"], color="tab:blue", linestyle=":")
        axis.set(title=SCENARIOS[scenario], xlabel="День", ylabel="Событий/сутки")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(scenarios):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    figure.suptitle(f"Пример №0 каждого сценария: прогноз зафиксирован в день {config['prefix_days']}")
    figure.tight_layout(rect=(0, 0.07 / rows, 1, 0.94))
    figure.savefig(output / "baseline_forecasts.png", dpi=170)
    plt.close(figure)

    report = [
        "# Этап 2. Фоновая нестационарность и оценка коэффициента ветвления", "",
        f"Эксперимент: {config['independent_trajectories']} независимых потоков, {config['window_fits']} подгонок, "
        f"по {config['replicates_per_scenario_rate']} повторов на сценарий и интенсивность. Seed={config['seed']}. "
        f"Интенсивности {config['target_rates']} событий/сутки; окна {windows} суток. "
        f"Обучающий префикс фона — {config['prefix_days']} суток, затем все окна начинаются на одной границе. "
        "Окна вложены, а пять способов используют одни и те же события; независимость — между потоками.", "",
        "## Протокол", "",
        "Интенсивность Хоукса: λ(t)=μ(t)+nβΣ exp(−β(t−tᵢ)), tᵢ<t. "
        f"β={config['decay_per_day']:g}/сутки известен и верен. Оценивается постоянный n∈[0,0.995] "
        "в каждом завершённом окне с учётом наблюдённой истории до левой границы. "
        "В вариантах со свободным уровнем μ(t)=a×форма(t), форма нормируется до среднего 1 внутри окна; "
        "a и n оцениваются совместно. Likelihood делится на число событий для численной устойчивости. "
        "Архивный оцениватель и результаты этапа 1 не изменялись.", "",
        "Пуассоновские сценарии имеют истинный n=0, кусочно постоянный суточный фон и точные условно "
        "равномерные внутрисуточные моменты событий. Профиль нормирован на заданную среднюю интенсивность "
        "за весь горизонт. Это не принудительная фиксация реализованного числа событий. "
        "Сезонность включает неделю и год, плавный тренд — экспоненциальный рост. "
        "Скачок в четыре раза размещён строго после обучения, в первые 1–120 суток проверяемого участка "
        "(с ограничением по длине максимального окна). "
        "У стационарных контролей Хоукса n=0.5 и 0.9, μ=rate×(1−n), прогрев 8/[β(1−n)].", "",
        "| Метод | Что известно и что подгоняется |", "|---|---|",
        "| constant_joint | Постоянный неизвестный фон и n — по завершённому окну |",
        "| oracle_fixed | Весь истинный μ(t) известен, подгоняется только n |",
        "| oracle_shape_joint | Истинная форма μ(t) известна, общий уровень и n подгоняются по окну |",
        "| past_fixed | Весь прогноз μ̂(t) зафиксирован по префиксу, подгоняется только n |",
        "| past_shape_joint | Форма прогноза зафиксирована по префиксу, уровень и n подгоняются по окну |", "",
        "Два oracle — привилегированные диагностические ориентиры, включая знание будущего скачка. "
        "Разница constant_joint/oracle_fixed объединяет эффект правильной формы и известного уровня. "
        "oracle_shape_joint отделяет эффект формы при одинаковом числе оцениваемых параметров.", "",
        "Прогноз фона — пуассоновская GLM: день недели, две годовые гармоники, линейный тренд в log μ "
        "и свободная константа. На вход поступают исключительно суточные счётчики префикса; сценарий, "
        "истинные n и μ, будущие события и будущие счётчики недоступны. Формула дизайна задана до эксперимента. "
        "Слабый L2-штраф 1e−5, кроме константы, и границы коэффициентов описаны в run_config.json. "
        "Фон не переобучается между окнами. В past_shape_joint уровень всё же оценивается на текущем "
        "завершённом окне: это причинная оценка на его правой границе, но не полностью замороженный фон.", "",
        "Для Хоукса пуассоновский прогноз счётчиков оценивает общую частоту, а не интенсивность иммиграции μ. "
        "Если зафиксировать этот уровень как μ, самовозбуждение может подавляться. Поэтому малые оценки "
        "на n=0 проверяются вместе с сохранением ненулевого n на положительных контролях.", "",
        "## Результаты", "",
        f"Ниже средняя оценка n среди валидных подгонок при целевой интенсивности {rate:g}/сутки. "
        "Полная сетка и доли валидности — в summary.csv.", "",
        "| Сценарий | Окно | Постоянный | Истинный целиком | Истинная форма | Прошлый целиком | Прошлая форма |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    shown_windows = sorted(set([min(windows, key=lambda w: abs(w - 90)), max(windows)]))
    for scenario in scenarios:
        for window in shown_windows:
            cell = selected[(selected.scenario == scenario) & (selected.window_days == window)].set_index("method")
            values = " | ".join(f"{cell.loc[m, 'mean_n_hat_valid']:.3f}" for m in METHODS)
            report.append(f"| {SCENARIOS[scenario]} | {window} | {values} |")
    longest = selected[selected.window_days == max(windows)]

    def mean_n(scenario, method):
        return float(longest[(longest.scenario == scenario) & (longest.method == method)].mean_n_hat_valid.iloc[0])

    report.extend(["", f"Наблюдения при {rate:g}/сутки и окне {max(windows)} суток:", ""])
    if "seasonal" in scenarios:
        report.append(f"- На сезонном Пуассоне среднее n̂ снижается с {mean_n('seasonal', 'constant_joint'):.3f} "
                      f"до {mean_n('seasonal', 'past_shape_joint'):.3f}, если использовать форму по прошлому "
                      f"и подгонять уровень. При известной истинной форме оно равно {mean_n('seasonal', 'oracle_shape_joint'):.3f}.")
    if "hawkes_n0p9" in scenarios:
        report.append(f"- На настоящем Хоуксе n=0.9 форма по прошлому сохраняет высокий коэффициент: "
                      f"среднее n̂={mean_n('hawkes_n0p9', 'past_shape_joint'):.3f}. Если зафиксировать весь "
                      f"пуассоновский прогноз как фон, оно падает до {mean_n('hawkes_n0p9', 'past_fixed'):.3f}. "
                      "Малая оценка при фиксированном прогнозе сама по себе не означает хорошую коррекцию фона.")
    if "step_up" in scenarios and "step_down" in scenarios:
        report.append(f"- Новый скачок не устранён: при форме по прошлому n̂ в среднем "
                      f"{mean_n('step_up', 'past_shape_joint'):.3f} для роста и {mean_n('step_down', 'past_shape_joint'):.3f} "
                      "для падения при истинном n=0. Предыдущий календарный профиль не описывает новый разрыв.")
    if "homogeneous" in scenarios:
        report.append(f"- Лишняя гибкость фона имеет цену: на однородном Пуассоне среднее n̂ меняется "
                      f"с {mean_n('homogeneous', 'constant_joint'):.3f} до {mean_n('homogeneous', 'past_shape_joint'):.3f}. "
                      "Поэтому сезонную модель нельзя считать безусловным улучшением для всех потоков.")
    report.extend([
        "", "Чувствительность к плотности потока: форма фона по прошлому со свободным уровнем, "
        f"окно {max(windows)} суток. Вероятность ошибки ≤0.1 включает неудачные подгонки как ошибки.", "",
        "| Сценарий | Поток/сутки | Среднее n̂ | MAE | Валидно | Ошибка ≤0.1 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for scenario in ("seasonal", "hawkes_n0p9"):
        data = summary[(summary.scenario == scenario) & (summary.window_days == max(windows)) &
                       (summary.method == "past_shape_joint")].sort_values("target_rate")
        for row in data.itertuples():
            report.append(f"| {SCENARIOS[scenario]} | {row.target_rate:g} | {row.mean_n_hat_valid:.3f} | "
                          f"{row.mae_valid:.3f} | {row.valid_fraction:.1%} | {row.accuracy_within_0p10_all_trials:.1%} |")
    report.extend(["", "Редкие потоки остаются трудными даже после коррекции формы. "
                   "Длинное окно снижает ошибку стационарной оценки, но не проверяет задержку при меняющемся n."])
    report.extend([
        "", "![Оценки n](n_background_comparison.png)", "",
        "![Прогнозы фоновой интенсивности](baseline_forecasts.png)", "",
        f"Парное уменьшение MAE при переходе от постоянного фона к форме по прошлому, "
        f"окно {max(windows)} суток, поток {rate:g}/сутки. Положительное значение означает улучшение. "
        "Интервалы Монте-Карло 95% рассчитаны по независимым разностям на потоках, где обе подгонки валидны; "
        "это описательные интервалы без поправки на множество сравнений.", "",
        "| Сценарий | Пар валидно | Уменьшение MAE | Интервал Монте-Карло |",
        "|---|---:|---:|---:|",
    ])
    for scenario in scenarios:
        row = pairs[(pairs.scenario == scenario) & (pairs.target_rate == rate) &
                    (pairs.window_days == max(windows)) & (pairs.method == "past_shape_joint")].iloc[0]
        report.append(f"| {SCENARIOS[scenario]} | {int(row.paired_valid_estimates)} | {row.mae_reduction_common_valid:+.4f} | "
                      f"[{row.mae_reduction_mc_ci95_lower:+.4f}, {row.mae_reduction_mc_ci95_upper:+.4f}] |")
    report.extend([
        "", "## Как читать метрики и ограничения", "",
        "MAE, RMSE, bias и среднее n рассчитаны только среди валидных оценок; неудачи сохранены как NaN, "
        "а не как нулевое n. Доля |n̂−n|≤0.1 рассчитана по всем реализациям, включая неудачи как ошибки. "
        "Таблица содержит долю валидности и отдельные причины отказа, квантили n̂, интервалы Монте-Карло "
        "для bias и Wilson-интервалы для долей. Относительная ошибка прогноза фона сравнивается с истинной μ; "
        "на Хоуксе она включает отличие общей интенсивности от иммиграционной.", "",
        "Доли n̂≥0.5 и n̂≥0.9 — описательные частоты высоких оценок при фиксированных порогах. "
        "Они не являются результатом калибровки детектора на 5% ложных тревог и не измеряют предупреждение "
        "о будущем переходе. Неудачная подгонка не создаёт высокий сигнал, поэтому одновременно нужно "
        "смотреть долю валидности и частоту высоких оценок среди валидных. Вложенные окна нельзя складывать "
        "как независимые испытания. Доверительные интервалы отдельных n здесь не строятся.", "",
        "Новый скачок после обучающего участка нельзя предсказать только по прежнему фону. "
        "Oracle показывает достижимое при знании фона, а прогноз по прошлому — ограничение выбранной GLM. "
        "Результат не доказывает, что любой сезонный фон восстановлен верно или что агрегация городских "
        "событий имеет такое ядро. Все проверки здесь синтетические; CODE и новые городские выгрузки "
        "сохраняют статус эмпирической иллюстрации зарегистрированных событий.", "",
        "Следующий этап — отдельно оценивать текущее высокое n и предупреждение о будущем пересечении. "
        "Выбор окон и признаков, калибровка 5% ложных тревог и повторное нейросетевое обучение должны "
        "использовать выделенные обучение/валидацию, затем новый финальный тест. Эта диагностика не "
        "заменяет независимую проверку детектора.", "",
        "Смещение branching ratio при неверном задании иммиграционной интенсивности обсуждается в "
        "[Wheatley, Filimonov, Sornette (2014)](https://arxiv.org/abs/1407.7118). "
        "Численные результаты в этом отчёте получены локальным экспериментом.", "",
        "## Артефакты и воспроизведение", "",
        "- replicates.csv — все оценки, причины отказа, seed и ошибки прогноза.",
        "- summary.csv — показатели по сценарию, интенсивности, окну и способу задания фона.",
        "- paired_comparison.csv — парные разности MAE относительно постоянного фона.",
        "- baseline_examples.csv — префикс и прогноз реализации №0 для каждого сценария и интенсивности.",
        "- run_config.json — фиксированный протокол, версии и SHA-256 сырых результатов и примеров.", "",
        "Проверка: `.\\.venv\\Scripts\\python.exe scripts/verify_background_comparison.py`. "
        "Отчёт можно пересоздать функцией background_report.write_background_report без новых симуляций.", "",
    ])
    (output / "analysis.md").write_text("\n".join(report), encoding="utf-8")
