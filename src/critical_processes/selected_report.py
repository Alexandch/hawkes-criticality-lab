from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .selected_training import METHOD_NAMES


def write_selected_report(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    frozen = json.loads((output / "frozen_plan.json").read_text(encoding="utf-8"))
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    table = pd.read_csv(output / "test_metrics.csv")
    family = pd.read_csv(output / "family_test_metrics.csv")
    trajectories = pd.read_csv(output / "trajectory_test_metrics.csv", dtype={"clean_detection": "boolean", "has_positive": "boolean"})
    selected_table = table[table.member.isin(["ensemble", "statistic"])]
    rng = np.random.default_rng(protocol["data_seed"] + 1234)
    paired_rows = []
    for task in ("state", "warning"):
        frames = {method: trajectories[(trajectories.task == task) & (trajectories.method == method) &
                    trajectories.member.isin(["ensemble", "statistic"])].set_index("trajectory_id") for method in METHOD_NAMES}
        reference = frames["counts"]
        for method in ("constant_mle", "calendar_mle"):
            other = frames[method].loc[reference.index]
            if task == "state":
                positive = reference.positive_windows.to_numpy()
                difference = other.true_positive.to_numpy() - reference.true_positive.to_numpy()
                denominator = positive.sum()
            else:
                keep = reference.has_positive.to_numpy(dtype=bool)
                reference_clean = reference.loc[keep, "clean_detection"].astype(bool).to_numpy()
                other_clean = other.loc[keep, "clean_detection"].astype(bool).to_numpy()
                difference = other_clean.astype(int) - reference_clean.astype(int)
                positive = np.ones(len(difference), dtype=int)
                denominator = len(difference)
            draws = []
            for _ in range(protocol["bootstrap_samples"]):
                indices = rng.integers(0, len(difference), len(difference))
                if positive[indices].sum():
                    draws.append(difference[indices].sum() / positive[indices].sum())
            lower, upper = np.quantile(draws, [.025, .975]) if draws else (np.nan, np.nan)
            paired_rows.append({"task": task, "method": method, "reference": "counts",
                "utility_difference": difference.sum() / denominator if denominator else np.nan,
                "paired_trajectory_bootstrap_ci95_lower": lower, "paired_trajectory_bootstrap_ci95_upper": upper})
    pd.DataFrame(paired_rows).to_csv(output / "paired_comparison.csv", index=False)
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.8), layout="constrained")
    names = list(METHOD_NAMES)
    titles = ["MLP\nсчётчики", "MLP\nобычный\nMLE", "MLP\nкалендарный\nMLE", "Обычный\nMLE", "Календарный\nMLE"]
    colors = ["#477d94", "#16876d", "#7664a0", "#a97b36", "#75816a"]
    for index, method in enumerate(names):
        for axis_index, task in ((0, "state"), (1, "warning")):
            row = selected_table[(selected_table.task == task) & (selected_table.method == method)].iloc[0]
            metric = metrics[row.stream]
            value = metric["window_sensitivity"] if task == "state" else metric["clean_event_detection_probability"]
            interval = metric["window_sensitivity_trajectory_bootstrap_ci95"] if task == "state" else metric["clean_detection_ci95"]
            axes[axis_index].bar(index, 100 * value, color=colors[index], width=.65)
            axes[axis_index].errorbar(index, value * 100, yerr=[[100 * max(0, value - interval[0])], [100 * max(0, interval[1] - value)]], color="black", fmt="none", capsize=3)
            axes[axis_index].text(index, max(value, interval[1]) * 100 + .3, f"{100*value:.1f}%", ha="center", fontsize=9)
            axes[2].bar(index + (-.18 if task == "state" else .18), 100 * metric["maximum_family_false_alarm_probability"],
                        width=.34, color=colors[index], hatch=None if task == "state" else "//",
                        label=("Состояние" if task == "state" else "Предупреждение") if index == 0 else None)
    for axis in axes:
        axis.set_xticks(range(len(names)), titles, fontsize=9)
        axis.grid(axis="y", alpha=.2)
        axis.set_ylabel("%")
    state_limit = max(10, 5 * np.ceil(max(metrics[row.stream]["window_sensitivity_trajectory_bootstrap_ci95"][1] for row in selected_table[selected_table.task == "state"].itertuples()) * 100 / 5) + 3)
    axes[0].set(title="Текущее высокое n\nЧувствительность по положительным окнам", ylim=(0, state_limit))
    warning_limit = max(3, np.ceil(max(metrics[row.stream]["clean_detection_ci95"][1] for row in selected_table[selected_table.task == "warning"].itertuples()) * 100) + 1)
    axes[1].set(title="Раннее предупреждение\nПереходы с правильной первой тревогой", ylim=(0, warning_limit))
    axes[2].axhline(protocol["alpha"] * 100, color="black", linestyle=":", label="Цель 5%")
    axes[2].set(title="Хотя бы одна ложная тревога\nМаксимум по семействам")
    axes[2].legend(fontsize=8)
    figure.suptitle("Окна выбраны на валидации; ансамбль всех seed; один новый финальный тест")
    figure.savefig(output / "final_comparison.png", dpi=170)
    plt.close(figure)

    def pc(value):
        return "—" if value is None or not np.isfinite(value) else f"{value:.1%}"
    primary_rows = {task: table[(table.task == task) & table.primary].iloc[0] for task in ("state", "warning")}
    primary_state = metrics[primary_rows["state"].stream]
    primary_warning = metrics[primary_rows["warning"].stream]
    state_ci = primary_state["window_sensitivity_trajectory_bootstrap_ci95"]
    warning_ci = primary_warning["event_detection_ci95"]
    report = [
        "# Этап 4. Выбор признаков и окон, несколько seed, замороженный финальный тест", "",
        f"Фиксированная сетка окон: {protocol['windows']} суток. Seed инициализации сетей: {protocol['model_seeds']}. "
        f"Эпох: {protocol['epochs']}. Размеры на каждое из {len(protocol['families'])} семейств: {protocol['sizes']}. "
        f"Seed данных {protocol['data_seed']}, отдельно заранее задан seed финального теста {protocol['final_test_seed']}.", "",
        "## Основной результат", "",
        f"Выбранный до теста метод состояния — {METHOD_NAMES[primary_rows['state'].method]}, "
        f"окно {primary_rows['state'].window_days} суток. AUC={primary_state['auc_valid_only']:.3f}, "
        f"но чувствительность по высоким окнам лишь {pc(primary_state['window_sensitivity'])} "
        f"(bootstrap 95%: {pc(state_ci[0])}–{pc(state_ci[1])}). "
        f"Хотя бы одно правильное распознавание имеется на {primary_state['detected_trajectories']} из "
        f"{primary_state['positive_trajectories']} положительных потоков; это другой знаменатель.", "",
        f"Выбранный до теста метод предупреждения — {METHOD_NAMES[primary_rows['warning'].method]}, "
        f"окно {primary_rows['warning'].window_days} суток. Обнаружено {primary_warning['detected_trajectories']} из "
        f"{primary_warning['positive_trajectories']} будущих переходов, Wilson 95%: "
        f"[{pc(warning_ci[0])}, {pc(warning_ci[1])}]. Это не доказывает, что популяционная вероятность "
        "обнаружения равна нулю.", "",
        f"На этой финальной выборке максимальная семейная доля ложных тревог среди ансамблей и "
        f"классических ориентиров — {pc(selected_table.maximum_family_false_alarm_probability.max())}; "
        f"среди всех отдельных seed тоже учтённых в калибровке — {pc(table.maximum_family_false_alarm_probability.max())}. "
        "Чувствительность раннего предупреждения при таком контроле пока недостаточна. Порог после "
        "просмотра теста не менялся; метод с лучшим тестовым числом обнаружений не подменяет первичный.", "",
        "## Разделение данных и фиксация решений", "",
        "Используются пять непересекающихся наборов полных потоков: train, validation_calibration, validation, "
        "calibration и test. Только train используется для градиентов и нормировки. Предварительные пороги "
        "выбираются на validation_calibration, окна и первичный метод — на validation. Итоговые пороги "
        "выбираются на отдельной calibration уже после выбора окон. Тест до этого не генерируется. "
        "selection.json и frozen_plan.json сохраняют выбор, все пороги и SHA-256 весов до появления test.npz. "
        "final_test_started.json связывает тест с неизменяемым планом; final_test_complete.json фиксирует результаты. "
        "Повтор команды после завершения возвращает сохранённые метрики без нового теста.", "",
        "Для каждого набора признаков выбирается одно окно по ансамблю всех seed: сначала допустимость "
        "максимальной валидационной траекторной частоты ложных тревог ≤5%, затем качество (state TPR "
        "или warning clean recall), AUC и более короткое окно. Если допустимого варианта нет, сначала "
        "минимизируется нарушение и оно сохраняется явно. Первичный метод для каждой задачи выбирается "
        "тем же правилом на валидации. Лучший seed по тесту не выбирается. Все нейросетевые seed имеют "
        "одни и те же данные и различаются инициализацией; их разброс не является интервалом по независимым тестовым выборкам.", "",
        "## Признаки и модельные предположения", "",
        f"Перед мониторингом имеется {protocol['prefix_days']} суток наблюдаемой предыстории. По её счётчикам "
        "однократно обучается Poisson GLM (день недели, две годовые гармоники, log-тренд), без сценария, "
        "истинного n и будущих событий. Её форма заморожена на весь мониторинг; уровень оценивается "
        "вместе с n внутри текущего завершённого окна. Прогноз общей частоты не фиксируется как иммиграционная μ.", "",
        "Для имеющихся городских выгрузок за два года такая отдельная двухлетняя предыстория и последующий "
        "мониторинг пока не обеспечены. Перенос потребует более длинного ряда или отдельной проверки "
        "более короткого фонового протокола; эти городские данные здесь не оценивались.", "",
        "Наборы: 16 признаков счётчиков (три последние стадии и выбранное окно); 39 признаков с обычными "
        "оконными MLE; 64 признака с дополнительными календарными MLE и флагом прогноза. MLE имеет "
        "известное верное β=1/сутки. Быстрый Newton с аналитическим Hessian решает тот же условный likelihood, "
        "что и прежний SciPy-оцениватель, с резервным SciPy-решением при несходимости. Curvature SE — "
        "признак информации, не откалиброванный доверительный интервал. Неудачи имеют отдельные флаги; "
        "их конечные заполнители для сети не трактуются как оценённый нулевой n.", "",
        "Классический ориентир использует текущий n для состояния и проекцию n по трём последним стадиям "
        "для предупреждения; окно тоже выбирается на валидации. Сохраняется возбуждение между стадиями. "
        "μ=rate×(1−n) поддерживает локальную стационарную среднюю, но не фиксирует число событий при переходах. "
        "Пуассоновские скачки происходят после фонового обучения. Генератор с префиксом отличается от этапа 3, "
        "поэтому цифры этапов нельзя читать как чистый эффект только нового обучения.", "",
        f"Состояние — n(t−)≥{protocol['critical_level']}; предупреждение — первое пересечение строго после t "
        f"в течение {protocol['forecast_stages'] * protocol['stage_days']} суток с полностью наблюдённым горизонтом. "
        "Изначально высокий режим исключён из задачи первого будущего перехода. Время решений в таблицах "
        "отсчитывается от начала мониторинга после предыстории. Порог 0.9 ниже математической границы n=1.", "",
        "## Контроль ложных тревог", "",
        "Для каждого семейства берётся один максимум score на отрицательных eligible-решениях каждого "
        "независимого потока. Ранг k выбирается до просмотра значений: верхний квантиль "
        "Beta(m+1−k,k) должен быть не выше 0.05. Затем порог ставится строго выше k-й порядковой статистики. "
        "Берётся максимум семейных порогов; совпадающие scores обрабатываются консервативно. "
        "При нехватке калибровочных данных используется воздержание от всех тревог, а не неподтверждённый порог.", "",
        f"На финальной calibration вероятность ошибки границы распределяется между всеми {frozen['protected_score_streams']} "
        f"заранее фиксированными потоками scores и активными семействами. Совместный уровень доверия — "
        f"{pc(frozen['simultaneous_calibration_confidence'])}. Это утверждение о популяционных траекторных ложных "
        "тревогах при независимой калибровке и той же генеративной модели. Оно не требует, чтобы каждая "
        "конечная тестовая доля обязательно оказалась ≤5%, и не распространяется на неизвестные городские процессы.", "",
        "Ранговые непараметрические границы описаны в [NIST](https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/tolelimi.htm); "
        "[JMP](https://www.jmp.com/support/help/en/19.2/jmp/statistical-details-for-tolerance-intervals.shtml) "
        "приводит связь односторонней границы с биномиальным квантилем. Формула и поправка на все сохранённые "
        "scores реализованы в confident_calibration.py и проверяются тестами.", "",
        "## Выбор до финального теста", "",
        "| Задача | Набор | Окно | Качество validation | Макс. ложные validation | Допустим | Первичный |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in selection["selected"]:
        report.append(f"| {row['task']} | {METHOD_NAMES[row['method']]} | {int(row['window_days'])} | {pc(row['utility'])} | "
                      f"{pc(row['maximum_family_false_alarm_probability'])} | {row['validation_feasible']} | "
                      f"{selection['primary_method_by_task'][row['task']] == row['method']} |")
    report.extend(["", "## Финальный тест", "",
        "Состояние: чувствительность по положительным окнам. Предупреждение: обычный и clean recall "
        "по переходам. Clean recall требует, чтобы первая eligible-тревога была правильной; позднее "
        "обнаружение не отменяет прежнюю ложную тревогу. Знаменатели этих задач различаются.", "",
        "| Задача | Метод | Окно | TPR окон | Recall потоков | Clean recall | Макс. ложные |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in selected_table.itertuples():
        metric = metrics[row.stream]
        report.append(f"| {row.task} | {METHOD_NAMES[row.method]} | {row.window_days} | {pc(metric['window_sensitivity'])} | "
                      f"{pc(metric['event_detection_probability'])} | {pc(metric['clean_event_detection_probability'])} | "
                      f"{pc(metric['maximum_family_false_alarm_probability'])} |")
    report.extend(["", "![Финальное сравнение](final_comparison.png)", "",
        "## Разброс и сравнение", "",
        "Таблица test_metrics.csv содержит все отдельные seed и ансамбли. Интервалы оконных метрик "
        "получены bootstrap полных потоков; для событий и семейных траекторных долей сохранены Wilson-интервалы. "
        "Нельзя объединять один и тот же тест для трёх seed как три независимые выборки. Парные разности "
        "ниже вычислены на тех же потоках относительно ансамбля по счётчикам; интервалы описательные, "
        "без поправки на четыре сравнения.", "",
        "У нулевого числа успехов обычный bootstrap наблюдаемых окон может дать вырожденный интервал "
        "[0,0]; он не доказывает нулевую популяционную чувствительность. Для нулевых обнаружений переходов "
        "используется также невырожденная верхняя Wilson-граница по независимым потокам.", "",
        "Разброс качества между инициализациями при одних и тех же данных и выбранном окне "
        "(для состояния — TPR окон, для предупреждения — clean recall):", "",
        "| Задача | Набор | Минимум seed | Медиана seed | Максимум seed | Ансамбль |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for task in ("state", "warning"):
        for method in ("counts", "constant_mle", "calendar_mle"):
            data = table[(table.task == task) & (table.method == method) & table.member.isin([str(seed) for seed in protocol["model_seeds"]])]
            key = "window_sensitivity" if task == "state" else "clean_event_detection_probability"
            values = data[key]
            ensemble = selected_table[(selected_table.task == task) & (selected_table.method == method)].iloc[0][key]
            report.append(f"| {task} | {METHOD_NAMES[method]} | {pc(values.min())} | {pc(values.median())} | {pc(values.max())} | {pc(ensemble)} |")
    report.extend([
        "", "Парное сравнение с ансамблем по счётчикам:", "",
        "| Задача | Метод | Разность качества | Парный bootstrap 95% |",
        "|---|---|---:|---:|",
    ])
    for row in paired_rows:
        report.append(f"| {row['task']} | {METHOD_NAMES[row['method']]} | {row['utility_difference']:+.3f} | "
                      f"[{row['paired_trajectory_bootstrap_ci95_lower']:+.3f}, {row['paired_trajectory_bootstrap_ci95_upper']:+.3f}] |")
    report.extend(["", "Метрики финального теста не используются для смены окна, метода, seed или порога. "
        "Следующие изменения потребуют новых обучающих решений и нового финального теста. Реальные "
        "зарегистрированные преступления остаются эмпирической иллюстрацией и в этом эксперименте не используются.", "",
        "## Артефакты и воспроизведение", "",
        "- protocol.json — параметры и SHA-256 исходного кода до генерации данных.",
        "- validation_candidates.csv, validation_calibration.csv, selection.json — все варианты и правило выбора.",
        "- frozen_plan.json, calibration.csv — выбранные веса, окна и защищённые пороги до теста.",
        "- data/*.npz — полные непересекающиеся потоки, признаки и scores с seed.",
        "- models/*.pt, training/*.csv — все инициализации и кривые обучения.",
        "- metrics.json, test_metrics.csv, family_test_metrics.csv, trajectory_test_metrics.csv — финальные результаты.",
        "- primary_*_decisions.csv — решения заранее выбранных первичных моделей.",
        "- paired_comparison.csv — сравнение ансамблей на одних и тех же тестовых потоках.", "",
        "Проверка: `.\\.venv\\Scripts\\python.exe scripts/verify_selected_training.py`. "
        "Перестроение отчёта: selected_report.write_selected_report, без нового выбора и теста.", "",
    ])
    (output / "analysis.md").write_text("\n".join(report), encoding="utf-8")
