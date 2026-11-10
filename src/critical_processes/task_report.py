from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .task_audit import AUDIT_FAMILY_NAMES, SCORE_NAMES, load_audit_batch
from .task_protocol import state_warning_targets


def write_task_report(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    family = pd.read_csv(output / "family_test_metrics.csv")
    shared = pd.read_csv(output / "shared_prefix_summary.csv")
    names = list(SCORE_NAMES)
    titles = ["MLP\nсчётчики", "MLP\nс MLE", "Тренд n", "Текущий\nMLE n", "Истинный\nтекущий n"]
    colors = ["#477d94", "#16876d", "#bc6c3b", "#7664a0", "#888888"]
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.8), layout="constrained")
    for index, method in enumerate(names):
        state, warning = metrics[method]["state"], metrics[method]["warning"]
        for axis, value, interval in ((axes[0], state["window_sensitivity"], state["window_sensitivity_trajectory_bootstrap_ci95"]),
                                       (axes[1], warning["event_detection_probability"], warning["event_detection_ci95"])):
            low, high = interval
            axis.bar(index, 100 * value, color=colors[index], width=0.65)
            axis.errorbar(index, 100 * value, yerr=[[100 * max(0, value - low)], [100 * max(0, high - value)]], fmt="none", color="black", capsize=3)
            axis.text(index, min(103, 100 * value + 2), f"{value:.1%}", ha="center", fontsize=9)
        for offset, task, hatch in ((-0.19, "state", None), (0.19, "warning", "//")):
            value = metrics[method][task]["maximum_family_false_alarm_probability"]
            axes[2].bar(index + offset, 100 * value, width=0.36, color=colors[index], hatch=hatch,
                        label="Состояние" if index == 0 and task == "state" else ("Предупреждение" if index == 0 else None))
    for axis in axes:
        axis.set_xticks(np.arange(len(names)), titles, fontsize=9)
        axis.grid(axis="y", alpha=0.2)
        axis.set_ylabel("%")
    axes[0].set(title="Высокое n в завершённом интервале\nЧувствительность по положительным окнам", ylim=(0, 110))
    warning_upper = max(metrics[method]["warning"]["event_detection_ci95"][1] for method in names)
    axes[1].set(title="Предупреждение до первого пересечения\nДоля обнаруженных переходов", ylim=(0, max(10, 5 * np.ceil(100 * warning_upper / 5) + 5)))
    axes[2].set(title="Хотя бы одна ложная тревога\nМаксимум по семействам")
    axes[2].axhline(100 * config["target_false_alarm_probability"], color="black", linestyle=":", label="Цель калибровки")
    axes[2].legend(fontsize=8)
    figure.suptitle("Разные цели оценки; веса сетей заморожены, пороги откалиброваны отдельно")
    figure.savefig(output / "state_warning_comparison.png", dpi=170)
    plt.close(figure)

    batch = load_audit_batch(output / "data" / "test.npz")
    protocol = {key: config[key] for key in ("stage_days", "lookback", "forecast_stages", "critical_level")}
    targets = state_warning_targets(batch.base.ratios, **protocol)
    i = int(np.flatnonzero(batch.base.families == "hawkes_rising")[0])
    days = targets["state"].decision_days
    figure, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
    stage_days = config["stage_days"]
    boundaries = np.arange(config["stages"] + 1) * stage_days
    ratios = batch.base.ratios[i]
    axes[0].stairs(ratios, boundaries, baseline=None, color="black", label="Истинный параметр n по стадиям")
    fits = batch.short_fits[i, config["lookback"] - 1:]
    usable = fits[:, 4] > 0
    axes[0].plot(days[usable], fits[usable, 0], "o--", color="tab:purple", label="MLE по последним завершённым 90 суткам" if stage_days == 90 else f"MLE по последним {stage_days} суткам")
    axes[0].axhline(config["critical_level"], linestyle=":", color="gray")
    crossing = targets["state"].first_crossing[i]
    for axis in axes:
        axis.axvline(crossing, color="tab:red", linestyle=":", label="Первое пересечение")
        axis.grid(alpha=0.2)
    axes[0].set(ylabel="n", ylim=(0, 1.03), title=f"Разметка одного потока: {batch.base.ids[i]}")
    axes[0].legend(loc="upper left", fontsize=8)
    for offset, task, color in ((0, "state", "tab:green"), (1, "warning", "tab:orange")):
        target = targets[task]
        for j, day in enumerate(days):
            axis_marker = "x" if not target.eligible[i, j] else ("o" if target.labels[i, j] else "_")
            axes[1].scatter(day, offset, marker=axis_marker, color=color, s=65)
    axes[1].set(yticks=[0, 1], yticklabels=["Высокое n сейчас", "Переход в будущем"], xlabel="День решения (правая граница наблюдений)",
                ylim=(-0.55, 1.55), title="● — положительная цель; _ — отрицательная; × — исключено из оценки")
    figure.savefig(output / "target_timeline.png", dpi=170)
    plt.close(figure)

    def percent(value):
        return "—" if value is None or not np.isfinite(value) else f"{value:.1%}"

    def number(value):
        return "—" if value is None or not np.isfinite(value) else f"{value:.3f}"

    report = [
        "# Этап 3. Текущее высокое n и раннее предупреждение — раздельная проверка", "",
        f"Новые данные: {config['sizes_per_family']['calibration']} калибровочных и "
        f"{config['sizes_per_family']['test']} тестовых потоков на каждое из {len(config['families'])} семейств "
        f"({config['total_independent_main_trajectories']} независимых потоков всего). "
        f"Seed={config['seed']}. Дополнительно {config['sizes_per_family']['counterfactual']} независимых общих "
        "префиксов с двумя возможными продолжениями каждый. Две ветви одного префикса не считаются независимыми реализациями.", "",
        "## Что именно оценивается", "",
        f"Порог высокого режима n*={config['critical_level']}; это операционный порог ниже математической "
        "границы стационарности n=1. Пересечение n* не означает доказанного достижения n=1.", "",
        f"1. **Состояние.** На правой границе t очередного интервала длиной {stage_days} суток оценивается "
        "n(t−) последнего завершённого интервала. Признаки используют события строго до t. "
        "Параметр следующей стадии не становится меткой текущего состояния. У снижающегося режима "
        "положительные метки сменяются отрицательными; постоянно высокий режим тоже включён.",
        f"2. **Предупреждение.** Для первого пересечения τ положительная метка ставится только при "
        f"0<τ−t≤{config['forecast_stages'] * stage_days} суток. Решения при t≥τ исключены, даже если позже n "
        "снова опустился. Для каждого решения требуется полный будущий горизонт t+H≤T. "
        "Последние решения без него исключены, а не превращены в отрицательные. Изначально высокий "
        "режим не имеет наблюдаемого первого перехода и не входит в эту задачу.", "",
        f"Первое решение возможно после {config['lookback'] * stage_days} суток наблюдения. "
        "Состояние размечается по завершённой стадии; первый положительный state-интервал после нового "
        f"пересечения доступен через {stage_days} суток. Это ограничение частоты решений, а не раннее обнаружение. "
        "Для режима, высокого с начала наблюдения, задержка относительно неизвестного начала не рассчитывается.", "",
        "![Разметка целей](target_timeline.png)", "",
        "## Методы и данные", "",
        "Две прежние MLP и формула тренда n используются без нового обучения и без изменения исходных "
        "весов. Веса скопированы в каталог отчёта и проверяются SHA-256. Их scores обучались для "
        "предупреждений, поэтому эта проверка не заменяет обучение отдельного классификатора состояния. "
        "Статистический ориентир — MLE n по последнему завершённому окну с неизвестным постоянным фоном "
        "и известным β=1/сутки. Коррекция фоновой сезонности из этапа 2 в эти старые признаки не добавлялась. "
        "Истинный текущий n — привилегированная диагностическая оценка, недоступная по реальным данным; "
        "она не использует будущий n.", "",
        "Присутствуют плавные и внезапные пересечения, рост без пересечения, постоянные низкий, высокий "
        "и близкий к порогу режимы, снижение из высокого n и шесть пуассоновских контролей с внешним "
        "сезонным, трендовым или скачкообразным фоном. Между стадиями сохраняется возбуждение; "
        "амплитуда потомков задаётся n в момент родительского события. μ[k]=rate×(1−n[k]) сохраняет "
        "локальную стационарную среднюю, но не фиксирует реализованную интенсивность при переходах. "
        f"Интенсивности выбираются от {config['rate_min']:g} до {config['rate_max']:g} событий/сутки "
        "с дополнительными опорными 2, 20 и 50 внутри этого диапазона.", "",
        "## Раздельная калибровка и неопределённость", "",
        f"Для каждой задачи и метода порог выбирается только на новой калибровочной части с целью "
        f"{config['target_false_alarm_probability']:.0%} траекторий хотя бы с одной ложной тревогой в каждом "
        "семействе. Берётся максимум семейных квантилей отрицательных максимумов по независимым потокам; "
        "положительные и недопустимые по временному протоколу решения исключаются. Семейства без отрицательных решений явно "
        "помечены как исключённые, их доля ложных тревог не объявляется нулевой. Если отрицательные "
        "scores насыщаются на единице, разрешён порог выше 1: такой метод воздерживается от всех сигналов.", "",
        "Веса, признаки и пороги по тесту не выбирались. Цель 5% — правило калибровки, а не гарантия "
        "наблюдаемой частоты на новом тесте. Внутри потока решения зависимы: интервалы чувствительности "
        "и оконной частоты ложных тревог построены bootstrap целых траекторий. Для вероятности обнаружить "
        "переход и семейных траекторных ложных тревог применены Wilson-интервалы. AUC — описательная "
        "оценка по валидным eligible-решениям без интервала из предположения независимости окон. "
        "Пропуск MLE — явное воздержание, положительная цель при нём остаётся пропущенной. "
        "AUC исключает невалидные scores, поэтому рядом сохранена их доля.", "",
        "## Основные результаты", "",
        "Чувствительность состояния считается по положительным окнам; обнаружение перехода — хотя бы "
        "одно правильное предупреждение на положительную траекторию. Эти две колонки имеют разные знаменатели.", "",
        "| Метод | State TPR окон | State AUC | Warning recall переходов | Warning AUC | Макс. ложные state | Макс. ложные warning |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in names:
        a, b = metrics[method]["state"], metrics[method]["warning"]
        report.append(f"| {SCORE_NAMES[method]} | {percent(a['window_sensitivity'])} | {number(a['auc_valid_only'])} | "
                      f"{percent(b['event_detection_probability'])} | {number(b['auc_valid_only'])} | "
                      f"{percent(a['maximum_family_false_alarm_probability'])} | {percent(b['maximum_family_false_alarm_probability'])} |")
    current = metrics["current_mle"]["state"]
    warning_mlp = metrics["hawkes_mlp"]["warning"]
    warning_low, warning_high = warning_mlp["event_detection_ci95"]
    report.extend([
        "", "Наблюдения на этом тесте:", "",
        f"- Текущая MLE-оценка имеет state AUC={number(current['auc_valid_only'])}, но при откалиброванном "
        f"пороге {current['threshold']:.4f} её чувствительность по высоким окнам — "
        f"{percent(current['window_sensitivity'])}. Высокий AUC не гарантирует полезной чувствительности "
        "в выбранной точке с ограничением ложных тревог.",
        f"- MLP с MLE-признаками обнаружила {warning_mlp['detected_trajectories']} из "
        f"{warning_mlp['positive_trajectories']} будущих переходов: {percent(warning_mlp['event_detection_probability'])}, "
        f"Wilson-интервал [{percent(warning_low)}, {percent(warning_high)}]. Максимальная семейная доля "
        f"траекторий с ложной тревогой — {percent(warning_mlp['maximum_family_false_alarm_probability'])} "
        f"при цели {percent(config['target_false_alarm_probability'])}; требуемое сочетание качества пока не подтверждено.",
        f"- Даже диагностический истинный текущий n даёт state AUC="
        f"{number(metrics['oracle_current_n']['state']['auc_valid_only'])}, но при этой калибровке предупреждений "
        f"обнаруживает {metrics['oracle_current_n']['warning']['detected_trajectories']} будущих переходов. "
        "Это результат выбранного правила для текущего n, а не доказательство невозможности иных прогнозов.",
        "", "Наибольшая тестовая доля ложных тревог каждого метода; интервалы здесь относятся к "
        "отдельному указанному семейству, а не являются одновременной областью для максимума по всем семействам.", "",
        "| Метод | Задача | Семейство | Ложные / отрицательные потоки | Доля | Wilson 95% |",
        "|---|---|---|---:|---:|---:|",
    ])
    for method in names:
        for task in ("state", "warning"):
            data = family[(family.method == method) & (family.task == task)].dropna(subset=["probability_false_alarm"])
            row = data.loc[data.probability_false_alarm.idxmax()]
            report.append(f"| {SCORE_NAMES[method]} | {task} | {AUDIT_FAMILY_NAMES[row.family]} | "
                          f"{int(row.false_alarm_trajectories)} / {int(row.negative_trajectories)} | "
                          f"{percent(row.probability_false_alarm)} | "
                          f"[{percent(row.false_alarm_ci95_lower)}, {percent(row.false_alarm_ci95_upper)}] |")
    report.extend(["", "![Сравнение задач](state_warning_comparison.png)", "",
        "Вероятность хотя бы одного правильного предупреждения отдельно для плавных и внезапных "
        "переходов. Clean recall требует, чтобы первая eligible-тревога попала в положительное окно; "
        "позднее верное предупреждение не стирает предыдущую ложную тревогу.", "",
        "| Метод | Плавный переход | Внезапный переход | Clean recall всех переходов | Медиана упреждения обнаруженных |",
        "|---|---:|---:|---:|---:|",
    ])
    for method in names:
        selected = family[(family.method == method) & (family.task == "warning")].set_index("family")
        warning = metrics[method]["warning"]
        lead = warning['median_warning_lead_days_detected_only']
        lead_text = "—" if lead is None else f"{lead:g} суток"
        report.append(f"| {SCORE_NAMES[method]} | {percent(selected.loc['hawkes_rising', 'event_detection_probability'])} | "
                      f"{percent(selected.loc['hawkes_abrupt', 'event_detection_probability'])} | "
                      f"{percent(warning['clean_event_detection_probability'])} | {lead_text} |")
    report.extend(["", "## Общая история и разные возможные будущие режимы", "",
        "Для каждого дополнительного префикса n плавно растёт, оставаясь ниже порога. После одного "
        "следующего низкого интервала возможны два продолжения: пересечение порога или сохранение "
        "низкого режима. В момент решения вся наблюдаемая история одинакова, а метки будущего противоположны. "
        "Будущие события не нужны для расчёта этих scores: это проверка различимости двух допустимых "
        "режимных продолжений по одному и тому же прошлому, а не ещё один независимый тест предсказания.", "",
        "| Метод | TPR в пересекающей ветви | FPR в непересекающей ветви | AUC двух возможных будущих |",
        "|---|---:|---:|---:|",
    ])
    for row in shared.itertuples():
        report.append(f"| {SCORE_NAMES[row.method]} | {percent(row.sensitivity_crossing_branch)} | "
                      f"{percent(row.false_alarm_probability_no_crossing_branch)} | {row.auc_paired_possible_futures:.3f} |")
    report.extend(["", "Равенство TPR и FPR и AUC=0.5 здесь следуют из одинакового прошлого и равного веса "
        "двух ветвей, а не из ошибки обучения. Это не доказывает невозможность прогнозирования плавных "
        "переходов при других предположениях. Оно показывает, что предупреждение требует информации "
        "о законе изменения n, помимо точной оценки текущего значения.", "",
        "## Что следует из проверки", "",
        "Хорошее разделение текущих высоких и низких n по AUC само по себе не является подтверждением "
        "раннего предупреждения. Ранние результаты учитываются только до первого пересечения, в заданном "
        "горизонте и вместе с ложными тревогами. Совместно следует смотреть recall, clean recall, интервалы "
        "и семейные ложные тревоги; упреждение только обнаруженных переходов не характеризует все переходы. "
        "Постоянно высокий и падающий режимы не превращаются в положительные примеры первого будущего перехода.", "",
        "Следующий этап — выбор окон и признаков на обучении/валидации, обучение детекторов под явно "
        "выбранную задачу, несколько seed, новая калибровка и один новый финальный тест. Прежние тесты "
        "остаются диагностическими данными и не должны повторно называться независимым финальным подтверждением. "
        "Городские данные здесь не используются; выводы относятся к заданным синтетическим режимам.", "",
        "## Артефакты", "",
        "- metrics.json — раздельные метрики, интервалы, доли валидности и пороги.",
        "- calibration.csv и thresholds.json — калибровка по задаче и семейству.",
        "- family_test_metrics.csv и trajectory_test_metrics.csv — показатели с явными знаменателями.",
        "- *_state_test_scores.csv и *_warning_test_scores.csv — все решения, метки и маски eligible.",
        "- data/*.npz — новые данные, причинные признаки, scores и seed; models/*.pt — копии замороженных весов.",
        "- shared_prefix_scores.csv и shared_prefix_summary.csv — диагностика одинакового прошлого.",
        "- run_config.json — определения задач, допущения, версии, seed и SHA-256.", "",
        "Проверка: `.\\.venv\\Scripts\\python.exe scripts/verify_task_audit.py`. "
        "Перестроение отчёта: task_report.write_task_report без повторной генерации данных.", "",
    ])
    (output / "analysis.md").write_text("\n".join(report), encoding="utf-8")
