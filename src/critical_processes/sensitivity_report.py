from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .sensitivity_diagnostics import METHOD_NAMES, POLICY_NAMES
from .task_audit import AUDIT_FAMILY_NAMES


def write_sensitivity_report(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    table = pd.read_csv(output / "operating_metrics.csv")
    curves = pd.read_csv(output / "threshold_curves.csv")
    policies = pd.read_csv(output / "calibration_policies.csv")
    conditional = pd.read_csv(output / "conditional_auc.csv")
    effects = pd.read_csv(output / "paired_effects.csv")
    raw = pd.read_csv(output / "estimation_windows.csv")
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 6.5))
    methods = ("frozen_primary", "paired_known", "paired_constant", "paired_calendar")
    colors = dict(zip(methods, ("#477d94", "#333333", "#bc6c3b", "#7664a0")))
    for axis, task in zip(axes, ("state", "warning")):
        key = "window_sensitivity" if task == "state" else "clean_recall"
        for method in methods:
            sample = curves[(curves.task == task) & (curves.method == method)].sort_values("threshold", ascending=False)
            axis.plot(100 * sample.maximum_family_false_alarm_probability, 100 * sample[key], color=colors[method], label=METHOD_NAMES[method])
        primary = table[(table.task == task) & (table.method == "frozen_primary")]
        for policy, label, marker in (("frozen_stage4", "этап 4", "s"), ("family_nominal", "номинал 5%", "o"),
                                     ("family_confident_source_scope", "доверие / охват", "D")):
            row = primary[primary.policy == policy].iloc[0]
            value = row.window_sensitivity if task == "state" else row.clean_event_detection_probability
            x, y = row.maximum_family_false_alarm_probability * 100, value * 100
            if x <= 20:
                axis.scatter(x, y, color=colors["frozen_primary"], marker=marker, zorder=5,
                             label=f"Основная модель: {label}")
        axis.axvline(5, color="gray", linestyle=":")
        axis.set(xlim=(-.3, 20), ylim=(-1, 102), xlabel="Макс. траекторные ложные тревоги по семействам, %",
                 ylabel="TPR высоких окон, %" if task == "state" else "Clean recall переходов, %",
                 title="Текущее состояние" if task == "state" else "Предупреждение о будущем переходе")
        axis.grid(alpha=.2)
        if task == "warning":
            limit = curves[(curves.task == task) & curves.method.isin(methods)].clean_recall.max() * 100
            axis.set_ylim(-1, min(102, max(20, 10 * np.ceil(limit / 10) + 5)))
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    figure.suptitle("Новые диагностические потоки: одинаковые scores, разные пороги\nПоказана область до 20% ложных тревог; сетка порогов задана по calibration")
    figure.tight_layout(rect=(0, .18, 1, 1))
    figure.savefig(output / "threshold_tradeoff.png", dpi=170)
    plt.close(figure)

    windows = protocol["generator"]["windows"]
    shown_windows = sorted(set([protocol["generator"]["stage_days"], max(windows)]))
    figure, axes = plt.subplots(1, len(shown_windows), figsize=(6 * len(shown_windows), 4.8), squeeze=False, layout="constrained")
    for axis, window in zip(axes.flat, shown_windows):
        for family, label, color in (("hawkes_high", "Высокий n", "tab:green"), ("hawkes_near_threshold", "Близкий n ниже порога", "tab:red")):
            sample = raw[(raw.window_days == window) & (raw.variant == "calendar") & (raw.family == family)]
            for column, style, suffix in (("n_hat", "-", "оценка"), ("true_current_n", ":", "истина")):
                values = np.sort(sample[column].dropna().to_numpy())
                axis.step(values, np.arange(1, len(values) + 1) / max(1, len(values)), where="post", color=color, linestyle=style, label=f"{label}: {suffix}")
        axis.axvline(protocol["generator"]["critical_level"], color="black", linestyle="--", linewidth=.8)
        axis.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="n", ylabel="Эмпирическая функция распределения", title=f"Календарный MLE, окно {window} суток")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8, loc="upper left")
    figure.suptitle("Перекрытие оценок у высокого и близкого к порогу режима\nОконные значения внутри потока зависимы; график описательный")
    figure.savefig(output / "n_overlap.png", dpi=170)
    plt.close(figure)

    def pc(value):
        return "—" if value is None or not np.isfinite(value) else f"{value:.1%}"

    def row(task, method, policy):
        return table[(table.task == task) & (table.method == method) & (table.policy == policy)].iloc[0]

    state_nominal = row("state", "frozen_primary", "family_nominal")
    state_strict = row("state", "frozen_primary", "family_confident_source_scope")
    warning_nominal = row("warning", "frozen_primary", "family_nominal")
    warning_strict = row("warning", "frozen_primary", "family_confident_source_scope")
    state_known = row("state", "paired_known", "family_confident_source_scope")
    state_estimated = row("state", "paired_calendar", "family_confident_source_scope")
    warning_known = row("warning", "paired_known", "family_confident_source_scope")
    warning_estimated = row("warning", "paired_calendar", "family_confident_source_scope")
    report = [
        "# Этап 5. Где теряется чувствительность: n, модель предупреждения и порог", "",
        f"Новые диагностические данные: {protocol['sizes']} потоков на каждое из 13 семейств. "
        f"Seed данных {protocol['data_seed']}, seed оценки {protocol['evaluation_seed']}. "
        "Все исходные веса, выбор и финальные результаты этапа 4 проверяются SHA-256 и остаются неизменными. "
        "Старый holdout не используется для подгонки или выбора; из его архива читаются только seed для исключения совпадений.", "",
        "## Что установлено", "",
        f"**Ошибки информации об n существенны.** При одинаковой архитектуре и строгом правиле "
        f"калибровки парные модели дают для состояния {pc(state_known.window_sensitivity)} TPR с истинными "
        f"n последних трёх стадий и {pc(state_estimated.window_sensitivity)} с календарными MLE. "
        f"Для предупреждения clean recall — {pc(warning_known.clean_event_detection_probability)} против "
        f"{pc(warning_estimated.clean_event_detection_probability)}; максимальные семейные ложные тревоги "
        f"соответственно {pc(warning_known.maximum_family_false_alarm_probability)} и "
        f"{pc(warning_estimated.maximum_family_false_alarm_probability)}. Это контролируемое сравнение "
        "источников n в малой модели, а не точное разложение ошибки исходной сети.", "",
        f"**Строгий порог тоже снижает чувствительность.** У одного и того же основного state-score "
        f"TPR меняется с {pc(state_nominal.window_sensitivity)} при номинальных семейных 5% до "
        f"{pc(state_strict.window_sensitivity)} при доверительной поправке на охват этапа 4. "
        f"Максимальные ложные тревоги на новых данных — {pc(state_nominal.maximum_family_false_alarm_probability)} "
        f"и {pc(state_strict.maximum_family_false_alarm_probability)}. Сравниваемые правила имеют разную "
        "статистическую гарантию; номинальный вариант не подменяет прежний рабочий порог.", "",
        f"**Плохое предупреждение не объясняется только строгостью порога.** У основной модели его "
        f"ослабление до номинальных семейных 5% даёт clean recall всего "
        f"{pc(warning_nominal.clean_event_detection_probability)} при "
        f"{pc(warning_nominal.maximum_family_false_alarm_probability)} ложных тревог, против "
        f"{pc(warning_strict.clean_event_detection_probability)} и "
        f"{pc(warning_strict.maximum_family_false_alarm_probability)} у доверительного варианта. "
        "Предельные отрицательные режимы и временная неопределённость будущего остаются частью задачи.", "",
        "Разности этих экспериментов взаимодействуют и не складываются в проценты единой причины. "
        "Результаты ниже относятся к новым диагностическим потокам, а не к повторной оценке старого финального теста.", "",
        "## Что сравнивается", "",
        "1. **Один score, разные пороги.** Основные модели этапа 4 сохранены. На одной новой calibration "
        "сравниваются общий квантиль максимумов потоков, семейные номинальные 5%, доверительная "
        "граница для одной оценки и для охвата всех оценок этапа 4. Старый порог показан отдельно: "
        "его отличие от новой калибровки включает случайность калибровочных данных.",
        "2. **Одна архитектура, разные сведения о n.** Шесть новых диагностических MLP получают восемь "
        "признаков: последние три n, наклон, линейную проекцию и три флага валидности. Источник n — "
        "истинная прошлая история, обычный MLE или календарный MLE по последним завершённым стадиям. "
        "Данные, архитектура, начальные веса, число эпох и цели одинаковы; нормировка обучается на train "
        "отдельно для каждого источника. Разница отражает замену информации о n, включая пропуски. "
        "Она не является точным процентным разложением ошибки исходной MLP с 64 признаками.",
        "3. **Разделимость и сглаживание.** Сохраняются ошибки MLE для всех окон, распределения высоких "
        "и близких к порогу режимов, условные AUC и семейства, задающие порог. Среднее истинного n "
        "по окну — описательный ориентир сглаживания, а не утверждение, что MLE при меняющемся n "
        "обязан оценивать именно арифметическое среднее.", "",
        f"Парные модели обучаются {protocol['epochs']} эпох при фиксированном seed {protocol['model_seed']}; "
        "hyperparameter search по diagnostic evaluation не проводится. Истинные n доступны только "
        "диагностическим ориентирам. Они не являются реализуемыми детекторами на реальных данных.", "",
        "Sigmoid-scores сетей не проверялись как откалиброванные вероятности. Порог 0.5 — исходный "
        "классификационный ориентир; в остальных вариантах калибруется правило тревоги, а не вероятность n.", "",
        "## Цена разных правил калибровки", "",
        "Все частоты ниже проверяются на полной новой evaluation, включая контроль близкого к порогу n. "
        "Порог «без близкого контроля» исключает это семейство только из calibration: его ложные тревоги "
        "остаются в оценке качества. Преимущество после исключения нельзя объявлять улучшением на полной задаче.", "",
        "| Задача | Порог | Значение | TPR окон | Recall переходов / потоков | Clean recall | Макс. ложные |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for task in ("state", "warning"):
        for policy in ("frozen_stage4", "fixed_score", "pooled_trajectory", "family_nominal", "family_confident", "family_confident_source_scope", "family_nominal_without_near"):
            item = row(task, "frozen_primary", policy)
            report.append(f"| {task} | {POLICY_NAMES[policy]} | {item.threshold:.6f} | {pc(item.window_sensitivity)} | "
                          f"{pc(item.event_detection_probability)} | {pc(item.clean_event_detection_probability)} | {pc(item.maximum_family_false_alarm_probability)} |")
    report.extend(["", "![Пороги и чувствительность](threshold_tradeoff.png)", "",
        "Доверительные варианты используют ту же ранговую границу, что и этап 4. Вариант «одна оценка» "
        "защищает семейства одного score; следующий дополнительно делит вероятность ошибки на охват "
        f"{protocol['source_protected_streams']} оценок исходного эксперимента. Это контролируемое сравнение "
        "правил. Оно не означает одновременную гарантию для всех диагностических политик, включая "
        "номинальные и специально ослабленные. Нельзя автоматически интерпретировать рост recall при "
        "ослаблении правила как улучшение при прежнем ограничении ложных тревог.", "",
        "Clean recall не обязан монотонно расти при снижении порога: ранняя ложная тревога может "
        "опередить последующее правильное предупреждение. Повышение порога иногда удаляет такую "
        "преждевременную тревогу. Поэтому показанные кривые clean recall не являются обычной ROC-кривой.", "",
        "## Парные модели с известной и оценённой историей n", "",
        "Для состояния качество — TPR высоких окон; для предупреждения — clean recall по переходам. "
        "Столбец ложных тревог имеет траекторный знаменатель и считается по худшему семейству.", "",
        "| Задача | Источник | Политика | Качество | Макс. ложные | AUC |",
        "|---|---|---|---:|---:|---:|",
    ])
    for task in ("state", "warning"):
        for policy in ("family_nominal", "family_confident_source_scope"):
            for method in ("paired_known", "paired_constant", "paired_calendar"):
                item = row(task, method, policy)
                value = item.window_sensitivity if task == "state" else item.clean_event_detection_probability
                report.append(f"| {task} | {METHOD_NAMES[method]} | {POLICY_NAMES[policy]} | {pc(value)} | {pc(item.maximum_family_false_alarm_probability)} | {item.auc_valid_only:.3f} |")
    report.extend(["", "Прямые ориентиры с истинным n не обучают будущую метку: текущий n использует "
        "только последнее завершённое значение; линейная проекция — три прошлых значения. Даже точная "
        "история не задаёт закон будущего перехода. Линейная проекция предполагает продолжение наклона, "
        "а парная MLP учится на конкретном генераторе; ни один ориентир не является универсальным верхним пределом.", "",
        "| Задача | Ориентир | Политика | TPR окон | Recall | Макс. ложные |",
        "|---|---|---|---:|---:|---:|",
    ])
    for task, method in (("state", "oracle_current"), ("warning", "oracle_current"), ("warning", "oracle_linear")):
        for policy in ("fixed_score", "family_nominal"):
            item = row(task, method, policy)
            report.append(f"| {task} | {METHOD_NAMES[method]} | {POLICY_NAMES[policy]} | {pc(item.window_sensitivity)} | {pc(item.event_detection_probability)} | {pc(item.maximum_family_false_alarm_probability)} |")
    report.extend(["", "## Перекрытие оценок n", "", "![Перекрытие n](n_overlap.png)", "",
        "MAE ниже вычислена среди валидных оценок календарного MLE. Неудачи не заменяются нулевым "
        "параметром; валидность показана отдельно. Граничная доля — n̂≥0.995−10⁻⁸.", "",
        "| Семейство | Окно | Валидно | MAE к текущему n | Доля на верхней границе |",
        "|---|---:|---:|---:|---:|",
    ])
    for family in ("hawkes_high", "hawkes_near_threshold"):
        for window in windows:
            frame = raw[(raw.family == family) & (raw.window_days == window) & (raw.variant == "calendar")]
            valid = frame[frame.fit_valid]
            mae = (valid.n_hat - valid.true_current_n).abs().mean()
            report.append(f"| {AUDIT_FAMILY_NAMES[family]} | {window} | {pc(frame.fit_valid.mean())} | {mae:.3f} | {pc((valid.n_hat >= .995-1e-8).mean())} |")
    report.extend(["", "Для меняющегося n длинное окно уменьшает шум и одновременно усредняет прошлые "
        "режимы. В таблице — календарный MLE на тех же динамических потоках; ошибка и сглаживание "
        "не являются аддитивным разложением.", "",
        "| Семейство | Окно | MAE текущего n среди валидных | Средний модуль сдвига n по окну |",
        "|---|---:|---:|---:|",
    ])
    for family in ("hawkes_rising", "hawkes_falling"):
        for window in windows:
            frame = raw[(raw.family == family) & (raw.window_days == window) & (raw.variant == "calendar")]
            mae = (frame.n_hat - frame.true_current_n).abs().mean()
            smoothing = (frame.true_window_mean_n - frame.true_current_n).abs().mean()
            report.append(f"| {AUDIT_FAMILY_NAMES[family]} | {window} | {mae:.3f} | {smoothing:.3f} |")
    report.extend(["", "Условные AUC основной модели: одинаковая положительная группа сравнивается с "
        "разными отрицательными. Общий AUC по смеси может скрывать трудную пару, которая задаёт "
        "траекторный порог. AUC здесь описательная, без интервала из предположения независимости окон.", "",
        "| Задача | Положительное семейство | Отрицательное семейство | AUC |",
        "|---|---|---|---:|",
    ])
    for task, positive in (("state", "hawkes_high"), ("warning", "hawkes_rising"), ("warning", "hawkes_abrupt")):
        for negative in ("homogeneous", "hawkes_near_threshold", "hawkes_subthreshold"):
            sample = conditional[(conditional.task == task) & (conditional.method == "frozen_primary") &
                (conditional.positive_family == positive) & (conditional.negative_family == negative)]
            if len(sample):
                report.append(f"| {task} | {AUDIT_FAMILY_NAMES[positive]} | {AUDIT_FAMILY_NAMES[negative]} | {sample.auc_valid_only.iloc[0]:.3f} |")
    report.extend(["", "## Какое семейство задаёт порог", "",
        "| Задача | Правило | Семейство с максимальной границей | Отрицательных потоков | Порог |",
        "|---|---|---|---:|---:|",
    ])
    binding = policies[(policies.method == "frozen_primary") & policies.binding &
        policies.policy.isin(["family_nominal", "family_confident", "family_confident_source_scope", "family_nominal_without_near"])]
    for item in binding.itertuples():
        report.append(f"| {item.task} | {POLICY_NAMES[item.policy]} | {AUDIT_FAMILY_NAMES.get(item.family, item.family)} | {int(item.negative_trajectories)} | {item.family_cutoff:.6f} |")
    for task in ("state", "warning"):
        selected_binding = binding[(binding.task == task) & (binding.policy == "family_confident_source_scope")]
        families = ", ".join(AUDIT_FAMILY_NAMES.get(name, name) for name in selected_binding.family)
        report.extend(["", f"Для задачи {task} максимальную доверительную границу на этой calibration задаёт: {families}."])
    without_near = row("warning", "frozen_primary", "family_nominal_without_near")
    if without_near.threshold == warning_nominal.threshold:
        report.extend(["", "Исключение стабильного близкого к порогу контроля из calibration не изменило "
            "порог предупреждения. Его задаёт другое отрицательное семейство; проблема не сводится "
            "к наличию стабильного n чуть ниже 0.9."])
    report.extend(["", "Для текущего состояния следует различать номинальный и более высокий доверительный "
        "квантили: их могут задавать разные отрицательные семейства. Для предупреждения важно отдельно "
        "сравнивать постепенный рост с пересечением и рост без пересечения. Исключение одного трудного "
        "семейства не заменяет проверку на остальных."])
    report.extend(["", "## Парные эффекты и пределы вывода", "",
        "Положительная разность означает больше TPR состояния или clean recall предупреждения у первого "
        "варианта. Интервалы получены bootstrap полных потоков на одинаковых траекториях; они описательные "
        "и не исправлены на множество диагностических сравнений. Изменение политики одновременно "
        "изменяет частоту ложных тревог; она сохранена в paired_effects.csv.", "",
        "| Задача | Контраст | Разность | Bootstrap 95% |",
        "|---|---|---:|---:|",
    ])
    for item in effects[effects.contrast.isin(["confidence_cost", "multiplicity_cost", "near_control_cost", "known_vs_calendar"])].itertuples():
        report.append(f"| {item.task} | {item.contrast} ({item.proposed_policy}) | {item.utility_difference:+.3f} | [{item.paired_ci95_lower:+.3f}, {item.paired_ci95_upper:+.3f}] |")
    report.extend(["", "Это диагностическая проверка механизмов. Она не выбирает новый рабочий порог и "
        "не превращает новый evaluation в независимый финальный тест после просмотра кривых. Новый "
        "метод или правило потребуется отдельно выбрать на обучении/валидации и проверить на новом holdout. "
        "На городские зарегистрированные события эти результаты автоматически не переносятся.", "",
        "## Артефакты", "",
        "- operating_metrics.csv и metrics.json — все пороги, полнота, ложные тревоги и интервалы.",
        "- calibration_policies.csv, source_calibration.csv — ранги и семейства, задающие пороги.",
        "- threshold_curves.csv — сетка из calibration, описательные кривые на evaluation.",
        "- paired_effects.csv — парные различия политик и источников n.",
        "- estimation_windows.csv, estimation_summary.csv — ошибки n и наклона, сглаживание и пропуски.",
        "- conditional_auc.csv — разделимость по положительным и отрицательным семействам.",
        "- data/*.npz — новые данные и scores; paired_models/*.pt — только диагностические модели.",
        "- protocol.json и run_config.json — фиксированные правила, seed и проверки неизменности этапа 4.", "",
        "Проверка: `.\\.venv\\Scripts\\python.exe scripts/verify_sensitivity_diagnostics.py`. "
        "Перестроение отчёта — sensitivity_report.write_sensitivity_report, без подстройки моделей.", "",
    ])
    (output / "analysis.md").write_text("\n".join(report), encoding="utf-8")
