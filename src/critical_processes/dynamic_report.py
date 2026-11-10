from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {"constant90": "Обычный MLE 90", "calendar90": "Календарный MLE 90",
          "calendar360": "Календарный MLE 360", "dynamic_calendar": "Причинный фильтр"}


def write_dynamic_report(output):
    output = Path(output)
    protocol = json.loads((output/"protocol.json").read_text(encoding="utf-8"))
    frozen = json.loads((output/"frozen_plan.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(output/"estimation_summary.csv")
    errors = pd.read_csv(output/"paired_errors.csv")
    delays = pd.read_csv(output/"abrupt_delay.csv")
    coverage = pd.read_csv(output/"interval_coverage.csv")
    metrics = pd.read_csv(output/"operating_metrics.csv")
    methods = list(LABELS)
    selected = frozen["selected_filter"]
    gate = frozen["validation_gate"]
    lines = ["# R6. Причинная динамическая оценка коэффициента ветвления", "",
        "Отдельный завершённый эксперимент. Результаты R4 и R5 не заменяются и не используются для нового подбора.", "",
        "## Протокол", "",
        f"Семейств: 13. На семейство: обучение {protocol['sizes']['train']}, валидация {protocol['sizes']['validation']}, "
        f"калибровка {protocol['sizes']['calibration']}, финальная проверка {protocol['sizes']['evaluation']}. "
        "Фон оценивается по отдельной предыстории; известная скорость затухания β=1/сутки. Оценки коротких блоков не перекрываются, шаг решений — 90 суток.", "",
        "Один фильтр локального линейного тренда оценивает n и наклон только по завершённым блокам. "
        "Дисперсия наблюдения обучена по фактическим квадратам ошибок в ячейках оценённого n и curvature SE; "
        "curvature SE не объявляется готовой дисперсией ошибки. Ковариация фильтра служит рабочим масштабом. "
        "Проекция состояния на допустимые границы делает фильтр приближённым.", "",
        f"На валидации выбраны q={selected['q']:.3g}, затухание наклона ρ={selected['damping']:.1f} из 8 заранее заданных сочетаний. "
        "Критерий — средняя MAE по семействам, затем задержка. Истинный n используется для обучения и проверки, но не подаётся фильтру при оценке.", "",
        "Наклон сравнивается с причинной секущей (n[k]−n[k−2])/2 на один 90-суточный шаг. "
        "Это не производная ступенчатой функции n и не знание будущего изменения.", "",
        "## Ошибка текущего n на новом holdout", "",
        "| Метод | MAE, среднее по 13 семействам | MAE, высокий/близкий n | MAE наклона | Доля допустимых оценок |",
        "|---|---:|---:|---:|---:|"]
    for method in methods:
        rows = summary[summary.method == method]
        nh = rows[rows.family.isin(["hawkes_high", "hawkes_near_threshold"])].mae.mean()
        lines.append(f"| {LABELS[method]} | {rows.mae.mean():.4f} | {nh:.4f} | {rows.slope_mae.mean():.4f} | {rows.valid_fraction.mean():.2%} |")
    lines += ["", "Парная разность MAE: фильтр минус исходная оценка. Отрицательное значение означает меньшую ошибку. "
        "Интервалы получены бутстрепом целых траекторий и условны на выбранных настройках и обучении.", "",
        "| Группа | Сравнение | Разность MAE | Парный 95% интервал |", "|---|---|---:|---|" ]
    for row in errors.itertuples():
        lines.append(f"| {row.group} | Фильтр − {LABELS[row.reference]} | {row.dynamic_minus_reference_mae:.4f} | "
                     f"[{row.paired_bootstrap_ci95_lower:.4f}; {row.paired_bootstrap_ci95_upper:.4f}] |")
    lines += ["", "![Ошибка по семействам](estimation_comparison.png)", "", "## Задержка при внезапном переходе", "",
        "Используется пересечение оценки n≥0.9 после истинного перехода. Первый доступный завершённый высокий блок появляется через 90 суток. "
        "Пропущенные переходы не удаляются: их задержка заменена заранее заданным пределом T−τ+90 суток. "
        "Средняя ограниченная задержка сравнима между методами, но не является несмещённой оценкой неограниченного времени обнаружения.", "",
        "| Метод | Обнаружено | Средняя задержка с пределом, сутки | Медиана среди обнаруженных, сутки |", "|---|---:|---:|---:|"]
    for method in methods:
        rows = delays[delays.method == method]
        hit = rows[rows.detected]
        median = f"{hit.delay_or_censor_cap_days.median():.0f}" if len(hit) else "—"
        lines.append(f"| {LABELS[method]} | {len(hit)}/{len(rows)} | {rows.delay_or_censor_cap_days.mean():.1f} | {median} |")
    lines += ["", "## Неопределённость", "",
        "На независимой калибровке строится максимум нормированной абсолютной ошибки по всей траектории. "
        "Общий множитель — максимум семейных ранговых квантилей ceil((m+1)×0.95). Номинальное покрытие 95% относится "
        "к целой новой траектории того же семейства при обменности; для n и наклона оно заявляется раздельно. "
        "Это не совместный интервал для двух параметров и не условная гарантия при каждом моменте времени. "
        "Пропуски исключены из области покрытия, а доля допустимых оценок сохранена. "
        "Если допустимых калибровочных траекторий недостаточно для нужного ранга, используется весь допустимый диапазон параметра.", "",
        "| Метод | Минимальное семейное покрытие траектории n | Средняя ширина интервала n | Минимальное покрытие наклона |",
        "|---|---:|---:|---:|"]
    for method in methods:
        n = coverage[(coverage.method == method) & (coverage.target == "n")]
        s = coverage[(coverage.method == method) & (coverage.target == "slope")]
        lines.append(f"| {LABELS[method]} | {n.trajectory_coverage.min():.2%} | {n.mean_width.mean():.4f} | {s.trajectory_coverage.min():.2%} |")
    lines += ["", "Семейные доверительные интервалы эмпирического покрытия сохранены в `interval_coverage.csv`. "
        "Высокое покрытие при широкой полосе не означает точную оценку n.", "", "## Проверка обнаружения", "",
        "После изменения признаков MLP обучены заново: одинаковые 8 признаков истории оценённого n, одинаковая архитектура, "
        f"{protocol['epochs']} эпох, seed {protocol['model_seeds']}. У фильтра используется оценённый наклон состояния; "
        "у исходного MLE — секущая трёх блоков. Все инициализации и заранее заданное среднее сохранены.", "",
        f"Основная калибровка контролирует цель 5% ложных тревог на траекторию отдельно по семействам, с совместным уровнем доверия 95% "
        f"для {protocol['protected_score_streams']} оценок. Доверие распределено между оценками и активными семействами. "
        "Номинальная семейная калибровка приведена отдельно как заранее объявленная проверка чувствительности. "
        "Порог, параметры фильтра, интервалы и веса заморожены до генерации holdout.", "",
        "| Задача | Метод | Полнота состояния / чистое предупреждение | Макс. семейная ложная тревога | AUC |",
        "|---|---|---:|---:|---:|"]
    wanted = ["stat_calendar90", "stat_calendar360", "stat_dynamic_calendar", "mlp_calendar90_ensemble", "mlp_dynamic_calendar_ensemble"]
    for row in metrics[(metrics.policy == "family_confident") & metrics.method.isin(wanted)].itertuples():
        utility = row.window_sensitivity if row.task == "state" else row.clean_event_detection_probability
        lines.append(f"| {row.task} | {row.method} | {utility:.2%} | {row.maximum_family_false_alarm_probability:.2%} | {row.auc_valid_only:.3f} |")
    lines += ["", "Полнота предупреждения требует первой допустимой тревоги внутри окна 0<τ−t≤180 суток. "
        "Внезапные и плавные переходы учитываются вместе; ложная тревога до правильного предупреждения не отменяется. "
        "Итоги отдельных seed и обеих калибровок находятся в `operating_metrics.csv`.", "",
        "## Решение по заранее установленному критерию", "",
        f"Условие перехода к рекомендации нового метода по валидации: **{'выполнено' if gate['passed'] else 'не выполнено'}**.", ""]
    for key, passed in gate["checks"].items():
        lines.append(f"- {'Выполнено' if passed else 'Не выполнено'}: `{key}`.")
    lines += ["", "Решение принято до holdout. Диагностическое сравнение переобученных сетей сохранено независимо от результата. "
        "Этот опыт не меняет выбранные модели R4 и не обосновывает перенос на городские данные. "
        "Если критерий не выполнен, добавление фильтра в основной метод не рекомендуется; новый перебор на этом holdout не проводится.", "",
        "## Воспроизведение", "", "```powershell", ".\\.venv\\Scripts\\python.exe -m critical_processes.dynamic_experiment --workers 4",
        ".\\.venv\\Scripts\\python.exe scripts/verify_dynamic_estimation.py", "```", "",
        "Повторный запуск завершённого протокола проверяет хеши и возвращает сохранённый результат. "
        "Изменение параметров или кода требует отдельного каталога. Точные семена всех потоков и их непересечение с R4/R5 проверяются отдельно.", "",
        "Основа рекурсивной оценки: R. E. Kalman (1960), [A New Approach to Linear Filtering and Prediction Problems](https://doi.org/10.1115/1.3662552). "
        "Конкретная калибровка ошибок и ограничения состояния — реализация данного проекта, без заявления оптимальности для Хоукса.", ""]
    (output/"analysis.md").write_text("\n".join(lines), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    families = sorted(summary.family.unique())
    x = np.arange(len(families))
    for method in methods:
        rows = summary[summary.method == method].set_index("family").loc[families]
        axes[0].plot(x, rows.mae, marker="o", ms=3, label=LABELS[method])
        axes[1].plot(x, rows.slope_mae, marker="o", ms=3, label=LABELS[method])
    for ax, title in zip(axes, ("MAE текущего n", "MAE наклона за стадию")):
        ax.set_xticks(x, families, rotation=70, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/"estimation_comparison.png", dpi=160)
    plt.close(fig)
    with np.load(output/"data/evaluation.npz") as data, np.load(output/"data/estimate_paths.npz") as paths:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        for ax, family in zip(axes, ("hawkes_rising", "hawkes_abrupt", "hawkes_falling")):
            # Deterministic representative by observable design rate, not error.
            candidates = np.flatnonzero(data["families"] == family)
            index = candidates[np.argmin(np.abs(np.log(data["rates"][candidates]/20.)))]
            days = np.arange(3, data["ratios"].shape[1]+1)*protocol["generator"]["stage_days"]
            ax.step(days, data["ratios"][index, 2:], where="pre", color="black", label="Истинный n")
            for method in ("calendar90", "calendar360", "dynamic_calendar"):
                ax.plot(days, paths[f"{method}__n"][index], marker=".", label=LABELS[method])
            ax.axhline(.9, color="gray", ls=":")
            ax.set_title(f"{family}\nrate={data['rates'][index]:.1f}/сутки", fontsize=10)
            ax.set_xlabel("Сутки мониторинга")
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output/"causal_examples.png", dpi=160)
        plt.close(fig)
