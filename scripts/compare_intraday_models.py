from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare daily and hourly baseline controls on matching city cohorts.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/intraday_comparison"))
    args = parser.parse_args()
    rows = []
    quarter_rows = []
    hourly_rows = []
    for city, prefix in (("Chicago", "chicago"), ("New York City", "nyc")):
        cohorts = []
        for baseline, suffix in (("weekday+month", "daily_control"), ("weekday+month+hour", "hourly")):
            source = args.results_root / f"{prefix}_2024_2025_{suffix}" / "estimates.json"
            estimates = json.loads(source.read_text(encoding="utf-8"))
            fit = estimates["seasonal_hawkes_backtest"]
            if not fit["converged"]:
                raise ValueError(f"Fit did not converge: {source}")
            cohorts.append((estimates["events"], fit["training_events"], fit["test_events"], fit["training_days"], fit["test_days"]))
            rows.append({
                "city": city,
                "baseline": baseline,
                "events": estimates["events"],
                "training_events": fit["training_events"],
                "test_events": fit["test_events"],
                "branching_ratio": fit["branching_ratio"],
                "lr_support_lower": fit["branching_ratio_ci95_lower"],
                "lr_support_upper": fit["branching_ratio_ci95_upper"],
                "decay_per_day": fit["decay_per_day"],
                "excitation_at_zero_boundary": fit["excitation_at_zero_boundary"],
                "hawkes_test_ll": fit["test_log_likelihood"],
                "poisson_test_ll": fit["calendar_poisson_test_log_likelihood"],
                "hawkes_gain_vs_poisson": fit["test_ll_gain_vs_calendar_poisson"],
                "hawkes_gain_per_event": fit["test_ll_gain_per_event_vs_calendar_poisson"],
            })
            for period, values in fit["quarterly_test_windows"].items():
                quarter_rows.append({"city": city, "baseline": baseline, "period": period, **values})
            if suffix == "hourly":
                for hour in range(24):
                    hourly_rows.append({
                        "city": city,
                        "hour": hour,
                        "poisson_multiplier": fit["poisson_hour_of_day_multipliers"][hour],
                        "hawkes_multiplier": fit["hawkes_hour_of_day_multipliers"][hour],
                    })
        if cohorts[0] != cohorts[1]:
            raise ValueError(f"Daily and hourly models use different cohorts for {city}")

    args.out.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.out / "comparison.csv", index=False)
    pd.DataFrame(quarter_rows).to_csv(args.out / "quarterly_comparison.csv", index=False)
    hourly = pd.DataFrame(hourly_rows)
    hourly.to_csv(args.out / "hourly_profiles.csv", index=False)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    positions = np.arange(2)
    for baseline, offset, color, label in (
        ("weekday+month", -0.15, "#b44c4c", "Без часа суток"),
        ("weekday+month+hour", 0.15, "#17816e", "С часом суток"),
    ):
        subset = comparison[comparison["baseline"] == baseline]
        n = subset["branching_ratio"].to_numpy()
        errors = np.vstack([np.maximum(0, n - subset["lr_support_lower"]), np.maximum(0, subset["lr_support_upper"] - n)])
        axes[0].errorbar(positions + offset, n, yerr=errors, fmt="o", color=color, capsize=4, label=label)
        axes[1].bar(positions + offset, subset["hawkes_gain_per_event"], width=0.28, color=color, label=label)
    for axis in axes[:2]:
        axis.set_xticks(positions, ["Chicago", "New York City"])
        axis.grid(axis="y", alpha=0.2)
    axes[0].axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axes[0].set(title="Оценка коэффициента ветвления", ylabel="n и LR-область поддержки", ylim=(-0.04, 1.04))
    axes[0].legend(fontsize=8)
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set(title="Преимущество Hawkes на 2025 годе", ylabel="Разность log-likelihood / событие")
    for city, color in (("Chicago", "#167383"), ("New York City", "#b46e21")):
        subset = hourly[hourly["city"] == city]
        axes[2].plot(subset["hour"], subset["poisson_multiplier"], marker=".", color=color, label=city)
    axes[2].axhline(1, color="#555555", linewidth=0.8, linestyle="--")
    axes[2].set(title="Суточный профиль Пуассона (2024)", xlabel="Час суток по времени источника", ylabel="Множитель, среднее = 1", xticks=[0, 4, 8, 12, 16, 20, 23])
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.2)
    figure.savefig(args.out / "intraday_comparison.png", dpi=180)
    plt.close(figure)

    report = [
        "# Проверка суточной сезонности: Chicago и New York City",
        "",
        "Обучение: 2024 год; условная проверка: 2025 год. Оба варианта пересчитаны одной версией кода с одинаковыми событиями, несколькими стартами оптимизации и одинаковой процедурой профилирования. Квартальные оценки получены без переобучения, с использованием только предшествующей истории событий.",
        "",
        "Исходная базовая интенсивность учитывает день недели и месяц. В новом варианте добавлены 23 индикатора часа суток. Базовая интенсивность постоянна внутри часа; ее интеграл считается с экспозицией 1/24 суток. Пуассон и Hawkes в каждой паре используют одинаковую спецификацию календаря и один и тот же test-период. Единица всех интенсивностей: события/сутки.",
        "",
        "| Город | База | n | LR-область поддержки | Прирост test LL к Пуассону | Прирост на событие |",
        "| --- | --- | ---: | --- | ---: | ---: |",
    ]
    for row in rows:
        label = "Месяц + день недели" if row["baseline"] == "weekday+month" else "+ час суток"
        report.append(f"| {row['city']} | {label} | {row['branching_ratio']:.4f} | [{row['lr_support_lower']:.4f}; {row['lr_support_upper']:.4f}] | {row['hawkes_gain_vs_poisson']:.2f} | {row['hawkes_gain_per_event']:.6f} |")
    chicago = next(row for row in rows if row["city"] == "Chicago" and row["baseline"] == "weekday+month+hour")
    new_york = next(row for row in rows if row["city"] == "New York City" and row["baseline"] == "weekday+month+hour")
    report.extend([
        "",
        f"В Chicago почасовая модель дает n={chicago['branching_ratio']:.4f}, с областью поддержки, включающей ноль. Найденное слабое возбуждение затухает медленно (beta={chicago['decay_per_day']:.4f} в день), а на тестовом году Hawkes уступает почасовому Пуассону на {abs(chicago['hawkes_gain_vs_poisson']):.2f} log-likelihood. Прежний выигрыш после учета только дня недели и месяца не сохраняется.",
        "",
        f"Для New York City n снижается до {new_york['branching_ratio']:.4f}. Выигрыш сохраняется, но составляет {new_york['hawkes_gain_per_event']:.6f} на событие при beta={new_york['decay_per_day']:.4f} в день. Таким образом, большая часть прежнего преимущества объясняется добавленной суточной структурой базы; оставшуюся зависимость нужно проверять отдельно.",
    ])
    report.extend([
        "",
        "Границы получены по профилю likelihood с LR-порогом 3.8415, поиском на всем диапазоне n и несколькими стартами для параметров-помех. Это оболочка области поддержки; сетка профиля сохранена в каждом estimates.json. Она может охватывать несколько областей. На границе n=0 скорость затухания не идентифицируется, а номинальное 95%-покрытие такой оболочки не гарантируется. Узкую область вокруг одного локального максимума нельзя автоматически считать надежным доверительным интервалом.",
        "",
        "Изменение оценки n и прироста likelihood после добавления часа суток показывает чувствительность прежних выводов к выбранной базовой интенсивности. Нулевой выигрыш следует читать как отсутствие практического преимущества данной Hawkes-спецификации над соответствующим Пуассоном. Это не доказывает отсутствия любых зависимостей в данных. Небольшой оставшийся выигрыш тоже не доказывает критичности или причинного каскада.",
        "",
        "## Проверка по кварталам",
        "",
        "Разность условного test log-likelihood Hawkes минус Пуассон с почасовой базой:",
        "",
        "| Город | Квартал 2025 | Событий | Разность LL |",
        "| --- | --- | ---: | ---: |",
    ])
    for row in quarter_rows:
        if row["baseline"] == "weekday+month+hour":
            report.append(f"| {row['city']} | {row['period']} | {int(row['events'])} | {row['hawkes_gain_vs_calendar_poisson']:.2f} |")
    report.extend([
        "",
        "## Ограничения и следующий этап",
        "",
        "Временные метки имеют минутную точность, а совпадающие времена обрабатываются одной группой без выдуманного порядка. Это приближение для округленных данных, не точная likelihood механизма округления. Отдельно остаются неучтенными внутричасовая концентрация на круглых отметках, различия между районами и категориями событий, тренды и особенности регистрации. Шкала — локальное время источника с сутками по 24 часа; неоднозначности переходов на летнее время не восстанавливаются из CSV без часового пояса.",
        "",
        "Следующая проверка детектора должна включать отрицательные контроли: пуассоновские потоки с суточной сезонностью, трендом и изменением уровня интенсивности. Частоту ложных тревог и время упреждения следует измерять на симуляциях с известным режимом. Для реальных зарегистрированных событий истинные моменты критичности не размечены.",
        "",
        "Обоснование контроля нестационарной базы: [Filimonov & Sornette, Apparent criticality and calibration issues in the Hawkes self-excited point process model](https://arxiv.org/abs/1308.6756). В работе рассматриваются смещения оценки n из-за изменений режима и сезонности; здесь этот риск проверяется отдельным сравнением на двух городских рядах.",
        "",
        "Источники исходных выгрузок: [City of Chicago](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2) и [NYPD Complaint Data Historic](https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i). Это прямые городские выгрузки за 2024–2025, а не обновленная версия гармонизированного CODE. Фильтры и параметры повторного запуска сохранены в run_config.json каждого нового расчета.",
        "",
        "![Сравнение базовой интенсивности](intraday_comparison.png)",
        "",
        "Повторное построение отчета: `.\\.venv\\Scripts\\python.exe scripts/compare_intraday_models.py`.",
    ])
    (args.out / "analysis.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(comparison.to_string(index=False))
    print(f"Saved comparison to {args.out.resolve()}")


if __name__ == "__main__":
    main()
