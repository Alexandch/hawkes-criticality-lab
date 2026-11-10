# Реестр результатов и источников для отчёта

Состояние на 10 ноября 2026 года. Протоколы ниже самостоятельные: их тестовые выборки, единицы сравнения и определения метрик нельзя механически объединять. Новые проверки при завершении работы добавляются отдельной строкой.

## 1. Экспериментальные блоки

| ID | Содержание | Основной источник | Статус для отчёта |
|---|---|---|---|
| P0 | Показатели потока при росте n; 30 повторов на уровень | [simulation_v2](../results/simulation_v2/analysis.md) | Сохранённый пилот; основной рисунок заменён P1 |
| P1 | Итоговые показатели; 500 повторов на 8 уровней, 4 000 потоков | [indicators_final](../results/indicators_final/analysis.md) | Основной рисунок; все потоки воспроизведены |
| R1 | Точность MLE при постоянном n, одинаковой ожидаемой интенсивности; 10 000 потоков, 40 000 подгонок | [n_estimation_accuracy](../results/n_estimation_accuracy/analysis.md) | Основной результат |
| R2 | Влияние фона; 9 600 потоков, 192 000 подгонок | [background_comparison](../results/background_comparison/analysis.md) | Основной результат |
| R3 | Разделение состояния и первого будущего пересечения; 6 500 потоков и 300 общих префиксов | [state_warning_audit](../results/state_warning_audit/analysis.md) | Определения целей и диагностический аудит |
| R4 | Выбор окон на валидации, три seed, один новый финальный тест; 13 650 потоков | [selected_detectors](../results/selected_detectors/analysis.md) | Итог реализованных детекторов до заключительного метода |
| R5 | Ошибки n и калибровки на 9 750 новых потоках; 92 варианта правил | [sensitivity_diagnostics](../results/sensitivity_diagnostics/analysis.md) | Диагностическая проверка механизмов |
| E1 | Календарные базы на Chicago и NYC, обучение 2024, проверка 2025 | [intraday_comparison](../results/intraday_comparison/analysis.md) | Эмпирическая иллюстрация |
| R6 | Причинная динамическая оценка n; 11 700 потоков, 48 правил обнаружения | [dynamic_estimation](../results/dynamic_estimation/analysis.md), [заранее заданный протокол](DYNAMIC_PROTOCOL.md) | Выполнен; умеренный выигрыш MAE, рост задержки, предупреждение остаётся слабым |
| R7 | Замороженное применение к 14 городам CODE и четырём порталам; 22 блока, 220 решений | [empirical_transfer](../results/empirical_transfer/analysis.md) | Выполнено без выдуманных меток критичности |
| R8 | 800 000 полных генеалогических деревьев | [genealogical_clusters](../results/genealogical_clusters/analysis.md) | Выполнены все полные воспроизведения |
| R9 | Расширенная MLP: обучение 7 800 потоков, 11 700 новых потоков в частях, 26 правил с ориентирами | [extended_neural](../results/extended_neural/analysis.md) | Новый независимый тест; R4–R6 неизменны |

## 2. Числа, использованные в основном тексте

| Утверждение | Значение | Где проверить | Обязательная оговорка |
|---|---|---|---|
| R1: MAE n=0.95 при 20 событиях/сутки, окно 90 | 0.059143 | [summary.csv](../results/n_estimation_accuracy/summary.csv): `true_n=0.95, target_rate=20, window_days=90`, `mae_valid` | Среди валидных подгонок; β известен |
| R1: то же, окно 360 | 0.017184 | Та же таблица, `window_days=360` | Валидно 99.2% подгонок |
| R1: AUC n=0.9 против 0.95, 20/сутки, 90/360 суток | 0.719 / 0.940 | [distinguishability.csv](../results/n_estimation_accuracy/distinguishability.csv) | Условно на валидности оценок, не раннее предупреждение |
| R2: сезонный Пуассон, постоянный фон | Среднее n̂=0.909405 | [summary.csv](../results/background_comparison/summary.csv): `seasonal, 20, 360, constant_joint` | Истинный n=0 |
| R2: тот же поток, форма по прошлому со свободным уровнем | Среднее n̂=0.023715 | Та же таблица, `past_shape_joint` | Форма обучена на отдельном 730-дневном префиксе |
| R2: Хоукс n=0.9, форма по прошлому со свободным уровнем | Среднее n̂=0.894580 | Та же таблица, `hawkes_n0p9, 20, 360, past_shape_joint` | Положительный контроль |
| R2: тот же Хоукс, весь прогноз фона фиксирован | Среднее n̂=0.170038 | Та же таблица, `past_fixed` | Прогноз общей частоты не равен иммиграционной μ |
| R4: первичная модель состояния | AUC 0.972; TPR окон 2.9% | [metrics.json](../results/selected_detectors/metrics.json): `state__calendar_mle__ensemble` | Окно 360, модель выбрана до теста |
| R4: первичная модель предупреждения | 0 из 600 переходов | Тот же JSON: `warning__constant_mle__ensemble` | Верхняя Wilson-граница 95% около 0.64%; не доказательство нулевой популяционной вероятности |
| R4: максимум траекторных ложных тревог | 1.0% у ансамблей/статистики; 1.7% с учётом отдельных seed | [test_metrics.csv](../results/selected_detectors/test_metrics.csv) | Максимум по семействам; конечная выборочная доля |
| R5: известная и календарная история n в парных моделях предупреждения | Clean recall 31.2% / 0.2% | [operating_metrics.csv](../results/sensitivity_diagnostics/operating_metrics.csv): `warning, paired_known/paired_calendar, family_confident_source_scope` | Истинный n — недоступный на реальных данных ориентир; это другая модель и новая диагностическая выборка |
| R5: номинальная и строгая калибровка одного state-score | TPR 12.9% / 2.2%; ложные 4.3% / 0.3% | Та же таблица: `state, frozen_primary, family_nominal/family_confident_source_scope` | Правила имеют разные статистические гарантии |
| R5: предупреждение после номинальной семейной калибровки | Clean recall 1.3%, максимум ложных 4.0% | Та же таблица: `warning, frozen_primary, family_nominal` | Не новый рабочий порог и не результат старого финального теста |
| E1: Chicago, n до/после часа суток | 0.4752 / 0.0368 | [Сводный анализ](../results/intraday_comparison/analysis.md), исходные JSON указаны там | Выбранные спецификации моделей; не измеренная истинная критичность |
| E1: NYC, n до/после часа суток | 0.7480 / 0.2943 | Тот же анализ | Остаётся небольшой положительный test LL |

Точность отображения в отчёте может быть ниже точности CSV. Например, 0.023715 показывается как 0.024, а 0.028605 — как 2.9%. Округление не должно менять интерпретацию знаменателя.

### Дополнение после R6 и P1

| Утверждение | Значение | Где проверить | Ограничение |
|---|---|---|---|
| P1: дисперсия при n=0 / 0.97 | 1.9907 / 27.5222 | [summary.csv](../results/indicators_final/summary.csv): `variance` | Средние по 500 потокам; β=1, rate=2 |
| P1: ACF(1) при n=0 / 0.97 | −0.0039 / 0.7706 | Та же таблица: `lag1_autocorrelation` | Суточные количества, не специфический тест критичности |
| R6: календарный MLE 90 / фильтр, MAE | 0.0875 / 0.0817 | [estimation_summary.csv](../results/dynamic_estimation/estimation_summary.csv): среднее `mae` по 13 семействам | Валидные оценки; доли 98.37% / 99.72% |
| R6: то же, высокий и близкий к порогу n | 0.1008 / 0.0872 | Та же таблица, семьи `hawkes_high`, `hawkes_near_threshold` | Отдельная группа, не вся выборка |
| R6: парная разность MAE фильтр − MLE 90 | −0.0058 [−0.0065; −0.0051] | [paired_errors.csv](../results/dynamic_estimation/paired_errors.csv): `group=all, reference=calendar90` | Бутстреп целых траекторий, условно на обучении |
| R6: ограниченная задержка MLE 90 / фильтр | 213.9 / 272.4 суток | [abrupt_delay.csv](../results/dynamic_estimation/abrupt_delay.csv): среднее `delay_or_censor_cap_days` | Только внезапные переходы; пропуски включены с пределом |
| R6: чистые предупреждения сети с фильтром | 2/600 = 0.33% | [metrics.json](../results/dynamic_estimation/metrics.json): `warning__mlp_dynamic_calendar_ensemble__family_confident` | Новая выборка и новая калибровка; модели R4 не заменяются |
| R6: полоса n фильтра | Мин. покрытие 96.33%, средняя ширина 0.7909 | [interval_coverage.csv](../results/dynamic_estimation/interval_coverage.csv) | Покрытие целой траектории, отдельно для n; широкая полоса малоинформативна |

## 3. Завершающие результаты и исходные данные

Завершающие результаты:

| Утверждение | Значение | Где проверить | Ограничение |
|---|---|---|---|
| R8: полный размер при n=0.75 / 0.97 | 3.97421 / 32.63428 | [summary.csv](../results/genealogical_clusters/summary.csv) | 100 000 независимых корней на уровень, дерево не обрезано |
| R8: n=0.97, хвост и длительность | P(S≥100)=5.566%, 95% длительность 25.5713 суток | Та же таблица | Генеалогия известна только в имитации |
| R9: обучение | 7 800 потоков | [frozen_plan.json](../results/extended_neural/frozen_plan.json): `training_trajectories` | Только старые train и новые train; без старых holdout |
| R9: строгий ансамбль состояния | TPR=3.8411%, AUC≈0.975 | [metrics.json](../results/extended_neural/metrics.json): `state__ensemble__family_confident` | Максимальная семейная ложная тревога 1% |
| R9: строгий ансамбль предупреждения | 5/600 = 0.8333%, Wilson [0.3565%; 1.9358%] | Та же таблица: `warning__ensemble__family_confident` | Максимальная семейная ложная тревога 0.6667% |
| R9: номинальное предупреждение | 27/600 = 4.5%, макс. ложная тревога 5% | Та же таблица: `warning__ensemble__family_nominal_sensitivity` | Номинальная калибровка без той же совместной доверительной гарантии |
| R9 против замороженного R4, общий тест | TPR +0.5571 п.п. [0.1146; 0.9904], предупреждение +0.6667 п.п. [0; 1.4706] | [paired_detector_differences.csv](../results/project_completion/paired_detector_differences.csv) | Условно на моделях/калибровках, 3000 бутстрепов целых потоков; менялось несколько компонентов |
| Вспомогательная оценка n | MAE=0.0426105 | [auxiliary_n_errors.csv](../results/extended_neural/auxiliary_n_errors.csv): среднее 13 семейств | На симуляции, не истинный городской n |
| Городские выгрузки | 7 395 987 сырых строк, 7 146 403 дела/отчёта | [data_totals.json](../results/project_completion/data_totals.json) | 2019–2025 и январь–сентябрь 2026 |
| CODE 2018–2022 | 9 543 104 записи, 17 подготовленных городов, 14 допущенных | [panel_manifest.json](../data/processed/expanded_20261110/panel_manifest.json) | uid дополнен годом, у трёх городов пропуски месяцев |
| R7: эмпирическое применение | 22 блока, 220 решений, ни одной пороговой тревоги R4/R9 | [blocks.csv](../results/empirical_transfer/blocks.csv), [decisions.csv](../results/empirical_transfer/decisions.csv) | Истинных n и переходов нет, accuracy/TPR/FPR не измеряются |

- [data/README.md](../data/README.md) — происхождение выгрузок, поля времени и правила подготовки.
- [Манифест CODE](../data/processed/code_core_2020_2021_17cities.csv.manifest.json) — состав подготовленной панели 2020–2021.
- [Chicago, почасовая модель](../results/chicago_2024_2025_hourly/run_config.json) и [NYC, почасовая модель](../results/nyc_2024_2025_hourly/run_config.json) — параметры городского анализа.

Подготовленная панель CODE содержит 3 753 014 записей в 17 городах. Это характеристика набора после подготовки, а не число записей единого проведённого теста детектора.

### R10: наблюдаемый прогноз район × 6 часов

Условия заранее записаны в [SPATIOTEMPORAL_PROTOCOL.md](SPATIOTEMPORAL_PROTOCOL.md), машинный протокол и выбор — в [frozen_plan.json](../results/spatiotemporal_forecast/frozen_plan.json). Хеши полного результата — в [complete.json](../results/spatiotemporal_forecast/complete.json). Выбор параметров и порогов не пересматривается по тестам.

| Утверждение | Значение | Где проверить | Ограничение |
|---|---|---|---|
| Районная панель | 1 176 356 Chicago + 423 703 SF, 22 + 10 округов | [Chicago](../data/processed/spatiotemporal_20261110/chicago.json), [SF](../data/processed/spatiotemporal_20261110/san_francisco.json) | Выбранные группы преступлений, одна запись на дело после очистки |
| Поздние отчёты SF | 54.974% позже 6 часов, 11.446% позже 7 суток | Метаданные SF, `report_delay_fraction_*` | Исторические исправления неизвестны, дата регистрации Chicago отсутствует |
| Chicago 2026, MAE нейросети / календаря / адаптивного фона | 1.857868 / 1.988378 / 1.871230 | [test_metrics.csv](../results/spatiotemporal_forecast/test_metrics.csv), лаг 7, `test2026` | Событий на округ / 6 часов, январь–сентябрь |
| SF 2026, те же MAE | 1.436770 / 2.467611 / 1.415489 | Та же таблица | Адаптивный фон лучше заранее выбранной нейросети |
| NN минус адаптивный фон, MAE 2026 | Chicago −0.01336 [−0.01879; −0.00809], SF +0.02128 [0.00938; 0.03409] | [neural_vs_recent.csv](../results/spatiotemporal_summary/neural_vs_recent.csv) | Описательный парный бутстреп 14 суток, 1000 повторов, условно на моделях |
| Выбранный метод всплеска 2026 | Chicago 22/971, FPR 0.4641%; SF 0/62, FPR 0.1105% | `test_metrics.csv`: `spike_primary=True`, лаг 7 | Порог на калибровке 2024, FPR ячеек; высокий n не является меткой |
| Покрытие NN 95% интервалов 2026 | 97.5358% / 98.4707%, ширина 9.1521 / 6.6467 | Та же таблица | Интервалы широкие, модельные, не точный прогноз |
| Выдача следующего интервала | 32 округа, 1 октября 2026 00:00–06:00 local | [next_interval_forecasts.csv](../results/spatiotemporal_summary/next_interval_forecasts.csv) | Пример на границе архива, `observed_count` пуст; не текущий прогноз |

[provenance.json](../results/spatiotemporal_summary/provenance.json) связывает дополнительное парное сравнение и пример выдачи с замороженным R10, кодом выдачи и скриптом сводки. Эти результаты не изменяют финальные модели или пороги. Индивидуальные координаты и адреса не скачиваются; граф использует агрегированные обучающие центры округов.

## 4. Рисунки

`scripts/prepare_report_assets.py` копирует выбранные PNG в `report/assets/`, не меняя исходные результаты. `assets/manifest.json` хранит путь исходного PNG, путь копии и SHA-256. Подпись каждого рисунка в `REPORT.md` указывает его экспериментальный блок. Основной рисунок показателей использует P1 (500 повторов), прежний P0 сохранён отдельно. P1 и R1 — разные эксперименты.

## 5. Проверки воспроизводимости

Из корня проекта:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe scripts/verify_report.py
.\.venv\Scripts\python.exe scripts/verify_n_accuracy.py
.\.venv\Scripts\python.exe scripts/verify_background_comparison.py
.\.venv\Scripts\python.exe scripts/verify_task_audit.py
.\.venv\Scripts\python.exe scripts/verify_selected_training.py
.\.venv\Scripts\python.exe scripts/verify_sensitivity_diagnostics.py
.\.venv\Scripts\python.exe scripts/verify_dynamic_estimation.py
.\.venv\Scripts\python.exe scripts/finalize_indicators.py --verify
.\.venv\Scripts\python.exe scripts/complete_genealogy.py --verify
.\.venv\Scripts\python.exe scripts/verify_extended_training.py
.\.venv\Scripts\python.exe scripts/verify_empirical_transfer.py
.\.venv\Scripts\python.exe scripts/verify_spatiotemporal.py
```

После завершения R7–R9 прошли все 82 теста. Для R9 проверены 7 800 обучающих потоков, 11 700 новых потоков, все 26 правил, эпохи валидации и 26 точных воспроизведений. Все 800 000 кластеров R8 воспроизведены полностью. Проверка R7 контролирует 35 исходных выгрузок, 21 подготовленный ряд, признаки всех 22 блоков и 220 решений без выдуманных меток. Численные выходы float32 допускают малое отличие округления при разном числе потоков BLAS; решения о тревогах должны совпадать точно.

После добавления R10 прошли все 90 тестов. Независимая проверка R10 повторила очистку 1 621 231 исходной строки, обе точные сетки, доступность отчётов SF, все прогнозы, замороженные параметры, пороги, 40 сравнений и парные интервалы. Проверка выдачи следующего интервала дополнительно отклоняет чужую панель и неоднозначное обозначение часового пояса.

Ранее проверены 11 700 потоков R6, 8 настроек, 48 правил и 26 воспроизведений; все 4 000 потоков P1 воспроизведены полностью. Неизменность данных, весов и ключевых результатов R4–R6 подтверждена SHA-256. Прежние проверки R4 и R5 воспроизводили 130 и 78 потоков; их полный новый запуск не подменяется проверкой хешей.
