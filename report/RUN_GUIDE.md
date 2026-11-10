# Запуск, данные и сдаваемые материалы

## Свежий клон GitHub

Название программного проекта — **Hawkes Criticality Lab**, пакет установки `hawkes-criticality-lab`; Python-импорты `critical_processes` и прежняя команда `critical-processes` сохранены. В репозиторий включены исходники, тесты, модели и компактные результаты. Большие CSV и имитационные NPZ-кэши хранятся отдельно и не загружаются в Git.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[neural,fast]"
.\.venv\Scripts\python.exe scripts/verify_repository_snapshot.py
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Проверка репозитория контролирует опубликованные байты, код, рисунки и компактные численные результаты. Она явно перечисляет внешние артефакты и не подменяет полную проверку исходных данных. Раздел ниже о командах `verify_report.py`, `verify_extended_training.py` и `verify_empirical_transfer.py` относится к полному архивному снимку, включающему большие кэши.

Полное повторение нейросетевой цепочки на новых локальных каталогах:

```powershell
.\.venv\Scripts\python.exe scripts/reproduce_neural_pipeline.py --stage all --out runs/reproduced --workers 4
```

Команда заново моделирует десятки тысяч потоков и требует времени и места на диске. Она не изменяет архивные `results/`. Метаданные путей и хеши новых контейнеров могут отличаться от исходного архива; семена и статистический протокол сохраняются.

Для получения реальных данных в свежем клоне сначала нужны также исходные CODE 2020–2021:

```powershell
.\.venv\Scripts\python.exe scripts/download_code_history.py
.\.venv\Scripts\python.exe scripts/download_expanded_data.py --out data/raw/expanded_local
.\.venv\Scripts\python.exe scripts/prepare_expanded_panel.py --raw data/raw/expanded_local --out data/processed/expanded_local
```

Повторная загрузка в другую дату создаёт новый снимок изменяющихся порталов, а не обещает побайтовое совпадение с выгрузкой 10 ноября 2026. Для применения к заново обученным моделям функция `run_transfer` принимает пути `old_model`, `new_model`, `panel` и отдельный `out`. Архивные веса и пороги не перенастраиваются на новых городских данных.

## Прогноз район × 6 часов

Данные R10 имеют отдельную схему с районом. Сохранённый снимок: [исходные манифесты](../data/raw/spatiotemporal_20261110/), [метаданные сеток](../data/processed/spatiotemporal_20261110/). Самих больших CSV.GZ и NPZ в Git нет. В свежем клоне следует скачать новый снимок в новые каталоги, затем обучить и проверить его отдельно:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_spatiotemporal_data.py --raw data/raw/spatial_local --out data/processed/spatial_local
.\.venv\Scripts\python.exe -m critical_processes.spatial_experiment --panel-dir data/processed/spatial_local --out runs/spatial_local
.\.venv\Scripts\python.exe scripts/verify_spatiotemporal.py --out runs/spatial_local --panels data/processed/spatial_local
.\.venv\Scripts\python.exe -m critical_processes.district_forecast --city chicago --models runs/spatial_local --panels data/processed/spatial_local --out runs/chicago_forecast.csv
```

Те же команды работают для `--city san_francisco`. Загрузка позднее 10 ноября создаёт другой снимок изменяемых источников; она не обещает совпадение с опубликованными числами. Архивные манифесты без исходных CSV нельзя считать полноценным кэшем и перезаписывать новыми данными. Основной прогон — до 100 эпох, три seed, оба города, два информационных лага; CPU-обучение требует времени.

На этой машине полный сохранённый архив уже есть:

```powershell
.\.venv\Scripts\python.exe scripts/verify_spatiotemporal.py
.\.venv\Scripts\python.exe -m critical_processes.district_forecast --city chicago --out runs/chicago_forecast.csv
.\.venv\Scripts\python.exe -m critical_processes.district_forecast --city san_francisco --out runs/sf_forecast.csv
```

Полная независимая проверка пересобирает сетки из 16 исходных CSV, причинные признаки, выбранные параметры, прогнозы и все 40 сравнений. Выдача содержит округ, начало/конец интервала, ожидаемое количество, рабочие границы 95%, вероятность и сигнал всплеска. Модель количества и модель всплеска могут различаться. По умолчанию следующий интервал — **1 октября 2026, 00:00–06:00 местного времени**, граница архива; это не актуальный прогноз на ноябрь. `--asof` задаёт другую шестичасовую границу внутри доступного архива с достаточной предысторией. Фактическое будущее количество в выдаче пусто.

Готовые материалы: [полное сравнение](../results/spatiotemporal_forecast/analysis.md), [первые недели отложенных периодов](../results/spatiotemporal_forecast/forecast_examples.csv), [выдача 32 округов](../results/spatiotemporal_summary/next_interval_forecasts.csv). Основные ограничения — задержки регистрации и исторические правки. Количество, эмпирический всплеск и скрытая критичность n имеют разные метки и метрики.

## Что открыть для защиты

1. [REPORT.md](REPORT.md) — итоговый отчёт.
2. [DEFENSE.md](DEFENSE.md) — выступление и ответы.
3. [ACCEPTANCE.md](ACCEPTANCE.md) — выполнение исходных условий.
4. [EVIDENCE.md](EVIDENCE.md) — источники всех основных чисел.

Markdown-отчёт читается вместе с папкой `assets/`. При передаче преподавателю следует сохранять относительное расположение файлов. Код и полные результаты находятся в проекте. Персональные сведения о студенте можно добавить перед сдачей, если преподаватель их потребует.

## Где настоящие CSV

- [Сырые новые данные](../data/raw/expanded_20261110/) — 32 городских CSV.GZ и три дополнительных CODE core.
- [Подготовленные данные](../data/processed/expanded_20261110/) — отдельные `portal_*.csv.gz` и `code_*.csv.gz`, годовые манифесты, суточные количества и NPZ-кэши.
- [Общие годовые объёмы](../results/project_completion/yearly_data_coverage.csv) — экспозиция каждого года, включая неполный 2026.
- [Результаты реального применения](../results/empirical_transfer/decisions.csv) — оценки и пороговые срабатывания по датам.

CSV.GZ — обычный CSV, сжатый gzip. `pandas.read_csv` открывает его напрямую. Городские единицы — дела/отчёты, CODE — записи правонарушений; их абсолютные количества нельзя сравнивать как одинаковую меру риска. Данные 2026 охватывают январь–сентябрь.

## Проверить результаты на этой машине

В PowerShell из корня проекта. Активировать `.venv` и менять ExecutionPolicy не требуется:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/verify_extended_training.py
.\.venv\Scripts\python.exe scripts/verify_empirical_transfer.py
.\.venv\Scripts\python.exe scripts/verify_spatiotemporal.py
.\.venv\Scripts\python.exe scripts/complete_genealogy.py --verify
.\.venv\Scripts\python.exe scripts/finalize_indicators.py --verify
.\.venv\Scripts\python.exe scripts/verify_report.py
```

Проверки имитации, признаков и больших реальных рядов могут занять несколько минут. Они не загружают свежие данные вместо сохранённого снимка и не подбирают новые пороги.

Проверенное окружение сохранено в [environment.json](../reproducibility/environment.json), точные версии зависимостей — в [requirements-lock.txt](../requirements-lock.txt). Это снимок фактически использованного окружения, без привязки к абсолютному пути локальной установки проекта.

## Повторить завершённые вычисления

```powershell
.\.venv\Scripts\python.exe -m critical_processes.extended_training
.\.venv\Scripts\python.exe -m critical_processes.empirical_transfer
```

Для уже завершённых протоколов команды проверяют хеши и возвращают сохранённый результат. Если код, параметры или исходные веса изменены, нужен отдельный каталог и новый независимый тест, а не перезапись прежнего финального результата.

Источники и сборка новых данных:

```powershell
.\.venv\Scripts\python.exe scripts/download_expanded_data.py
.\.venv\Scripts\python.exe scripts/prepare_expanded_panel.py
```

Успешные сохранённые выгрузки используются как кэш. Это снимок на 10 ноября 2026, не команда обновления частично изменившегося источника поверх старого снимка. Для нового снимка задаются отдельные каталоги.

## Как трактовать детектор

Программный детектор реализован. Его строгий вариант даёт 5/600 чистых предупреждений в финальной симуляционной проверке; качество низкое. Номинальный вариант даёт 27/600 и имеет другую строгость калибровочного доверия. На реальных данных нет истинных меток: нулевое число срабатываний не равно нулевой ошибке и не доказывает отсутствие критичности.
