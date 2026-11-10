from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NAMES={'seasonal':'Календарный фон','recent_calendar':'Фон + недавний уровень','temporal_hawkes':'Временная самовозбуждающаяся','spatial_hawkes':'Пространственная самовозбуждающаяся','neural_ensemble':'Нейросетевой ансамбль'}


def write_spatial_report(output,panel_dir):
    output,panel_dir=Path(output),Path(panel_dir)
    config=json.loads((output/'protocol.json').read_text(encoding='utf-8'));frozen=json.loads((output/'frozen_plan.json').read_text(encoding='utf-8'))
    table=pd.read_csv(output/'test_metrics.csv');effects=pd.read_csv(output/'paired_count_errors.csv');hours=config['bin_hours']
    lines=['# R10. Прогноз числа происшествий по районам и времени','',
        f'Цель — количество зарегистрированных происшествий выбранных категорий в следующем {hours}-часовом интервале для каждого полицейского округа. '
        'Это наблюдаемая целевая переменная, в отличие от скрытого коэффициента ветвления n. Всплеск определяется как строгое превышение 95% квантили '
        'исторических количеств того же района, дня недели и времени суток; квантиль оценивается только по обучению.','',
        'Использованы Chicago (22 округа) и San Francisco (10). Выбраны сопоставимые группы: кражи, кражи со взломом, грабежи, угон транспорта и нападения; '
        'Chicago Assault/Battery объединены в общую категорию нападений. Определения источников не тождественны; сравнение качества проводится внутри города. '
        'Считается одно дело/отчёт, а не потерпевший или каждая строка правонарушения. Неизвестные и конфликтующие районы исключаются с учётом в QA.','',
        '## Протокол и доступность данных','',
        'Обучение: 2019–2023 (первые 60 суток исключены для прогрева признаков). Выбор параметров: январь–июнь 2024. '
        'Калибровка тревог: июль–декабрь 2024. Веса, выбор модели и пороги фиксируются до расчёта итоговых метрик 2025 и января–сентября 2026. '
        'Случайного перемешивания будущих и прошлых временных строк нет. Внутри теста разрешены только прошлые наблюдения, по мере их появления.','',
        'Основной сценарий задаёт информационный лаг 7 суток: признаки не используют события последних семи суток до прогнозируемого интервала. '
        'Сценарий без лага показывает ретроспективный результат при более сильном предположении доступности. '
        'В San Francisco дополнительно используются реальные report_datetime: прошлые события становятся признаками лишь после регистрации, '
        'а обучающие, валидационные и калибровочные цели включают только отчёты, известные к датам соответствующего решения. '
        'Архив скачан одной современной версией, а не историческими версиями: доступность данных реконструирована не полностью. '
        'Поздние регистрации частично учтены по report_datetime SF, но исторические правки неизвестны; у Chicago время первичной регистрации отсутствует. '
        'Поэтому даже лаг 7 суток не доказывает операционную применимость в реальном времени.','']
    for city in config['city_names']:
        meta=json.loads((panel_dir/(city+'.json')).read_text(encoding='utf-8'))
        text=f"{city}: {meta.get('incidents','—'):,} дел/отчётов" if isinstance(meta.get('incidents'),int) else city
        text+=f", доля меток в полночь {meta.get('fraction_midnight',0):.1%}."
        if meta.get('report_delay_fraction_over7days') is not None:
            text+=f" Доля регистрации позже 7 суток — {meta['report_delay_fraction_over7days']:.1%}, позже 6 часов — {meta['report_delay_fraction_over6h']:.1%}."
        lines.append(text+'\n')
    lines += ['## Модели','',
        'Сравниваются сезонный районный фон, его адаптация к наблюдаемому уровню последних 28 суток, временная и пространственная дискретные '
        'самовозбуждающиеся модели и нейросетевой ансамбль. Фон учитывает район, день недели, слот суток и месяц. '
        'Граф районов строится по трём ближайшим центрам событий обучения 2019–2023; он фиксирован и не означает точное соседство административных границ.','',
        'В самовозбуждающейся модели среднее имеет вид μ[d,t]=c B[d,t]+a X[d,t]+b Σ W′[d,j]X[j,t]. '
        'Память экспоненциальная и сдвинута на информационный лаг. Матрица W′ масштабируется обучающими частотами районов. '
        'c оценивается свободно, a,b неотрицательны, a+b≤0.99. При постоянном фоне это ограничивает спектральный радиус дискретного ядра; '
        'коэффициенты не выдаются за истинный непрерывный n прежних экспериментов.','',
        'Нейросеть использует embedding района, календарь и причинные суммы/отношения собственных, соседних и общегородских прошлых количеств. '
        'Три заранее заданных seed, Poisson NLL, эпоха по валидации, среднее всех сетей. Для вероятностей всплеска и интервалов у каждого метода '
        'по обучению подбирается рабочая отрицательно-биномиальная дисперсия. Номинальное покрытие проверяется фактически.','',
        'Модель числа событий выбирается по валидационному NB NLL, модель всплеска — по validation average precision. '
        'Для тревоги калибруется 5% ложноположительных районно-временных ячеек среди отрицательных ячеек. '
        'Это другая единица контроля, чем хотя бы одна тревога на целую траекторию в R4–R9. Дрейф распределения может нарушить 5% на тесте; '
        'сохраняются общий и максимальный районный FPR, без заявления гарантии для будущих городов.','',
        '## Отложенные результаты, информационный лаг 7 суток','',
        '| Город / период | Метод | MAE количества | NB NLL | AP всплеска | Полнота всплеска | FPR ячеек |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in table[table.information_lag_days==7].itertuples():
        star=' *' if row.count_primary else ''
        lines.append(f'| {row.city} / {row.split} | {NAMES[row.method]}{star} | {row.mae:.3f} | {row.nb_nll:.3f} | {row.spike_average_precision:.3f} | {row.spike_recall:.1%} | {row.spike_false_positive_rate:.1%} |')
    lines += ['','Звёздочка обозначает заранее выбранный метод количества, не победителя по тесту. '
        'Для AP базовый ориентир — фактическая доля всплесков; она записана в `test_metrics.csv`. '
        'Сценарий без информационного лага, покрытие/ширина интервалов, районные метрики и захват событий в наиболее нагруженных районах сохранены отдельно.','',
        '| Город / период | Выбранный count-метод | Разность MAE с сезонным фоном | 95% блочный интервал |',
        '|---|---|---:|---|']
    for row in effects[effects.information_lag_days==7].itertuples():
        lines.append(f'| {row.city} / {row.split} | {row.primary} | {row.primary_minus_seasonal_mae:.4f} | [{row.block_bootstrap_ci95_lower:.4f}; {row.block_bootstrap_ci95_upper:.4f}] |')
    lines += ['','Интервалы получены пересэмплированием полных 14-суточных блоков вместе со всеми районами и условны на выбранных моделях. '
        'Отрицательная разность означает меньшую ошибку. Округа и соседние временные интервалы не считаются независимыми повторениями.','',
        '![Сравнение пространственно-временных прогнозов](count_comparison.png)','',
        '![Первый отложенный недельный пример](district_time_example.png)','',
        'График примера использует заранее заданную первую неделю января 2025. `forecast_examples.csv` содержит первые семь суток каждого теста: '
        'город, район, начало/конец будущего интервала, ожидаемое и фактическое количество, вероятность и факт всплеска. '
        'Здесь нет обещания точного момента отдельного преступления и нет вывода о математической критичности n.','',
        '## Воспроизведение','',
        '```powershell','.\\.venv\\Scripts\\python.exe scripts/prepare_spatiotemporal_data.py',
        '.\\.venv\\Scripts\\python.exe -m critical_processes.spatial_experiment',
        '.\\.venv\\Scripts\\python.exe scripts/verify_spatiotemporal.py','```','']
    (output/'analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for row,city in enumerate(config['city_names']):
        for column,split in enumerate(('test2025','test2026')):
            values=table[(table.city==city)&(table.split==split)&(table.information_lag_days==7)].set_index('method').loc[list(NAMES)]
            ax=axes[row,column];ax.bar(np.arange(5),values.mae,color=['#7d8b9b','#4b9b9b','#59965b','#ad8b3f','#6657a3'])
            ax.set_xticks(np.arange(5),['Calendar','Recent','Temporal','Spatial','Neural'],rotation=20)
            ax.set_title(f'{city}, {split}, лаг 7 суток');ax.set_ylabel('MAE событий / район / интервал');ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(output/'count_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,7))
    for row,city in enumerate(config['city_names']):
        method=frozen['cities'][city]['7']['count_primary']
        with np.load(output/city/'predictions_lag7.npz') as data:
            dates=pd.DatetimeIndex(data['time_ns'].astype('datetime64[ns]'));use=(dates>=pd.Timestamp('2025-01-01'))&(dates<pd.Timestamp('2025-01-08'))
            observed,mean=data['counts'][use].T,data[method][use].T;labels=data['districts']
            vmax=max(float(np.quantile(observed,.99)),float(np.quantile(mean,.99)),1.)
            for column,array in enumerate((observed,mean)):
                ax=axes[row,column];im=ax.imshow(array,aspect='auto',vmin=0,vmax=vmax,cmap='YlOrRd')
                ax.set_yticks(np.arange(len(labels)),labels,fontsize=6);ax.set_xlabel(f'{hours}-часовые интервалы первой недели 2025');ax.set_ylabel('Полицейский округ')
                ax.set_title(city+(': фактические количества' if column==0 else ': прогноз '+method));fig.colorbar(im,ax=ax,pad=.01)
    fig.tight_layout();fig.savefig(output/'district_time_example.png',dpi=160);plt.close(fig)
