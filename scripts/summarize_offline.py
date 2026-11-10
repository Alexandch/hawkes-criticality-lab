"""Describe the frozen R11 result; never select a new model from October."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from critical_processes.selected_training import sha,write_json
from critical_processes.offline_forecast import load_categories


def main():
    source=Path('results/offline_readiness');out=Path('results/offline_summary');out.mkdir(exist_ok=True)
    table=pd.read_csv(source/'metrics.csv');frozen=json.loads((source/'frozen_plan.json').read_text(encoding='utf-8'))
    journal=json.loads((source/'forecast_journal.json').read_text(encoding='utf-8'));headlines=[];calibration=[];reliability=[]
    for city in table.city.unique():
        for group in table.group.unique():
            for split in table.split.unique():
                use=table[(table.city==city)&(table.group==group)&(table.split==split)&(table.scenario=='primary')]
                if use.empty:continue
                count=use[use.count_primary].iloc[0];rise=use[use.rise_primary].iloc[0]
                recent=use[use.method=='recent_calendar'].iloc[0];neural=use[use.method=='neural_nb'].iloc[0]
                headlines.append(dict(city=city,group=group,split=split,label_status=count.label_status,count_method=count.method,count_mae=count.mae,
                    recent_mae=recent.mae,neural_mae=neural.mae,rise_method=rise.method,prevalence=rise.prevalence,recall=rise.recall,
                    precision=rise.precision,fpr=rise.fpr,maximum_district_fpr=rise.maximum_district_fpr,true_positives=rise.true_positives,
                    false_positives=rise.false_positives,positives=rise.positives,ap=rise.ap,brier=rise.brier))
    headline=pd.DataFrame(headlines);headline.to_csv(out/'headline.csv',index=False)
    for city in table.city.unique():
        panel=load_categories(Path('data/processed/offline_20261110')/(city+'.npz'))
        for entry in journal['entries']:
            if entry['city']!=city:continue
            plan=frozen['cities'][city][entry['asof']]
            with np.load(source/entry['file']) as data:
                index=np.searchsorted(panel.time_ns,data['time_ns']);y=panel.counts[index];labels=y>=data['cutoff']
                for g,group in enumerate(panel.groups):
                    for method in ('seasonal','recent_calendar','neural_nb','neural_direct'):
                        raw=data['raw_'+method][:,:,g];probability=data['risk_'+method][:,:,g];truth=labels[:,:,g]
                        calibration.append(dict(city=city,scenario=entry['scenario'],month=entry['month'],group=str(group),method=method,
                            raw_brier=float(((raw-truth)**2).mean()),calibrated_brier=float(((probability-truth)**2).mean()),
                            mean_probability=float(probability.mean()),observed_frequency=float(truth.mean()),cells=truth.size))
                if entry['scenario']=='primary' and (entry['month'].startswith('2026-') and entry['month']<='2026-10'):
                    method=plan['rise_primary']['all'];values=data['risk_'+method][:,:,0];truth=labels[:,:,0]
                    if entry['month']=='2026-10':values=values[:56];truth=truth[:56]
                    label='new_october_first14' if entry['month']=='2026-10' else 'development2026'
                    edges=np.array([0,.01,.02,.05,.1,.2,.5,1.000001]);bins=np.digitize(values,edges)-1
                    for number in range(len(edges)-1):
                        mask=bins==number
                        if mask.any():reliability.append(dict(city=city,split=label,bin=number,cells=int(mask.sum()),probability_sum=float(values[mask].sum()),positive_count=int(truth[mask].sum())))
    pd.DataFrame(calibration).to_csv(out/'probability_calibration.csv',index=False)
    reliable=pd.DataFrame(reliability).groupby(['city','split','bin'],as_index=False)[['cells','probability_sum','positive_count']].sum()
    reliable['mean_probability']=reliable.probability_sum/reliable.cells;reliable['observed_frequency']=reliable.positive_count/reliable.cells
    reliable.to_csv(out/'reliability.csv',index=False)
    effects=pd.read_csv(source/'paired_differences.csv');small=effects.blocks<4
    effects.loc[small,['mae_ci_low','mae_ci_high','brier_ci_low','brier_ci_high']]=np.nan
    effects['interval_status']=np.where(small,'insufficient_independent_blocks','conditional_block_bootstrap')
    effects.to_csv(out/'interpretable_differences.csv',index=False)
    periods=['development2026','new_october_first14'];cities=list(table.city.unique())
    labels=['Адаптивный фон','Нейросеть'];colors=['#368f8b','#6b58a7']
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for row,city in enumerate(cities):
        for column,split in enumerate(periods):
            ax=axes[row,column];subset=table[(table.city==city)&(table.scenario=='primary')&(table.group=='all')&(table.split==split)].set_index('method')
            ax.bar(labels,[subset.loc['recent_calendar','mae'],subset.loc['neural_nb','mae']],color=colors)
            ax.set_title(city+' / '+('январь–сентябрь 2026' if column==0 else '1–14 октября, предварительно'));ax.set_ylabel('MAE событий / округ / 6 часов');ax.grid(axis='y',alpha=.2)
    fig.suptitle('R11: количество, основной сценарий доступности');fig.tight_layout();fig.savefig(out/'offline_count.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,city in zip(axes,cities):
        subset=table[(table.city==city)&(table.scenario=='primary')&(table.group=='all')&(table.split=='new_october_first14')]
        for _,record in subset.iterrows():
            ax.scatter(100*record.recall,100*(record.precision if pd.notna(record.precision) else 0),s=60)
            ax.annotate(record.method,(100*record.recall,100*(record.precision if pd.notna(record.precision) else 0)),xytext=(4,4),textcoords='offset points',fontsize=8)
        ax.set_title(city);ax.set_xlabel('Полнота существенного превышения, %');ax.set_ylabel('Доля верных тревог, %');ax.grid(alpha=.2)
        ax.set_xlim(0,max(10,145*subset.recall.max()));ax.set_ylim(0,max(35,125*subset.precision.fillna(0).max()))
    fig.suptitle('R11: 1–14 октября, новая метка и предварительные исходы');fig.tight_layout();fig.savefig(out/'offline_alarms.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,city in zip(axes,cities):
        for split,label in zip(periods,['Разработка: январь–сентябрь','Контроль: 1–14 октября']):
            subset=reliable[(reliable.city==city)&(reliable.split==split)]
            ax.plot(subset.mean_probability,subset.observed_frequency,'o-',label=label)
        ax.plot([0,1],[0,1],'--',color='gray',linewidth=1);ax.set_title(city);ax.set_xlabel('Средняя прогнозируемая вероятность');ax.set_ylabel('Наблюдаемая частота');ax.set_xlim(0,.5);ax.set_ylim(0,.5);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('R11: проверка вероятностей выбранных методов; ячейки зависимы');fig.tight_layout();fig.savefig(out/'offline_reliability.png',dpi=160);plt.close(fig)
    text=['# R11. Проверка перед внедрением без институционального пилота','',
        'Результаты получены самостоятельным ретроспективным прогоном. Модели, нормировка, эпохи, калибровка и районные пороги фиксируются по прошлым блокам. 2025 — сентябрь 2026 уже просматривались в R10; они не считаются новым независимым тестом.',
        '', 'Новая цель: число происшествий и N ≥ ceil(B_adapt + max(2,0.5 B_adapt)) на округ × 6 часов. Это существенное превышение доступного фонового уровня, не прежняя 95% историческая квантиль R10 и не скрытый n. Поэтому проценты полноты R10/R11 нельзя напрямую объявлять улучшением одного детектора.',
        '', 'Основная доступность — инцидент старше 7 суток; SF дополнительно report_datetime + условные 3 суток публикации. Стресс — 14 и 7 суток соответственно. Даты публикации исторически не восстановлены. Обучающие/валидационные/калибровочные цели имеют 28 суток на накопление регистраций, но это не гарантирует окончательную полноту.',
        '', 'Новый октябрьский снимок предварительный. Основной срез 1–14 октября имеет всего один 14-суточный блок: информативный бутстреп-интервал по нему невозможен. В интерпретируемой таблице соответствующие границы оставлены пустыми. Полный октябрь сохранён отдельно с более незрелыми метками.',
        '', '## Основные результаты, все выбранные категории', '',
        '| Город / период | Count-метод | MAE | MAE адаптивного фона | Метод превышения | Полнота | Precision | FPR | Верных / ложных тревог |',
        '|---|---|---:|---:|---|---:|---:|---:|---:|']
    for _,record in headline[(headline.group=='all')&headline.split.isin(periods)].iterrows():
        precision='нет тревог' if pd.isna(record.precision) else f'{100*record.precision:.1f}%'
        text.append(f'| {record.city} / {record.split} | {record.count_method} | {record.count_mae:.3f} | {record.recent_mae:.3f} | {record.rise_method} | {100*record.recall:.1f}% | {precision} | {100*record.fpr:.1f}% | {int(record.true_positives)} / {int(record.false_positives)} |')
    text += ['', 'Count- и rise-методы выбраны на валидации каждого винтажа. Порог тревоги оценивается на отдельной части калибровки каждого округа и группы, с целью 5% FPR отрицательных ячеек. Это не означает 95% верных тревог или сохранение 5% при дрейфе. Абсолютные числа тревог и частота цели показаны в CSV.',
        '', '![Количество](offline_count.png)','', '![Тревоги](offline_alarms.png)','', '![Вероятности](offline_reliability.png)',
        '', 'Группы property и assault_robbery разбираются отдельно в headline.csv и исходных таблицах; группа all включает обе и не является независимым дополнительным повторением. Месячные флаги дрейфа диагностические, не автоматические команды для оперативных служб.',
        '', '## Что реализовано и что остаётся', '',
        'Реализованы отдельная наблюдаемая цель, категории, причинные признаки с известным календарём, обновление по времени, multihead-сеть, калибровка вероятностей/районных порогов, журнал прогнозов без исходов, задержки и стресс-сценарий, районный/месячный контроль ошибок и реальный журнал первых наблюдений публичных строк.',
        '', 'Статус — исследовательский прототип. Реального пилота нет. Нельзя подтвердить пользу вмешательства, полностью восстановить доступность архива или объявить готовность к оперативным решениям. Самостоятельное наблюдение публичных обновлений начато 10 ноября; оно не подменяет исторические даты публикации.',
        '', 'Команды и ограничения — report/OFFLINE_GUIDE.md и report/OFFLINE_PROTOCOL.md. Источники чисел — results/offline_readiness; текущая сводка не меняет модели после просмотра октября.']
    (out/'analysis.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    write_json(out/'provenance.json',dict(source_sha256={'results/offline_readiness/complete.json':sha(source/'complete.json'),'scripts/summarize_offline.py':sha(Path(__file__))},
        artifact_sha256={path.name:sha(path) for path in out.iterdir() if path.is_file() and path.name!='provenance.json'}))
    print(headline[(headline.group=='all')&headline.split.isin(periods)].to_string(index=False))


if __name__=='__main__':main()
