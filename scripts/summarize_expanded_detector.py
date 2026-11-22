"""Create R12 figures and an explicit accounting of training versus test cases."""
from __future__ import annotations
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from critical_processes.selected_training import sha,write_json
from critical_processes.expanded_experiment import CITIES,METHODS
from critical_processes.expanded_forecast import chronological_masks,quarterly_release_bins
from critical_processes.offline_forecast import load_categories,available_targets


def main():
    root=Path(__file__).resolve().parents[1];out=root/'results/expanded_district_detector';assets=root/'report/assets'
    summary=[]
    for i,city in enumerate(CITIES):
        directory=root/('data/processed/offline_20261110' if i<2 else 'data/processed/expanded_districts_20261119')
        panel=load_categories(directory/(city+'.npz'));masks=chronological_masks(panel.dates,city)
        if city=='nyc':panel=replace(panel,report_bins=quarterly_release_bins(panel.report_bins,panel.time_ns[0],45))
        counts={}
        for key,cutoff in [('train','2024-01-01'),('validation','2024-07-01'),('probability_calibration','2025-01-01'),('alarm_calibration','2025-01-01')]:
            known=available_targets(panel,'2025-01-01' if city=='nyc' else cutoff,0 if city=='nyc' else 3);counts[key+'_cases']=int(known[masks[key],:,0].sum())
        counts.update({key+'_cases':int(panel.counts[masks[key],:,0].sum()) for key in ('test2025','test2026')})
        summary.append(dict(city=city,districts=len(panel.districts),source_panel_cases=int(panel.counts[:,:,0].sum()),
            common_scope_cases=int(panel.counts[panel.dates<pd.Timestamp('2026-10-01'),:,0].sum()),**counts))
    pd.DataFrame(summary).to_csv(out/'data_summary.csv',index=False)
    headline=pd.read_csv(out/'headline.csv');main=headline[(headline.scenario=='primary')&(headline.group=='all')].copy()
    main.to_csv(out/'headline_all.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.5),layout='constrained')
    for ax,(column,title,target) in zip(axes,[('precision','Верные среди тревог',30),('recall','Найденные всплески',10),('fpr','Тревоги среди интервалов без всплеска',5)]):
        for j,split in enumerate(('test2025','test2026')):
            values=main[main.split==split].set_index('city').loc[list(CITIES)]
            ax.bar(np.arange(4)+(j-.5)*.35,100*values[column],width=.35,label='2025' if j==0 else 'Янв–сен 2026')
        ax.axhline(target,color='#a22',linestyle='--',linewidth=1,label=f'Ориентир {target}%')
        ax.set_xticks(np.arange(4),['Chicago','SF','NYC','Seattle'],rotation=20);ax.set_title(title);ax.set_ylabel('%');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(fontsize=8);fig.suptitle('R12: выбранные по валидации модели, район × 6 часов\nРетроспективная проверка; данные 2026 предварительные')
    first=assets/'expanded_operating.png';fig.savefig(first,dpi=180);plt.close(fig)
    metrics=pd.read_csv(out/'metrics.csv');fig,axes=plt.subplots(2,2,figsize=(13.5,8),layout='constrained')
    labels=['Календарь','Адаптивный фон','Локальная НС','Общая НС','Общая НС + погода','Бустинг 15','Бустинг 31']
    for ax,city in zip(axes.ravel(),CITIES):
        subset=metrics[(metrics.city==city)&(metrics.scenario=='primary')&(metrics.group=='all')]
        for j,split in enumerate(('test2025','test2026')):
            values=subset[subset.split==split].set_index('method').loc[list(METHODS)]
            ax.bar(np.arange(len(METHODS))+(j-.5)*.35,values.ap,width=.35,label=split.replace('test',''))
        ax.set_title('New York' if city=='nyc' else city.replace('_',' ').title());ax.set_xticks(np.arange(7),labels,rotation=35,ha='right');ax.set_ylabel('Average precision');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0,0].legend();fig.suptitle('R12: вклад объединения городов, погоды и класса модели\nПоказаны все заранее заданные методы; выбор по тесту не выполняется')
    second=assets/'expanded_ablation.png';fig.savefig(second,dpi=180);plt.close(fig)
    write_json(assets/'expanded_manifest.json',dict(stage='R12',source_sha256={str(p.relative_to(root)).replace('\\','/'):sha(p) for p in [out/'headline.csv',out/'metrics.csv',out/'protocol.json']},
        renderer_sha256=sha(Path(__file__)),assets=[dict(file=str(p.relative_to(root)).replace('\\','/'),sha256=sha(p)) for p in [first,second]]))
    print(pd.DataFrame(summary).to_string(index=False));print(main[['city','split','method','precision','recall','fpr','practical_screen_passed']].to_string(index=False))


if __name__=='__main__':main()
