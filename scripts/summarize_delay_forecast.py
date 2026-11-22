"""R14 descriptive accounting and figures; no held-out parameter selection."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from critical_processes.offline_forecast import load_categories
from critical_processes.selected_training import sha,write_json


def main():
    root=Path(__file__).resolve().parents[1];out=root/'results/delay_forecast_detector';assets=root/'report/assets'
    panel=load_categories(root/'data/processed/delay_forecast_20261122/final/austin.npz');rows=[]
    for split,start,end in [('development_total','2021-04-01','2025-01-01'),('network_train','2021-10-01','2024-01-01'),
            ('checkpoint_validation','2024-02-01','2024-04-01'),('operating_selection','2024-07-01','2024-10-01'),
            ('confirmation','2024-10-01','2024-12-01'),('final_test','2025-01-01','2026-01-01')]:
        take=(panel.dates>=pd.Timestamp(start))&(panel.dates<pd.Timestamp(end))
        rows.append(dict(split=split,start=start,end_exclusive=end,cases=int(panel.counts[take,:,0].sum()),
            sector_cells=int(take.sum()*len(panel.districts)),sectors=len(panel.districts)))
    pd.DataFrame(rows).to_csv(out/'data_summary.csv',index=False)
    table=pd.read_csv(out/'metrics.csv');primary=table[(table.scenario=='primary')&(table.group=='all')]
    roles=['selected','baseline','weather_ablation','reporting_ablation'];labels=['Выбранная НС\nBCE + прогноз погоды','Простой фон\nс коррекцией задержки','НС NB2\nбез погоды','Простой фон\nбез коррекции']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(13,4.4),layout='constrained')
    for ax,metric,title,limit in zip(axes,['precision','recall','fpr'],['Верные тревоги, %','Полнота, %','FPR, %'],[30,10,5]):
        points=[primary[primary.role==role].iloc[0] for role in roles];values=np.array([r[metric] for r in points])*100
        lower=np.array([r[metric+'_ci_low'] for r in points])*100;upper=np.array([r[metric+'_ci_high'] for r in points])*100
        ax.barh(np.arange(4),values,color=['#315d8b','#669b85','#b78a3c','#929aa1'])
        ax.errorbar(values,np.arange(4),xerr=[values-lower,upper-values],fmt='none',ecolor='#222222',capsize=3)
        ax.set_yticks(np.arange(4),labels if metric=='precision' else ['']*4);ax.invert_yaxis();ax.set_title(title)
        ax.axvline(limit,color='#b44343',linestyle='--',label='Заданный ориентир');ax.grid(axis='x',alpha=.2)
        ax.set_xlim(0,max(limit*1.2,float(upper.max())*1.3))
        for i,v in enumerate(values):ax.text(v*.45,i,f'{v:.1f}',va='center',ha='center',fontsize=9,color='white',fontweight='bold')
    axes[0].legend(loc='lower right',fontsize=8)
    fig.suptitle('Austin 2025: 13 140 ячеек, 1 497 превышений; 27 временных блоков',fontsize=12)
    first=assets/'delay_forecast_operating.png';fig.savefig(first,dpi=180);plt.close(fig)
    months=pd.read_csv(out/'monthly_metrics.csv');fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    for role,label,color in zip(roles[:3],['Выбранная НС BCE + погода','Простой фон с коррекцией','НС NB2 без погоды'],['#315d8b','#669b85','#b78a3c']):
        sub=months[(months.scenario=='primary')&(months.group=='all')&(months.role==role)]
        for ax,metric in zip(axes,['fpr','recall']):ax.plot(np.arange(1,13),100*sub[metric],'-o',label=label,color=color,markersize=4)
    for ax,title,limit in zip(axes,['FPR по месяцам, %','Полнота по месяцам, %'],[5,10]):
        ax.axhline(limit,color='#b44343',linestyle='--');ax.set_title(title);ax.set_xlabel('Месяц 2025 года');ax.set_xticks(range(1,13));ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Заранее заданное ежемесячное обновление по доступному прошлому',fontsize=12)
    second=assets/'delay_forecast_monthly.png';fig.savefig(second,dpi=180);plt.close(fig)
    write_json(assets/'delay_forecast_manifest.json',dict(stage='R14',renderer_sha256=sha(Path(__file__)),
        source_sha256={p.relative_to(root).as_posix():sha(p) for p in [out/'metrics.csv',out/'monthly_metrics.csv',out/'frozen_plan.json']},
        assets=[dict(file=p.relative_to(root).as_posix(),sha256=sha(p)) for p in [first,second]]))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
