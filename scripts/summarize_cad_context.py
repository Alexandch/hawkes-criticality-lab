"""R16 source accounting and paired exploratory ablation figures."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from critical_processes.selected_training import sha,write_json


def main():
    root=Path(__file__).resolve().parents[1];out=root/'results/cad_context_detector';assets=root/'report/assets';table=pd.read_csv(out/'metrics.csv')
    primary=table.query("split=='reused2026' and scenario=='primary' and group=='all'")
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    methods=['base_nb','cad_nb','delay_nb'];labels=['NB2: парный контроль\nбез CAD','NB2: дополнительно\nмесячные CAD','Простой фон\nс коррекцией']
    fig,axes=plt.subplots(1,3,figsize=(13,4.4),layout='constrained')
    for ax,metric,title,limit in zip(axes,['precision','recall','fpr'],['Верные тревоги, %','Полнота, %','FPR, %'],[30,10,5]):
        rows=[primary[primary.method==m].iloc[0] for m in methods];values=np.array([r[metric] for r in rows])*100
        lower=np.array([r[metric+'_ci_low'] for r in rows])*100;upper=np.array([r[metric+'_ci_high'] for r in rows])*100
        ax.barh(range(3),values,color=['#929aa1','#315d8b','#669b85']);ax.errorbar(values,range(3),xerr=[values-lower,upper-values],fmt='none',ecolor='#222222',capsize=3)
        ax.set_yticks(range(3),labels if metric=='precision' else ['']*3);ax.set_ylim(2.5,-.5);ax.set_title(title);ax.axvline(limit,color='#b44343',linestyle='--');ax.grid(axis='x',alpha=.2)
        ax.set_xlim(0,max(limit*1.3,np.nanmax(upper)*1.15))
        for i,value in enumerate(values):
            if np.isfinite(value) and value>0:ax.text(value*.45,i,f'{value:.1f}',ha='center',va='center',fontsize=9,color='white',fontweight='bold')
            else:ax.text(.5,i,'нет тревог' if not np.isfinite(value) else '0.0',va='center',fontsize=9)
    fig.suptitle('R16: исследовательское сравнение на уже просмотренном январе–августе 2026',fontsize=12)
    first=assets/'cad_operating.png';fig.savefig(first,dpi=180);plt.close(fig)
    pairs=[]
    for split in ('development2025','reused2026'):
        for scenario in ('primary','delayed_stress'):
            subset=table[(table.split==split)&(table.scenario==scenario)&(table.group=='all')]
            base=subset[subset.method=='base_nb'].iloc[0];extra=subset[subset.method=='cad_nb'].iloc[0]
            pairs.append(dict(split=split,scenario=scenario,ap_difference=extra.ap-base.ap,brier_difference=extra.brier-base.brier,
                brier_ci_low=extra.brier_difference_ci_low,brier_ci_high=extra.brier_difference_ci_high))
    pairs=pd.DataFrame(pairs);pairs.to_csv(out/'paired_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained');labels=['2025: основной','2025: задержка','2026: основной','2026: задержка'];colors=['#315d8b','#669b85','#315d8b','#669b85']
    for ax,metric,title in zip(axes,['ap_difference','brier_difference'],['Разность AP: CAD − без CAD','Разность Brier: CAD − без CAD']):
        value=pairs[metric].to_numpy();ax.barh(range(4),value,color=colors)
        if metric=='brier_difference':ax.errorbar(value,range(4),xerr=[value-pairs.brier_ci_low,pairs.brier_ci_high-value],fmt='none',color='#222222',capsize=4)
        ax.axvline(0,color='#333333');ax.set_yticks(range(4),labels if metric=='ap_difference' else ['']*4);ax.set_ylim(3.5,-.5);ax.set_title(title);ax.grid(axis='x',alpha=.2)
        ax.set_xlabel('Больше — лучше' if metric=='ap_difference' else 'Меньше — лучше; 95% условный интервал')
    fig.suptitle('Парное сравнение признаков; оба периода уже использовались в проекте',fontsize=12)
    second=assets/'cad_paired.png';fig.savefig(second,dpi=180);plt.close(fig)
    data=[]
    for year in range(2021,2027):
        path=root/f'data/raw/cad_austin_20261122/cad_{year}.csv.gz.manifest.json';record=json.loads(path.read_text(encoding='utf-8'))
        data.append(dict(year=year,source_incidents=record['dispatched_incidents'],aggregate_rows=record['aggregate_rows']))
    pd.DataFrame(data).to_csv(out/'data_summary.csv',index=False)
    write_json(assets/'cad_manifest.json',dict(stage='R16',renderer_sha256=sha(Path(__file__)),
        source_sha256={p.relative_to(root).as_posix():sha(p) for p in [out/'metrics.csv',out/'frozen_plan.json']},
        assets=[dict(file=p.relative_to(root).as_posix(),sha256=sha(p)) for p in [first,second]]))
    print(primary[['method','true_positives','false_positives','precision','recall','fpr','ap','brier']].to_string(index=False));print(pairs.to_string(index=False))


if __name__=='__main__':main()
