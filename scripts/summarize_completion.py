from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from critical_processes.selected_training import sha,write_json


def main():
    output=Path('results/project_completion');output.mkdir(exist_ok=True)
    source=Path('results/extended_neural')
    metrics=json.loads((source/'metrics.json').read_text(encoding='utf-8'))
    trajectory=pd.read_csv(source/'trajectory_metrics.csv')
    rng=np.random.default_rng(20261130);pairs=[]
    for task in ('state','warning'):
        new=trajectory[(trajectory.stream==task+'__ensemble')&(trajectory.policy=='family_confident')].set_index('trajectory_id')
        old=trajectory[trajectory.stream=='old_primary_'+task].set_index('trajectory_id').reindex(new.index)
        if task=='state':
            a,b=new.true_positive.to_numpy(),old.true_positive.to_numpy();den=new.positive_windows.to_numpy()
        else:
            a,b=new.clean_detection.fillna(False).to_numpy(dtype=float),old.clean_detection.fillna(False).to_numpy(dtype=float);den=new.has_positive.to_numpy(dtype=int)
        draws=[]
        for _ in range(3000):
            index=rng.integers(0,len(a),len(a));d=den[index].sum()
            if d:draws.append(float((a[index]-b[index]).sum()/d))
        lo,hi=np.quantile(draws,[.025,.975])
        pairs.append(dict(task=task,new_minus_old=float((a-b).sum()/den.sum()),paired_bootstrap_ci95_lower=lo,paired_bootstrap_ci95_upper=hi,
            bootstrap_samples=3000,unit='whole trajectory, same new test; conditional on fitted models and calibrations'))
    pd.DataFrame(pairs).to_csv(output/'paired_detector_differences.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    labels=['R4 frozen','R9 strict','R9 nominal']
    for column,task in enumerate(('state','warning')):
        keys=['old_primary_'+task,task+'__ensemble__family_confident',task+'__ensemble__family_nominal_sensitivity']
        records=[metrics[key] for key in keys]
        field='window_sensitivity' if task=='state' else 'clean_event_detection_probability'
        interval='window_sensitivity_trajectory_bootstrap_ci95' if task=='state' else 'clean_detection_ci95'
        values=np.array([r[field] for r in records])*100
        lower=np.array([r[interval][0] for r in records])*100;upper=np.array([r[interval][1] for r in records])*100
        axes[0,column].bar(labels,values,yerr=[values-lower,upper-values],capsize=3,color=['#74869b','#29778e','#d39e36'])
        axes[0,column].set_title('TPR состояния, %' if task=='state' else 'Чистое предупреждение, %')
        for index,value in enumerate(values):axes[0,column].text(index,value,f'{value:.2f}',ha='center',va='bottom',fontsize=8)
        false=np.array([r['maximum_family_false_alarm_probability'] for r in records])*100
        axes[1,column].bar(labels,false,color=['#74869b','#29778e','#d39e36'])
        axes[1,column].axhline(5,color='black',ls=':',label='Цель 5%')
        axes[1,column].set_title('Макс. семейная доля ложных тревог, %');axes[1,column].legend(fontsize=8)
    for ax in axes.ravel():ax.grid(axis='y',alpha=.2)
    fig.suptitle('Новая общая финальная выборка: 3 900 потоков, 600 переходов')
    fig.tight_layout();fig.savefig(output/'extended_detector_comparison.png',dpi=160);plt.close(fig)
    panel=json.loads(Path('data/processed/expanded_20261110/panel_manifest.json').read_text(encoding='utf-8'))
    yearly=pd.DataFrame([dict(source=m['source'],unit=m['unit'],**row) for m in panel['streams'] for row in m['yearly']])
    yearly.to_csv(output/'yearly_data_coverage.csv',index=False)
    manifests=panel['streams']
    summary=dict(portal_raw_rows=sum(m['raw_rows'] for m in manifests if m['source'].startswith('portal:')),
        portal_prepared_cases=sum(m['events'] for m in manifests if m['source'].startswith('portal:')),
        code_prepared_offense_records=sum(m['events'] for m in manifests if m['source'].startswith('CODE:')),
        code_prepared_cities=sum(m['source'].startswith('CODE:') for m in manifests),
        code_eligible_cities=sum(m['source'].startswith('CODE:') and m['eligible_for_full_history'] for m in manifests))
    write_json(output/'data_totals.json',summary)
    import matplotlib.dates as mdates
    real=pd.read_csv('results/empirical_transfer/decisions.csv')
    real_blocks=pd.read_csv('results/empirical_transfer/blocks.csv')
    frozen=json.loads((source/'frozen_plan.json').read_text(encoding='utf-8'))
    latest=real_blocks[(real_blocks.period=='latest')&real_blocks.source.str.startswith('portal:')]
    fig,axes=plt.subplots(len(latest),2,figsize=(12,2.6*len(latest)),squeeze=False)
    for row,block in enumerate(latest.itertuples()):
        d=real[real.block_id==block.block_id];dates=pd.to_datetime(d.decision_date)
        for name,label in [('calendar_n90','MLE 90'),('calendar_n360','MLE 360'),('auxiliary_n','NN, оценка n')]:
            axes[row,0].plot(dates,d[name],label=label,marker='.',ms=4)
        axes[row,0].axhline(.9,ls=':',color='gray');axes[row,0].set_title(block.source+': оценки, истинный n неизвестен',fontsize=10)
        for task,label in [('state','Состояние'),('warning','Предупреждение')]:
            line,=axes[row,1].plot(dates,d['new_'+task+'_score'],label=label,marker='.',ms=4)
            axes[row,1].axhline(frozen['thresholds'][task+'__ensemble__family_confident'],color=line.get_color(),ls=':',alpha=.7)
        axes[row,1].set_title('Scores и замороженные пороги R9',fontsize=10)
        for ax in axes[row]:
            locator=mdates.AutoDateLocator(minticks=3,maxticks=6);ax.xaxis.set_major_locator(locator);ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            ax.set_ylim(-.02,1.03);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/'fresh_city_scores.png',dpi=160);plt.close(fig)
    for row in pairs:print(row)
    print(summary)
    write_json(output/'provenance.json',dict(source_sha256={name:sha(name) for name in
        ('results/extended_neural/complete.json','results/empirical_transfer/complete.json','results/genealogical_clusters/complete.json',
         'data/processed/expanded_20261110/panel_manifest.json','results/empirical_transfer/decisions.csv','scripts/summarize_completion.py')},
        artifact_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!='provenance.json'}))


if __name__=='__main__':main()
