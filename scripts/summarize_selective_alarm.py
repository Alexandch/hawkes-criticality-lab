"""R15 fixed comparisons and label-maturity diagnostics."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from critical_processes.selected_training import sha,write_json


def main():
    root=Path(__file__).resolve().parents[1];out=root/'results/selective_austin_detector';assets=root/'report/assets';table=pd.read_csv(out/'metrics.csv')
    main=table.query("scenario=='primary' and group=='all'")
    roles=['monthly_neural','selected','baseline'];labels=['NB2: прежнее\nмесячное правило','NB2: выбранное\nнедельное правило','Выбранный\nпростой фон']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(13,4.4),layout='constrained')
    for ax,metric,title,limit in zip(axes,['precision','recall','fpr'],['Верные тревоги, %','Полнота, %','FPR, %'],[30,10,5]):
        rows=[main[main.role==role].iloc[0] for role in roles];values=np.array([r[metric] for r in rows])*100
        lower=np.array([r[metric+'_ci_low'] for r in rows])*100;upper=np.array([r[metric+'_ci_high'] for r in rows])*100
        ax.barh(np.arange(3),values,color=['#929aa1','#315d8b','#669b85'])
        ax.errorbar(values,np.arange(3),xerr=[values-lower,upper-values],fmt='none',ecolor='#222222',capsize=3)
        ax.set_yticks(np.arange(3),labels if metric=='precision' else ['']*3);ax.set_ylim(2.5,-.5);ax.set_title(title)
        ax.axvline(limit,color='#b44343',linestyle='--');ax.grid(axis='x',alpha=.2);ax.set_xlim(0,max(limit*1.3,np.nanmax(upper)*1.15))
        for i,value in enumerate(values):
            if np.isfinite(value) and value>0:ax.text(value*.45,i,f'{value:.1f}',ha='center',va='center',fontsize=9,color='white',fontweight='bold')
            else:ax.text(.5,i,'нет тревог' if not np.isfinite(value) else '0.0',va='center',fontsize=9,color='#333333')
    fig.suptitle('Austin, январь–август 2026: отдельный контроль после фиксации правил',fontsize=12)
    first=assets/'selective_operating.png';fig.savefig(first,dpi=180);plt.close(fig)
    diagnostic=pd.read_csv(out/'maturity_diagnosis.csv');summary=diagnostic.groupby('maturity_days').agg(
        repeated_true_positives=('true_positives','sum'),repeated_unrecognized=('unrecognized_positives','sum'))
    summary['missed_fraction']=summary.repeated_unrecognized/summary.repeated_true_positives;summary.to_csv(out/'maturity_summary.csv')
    fig,ax=plt.subplots(figsize=(9,4.2),layout='constrained')
    for days,color in [(28,'#b44343'),(56,'#315d8b'),(90,'#669b85')]:
        sub=diagnostic[diagnostic.maturity_days==days];ax.plot(range(1,13),sub.missing_positive_fraction*100,'-o',label=f'Созревание {days} суток',color=color,markersize=4)
    ax.set_xticks(range(1,13));ax.set_xlabel('Месяц обновления в 2025 году');ax.set_ylabel('Ещё не распознанные превышения, %')
    ax.set_title('Диагностика по уже просмотренному 2025 году; окна перекрываются');ax.legend(fontsize=9);ax.grid(alpha=.2)
    second=assets/'selective_maturity.png';fig.savefig(second,dpi=180);plt.close(fig)
    write_json(assets/'selective_manifest.json',dict(stage='R15',renderer_sha256=sha(Path(__file__)),
        source_sha256={p.relative_to(root).as_posix():sha(p) for p in [out/'metrics.csv',out/'maturity_diagnosis.csv',out/'frozen_plan.json']},
        assets=[dict(file=p.relative_to(root).as_posix(),sha256=sha(p)) for p in [first,second]]))
    print(summary.to_string());print(main[['role','true_positives','false_positives','positives','precision','recall','fpr','blocks']].to_string(index=False))


if __name__=='__main__':main()
