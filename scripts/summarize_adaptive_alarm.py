"""Figures for the bounded R13 attempt; new-control outcomes never select a rule."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from critical_processes.selected_training import sha,write_json


def main():
    root=Path(__file__).resolve().parents[1];out=root/'results/adaptive_seattle_detector';assets=root/'report/assets'
    metrics=pd.read_csv(out/'metrics.csv');rows=metrics[(metrics.scenario=='primary')&(metrics.group=='all')]
    rows.to_csv(out/'headline.csv',index=False)
    roles=['static_neural','rolling_neural','static_adaptive','rolling_baseline'];labels=['НС R12','НС, обновление','Фон R12','Фон, обновление']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.5),layout='constrained')
    for ax,(column,title,target) in zip(axes,[('precision','Верные среди тревог',30),('recall','Найденные превышения',10),('fpr','Тревоги среди ячеек без превышения',5)]):
        for j,split in enumerate(('new_october_all','new_october_first21')):
            values=rows[rows.split==split].set_index('role').loc[roles]
            ax.bar(np.arange(4)+(j-.5)*.35,100*values[column],width=.35,label='Весь октябрь' if j==0 else '1–21 октября')
        ax.axhline(target,color='#a22',linestyle='--',linewidth=1,label=f'Ориентир {target}%');ax.set_xticks(np.arange(4),labels,rotation=25,ha='right')
        ax.set_title(title);ax.set_ylabel('%');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(fontsize=8);fig.suptitle('R13: Seattle, новый октябрьский контроль\nОдин месяц, предварительные метки; выбранное правило совпало с обновляемой НС')
    first=assets/'adaptive_october.png';fig.savefig(first,dpi=180);plt.close(fig)
    development=pd.read_csv(out/'development_metrics.csv');combined=development[development.split=='combined'];periods=development[development.split!='combined']
    fig,ax=plt.subplots(figsize=(8.5,5.2),layout='constrained')
    for method,color,label in [('local_neural','#2171b5','Нейросеть'),('recent_calendar','#238b45','Адаптивный фон'),('seasonal','#cb181d','Календарь')]:
        part=combined[combined.method==method]
        for _,row in part.iterrows():
            safe=(periods[periods.id==row.id].fpr<=.05).all()
            ax.scatter(100*row.recall,100*row.precision,color=color,marker='o' if safe else 'x',s=65,alpha=.8)
        ax.scatter([],[],color=color,label=label,s=50)
    frozen=json.loads((out/'frozen_plan.json').read_text(encoding='utf-8'));chosen=combined[combined.id==frozen['roles']['selected']['candidate']].iloc[0]
    ax.scatter(100*chosen.recall,100*chosen.precision,marker='*',s=220,color='#ffbf00',edgecolor='black',label='Выбрано до октября')
    ax.axhline(30,color='grey',linestyle='--');ax.axvline(10,color='grey',linestyle='--');ax.set_xlabel('Полнота, %');ax.set_ylabel('Верные среди тревог, %')
    ax.set_title('R13: совместный выбор модели и порога на разработке\n2025 и январь–сентябрь 2026 уже использовались ранее')
    ax.grid(alpha=.2);ax.legend(fontsize=9)
    ax.set_xlabel('Полнота, %\nКруг: FPR ≤ 5% в обоих периодах; крест: бюджет нарушен',fontsize=9)
    second=assets/'adaptive_development.png';fig.savefig(second,dpi=180);plt.close(fig)
    write_json(assets/'adaptive_manifest.json',dict(stage='R13',renderer_sha256=sha(Path(__file__)),
        source_sha256={str(p.relative_to(root)).replace('\\','/'):sha(p) for p in [out/'metrics.csv',out/'development_metrics.csv',out/'frozen_plan.json']},
        assets=[dict(file=str(p.relative_to(root)).replace('\\','/'),sha256=sha(p)) for p in [first,second]]))
    print(rows[['split','role','precision','recall','fpr','true_positives','false_positives']].to_string(index=False))


if __name__=='__main__':main()
