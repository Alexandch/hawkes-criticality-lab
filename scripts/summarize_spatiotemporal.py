from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from critical_processes.district_forecast import forecast_interval
from critical_processes.spatial_experiment import paired_block_improvement
from critical_processes.selected_training import sha,write_json


def main():
    source=Path('results/spatiotemporal_forecast');out=Path('results/spatiotemporal_summary');out.mkdir(exist_ok=True)
    config=json.loads((source/'protocol.json').read_text(encoding='utf-8'));table=pd.read_csv(source/'test_metrics.csv');effects=[];forecasts=[]
    for city in config['city_names']:
        with np.load(source/city/'predictions_lag7.npz') as data:
            dates=pd.DatetimeIndex(data['time_ns'].astype('datetime64[ns]'))
            for split,start,end in [('test2025','2025-01-01','2026-01-01'),('test2026','2026-01-01','2026-10-01')]:
                use=(dates>=pd.Timestamp(start))&(dates<pd.Timestamp(end))
                effect=paired_block_improvement(data['counts'][use],data['neural_ensemble'][use],data['recent_calendar'][use],24//config['bin_hours'],seed=181,bootstrap=1000)
                effects.append(dict(city=city,split=split,neural_minus_recent_mae=effect['primary_minus_seasonal_mae'],
                    block_ci95_lower=effect['block_bootstrap_ci95_lower'],block_ci95_upper=effect['block_bootstrap_ci95_upper'],
                    interpretation='descriptive paired comparison of prespecified methods; no new test-based model selection'))
        forecasts.append(forecast_interval(city))
    pd.DataFrame(effects).to_csv(out/'neural_vs_recent.csv',index=False)
    forecast=pd.concat(forecasts,ignore_index=True);forecast.to_csv(out/'next_interval_forecasts.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,city in zip(axes,config['city_names']):
        with np.load('data/processed/spatiotemporal_20261110/'+city+'.npz') as panel:coordinates=panel['centroids'];districts=panel['districts']
        values=forecast[forecast.city==city].set_index('district').loc[districts];size=values.expected_count.to_numpy()
        plot=ax.scatter(coordinates[:,1],coordinates[:,0],c=size,s=40+20*size,cmap='YlOrRd',edgecolor='black',linewidth=.3)
        for point,label in zip(coordinates,districts):ax.annotate(label,(point[1],point[0]),xytext=(3,3),textcoords='offset points',fontsize=6)
        ax.set_title(city+': 2026-10-01, 00–06 local');ax.set_xlabel('Долгота центра');ax.set_ylabel('Широта центра');ax.grid(alpha=.2)
        fig.colorbar(plot,ax=ax,label='Ожидаемое число выбранных происшествий')
    fig.suptitle('Исторический прогноз на границе архива; точки — центры, не границы округов',fontsize=10)
    fig.tight_layout();fig.savefig(out/'district_forecast_example.png',dpi=160);plt.close(fig)
    rows=[]
    for city in config['city_names']:
        for split in ('test2025','test2026'):
            d=table[(table.city==city)&(table.split==split)&table.information_lag_days.eq(7)].set_index('method')
            rows.append(dict(city=city,split=split,seasonal_mae=d.loc['seasonal','mae'],recent_mae=d.loc['recent_calendar','mae'],neural_mae=d.loc['neural_ensemble','mae'],
                neural_reduction_vs_seasonal_pct=100*(1-d.loc['neural_ensemble','mae']/d.loc['seasonal','mae'])))
    pd.DataFrame(rows).to_csv(out/'headline_metrics.csv',index=False)
    for row in rows:print(row)
    print(pd.DataFrame(effects).to_string(index=False))
    write_json(out/'provenance.json',dict(source_sha256={name:sha(name) for name in ('results/spatiotemporal_forecast/complete.json',
        'scripts/summarize_spatiotemporal.py','src/critical_processes/district_forecast.py')},
        artifact_sha256={path.name:sha(path) for path in out.iterdir() if path.is_file() and path.name!='provenance.json'}))


if __name__=='__main__':main()
