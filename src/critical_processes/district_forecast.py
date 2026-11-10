"""Apply frozen district-count models to an explicitly dated next interval."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from .selected_training import sha
from .spatiotemporal import (DistrictPanel,load_panel,calendar_prediction,forecast_features,recent_calendar,observed_memory,predict_count_network,spike_probability)


def forecast_interval(city,model_root='results/spatiotemporal_forecast',panel_dir='data/processed/spatiotemporal_20261110',asof=None,lag_days=7):
    root,panel_dir=Path(model_root),Path(panel_dir)
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'));frozen=json.loads((root/'frozen_plan.json').read_text(encoding='utf-8'))
    started=json.loads((root/'test_started.json').read_text(encoding='utf-8'))
    if frozen['protocol_sha256']!=sha(root/'protocol.json') or started['frozen_plan_sha256']!=sha(root/'frozen_plan.json'):
        raise ValueError('frozen protocol or model selection changed')
    if city not in protocol['city_names'] or lag_days not in protocol['information_lags']:raise ValueError('city/information lag not in frozen protocol')
    if sha(panel_dir/(city+'.npz'))!=protocol['panel_sha256'][city]:raise ValueError('panel differs from model archive; use matching data and models')
    for name,digest in frozen['model_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('frozen weights changed')
    panel=load_panel(panel_dir/(city+'.npz'),protocol['bin_hours']);hours=panel.bin_hours;bins_day=24//hours
    endpoint=panel.dates[-1]+pd.Timedelta(hours=hours);issue=endpoint if asof is None else pd.Timestamp(asof)
    if issue.tzinfo is not None:raise ValueError('asof must use the source local clock without a timezone suffix')
    if issue<pd.Timestamp('2025-01-01'):raise ValueError('models were not frozen before 2025-01-01')
    if issue>endpoint:raise ValueError('requested date exceeds available archive; obtain newer observations rather than filling missing dates with zeros')
    index=int(np.searchsorted(panel.time_ns,issue.value))
    if issue.value!=panel.time_ns[0]+index*hours*3600*10**9:raise ValueError('forecast start must align with the time grid')
    if index<35*bins_day:raise ValueError('need at least35 days observed history')
    if index==len(panel.counts):
        panel=DistrictPanel(np.vstack([panel.counts,np.zeros((1,panel.counts.shape[1]),dtype=int)]),np.r_[panel.time_ns,issue.value],panel.districts,panel.centroids,hours,
            panel.event_bins,panel.report_bins,panel.area_indices)
    plan=frozen['cities'][city][str(lag_days)];base,cutoffs=calendar_prediction(plan['calendar'],panel.dates);graph=np.asarray(plan['graph'])
    features=forecast_features(panel,base,graph,lag_days);cache={'seasonal':base[index],'recent_calendar':recent_calendar(panel,base,bins_day,lag_days)[index]}
    for method in ('temporal_hawkes','spatial_hawkes'):
        spec=plan['specifications'][method];memory=observed_memory(panel,spec['tau_bins'],lag_days*bins_day);scale=np.asarray(spec['normalizing_rates'])
        adjusted=scale[:,None]*graph/scale[None,:];values=np.stack([base[index],memory[index],adjusted@memory[index]],axis=-1)
        theta=np.asarray(spec['coefficients']);cache[method]=np.maximum(values[:,:len(theta)]@theta,1e-8)
    cache['neural_ensemble']=np.mean([predict_count_network(features[index:index+1],base[index:index+1],root/city/'models'/f'lag{lag_days}_seed{seed}.pt')[0]
        for seed in protocol['model_seeds']],axis=0)
    count_model,spike_model=plan['count_primary'],plan['spike_primary'];mean=cache[count_model];alpha=plan['dispersions'][count_model];r=1/alpha
    probability=spike_probability(cache[spike_model],cutoffs[index],plan['dispersions'][spike_model]);threshold=plan['thresholds'][spike_model]
    return pd.DataFrame(dict(city=city,district=panel.districts,forecast_start=str(issue),forecast_end=str(issue+pd.Timedelta(hours=hours)),
        local_time_zone='America/Chicago' if city=='chicago' else 'America/Los_Angeles',information_lag_days=lag_days,
        count_model=count_model,spike_model=spike_model,expected_count=mean,lower95=nbinom.ppf(.025,r,r/(r+mean)),upper95=nbinom.ppf(.975,r,r/(r+mean)),
        spike_cutoff=cutoffs[index],spike_probability=probability,spike_alarm=probability>=threshold,
        observed_count=np.full(len(mean),np.nan),
        interpretation='dated forecast using archived observations, not a live present-day forecast; historical record revisions are unknown'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--city',choices=['chicago','san_francisco'],required=True);parser.add_argument('--asof')
    parser.add_argument('--models',default='results/spatiotemporal_forecast');parser.add_argument('--panels',default='data/processed/spatiotemporal_20261110')
    parser.add_argument('--out',type=Path,default=Path('runs/district_forecast.csv'));args=parser.parse_args()
    result=forecast_interval(args.city,args.models,args.panels,args.asof);args.out.parent.mkdir(parents=True,exist_ok=True);result.to_csv(args.out,index=False)
    print(result[['district','forecast_start','forecast_end','expected_count','spike_probability']].to_string(index=False))
    print(f'Saved {len(result)} district forecasts to {args.out}')


if __name__=='__main__':main()
