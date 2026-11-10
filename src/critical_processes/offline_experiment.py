"""Run R11 rolling-origin forecasts, freeze a ledger, then reveal outcomes."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from .selected_training import sha,write_json
from .spatial_experiment import safe
from .spatiotemporal import fit_dispersion
from .offline_forecast import (load_categories,vintage_masks,available_targets,causal_inputs,rise_cutoff,
    fit_network,predict_network,calibrate_probability,apply_calibration,regional_thresholds,evaluate_forecast,paired_blocks)


METHODS=('seasonal','recent_calendar','neural_nb','neural_direct')
VINTAGES=('2024-01-01','2025-01-01','2026-01-01','2026-10-01')
SCENARIOS={'primary':dict(lag_days=7,publication_days=3),'delayed_stress':dict(lag_days=14,publication_days=7)}


def raw_forecasts(seasonal,adaptive,neural,direct,cutoff,dispersions):
    means={'seasonal':seasonal,'recent_calendar':adaptive,'neural_nb':neural,'neural_direct':neural}
    scores={}
    for method,mean in means.items():
        values=[]
        for group,alpha in enumerate(dispersions[method]):
            r=1/alpha;values.append(nbinom.sf(cutoff[:,:,group]-1,r,r/(r+mean[:,:,group])))
        scores[method]=np.stack(values,axis=2)
    scores['neural_direct']=direct
    return means,scores


def prepare_vintage(panel,asof,directory,seeds,epochs):
    import torch
    masks=vintage_masks(panel.dates,asof)
    for key in ('train','validation','probability_calibration','alarm_calibration'):
        if not masks[key].any():raise ValueError('empty chronological split: '+key)
    features,seasonal,adaptive,calendars=causal_inputs(panel,masks['train_end']);cutoff=rise_cutoff(adaptive)
    train_y=available_targets(panel,masks['train_end']);validation_y=available_targets(panel,masks['validation_end'])
    calibration_y=available_targets(panel,asof);learning=train_y.copy();learning[masks['validation']]=validation_y[masks['validation']]
    directory.mkdir(parents=True,exist_ok=True);records=[]
    for seed in seeds:
        path=directory/f'seed{seed}.pt'
        if path.exists():record=torch.load(path,map_location='cpu',weights_only=False)
        else:
            record,history=fit_network(features,learning,adaptive,masks['train'],masks['validation'],seed,epochs)
            record.update(asof=asof,district_ids=panel.districts.tolist(),group_ids=panel.groups.tolist())
            torch.save(record,path);history.to_csv(path.with_suffix('.csv'),index=False)
        records.append(record)
        print(f'R11 {directory.parent.name}/{asof}/seed{seed}: epoch {record["selected_epoch"]}',flush=True)
    outputs=[predict_network(features,adaptive,record) for record in records];neural=np.mean([v[0] for v in outputs],axis=0);direct=np.mean([v[1] for v in outputs],axis=0)
    means={'seasonal':seasonal,'recent_calendar':adaptive,'neural_nb':neural,'neural_direct':neural}
    dispersions={method:[fit_dispersion(train_y[masks['train'],:,group],mean[masks['train'],:,group]) for group in range(len(panel.groups))] for method,mean in means.items()}
    means,scores=raw_forecasts(seasonal,adaptive,neural,direct,cutoff,dispersions)
    validations=[];calibrations={};thresholds={};threshold_notes={}
    for method in METHODS:
        calibrations[method]=[];calibrated=np.empty_like(scores[method])
        for group,name in enumerate(panel.groups):
            use=masks['validation'];metric=evaluate_forecast(validation_y[use,:,group],means[method][use,:,group],scores[method][use,:,group],
                cutoff[use,:,group],np.ones(len(panel.districts)),dispersions[method][group])
            validations.append(dict(method=method,group=str(name),**metric))
            use=masks['probability_calibration'];calibration=calibrate_probability(scores[method][use,:,group],calibration_y[use,:,group]>=cutoff[use,:,group])
            calibrations[method].append(calibration);calibrated[:,:,group]=apply_calibration(scores[method][:,:,group],calibration)
        use=masks['alarm_calibration'];threshold,notes=regional_thresholds(calibrated[use],calibration_y[use]>=cutoff[use])
        thresholds[method]=threshold.tolist();threshold_notes[method]=notes
    table=pd.DataFrame(validations);count_primary={};rise_primary={}
    for group in panel.groups:
        subset=table[table.group==group]
        count_primary[str(group)]=subset[subset.method!='neural_direct'].sort_values(['nb_nll','method']).iloc[0]['method']
        rise_primary[str(group)]=subset.sort_values(['ap','method'],ascending=[False,True]).iloc[0]['method']
    plan=dict(asof=asof,train_end=masks['train_end'],validation_end=masks['validation_end'],calibration_end=asof,
        calendars=calendars,dispersions=dispersions,probability_calibration=calibrations,thresholds=thresholds,threshold_notes=threshold_notes,
        count_primary=count_primary,rise_primary=rise_primary,validation=validations,districts=panel.districts.tolist(),groups=panel.groups.tolist(),
        model_paths=[str(path.name) for path in sorted(directory.glob('*.pt'))],maturity_days=28,
        split_cells={name:int(masks[name].sum())*len(panel.districts)*len(panel.groups) for name in ('train','validation','probability_calibration','alarm_calibration')})
    return safe(plan)


def replay_vintage(panel,plan,directory,scenario):
    import torch
    features,seasonal,adaptive,_=causal_inputs(panel,plan['train_end'],**SCENARIOS[scenario],calendars=plan['calendars'])
    outputs=[predict_network(features,adaptive,torch.load(directory/name,map_location='cpu',weights_only=False)) for name in plan['model_paths']]
    neural=np.mean([v[0] for v in outputs],axis=0);direct=np.mean([v[1] for v in outputs],axis=0);cutoff=rise_cutoff(adaptive)
    # Both delay scenarios predict the same primary target; only the information supplied to models changes.
    if scenario!='primary':
        _,_,primary_adaptive,_=causal_inputs(panel,plan['train_end'],calendars=plan['calendars']);cutoff=rise_cutoff(primary_adaptive)
    means,scores=raw_forecasts(seasonal,adaptive,neural,direct,cutoff,plan['dispersions'])
    raw={method:values.copy() for method,values in scores.items()}
    for method in METHODS:
        for group,record in enumerate(plan['probability_calibration'][method]):scores[method][:,:,group]=apply_calibration(scores[method][:,:,group],record)
    return means,scores,raw,cutoff


def ledger_entries(out):
    journal=json.loads((out/'forecast_journal.json').read_text(encoding='utf-8'))
    for item in journal['entries']:
        if sha(out/item['file'])!=item['sha256']:raise ValueError('forecast ledger modified: '+item['file'])
        yield item


def evaluate_ledger(out,panel_dir,bootstrap=1000,destination=None):
    frozen=json.loads((out/'frozen_plan.json').read_text(encoding='utf-8'));panels={city:load_categories(panel_dir/(city+'.npz')) for city in frozen['cities']}
    monthly=[];regional=[];aggregate={};examples=[]
    for item in ledger_entries(out):
        city,scenario,asof=item['city'],item['scenario'],item['asof'];panel=panels[city];plan=frozen['cities'][city][asof]
        with np.load(out/item['file']) as saved:data={name:saved[name].copy() for name in saved.files}
        index=np.searchsorted(panel.time_ns,data['time_ns']);y=panel.counts[index];dates=panel.dates[index]
        for group,name in enumerate(panel.groups):
            for method in METHODS:
                threshold=np.array(plan['thresholds'][method])[:,group];alpha=plan['dispersions'][method][group]
                metric=evaluate_forecast(y[:,:,group],data['mean_'+method][:,:,group],data['risk_'+method][:,:,group],data['cutoff'][:,:,group],threshold,alpha)
                row=dict(city=city,scenario=scenario,asof=asof,month=item['month'],group=str(name),method=method,count_primary=plan['count_primary'][str(name)]==method,
                    rise_primary=plan['rise_primary'][str(name)]==method,**metric)
                row['alerts']=[]
                if metric['fpr'] is not None and metric['fpr']>.05:row['alerts'].append('pooled_fpr_over_target')
                if metric['maximum_district_fpr']>.10:row['alerts'].append('district_fpr_over_10pct')
                row['alerts']=';'.join(row['alerts']);monthly.append(row)
        key=(city,scenario)
        aggregate.setdefault(key,[]).append((data,y))
        if item['month']=='2026-10' and scenario=='primary':
            for t in range(min(7*4,len(dates))):
                for d,district in enumerate(panel.districts):
                    for g,group in enumerate(panel.groups):
                        count_method=plan['count_primary'][str(group)];risk_method=plan['rise_primary'][str(group)]
                        examples.append(dict(city=city,district=str(district),group=str(group),start=str(dates[t]),end=str(dates[t]+pd.Timedelta(hours=6)),
                            count_method=count_method,rise_method=risk_method,expected_count=float(data['mean_'+count_method][t,d,g]),
                            probability=float(data['risk_'+risk_method][t,d,g]),threshold=plan['thresholds'][risk_method][d][g],
                            observed_count=int(y[t,d,g]),rise_cutoff=int(data['cutoff'][t,d,g]),observed_rise=bool(y[t,d,g]>=data['cutoff'][t,d,g])))
    rows=[];effects=[]
    for (city,scenario),parts in aggregate.items():
        panel=panels[city];dates=pd.DatetimeIndex(np.concatenate([data['time_ns'] for data,y in parts]).astype('datetime64[ns]'))
        y=np.concatenate([y for data,y in parts]);cutoff=np.concatenate([data['cutoff'] for data,y in parts])
        for label,start,end,status in [('development2024','2024-01-01','2025-01-01','development'),('development2025','2025-01-01','2026-01-01','previously_inspected_period'),
            ('development2026','2026-01-01','2026-10-01','previously_inspected_period'),('new_october_first14','2026-10-01','2026-10-15','novel_preliminary'),('new_october_all','2026-10-01','2026-11-01','novel_immature_labels')]:
            use=(dates>=pd.Timestamp(start))&(dates<pd.Timestamp(end))
            if not use.any():continue
            asof='2026-10-01' if start>='2026-10-01' else ('2026-01-01' if start>='2026-01-01' else ('2025-01-01' if start>='2025-01-01' else '2024-01-01'))
            if asof not in frozen['cities'][city]:continue
            plan=frozen['cities'][city][asof]
            for group,name in enumerate(panel.groups):
                for method in METHODS:
                    mean=np.concatenate([data['mean_'+method] for data,yy in parts])[:,:,group];probability=np.concatenate([data['risk_'+method] for data,yy in parts])[:,:,group]
                    metric=evaluate_forecast(y[use,:,group],mean[use],probability[use],cutoff[use,:,group],np.array(plan['thresholds'][method])[:,group],plan['dispersions'][method][group])
                    rows.append(dict(city=city,scenario=scenario,split=label,label_status=status,asof=asof,group=str(name),method=method,
                        count_primary=plan['count_primary'][str(name)]==method,rise_primary=plan['rise_primary'][str(name)]==method,**metric))
                    for d,district in enumerate(panel.districts):
                        one=evaluate_forecast(y[use,d:d+1,group],mean[use,d:d+1],probability[use,d:d+1],cutoff[use,d:d+1,group],plan['thresholds'][method][d][group],plan['dispersions'][method][group])
                        regional.append(dict(city=city,scenario=scenario,split=label,group=str(name),method=method,district=str(district),**one))
                    if method in ('neural_nb','neural_direct'):
                        reference=np.concatenate([data['mean_recent_calendar'] for data,yy in parts])[:,:,group]
                        reference_p=np.concatenate([data['risk_recent_calendar'] for data,yy in parts])[:,:,group]
                        effect=paired_blocks(y[use,:,group],mean[use],reference[use],probability[use],reference_p[use],cutoff[use,:,group],bootstrap=bootstrap)
                        effects.append(dict(city=city,scenario=scenario,split=label,group=str(name),method=method,**effect))
    target=out if destination is None else Path(destination);target.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(target/'metrics.csv',index=False);pd.DataFrame(monthly).to_csv(target/'monthly_metrics.csv',index=False)
    pd.DataFrame(regional).to_csv(target/'district_metrics.csv',index=False);pd.DataFrame(effects).to_csv(target/'paired_differences.csv',index=False)
    pd.DataFrame(examples).to_csv(target/'october_examples.csv',index=False)
    return pd.DataFrame(rows)


def run_offline(panel_dir='data/processed/offline_20261110',out='results/offline_readiness',epochs=60,seeds=(211,223,227),vintages=VINTAGES,bootstrap=1000):
    import torch
    torch.set_num_threads(min(4,torch.get_num_threads()));panel_dir,out=Path(panel_dir),Path(out);out.mkdir(parents=True,exist_ok=True)
    cities=('chicago','san_francisco');code=('offline_forecast.py','offline_experiment.py')
    config=dict(stage='11_offline_readiness',epochs=epochs,seeds=list(seeds),vintages=list(vintages),methods=list(METHODS),scenarios=SCENARIOS,
        target='N >= ceil(adaptive_past_background + max(2,.5*adaptive_past_background)); group-specific, distinct from R10 historical95%',
        novel_control='October2026 first14days primary, full month supplementary; provisional registered-case labels',
        bootstrap_samples=bootstrap,panel_sha256={city:sha(panel_dir/(city+'.npz')) for city in cities},
        metadata_sha256={city:sha(panel_dir/(city+'.json')) for city in cities},code_sha256={name:sha(Path(__file__).parent/name) for name in code},
        previous_experiment_sha256=sha(Path('results/spatiotemporal_forecast/complete.json')),
        history_warning='historical publication times are assumed, not reconstructed; first-seen ledger begins only at actual snapshots')
    protocol=out/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text(encoding='utf-8'))!=config:raise ValueError('different R11 protocol; choose another output directory')
    if not protocol.exists():write_json(protocol,config)
    if (out/'complete.json').exists():
        complete=json.loads((out/'complete.json').read_text(encoding='utf-8'))
        for name,value in complete['artifact_sha256'].items():
            if sha(out/name)!=value:raise ValueError('R11 artifact changed: '+name)
        return pd.read_csv(out/'metrics.csv')
    panels={city:load_categories(panel_dir/(city+'.npz')) for city in cities};frozen=dict(protocol_sha256=sha(protocol),cities={},selection_used_october=False)
    for city,panel in panels.items():
        frozen['cities'][city]={}
        for asof in vintages:frozen['cities'][city][asof]=prepare_vintage(panel,asof,out/'models'/city/asof,seeds,epochs)
    frozen['model_sha256']={path.relative_to(out).as_posix():sha(path) for path in (out/'models').rglob('*.pt')}
    frozen_path=out/'frozen_plan.json'
    if frozen_path.exists() and json.loads(frozen_path.read_text(encoding='utf-8'))!=frozen:raise ValueError('R11 plan already frozen')
    write_json(frozen_path,frozen);entries=[]
    for city,panel in panels.items():
        for number,asof in enumerate(vintages):
            stop=pd.Timestamp(vintages[number+1]) if number+1<len(vintages) else pd.Timestamp('2026-11-01')
            month_keys=panel.dates.to_period('M').astype(str);use=(panel.dates>=pd.Timestamp(asof))&(panel.dates<stop)
            plan=frozen['cities'][city][asof]
            for scenario in SCENARIOS:
                means,scores,raw,cutoff=replay_vintage(panel,plan,out/'models'/city/asof,scenario)
                for month in sorted(set(month_keys[use])):
                    mask=use&(month_keys==month);path=out/'ledger'/city/f'{scenario}_{month}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                    values=dict(time_ns=panel.time_ns[mask],districts=panel.districts,groups=panel.groups,cutoff=cutoff[mask])
                    for method in METHODS:values.update({f'mean_{method}':means[method][mask],f'risk_{method}':scores[method][mask],f'raw_{method}':raw[method][mask]})
                    np.savez_compressed(path,**values)
                    entries.append(dict(city=city,scenario=scenario,asof=asof,month=month,file=path.relative_to(out).as_posix(),sha256=sha(path),forecast_cells=int(mask.sum())*len(panel.districts)*len(panel.groups)))
            print(f'Frozen forecast ledger {city}/{asof}',flush=True)
    write_json(out/'forecast_journal.json',dict(frozen_plan_sha256=sha(frozen_path),entries=entries,contains_outcomes=False,
        interpretation='retrospective sequential replay, not forecasts actually issued on historical dates'))
    write_json(out/'evaluation_started.json',dict(forecast_journal_sha256=sha(out/'forecast_journal.json'),frozen_plan_sha256=sha(frozen_path)))
    metrics=evaluate_ledger(out,panel_dir,bootstrap)
    write_json(out/'readiness.json',dict(status='research_prototype_offline_only',real_pilot_performed=False,
        historical_publication_times_verified=False,prospective_validation_performed=False,intervention_benefit_measured=False,
        first_seen_ledger_supported=True,remaining=['independent prospective observations','verified publication-time history','agreed action-specific utility and acceptable alarm budget','external impact and feedback-loop assessment'],
        note='Passing software tests and retrospective metrics does not authorize operational deployment.'))
    write_json(out/'complete.json',dict(artifact_sha256={path.relative_to(out).as_posix():sha(path) for path in out.rglob('*') if path.is_file() and path.name!='complete.json'}))
    return metrics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--panels',default='data/processed/offline_20261110');parser.add_argument('--out',default='results/offline_readiness');parser.add_argument('--epochs',type=int,default=60)
    args=parser.parse_args();run_offline(args.panels,args.out,args.epochs)


if __name__=='__main__':main()
