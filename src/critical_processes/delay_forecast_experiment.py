"""R14: a new information/model approach and a long, sealed Austin2025 test."""
from __future__ import annotations
import argparse,importlib.util,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import nbinom
from .selected_training import sha,write_json
from .spatial_experiment import safe
from .offline_forecast import load_categories,available_targets
from .spatiotemporal import fit_dispersion
from .adaptive_alarm import fit_policy,apply_policy,alarm_metrics,preliminary_screen
from .expanded_forecast import operating_intervals
from .delay_forecast import FIT_ASOF,fit_reporting_cdf,build_inputs,fit_network,predict_network

METHODS=('calendar_nb','naive_nb','delay_nb','bce_forecast','nb_plain','nb_forecast')
NEURAL=('bce_forecast','nb_plain','nb_forecast')
SEEDS=(601,613,619)
SCENARIOS={'primary':dict(lag_days=7,publication_days=7),'delayed_stress':dict(lag_days=14,publication_days=14)}


def preparer():
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('r14_preparer',root/'scripts/prepare_delay_forecast_data.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def score_models(inputs,models,alphas,cutoff):
    means=dict(calendar_nb=inputs['seasonal'],naive_nb=inputs['naive'],delay_nb=inputs['adaptive']);scores={}
    for name in ('calendar_nb','naive_nb','delay_nb'):
        scores[name]=np.stack([nbinom.sf(cutoff[:,:,g]-1,1/alphas[name][g],1/(1+alphas[name][g]*means[name][:,:,g])) for g in range(3)],axis=2)
    for name in NEURAL:
        x=inputs['features'] if name!='nb_plain' else inputs['features'][...,:-3]
        values=[predict_network(x,inputs['adaptive'],record) for record in models[name]]
        means[name]=np.mean([v[0] for v in values],axis=0)
        if name=='bce_forecast':scores[name]=np.mean([expit(v[1]) for v in values],axis=0)
        else:scores[name]=np.mean([nbinom.sf(cutoff-1,1/v[1],1/(1+v[1]*v[0])) for v in values],axis=0)
    return means,scores


def replay_policy(panel,scores,cutoff,method,alpha,scenario,start,end):
    dates=panel.dates;probability=np.zeros_like(scores[method]);alarm=np.zeros_like(probability,dtype=bool);journal=[]
    for asof in pd.date_range(start,pd.Timestamp(end)-pd.Timedelta(days=1),freq='MS'):
        take=np.asarray((dates>=asof)&(dates<asof+pd.DateOffset(months=1)))
        known=available_targets(panel,asof,SCENARIOS[scenario]['publication_days'])
        policy=fit_policy(scores[method],cutoff,known,dates,asof,180,alpha,28)
        probability[take],alarm[take]=apply_policy(scores[method][take],policy);journal.append(policy)
    return probability,alarm,journal


def choose(rows,allowed=None):
    eligible=[r for r in rows if allowed is None or r['method'] in allowed]
    def rank(row):
        metric=row['selection'];safe=metric['fpr']<=.05;feasible=safe and (metric['recall'] or 0.)>=.10
        if safe:return (0,0 if feasible else 1,-(float(metric['precision'] or 0.) if feasible else metric['fbeta_half']),-metric['fbeta_half'],row['id'])
        return (1,metric['fpr'],-metric['fbeta_half'],0.,row['id'])
    row=min(eligible,key=rank);metric=row['selection']
    return dict(id=row['id'],method=row['method'],alpha=row['alpha'],selection_status='fpr_recall_feasible' if metric['fpr']<=.05 and (metric['recall'] or 0.)>=.10 else 'predeclared_fallback_no_joint_point')


def fit_development(panel,weather,curve,out,epochs):
    import torch
    inputs=build_inputs(panel,weather,curve);cutoff=inputs['cutoff'];target=available_targets(panel,FIT_ASOF,7)
    models={};summaries=[];model_dir=out/'models';model_dir.mkdir(exist_ok=True)
    for name in NEURAL:
        models[name]=[];x=inputs['features'] if name!='nb_plain' else inputs['features'][...,:-3]
        for seed in SEEDS:
            path=model_dir/f'{name}_s{seed}.pt'
            if path.exists():record=torch.load(path,map_location='cpu',weights_only=False)
            else:
                record,history=fit_network(x,inputs['adaptive'],target,cutoff,panel.dates,'bce' if name=='bce_forecast' else 'nb',seed,epochs)
                torch.save(record,path);history.to_csv(path.with_suffix('.csv'),index=False)
            models[name].append(record);summaries.append(dict(model=name,seed=seed,epoch=record['selected_epoch'],validation_loss=record['validation_loss']))
            print(f'R14 fitted {name}/s{seed}: epoch {record["selected_epoch"]}',flush=True)
    train=(panel.dates>=pd.Timestamp('2021-10-01'))&(panel.dates<pd.Timestamp('2024-01-01'))
    alphas={name:[fit_dispersion(target[train,:,g],inputs[key][train,:,g]) for g in range(3)] for name,key in [('calendar_nb','seasonal'),('naive_nb','naive'),('delay_nb','adaptive')]}
    means,scores=score_models(inputs,models,alphas,cutoff);rows=[];policies={}
    for name in METHODS:
        for alpha in (.025,.05):
            probability,alarm,journal=replay_policy(panel,scores,cutoff,name,alpha,'primary','2024-07-01','2024-12-01')
            row=dict(id=f'{name}_a{int(alpha*1000):03}',method=name,alpha=alpha)
            for split,start,end in [('selection','2024-07-01','2024-10-01'),('confirmation','2024-10-01','2024-12-01')]:
                take=(panel.dates>=pd.Timestamp(start))&(panel.dates<pd.Timestamp(end))
                row[split]=alarm_metrics(panel.counts[take,:,0],cutoff[take,:,0],probability[take,:,0],alarm[take,:,0])
            rows.append(row);policies[row['id']]=journal
            print(f'R14 operating selection {row["id"]}: precision={row["selection"]["precision"]}, recall={row["selection"]["recall"]:.3f}, FPR={row["selection"]["fpr"]:.3f}',flush=True)
    roles=dict(selected=choose(rows),neural=choose(rows,NEURAL),baseline=choose(rows,['calendar_nb','naive_nb','delay_nb']),
        reporting_ablation=dict(method='naive_nb',alpha=.05),weather_ablation=dict(method='nb_plain',alpha=.05))
    pd.DataFrame(summaries).to_csv(out/'training_summary.csv',index=False)
    write_json(out/'development_candidates.json',safe(rows));write_json(out/'development_policy_journal.json',safe(policies))
    pd.DataFrame([dict(id=r['id'],method=r['method'],alpha=r['alpha'],split=split,**r[split]) for r in rows for split in ('selection','confirmation')]).to_csv(out/'development_metrics.csv',index=False)
    return dict(calendars=inputs['calendars'],alphas=alphas,roles=roles),models


def evaluate(out,panel_path,destination=None):
    panel=load_categories(panel_path);frozen=json.loads((out/'frozen_plan.json').read_text(encoding='utf-8'));journal=json.loads((out/'forecast_journal.json').read_text(encoding='utf-8'))
    rows=[];regional=[];months=[]
    for item in journal['entries']:
        if sha(out/item['file'])!=item['sha256']:raise ValueError('R14 forecast changed')
        with np.load(out/item['file']) as data:saved={key:data[key].copy() for key in data.files}
        index=np.searchsorted(panel.time_ns,saved['time_ns']);y=panel.counts[index];dates=panel.dates[index]
        for role in frozen['roles']:
            for g,group in enumerate(panel.groups):
                probability=saved['probability_'+role][:,:,g];alarm=saved['alarm_'+role][:,:,g];cutoff=saved['cutoff'][:,:,g];baseline=saved['probability_baseline'][:,:,g]
                metric=alarm_metrics(y[:,:,g],cutoff,probability,alarm);reference=alarm_metrics(y[:,:,g],cutoff,baseline,saved['alarm_baseline'][:,:,g])
                intervals=operating_intervals(y[:,:,g]>=cutoff,alarm,probability,baseline)
                row=dict(scenario=item['scenario'],role=role,group=str(group),**metric,baseline_ap=reference['ap'],baseline_brier=reference['brier'],**intervals)
                screen=preliminary_screen(row);row.update({k:v for k,v in screen.items() if k!='conditions'});row['failed_conditions']=';'.join(k for k,v in screen['conditions'].items() if not v);rows.append(row)
                for d,district in enumerate(panel.districts):regional.append(dict(scenario=item['scenario'],role=role,group=str(group),district=str(district),
                    **alarm_metrics(y[:,d:d+1,g],cutoff[:,d:d+1],probability[:,d:d+1],alarm[:,d:d+1])))
                for month in dates.to_period('M').unique():
                    take=np.asarray(dates.to_period('M')==month);months.append(dict(scenario=item['scenario'],role=role,group=str(group),month=str(month),
                        **alarm_metrics(y[take,:,g],cutoff[take],probability[take],alarm[take])))
    destination=out if destination is None else Path(destination);destination.mkdir(parents=True,exist_ok=True)
    table=pd.DataFrame(rows);table.to_csv(destination/'metrics.csv',index=False);pd.DataFrame(regional).to_csv(destination/'district_metrics.csv',index=False);pd.DataFrame(months).to_csv(destination/'monthly_metrics.csv',index=False)
    primary=table[(table.scenario=='primary')&(table.role=='selected')&(table.group=='all')].iloc[0]
    write_json(destination/'assessment.json',safe(dict(status='research_prototype_retrospective_new_city_test',selected=frozen['roles']['selected'],
        primary_point_passed=bool(primary.alarm_point_passed),primary_full_screen_passed=bool(primary.full_screen_passed),
        blocks=int(primary.blocks),new_geography='Austin, nine predeclared patrol sectors',new_period='2025 full year, first district forecast test in this project',
        source_publication_times_verified=False,prospective_forecasts_issued=False,warning='registered reports; assumed weekly publication; retrospective corrections and reporting bias remain')))
    return table


def run(out=Path('results/delay_forecast_detector'),development=Path('data/processed/delay_forecast_20261122/development'),
        raw_dir=Path('data/raw/delay_forecast_20261122'),final_dir=Path('data/processed/delay_forecast_20261122/final'),epochs=100):
    import torch
    torch.set_num_threads(4);out.mkdir(parents=True,exist_ok=True);root=Path(__file__).resolve().parents[2]
    code=('delay_forecast.py','delay_forecast_experiment.py','adaptive_alarm.py','offline_forecast.py','spatiotemporal.py','expanded_forecast.py')
    config=dict(stage='14_delay_aware_distributional_forecast',development_panel_sha256=sha(development/'austin.npz'),weather_sha256=sha(development/'austin_weather.npz'),
        code_sha256={name:sha(Path(__file__).parent/name) for name in code},preparation_sha256=sha(root/'scripts/prepare_delay_forecast_data.py'),
        methods=list(METHODS),seeds=list(SEEDS),epochs=epochs,scenarios=SCENARIOS,
        new_information='public archived GFS24h temperature forecasts; mean/range/missing flag, not actual future weather',
        reporting='mature pre2024 cohorts only; sector/group empirical delay CDF shrunk200cases; expected observed exposure in past windows',
        background='fixed training calendar plus delay-corrected past28day ratio; all methods share this causal reference target',
        training='events2021Oct-2023; targets known by2024Jul1; NN checkpoint validation2024Feb-Mar; January2024 excluded for forecast archive gaps',
        operating_choice='Jul-Sep2024 only; same180day mature-past calibration as R13; alpha.025/.05; FPR<=.05 & recall>=.10, maximize precision; predeclared F0.5 fallback',
        confirmation='Oct-Nov2024 reported but never changes selected policy',
        final_test='Austin2025 full year, downloaded only after final model/operating freeze; monthly updates use only mature past labels',
        criteria=dict(precision=.30,recall=.10,fpr=.05,ap_above_best_simple=True,brier_paired_ci_high_below_zero=True),
        note='new city and corrected background; percentages are not direct improvements on Seattle/R13 targets; registered cases, not true criticality')
    protocol=out/'protocol.json'
    if protocol.exists():
        if json.loads(protocol.read_text(encoding='utf-8'))!=safe(config):raise ValueError('R14 protocol/code/input changed')
        if (out/'complete.json').exists():
            for name,value in json.loads((out/'complete.json').read_text(encoding='utf-8'))['artifact_sha256'].items():
                if sha(out/name)!=value:raise ValueError('completed R14 artifact changed')
            return pd.read_csv(out/'metrics.csv')
    else:write_json(protocol,safe(config))
    panel=load_categories(development/'austin.npz')
    with np.load(development/'austin_weather.npz') as data:weather=data['features'].copy()
    prep=preparer()
    if not (out/'frozen_plan.json').exists():
        curve,profile=fit_reporting_cdf(panel);np.savez_compressed(out/'reporting_cdf.npz',curve=curve);write_json(out/'reporting_profile.json',profile)
        plan,models=fit_development(panel,weather,curve,out,epochs)
        plan.update(protocol_sha256=sha(protocol),frozen_utc=datetime.now(timezone.utc).isoformat(),reporting_cdf_sha256=sha(out/'reporting_cdf.npz'),
            model_sha256={p.relative_to(out).as_posix():sha(p) for p in (out/'models').glob('*.pt')},new_test_cases_downloaded=False)
        write_json(out/'frozen_plan.json',safe(plan))
    frozen=json.loads((out/'frozen_plan.json').read_text(encoding='utf-8'))
    with np.load(out/'reporting_cdf.npz') as data:curve=data['curve'].copy()
    models={name:[torch.load(out/'models'/f'{name}_s{seed}.pt',map_location='cpu',weights_only=False) for seed in SEEDS] for name in NEURAL}
    frozen_sha=sha(out/'frozen_plan.json');final_dir.mkdir(parents=True,exist_ok=True)
    prep.download_cases(2025,raw_dir,frozen_sha);prep.download_weather(2025,raw_dir,frozen_sha)
    prep.prepare(raw_dir,final_dir,[2025],frozen_sha,development/'austin.npz')
    final=load_categories(final_dir/'austin.npz')
    with np.load(final_dir/'austin_weather.npz') as data:final_weather=data['features'].copy()
    np.testing.assert_array_equal(final.counts[:len(panel.counts)],panel.counts)
    np.testing.assert_array_equal(final_weather[:len(weather)],weather)
    write_json(out/'new_input.json',dict(panel_sha256=sha(final_dir/'austin.npz'),weather_sha256=sha(final_dir/'austin_weather.npz'),
        metadata_sha256=sha(final_dir/'austin.json'),case_manifest_sha256=sha(raw_dir/'austin_2025.csv.gz.manifest.json'),
        weather_manifest_sha256=sha(raw_dir/'austin_weather_2025.json.manifest.json'),frozen_plan_sha256=frozen_sha))
    directory=out/'forecasts';directory.mkdir(exist_ok=True);entries=[];primary_cutoff=None
    for scenario,settings in SCENARIOS.items():
        inputs=build_inputs(final,final_weather,curve,frozen['calendars'],**settings)
        if scenario=='primary':primary_cutoff=inputs['cutoff']
        _,scores=score_models(inputs,models,frozen['alphas'],primary_cutoff);take=np.asarray(final.dates>=pd.Timestamp('2025-01-01'))
        arrays=dict(time_ns=final.time_ns[take],cutoff=primary_cutoff[take]);policies={}
        for role,selection in frozen['roles'].items():
            probability,alarm,journal=replay_policy(final,scores,primary_cutoff,selection['method'],selection['alpha'],scenario,'2025-01-01','2026-01-01')
            arrays['probability_'+role]=probability[take];arrays['alarm_'+role]=alarm[take];policies[role]=journal
        write_json(out/f'{scenario}_policy_journal.json',safe(policies));path=directory/(scenario+'.npz');np.savez_compressed(path,**arrays)
        entries.append(dict(file=path.relative_to(out).as_posix(),scenario=scenario,sha256=sha(path),contains_outcomes=False))
    write_json(out/'forecast_journal.json',dict(frozen_plan_sha256=frozen_sha,contains_outcomes=False,entries=entries))
    write_json(out/'test_started.json',dict(forecast_journal_sha256=sha(out/'forecast_journal.json'),frozen_plan_sha256=frozen_sha))
    metrics=evaluate(out,final_dir/'austin.npz')
    write_json(out/'complete.json',dict(successful_execution=True,protocol_sha256=sha(protocol),artifact_sha256={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='complete.json'}))
    print(metrics[(metrics.scenario=='primary')&(metrics.group=='all')][['role','precision','recall','fpr','ap','blocks','full_screen_passed']].to_string(index=False),flush=True)
    return metrics


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path('results/delay_forecast_detector'));parser.add_argument('--epochs',type=int,default=100)
    args=parser.parse_args();run(out=args.out,epochs=args.epochs)
