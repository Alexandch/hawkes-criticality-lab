"""R13 last bounded attempt: Seattle recalibration, frozen before new October data."""
from __future__ import annotations
import argparse,csv,gzip,json,shutil,urllib.parse,urllib.request
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from .selected_training import sha,write_json
from .spatial_experiment import safe
from .offline_forecast import CategoryPanel,load_categories,causal_inputs,available_targets,rise_cutoff,apply_calibration
from .expanded_forecast import predict_neural,operating_intervals
from .adaptive_alarm import fit_policy,apply_policy,alarm_metrics,choose_candidate,preliminary_screen

METHODS=('local_neural','recent_calendar','seasonal')
SCENARIOS={'primary':dict(lag_days=7,publication_days=3),'delayed_stress':dict(lag_days=14,publication_days=7)}
GROUPS=('all','property','assault_robbery')
SEEDS=(331,337,349)
END='2026-11-01'


def infer(panel,old,scenario):
    import torch
    frozen=json.loads((old/'frozen_plan.json').read_text(encoding='utf-8'));plan=frozen['cities']['seattle']
    features,seasonal,adaptive,_=causal_inputs(panel,'2024-01-01',**SCENARIOS[scenario],calendars=frozen['calendars']['seattle'])
    shape=adaptive.shape;category=np.empty((*shape,3),dtype='int16')
    offset=sum(len(frozen['cities'][city]['districts']) for city in ('chicago','san_francisco','nyc'))
    category[:,:,:,0]=np.arange(shape[1])[None,:,None]+offset;category[:,:,:,1]=3;category[:,:,:,2]=np.arange(shape[2])[None,None,:]
    values=[]
    for seed in SEEDS:
        path=old/'models'/f'local_seattle_s{seed}.pt'
        if sha(path)!=frozen['model_sha256'][path.relative_to(old).as_posix()]:raise ValueError('R12 model changed')
        record=torch.load(path,map_location='cpu',weights_only=False)
        values.append(predict_neural(features.reshape(-1,features.shape[-1]),category.reshape(-1,3),adaptive.ravel(),record))
    mean=np.mean([v[0] for v in values],axis=0).reshape(shape);raw=np.mean([v[1] for v in values],axis=0).reshape(shape)
    means=dict(local_neural=mean,recent_calendar=adaptive,seasonal=seasonal)
    return dict(means=means,raw_neural=raw,adaptive=adaptive,cutoff=rise_cutoff(adaptive))


def raw_scores(result,cutoff,old_plan):
    scores={'local_neural':result['raw_neural']}
    for method in ('recent_calendar','seasonal'):
        scores[method]=np.stack([nbinom.sf(cutoff[:,:,g]-1,1/old_plan['dispersions'][method][g],
            1/(1+old_plan['dispersions'][method][g]*result['means'][method][:,:,g])) for g in range(3)],axis=2)
    return scores


def static_forecast(raw,method,old_plan):
    probability=np.stack([apply_calibration(raw[:,:,g],record) for g,record in enumerate(old_plan['calibrations'][method])],axis=2)
    return probability,probability>=np.asarray(old_plan['thresholds'][method])


def candidates():
    return [dict(id=f'{method}_w{window}_a{int(alpha*1000):03}',method=method,window_days=window,alpha=alpha)
        for method in METHODS for window in (90,180,365) for alpha in (.01,.025,.05)]


def month_masks(dates,start='2025-01-01',end='2026-10-01'):
    for asof in pd.date_range(start,pd.Timestamp(end)-pd.Timedelta(days=1),freq='MS'):
        take=np.asarray((dates>=asof)&(dates<asof+pd.DateOffset(months=1)))
        if take.any():yield asof,take


def develop(panel,raw,cutoff):
    dates=panel.dates;known={str(asof):available_targets(panel,asof,3) for asof,_ in month_masks(dates)}
    rows=[];plans={}
    for config in candidates():
        probability=np.zeros_like(raw[config['method']]);alarm=np.zeros_like(probability,dtype=bool);monthly=[]
        for asof,take in month_masks(dates):
            policy=fit_policy(raw[config['method']],cutoff,known[str(asof)],dates,asof,config['window_days'],config['alpha'])
            probability[take],alarm[take]=apply_policy(raw[config['method']][take],policy);monthly.append(policy)
        row=dict(**config)
        for split,start,end in [('development2025','2025-01-01','2026-01-01'),('development2026','2026-01-01','2026-10-01'),('combined','2025-01-01','2026-10-01')]:
            take=(dates>=pd.Timestamp(start))&(dates<pd.Timestamp(end))
            row[split]=alarm_metrics(panel.counts[take,:,0],cutoff[take,:,0],probability[take,:,0],alarm[take,:,0])
        rows.append(row);plans[config['id']]=monthly
        print(f'R13 development {config["id"]}: FPR={row["combined"]["fpr"]:.3f}, recall={row["combined"]["recall"]:.3f}',flush=True)
    return rows,plans


def check_r12_replay(panel,result,raw,old,old_plan,scenario):
    with np.load(old/'forecasts'/f'seattle_{scenario}.npz') as saved:
        index=np.searchsorted(panel.time_ns,saved['time_ns'])
        for method in METHODS:
            probability,_=static_forecast(raw[method],method,old_plan)
            np.testing.assert_allclose(probability[index],saved['risk_'+method],rtol=2e-6,atol=2e-7)
            np.testing.assert_allclose(result['means'][method][index],saved['mean_'+method],rtol=2e-6,atol=2e-7)


def append_frame(history,frame,old_ids):
    frame=frame.copy();raw_rows=len(frame);repeat=frame.report_number.isin(old_ids)
    repeat_count=int(repeat.sum());frame=frame[~repeat].copy()
    frame['time']=pd.to_datetime(frame.offense_date,format='mixed',errors='coerce')
    frame['report']=pd.to_datetime(frame.report_date_time,format='mixed',errors='coerce')
    frame['district']=frame.precinct.str.strip().str.upper();frame['group']=np.where(frame.offense_sub_category.isin(['ASSAULT OFFENSES','AGGRAVATED ASSAULT','ROBBERY']),2,1)
    invalid=int((frame.report_number.isna()|frame.time.isna()).sum());frame=frame.dropna(subset=['report_number','time'])
    unknown=int((~frame.district.isin(history.districts)).sum());frame=frame[frame.district.isin(history.districts)].copy()
    conflicts=frame.groupby('report_number').district.nunique();bad=conflicts[conflicts>1].index;frame=frame[~frame.report_number.isin(bad)].copy()
    frame['group']=frame.groupby('report_number')['group'].transform('max');frame['report']=frame.groupby('report_number')['report'].transform('min')
    frame=frame.sort_values('time',kind='stable');duplicates=int(frame.report_number.duplicated().sum());frame=frame.drop_duplicates('report_number')
    start=pd.Timestamp('2026-10-01');end=pd.Timestamp(END)
    if not ((frame.time>=start)&(frame.time<end)).all():raise ValueError('new rows outside frozen October scope')
    if set(frame.district)!=set(history.districts):raise ValueError('incomplete new district coverage')
    dates=pd.date_range(history.dates[0],end,freq='6h',inclusive='left').as_unit('ns')
    bins=((frame.time.to_numpy(dtype='datetime64[ns]').astype('int64')-history.time_ns[0])//(6*3600*10**9)).astype('int32')
    areas=frame.district.map({d:i for i,d in enumerate(history.districts)}).to_numpy(dtype='int16');groups=frame.group.to_numpy(dtype='int8')
    reports=(frame.report.to_numpy(dtype='datetime64[ns]').astype('int64')-history.time_ns[0])//(6*3600*10**9)
    reports[frame.report.isna().to_numpy()]=len(dates)+10000;reports=np.maximum(reports,bins).astype('int32')
    counts=np.zeros((len(dates),len(history.districts),3),dtype='int32');counts[:len(history.counts)]=history.counts
    np.add.at(counts,(bins,areas,np.zeros(len(bins),dtype=int)),1);np.add.at(counts,(bins,areas,groups),1)
    panel=CategoryPanel(counts,dates.asi8,history.districts.copy(),history.groups.copy(),history.centroids.copy(),
        np.concatenate([history.event_bins,bins]),np.concatenate([history.area_indices,areas]),np.concatenate([history.group_indices,groups]),np.concatenate([history.report_bins,reports]))
    np.testing.assert_array_equal(panel.counts[:len(history.counts)],history.counts)
    return panel,dict(raw_rows=raw_rows,new_cases=len(frame),excluded_existing_history_rows=repeat_count,invalid_rows=invalid,
        unknown_district_rows=unknown,conflicting_cases=len(bad),duplicate_offense_rows=duplicates,missing_report_times=int(frame.report.isna().sum()),
        historical_cases=int(history.counts[:,:,0].sum()),all_panel_cases=int(counts[:,:,0].sum()))


def new_snapshot(history,history_raw,raw_dir,panel_dir,frozen_sha):
    raw_dir.mkdir(parents=True,exist_ok=True);panel_dir.mkdir(parents=True,exist_ok=True)
    path=raw_dir/'seattle_october.csv.gz';manifest=path.with_name(path.name+'.manifest.json')
    values=['LARCENY-THEFT','MOTOR VEHICLE THEFT','BURGLARY','ASSAULT OFFENSES','AGGRAVATED ASSAULT','ROBBERY']
    where="offense_date >= '2026-10-01T00:00:00' AND offense_date < '2026-11-01T00:00:00' AND offense_sub_category in("+','.join("'"+v+"'" for v in values)+')'
    base='https://data.seattle.gov/resource/tazs-3rd5'
    def get(url):
        return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'CriticalProcessesResearch/1.0'}),timeout=90)
    count_url=base+'.json?'+urllib.parse.urlencode({'$select':'count(*) as rows','$where':where})
    url=base+'.csv?'+urllib.parse.urlencode({'$select':'report_number,offense_date,report_date_time,precinct,offense_sub_category',
        '$where':where,'$order':'offense_date,report_number','$limit':100000})
    if manifest.exists():
        meta=json.loads(manifest.read_text(encoding='utf-8'))
        if sha(path)!=meta['sha256'] or meta['frozen_plan_sha256']!=frozen_sha:raise ValueError('new snapshot or pretest freeze changed')
    else:
        with get(count_url) as response:before=int(json.load(response)[0]['rows'])
        if not 0<before<100000:raise ValueError('unexpected new snapshot count')
        with get(url) as response,gzip.open(path.with_name(path.name+'.partial'),'wb') as output:shutil.copyfileobj(response,output)
        with gzip.open(path.with_name(path.name+'.partial'),'rt',encoding='utf-8-sig',newline='') as handle:actual=sum(1 for _ in csv.DictReader(handle))
        with get(count_url) as response:after=int(json.load(response)[0]['rows'])
        if before!=actual or after!=actual:raise ValueError('live source changed during download')
        path.with_name(path.name+'.partial').replace(path)
        meta=dict(url=url,count_url=count_url,rows=actual,sha256=sha(path),retrieved_utc=datetime.now(timezone.utc).isoformat(),
            frozen_plan_sha256=frozen_sha,label_status='retrospective registered-case snapshot; late reporting and edits possible')
        write_json(manifest,meta)
    ids=set()
    for year in range(2019,2027):
        file=history_raw/f'seattle_{year}.csv.gz';record=json.loads(file.with_name(file.name+'.manifest.json').read_text(encoding='utf-8'))
        if sha(file)!=record['sha256']:raise ValueError('historical raw snapshot changed')
        ids.update(pd.read_csv(file,usecols=['report_number'],dtype=str).report_number.dropna())
    frame=pd.read_csv(path,dtype=str);panel,quality=append_frame(history,frame,ids)
    target=panel_dir/'seattle.npz';np.savez_compressed(target,**vars(panel))
    write_json(panel_dir/'seattle.json',dict(**quality,panel_sha256=sha(target),raw_sha256=sha(path),frozen_plan_sha256=frozen_sha,
        source_manifest_sha256=sha(manifest),end_exclusive=END,history_unchanged=True))
    print(f'R13 new October input: {quality["raw_rows"]:,} rows, {quality["new_cases"]:,} new cases',flush=True)
    return panel


def evaluate(out,panel_path,destination=None):
    panel=load_categories(panel_path);frozen=json.loads((out/'frozen_plan.json').read_text(encoding='utf-8'));journal=json.loads((out/'forecast_journal.json').read_text(encoding='utf-8'))
    rows=[];regions=[]
    for item in journal['entries']:
        path=out/item['file']
        if sha(path)!=item['sha256']:raise ValueError('October forecasts changed')
        with np.load(path) as saved:data={k:saved[k].copy() for k in saved.files}
        index=np.searchsorted(panel.time_ns,data['time_ns']);dates=panel.dates[index];target=panel.counts[index]
        for split,end in [('new_october_all','2026-11-01'),('new_october_first21','2026-10-22')]:
            use=np.asarray(dates<pd.Timestamp(end))
            for role in frozen['roles']:
                for g,group in enumerate(GROUPS):
                    probability=data['probability_'+role][use,:,g];alarm=data['alarm_'+role][use,:,g];cutoff=data['cutoff'][use,:,g];y=target[use,:,g]
                    baseline=data['probability_rolling_baseline'][use,:,g]
                    metric=alarm_metrics(y,cutoff,probability,alarm);reference=alarm_metrics(y,cutoff,baseline,data['alarm_rolling_baseline'][use,:,g])
                    intervals=operating_intervals(y>=cutoff,alarm,probability,baseline)
                    row=dict(scenario=item['scenario'],split=split,role=role,group=group,**metric,baseline_ap=reference['ap'],baseline_brier=reference['brier'],**intervals)
                    screen=preliminary_screen(row);row.update({k:v for k,v in screen.items() if k!='conditions'});row['failed_conditions']=';'.join(k for k,v in screen['conditions'].items() if not v)
                    rows.append(row)
                    for d,district in enumerate(panel.districts):regions.append(dict(scenario=item['scenario'],split=split,role=role,group=group,district=str(district),
                        **alarm_metrics(y[:,d:d+1],cutoff[:,d:d+1],probability[:,d:d+1],alarm[:,d:d+1])))
    destination=out if destination is None else Path(destination);destination.mkdir(parents=True,exist_ok=True)
    result=pd.DataFrame(rows);result.to_csv(destination/'metrics.csv',index=False);pd.DataFrame(regions).to_csv(destination/'district_metrics.csv',index=False)
    primary=result[(result.scenario=='primary')&(result.role=='selected')&(result.group=='all')]
    assessment=dict(status='research_prototype_preliminary_new_control',primary_selection=frozen['roles']['selected'],
        new_control='October2026, full month provisional; first21days more mature',source_publication_times_verified=False,
        prospective_forecasts_issued=False,minimum_blocks_for_intervals=4,primary_point_conditions_passed=primary.alarm_point_passed.tolist(),
        primary_full_conditions_passed=primary.full_screen_passed.tolist(),
        interpretation='short new period cannot establish durable usefulness; no tuning on October outcomes')
    write_json(destination/'assessment.json',safe(assessment));return result


def run(history_path=Path('data/processed/expanded_districts_20261119/seattle.npz'),history_raw=Path('data/raw/expanded_districts_20261119'),
        old=Path('results/expanded_district_detector'),out=Path('results/adaptive_seattle_detector'),raw_dir=Path('data/raw/adaptive_seattle_20261122'),panel_dir=Path('data/processed/adaptive_seattle_20261122')):
    import torch
    torch.set_num_threads(4);out.mkdir(parents=True,exist_ok=True)
    old_protocol=json.loads((old/'protocol.json').read_text(encoding='utf-8'));old_frozen=json.loads((old/'frozen_plan.json').read_text(encoding='utf-8'));old_plan=old_frozen['cities']['seattle']
    if sha(history_path)!=old_protocol['panel_sha256']['seattle']:raise ValueError('R12 history changed')
    config=dict(stage='13_last_seattle_recalibration_attempt',history_sha256=sha(history_path),r12_complete_sha256=sha(old/'complete.json'),
        r12_plan_sha256=sha(old/'frozen_plan.json'),code_sha256={name:sha(Path(__file__).parent/name) for name in ('adaptive_alarm.py','adaptive_alarm_experiment.py','offline_forecast.py','expanded_forecast.py','spatiotemporal.py')},
        neural='R12 frozen local ensemble seeds331/337/349; no retraining or feature changes',candidates=candidates(),scenarios=SCENARIOS,
        development='all already inspected:2025 and Jan-Sep2026; both used for operating-point selection',target='unchanged R11/R12 excess above primary adaptive background',
        update='monthly; prior90/180/365days ending28days before update; chronological halves for probability and regional thresholds; targets known at update',
        selection='prefer FPR<=.05 and recall>=.10 in both development periods; maximize worst-period precision then pooledF0.5; FPR-safe fallback maximizesF0.5; no safe point fallback minimizes maximumFPR',
        primary_group='all; secondary groups reuse the primary selected configuration',
        final_control='Seattle October2026; full month preliminary, first21days mature sensitivity; fewer than4paired14dayblocks => no interval claim',
        criteria=dict(precision=.30,recall=.10,fpr=.05,ap_above_optimized_simple=True,brier_improvement=True),
        no_future_weather_or_zero_lag_sensitivity=True,planned_snapshot_date='2026-11-22')
    protocol=out/'protocol.json'
    if protocol.exists():
        if json.loads(protocol.read_text(encoding='utf-8'))!=safe(config):raise ValueError('R13 protocol/input changed; choose a new output directory')
        if (out/'complete.json').exists():
            complete=json.loads((out/'complete.json').read_text(encoding='utf-8'))
            for name,value in complete['artifact_sha256'].items():
                if sha(out/name)!=value:raise ValueError('completed R13 artifact changed: '+name)
            bound=json.loads((out/'new_input.json').read_text(encoding='utf-8'))
            if sha(panel_dir/'seattle.npz')!=bound['panel_sha256']:raise ValueError('completed R13 panel changed')
            return pd.read_csv(out/'metrics.csv')
    else:write_json(protocol,safe(config))
    history=load_categories(history_path);historical={};cutoff=None
    for scenario in SCENARIOS:
        result=infer(history,old,scenario)
        if scenario=='primary':cutoff=result['cutoff']
        raw=raw_scores(result,cutoff,old_plan);check_r12_replay(history,result,raw,old,old_plan,scenario)
        historical[scenario]=raw
    if not (out/'frozen_plan.json').exists():
        rows,policies=develop(history,historical['primary'],cutoff)
        write_json(out/'development_candidates.json',safe(rows));write_json(out/'development_policy_journal.json',safe(policies))
        pd.DataFrame([dict(id=r['id'],method=r['method'],window_days=r['window_days'],alpha=r['alpha'],split=split,**r[split]) for r in rows for split in ('development2025','development2026','combined')]).to_csv(out/'development_metrics.csv',index=False)
        roles=dict(selected=choose_candidate(rows),rolling_neural=choose_candidate(rows,['local_neural']),rolling_baseline=choose_candidate(rows,['recent_calendar','seasonal']),
            static_neural=dict(method='local_neural'),static_adaptive=dict(method='recent_calendar'))
        test_policies={}
        for scenario in SCENARIOS:
            known=available_targets(history,'2026-10-01',SCENARIOS[scenario]['publication_days']);test_policies[scenario]={}
            for role,selection in roles.items():
                if role.startswith('static'):continue
                test_policies[scenario][role]=fit_policy(historical[scenario][selection['method']],cutoff,known,history.dates,'2026-10-01',selection['window_days'],selection['alpha'])
        frozen=dict(protocol_sha256=sha(protocol),frozen_utc=datetime.now(timezone.utc).isoformat(),roles=roles,test_policies=test_policies,raw_new_data_downloaded=False,
            test_outcomes_used_for_selection=False,r12_forecasts_reproduced=True)
        write_json(out/'frozen_plan.json',safe(frozen))
    frozen=json.loads((out/'frozen_plan.json').read_text(encoding='utf-8'))
    if frozen['protocol_sha256']!=sha(protocol):raise ValueError('frozen R13 plan changed')
    panel=new_snapshot(history,history_raw,raw_dir,panel_dir,sha(out/'frozen_plan.json'))
    write_json(out/'new_input.json',dict(panel_sha256=sha(panel_dir/'seattle.npz'),metadata_sha256=sha(panel_dir/'seattle.json'),
        raw_manifest_sha256=sha(raw_dir/'seattle_october.csv.gz.manifest.json'),frozen_plan_sha256=sha(out/'frozen_plan.json')))
    entries=[];directory=out/'forecasts';directory.mkdir(exist_ok=True);primary_cutoff=None
    for scenario in SCENARIOS:
        result=infer(panel,old,scenario)
        if scenario=='primary':primary_cutoff=result['cutoff']
        raw=raw_scores(result,primary_cutoff,old_plan)
        for method in METHODS:np.testing.assert_allclose(raw[method][:len(history.counts)],historical[scenario][method],rtol=2e-6,atol=2e-7)
        use=np.asarray(panel.dates>=pd.Timestamp('2026-10-01'));arrays=dict(time_ns=panel.time_ns[use],cutoff=primary_cutoff[use])
        for role,selection in frozen['roles'].items():
            if role.startswith('static'):probability,alarm=static_forecast(raw[selection['method']][use],selection['method'],old_plan)
            else:probability,alarm=apply_policy(raw[selection['method']][use],frozen['test_policies'][scenario][role])
            arrays['probability_'+role]=probability;arrays['alarm_'+role]=alarm
        path=directory/(scenario+'.npz');np.savez_compressed(path,**arrays);entries.append(dict(file=path.relative_to(out).as_posix(),scenario=scenario,sha256=sha(path),contains_outcomes=False))
    write_json(out/'forecast_journal.json',dict(frozen_plan_sha256=sha(out/'frozen_plan.json'),contains_outcomes=False,entries=entries))
    write_json(out/'test_started.json',dict(forecast_journal_sha256=sha(out/'forecast_journal.json'),frozen_plan_sha256=sha(out/'frozen_plan.json')))
    metrics=evaluate(out,panel_dir/'seattle.npz')
    write_json(out/'complete.json',dict(successful_execution=True,protocol_sha256=sha(protocol),artifact_sha256={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='complete.json'}))
    print(metrics[(metrics.scenario=='primary')&(metrics.group=='all')][['split','role','precision','recall','fpr','ap','statistical_status']].to_string(index=False),flush=True)
    return metrics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path('results/adaptive_seattle_detector'))
    args=parser.parse_args();run(out=args.out)


if __name__=='__main__':main()
