"""R12 immutable four-city training, prediction freeze and held-period evaluation."""
from __future__ import annotations
import argparse, json, platform, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from .selected_training import sha,write_json
from .spatial_experiment import safe
from .spatiotemporal import fit_dispersion,average_precision,DistrictPanel,fit_calendar
from .offline_forecast import (load_categories,available_targets,causal_inputs,rise_cutoff,
    calibrate_probability,apply_calibration,regional_thresholds,evaluate_forecast)
from .expanded_forecast import (chronological_masks,quarterly_release_bins,delayed_weather,fit_neural,predict_neural,
    operating_intervals,practical_screen)

CITIES=('chicago','san_francisco','nyc','seattle')
METHODS=('seasonal','recent_calendar','local_neural','pooled_neural','weather_neural','boosted15','boosted31')
SEEDS=(331,337,349)
SCENARIOS={'primary':dict(lag_days=7,publication_days=3,weather_lag=5,nyc_quarter_release_days=45,nyc_lag_days=150),
    'delayed_stress':dict(lag_days=14,publication_days=7,weather_lag=10,nyc_quarter_release_days=90,nyc_lag_days=210)}


def assemble(panel_dir,history_dir,scenario='primary',calendars=None):
    settings=SCENARIOS[scenario];blocks=[];offset=0;plans={}
    for c,city in enumerate(CITIES):
        directory=history_dir if c<2 else panel_dir;panel=load_categories(directory/(city+'.npz'))
        publication_days=settings['publication_days'];chosen_calendar=None if calendars is None else calendars[city]
        if city=='nyc':
            panel=replace(panel,report_bins=quarterly_release_bins(panel.report_bins,panel.time_ns[0],settings['nyc_quarter_release_days']))
            publication_days=0
            if chosen_calendar is None:
                # The actual fit date is2025-01-01. Only pre2024 events enter the
                # calendar; their labels may mature before that actual fit date.
                known=available_targets(panel,'2025-01-01',0)
                chosen_calendar=[fit_calendar(DistrictPanel(known[:,:,g],panel.time_ns,panel.districts,panel.centroids,6),'2024-01-01') for g in range(3)]
        history_lag=settings['nyc_lag_days'] if city=='nyc' else settings['lag_days']
        features,seasonal,adaptive,fitted=causal_inputs(panel,'2024-01-01',history_lag,publication_days,chosen_calendar)
        dates=panel.dates;keep=dates<pd.Timestamp('2026-10-01');dates=dates[keep]
        features=features[keep];seasonal=seasonal[keep];adaptive=adaptive[keep]
        shape=adaptive.shape;category=np.empty((*shape,3),dtype='int16')
        category[:,:,:,0]=np.arange(len(panel.districts))[None,:,None]+offset
        category[:,:,:,1]=c;category[:,:,:,2]=np.arange(3)[None,None,:]
        with np.load(panel_dir/(city+'_weather.npz')) as w:weather=delayed_weather(dates,w['day_ns'],w['values'],settings['weather_lag'])
        weather=np.broadcast_to(weather[:,None,None,:],(*shape,weather.shape[-1]))
        x=np.concatenate([features,weather],axis=-1).reshape(-1,features.shape[-1]+8).astype('float32')
        masks=chronological_masks(dates,city);learning=np.zeros(shape,dtype='int32')
        for names,cutoff in [(('train',),'2024-01-01'),(('validation',),'2024-07-01'),
            (('probability_calibration','alarm_calibration'),'2025-01-01')]:
            known=available_targets(panel,'2025-01-01' if city=='nyc' else cutoff,publication_days)[keep]
            for name in names:learning[masks[name]]=known[masks[name]]
        blocks.append(dict(city=city,shape=shape,dates=dates,x=x,category=category.reshape(-1,3),
            adaptive=adaptive.ravel(),seasonal=seasonal.ravel(),target=learning.ravel(),
            cutoff=rise_cutoff(adaptive).ravel(),masks={k:np.broadcast_to(v[:,None,None],shape).ravel() for k,v in masks.items()},
            districts=panel.districts.tolist(),groups=panel.groups.tolist(),panel=panel))
        plans[city]=fitted;offset+=len(panel.districts)
        print(f'R12 features {scenario}/{city}: {shape}, {x.shape[1]} features',flush=True)
    combined={key:np.concatenate([block[key] for block in blocks]) for key in ('x','category','adaptive','seasonal','target','cutoff')}
    combined['masks']={key:np.concatenate([b['masks'][key] for b in blocks]) for key in blocks[0]['masks']}
    start=0
    for block in blocks:block['slice']=slice(start,start+len(block['x']));start+=len(block['x'])
    return combined,blocks,plans


def train_models(data,out,epochs,seeds):
    import joblib,torch
    from sklearn.ensemble import HistGradientBoostingClassifier
    x,cat,base,y=data['x'],data['category'],data['adaptive'],data['target'];masks=data['masks'];directory=out/'models';directory.mkdir(exist_ok=True)
    models={};log=[]
    specifications=[('local_'+city,c,False) for c,city in enumerate(CITIES)]+[('pooled',None,False),('weather',None,True)]
    for name,local,weather in specifications:
        width=x.shape[1] if weather else x.shape[1]-8;records=[]
        for seed in seeds:
            path=directory/f'{name}_s{seed}.pt'
            if path.exists():record=torch.load(path,map_location='cpu',weights_only=False)
            else:
                record,history=fit_neural(x[:,:width],cat,base,y,masks['train'],masks['validation'],seed,epochs,local)
                torch.save(record,path);history.to_csv(path.with_suffix('.csv'),index=False)
            records.append(record);log.append(dict(model=name,seed=seed,selected_epoch=record['selected_epoch'],validation_loss=record['validation_loss']))
            print(f'R12 trained {name}/s{seed}: selected epoch {record["selected_epoch"]}',flush=True)
        models[name]=records
    rng=np.random.default_rng(353);train=np.flatnonzero(masks['train']);chosen=np.concatenate([
        rng.choice(train[cat[train,1]==c],size=min(60000,int((cat[train,1]==c).sum())),replace=False) for c in range(4)])
    design=np.column_stack([x[chosen],cat[chosen]]);labels=y[chosen]>=data['cutoff'][chosen]
    for leaves in (15,31):
        name=f'boosted{leaves}';path=directory/(name+'.joblib')
        if path.exists():model=joblib.load(path)
        else:
            model=HistGradientBoostingClassifier(max_iter=160,learning_rate=.05,max_leaf_nodes=leaves,
                min_samples_leaf=100,l2_regularization=10.,max_bins=127,early_stopping=False,
                categorical_features=[False]*x.shape[1]+[True]*3,random_state=353)
            model.fit(design,labels);joblib.dump(model,path)
        models[name]=model;print(f'R12 trained {name}: {len(chosen):,} city-balanced rows',flush=True)
    pd.DataFrame(log).to_csv(out/'training_summary.csv',index=False)
    return models


def predict_models(data,models):
    x,cat,base=data['x'],data['category'],data['adaptive'];result={}
    for name in ('pooled','weather'):
        width=x.shape[1] if name=='weather' else x.shape[1]-8
        values=[predict_neural(x[:,:width],cat,base,record) for record in models[name]]
        result[name+'_neural']=(np.mean([v[0] for v in values],axis=0),np.mean([v[1] for v in values],axis=0))
    mean=np.empty(len(x));risk=np.empty(len(x))
    for c,city in enumerate(CITIES):
        use=cat[:,1]==c;values=[predict_neural(x[use,:-8],cat[use],base[use],record) for record in models['local_'+city]]
        mean[use]=np.mean([v[0] for v in values],axis=0);risk[use]=np.mean([v[1] for v in values],axis=0)
    result['local_neural']=(mean,risk)
    for leaves in (15,31):
        name=f'boosted{leaves}';risk=np.empty(len(x));model=models[name]
        for start in range(0,len(x),32768):
            sl=slice(start,start+32768);risk[sl]=model.predict_proba(np.column_stack([x[sl],cat[sl]]))[:,1]
        result[name]=(base,risk)
    return result


def fit_selection(data,blocks,prediction):
    plan={};validation=[]
    for block in blocks:
        city=block['city'];sl=block['slice'];shape=block['shape'];group_names=block['groups'];masks={k:v.reshape(shape)[:,0,0] for k,v in block['masks'].items()}
        y=block['target'].reshape(shape);cutoff=block['cutoff'].reshape(shape);means={};raw={};alpha={};calibrations={};thresholds={};notes={}
        for method in METHODS:
            means[method]=(block['seasonal'] if method=='seasonal' else block['adaptive'] if method=='recent_calendar' else prediction[method][0][sl]).reshape(shape)
            alpha[method]=[fit_dispersion(y[masks['train'],:,g],means[method][masks['train'],:,g]) for g in range(3)]
            if method in ('seasonal','recent_calendar'):
                raw[method]=np.stack([nbinom.sf(cutoff[:,:,g]-1,1/alpha[method][g],1/(1+alpha[method][g]*means[method][:,:,g])) for g in range(3)],axis=2)
            else:raw[method]=prediction[method][1][sl].reshape(shape)
            probability=np.empty(shape);calibrations[method]=[]
            for g,group in enumerate(group_names):
                use=masks['validation'];metric=evaluate_forecast(y[use,:,g],means[method][use,:,g],raw[method][use,:,g],cutoff[use,:,g],np.ones(shape[1]),alpha[method][g])
                validation.append(dict(city=city,method=method,group=group,**metric))
                use=masks['probability_calibration'];record=calibrate_probability(raw[method][use,:,g],y[use,:,g]>=cutoff[use,:,g]);calibrations[method].append(record)
                probability[:,:,g]=apply_calibration(raw[method][:,:,g],record)
            use=masks['alarm_calibration'];threshold,note=regional_thresholds(probability[use],y[use]>=cutoff[use])
            thresholds[method]=threshold.tolist();notes[method]=note
        table=pd.DataFrame([r for r in validation if r['city']==city]);selected={};baseline={}
        for group in group_names:
            part=table[table.group==group].sort_values(['ap','method'],ascending=[False,True])
            selected[group]=part.iloc[0]['method'];baseline[group]=part[part.method.isin(['seasonal','recent_calendar'])].iloc[0]['method']
        plan[city]=dict(selected=selected,baseline=baseline,dispersions=alpha,calibrations=calibrations,thresholds=thresholds,
            threshold_notes=notes,districts=block['districts'],groups=group_names)
    return safe(plan),pd.DataFrame(validation)


def freeze_forecasts(data,blocks,prediction,plan,out,scenario,primary_cutoffs):
    entries=[];directory=out/'forecasts';directory.mkdir(exist_ok=True)
    for block in blocks:
        city=block['city'];sl=block['slice'];shape=block['shape'];dates=block['dates'];keep=np.asarray(dates>=pd.Timestamp('2025-01-01'))
        cutoff=primary_cutoffs[city];record=plan[city];arrays={'time_ns':dates.as_unit('ns').asi8[keep],'cutoff':cutoff[keep]}
        for method in METHODS:
            mean=(block['seasonal'] if method=='seasonal' else block['adaptive'] if method=='recent_calendar' else prediction[method][0][sl]).reshape(shape)
            if method in ('seasonal','recent_calendar'):
                raw=np.stack([nbinom.sf(cutoff[:,:,g]-1,1/record['dispersions'][method][g],1/(1+record['dispersions'][method][g]*mean[:,:,g])) for g in range(3)],axis=2)
            else:raw=prediction[method][1][sl].reshape(shape)
            risk=np.stack([apply_calibration(raw[:,:,g],record['calibrations'][method][g]) for g in range(3)],axis=2)
            arrays['mean_'+method]=mean[keep].astype('float32');arrays['risk_'+method]=risk[keep].astype('float32')
        path=directory/f'{city}_{scenario}.npz';np.savez_compressed(path,**arrays)
        entries.append(dict(city=city,scenario=scenario,file=path.relative_to(out).as_posix(),sha256=sha(path),contains_outcomes=False))
    return entries


def evaluate(out,panel_dir,history_dir,bootstrap=1000,destination=None):
    plan=json.loads((out/'frozen_plan.json').read_text(encoding='utf-8'))['cities'];journal=json.loads((out/'forecast_journal.json').read_text(encoding='utf-8'))
    rows=[];regional=[];headline=[];monthly=[]
    for entry in journal['entries']:
        city=entry['city'];record=plan[city];path=out/entry['file']
        if sha(path)!=entry['sha256']:raise ValueError('forecast file changed')
        panel=load_categories((history_dir if city in CITIES[:2] else panel_dir)/(city+'.npz'))
        with np.load(path) as saved:values={k:saved[k].copy() for k in saved.files}
        dates=pd.DatetimeIndex(values['time_ns'].astype('datetime64[ns]'));indices=np.searchsorted(panel.time_ns,values['time_ns']);y=panel.counts[indices]
        for split in ('test2025','test2026'):
            use=chronological_masks(dates)[split]
            for g,group in enumerate(record['groups']):
                base=record['baseline'][group];base_risk=values['risk_'+base][use,:,g];labels=y[use,:,g]>=values['cutoff'][use,:,g]
                baseline_ap=average_precision(labels,base_risk)
                for method in METHODS:
                    mean=values['mean_'+method][use,:,g];risk=values['risk_'+method][use,:,g];threshold=np.array(record['thresholds'][method])[:,g]
                    metric=evaluate_forecast(y[use,:,g],mean,risk,values['cutoff'][use,:,g],threshold,record['dispersions'][method][g])
                    row=dict(city=city,scenario=entry['scenario'],split=split,group=group,method=method,selected=record['selected'][group]==method,
                        baseline=base,baseline_ap=baseline_ap,**metric);rows.append(row)
                    for d,district in enumerate(record['districts']):
                        one=evaluate_forecast(y[use,d:d+1,g],mean[:,d:d+1],risk[:,d:d+1],values['cutoff'][use,d:d+1,g],threshold[d],record['dispersions'][method][g])
                        regional.append(dict(city=city,scenario=entry['scenario'],split=split,group=group,method=method,district=district,**one))
                    if row['selected']:
                        intervals=operating_intervals(labels,risk>=threshold,risk,base_risk,bootstrap)
                        result=dict(**row,**intervals);result['practical_screen_passed']=practical_screen(result)['passed'];result['failed_requirements']=';'.join(k for k,v in practical_screen(result)['requirements'].items() if not v)
                        result['evaluation_status']='new_city_predeclared_holdout' if city in CITIES[2:] else 'previously_inspected_period'
                        if split=='test2026':result['evaluation_status']+=';preliminary_snapshot'
                        headline.append(result)
                for month in dates[use].to_period('M').unique():
                    take=np.asarray(dates.to_period('M')==month);method=record['selected'][group]
                    metric=evaluate_forecast(y[take,:,g],values['mean_'+method][take,:,g],values['risk_'+method][take,:,g],values['cutoff'][take,:,g],np.array(record['thresholds'][method])[:,g],record['dispersions'][method][g])
                    monthly.append(dict(city=city,scenario=entry['scenario'],month=str(month),group=group,method=method,**metric))
    destination=out if destination is None else Path(destination);destination.mkdir(parents=True,exist_ok=True)
    for name,table in [('metrics',rows),('district_metrics',regional),('headline',headline),('monthly_metrics',monthly)]:pd.DataFrame(table).to_csv(destination/(name+'.csv'),index=False)
    primary=[r for r in headline if r['scenario']=='primary' and r['group']=='all' and r['city'] in CITIES[2:]]
    status=dict(status='research_prototype_offline_only',practical_screen='precision>=.30, recall>=.10, FPR<=.05, AP>validation-chosen simple baseline, paired Brier improvement CI<0',
        selected_before_test=True,prospective_validation_performed=False,historical_publication_times_verified=False,
        novel_city_all_group_slices=len(primary),novel_city_all_group_slices_passing=sum(r['practical_screen_passed'] for r in primary),
        all_new_city_slices_pass=all(r['practical_screen_passed'] for r in primary),
        warning='an exploratory engineering screen, not proof of operational utility or intervention benefit')
    write_json(destination/'assessment.json',status)
    return pd.DataFrame(headline)


def run(panel_dir=Path('data/processed/expanded_districts_20261119'),history_dir=Path('data/processed/offline_20261110'),out=Path('results/expanded_district_detector'),epochs=120,seeds=SEEDS):
    import sklearn,torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4);threadpool_limits(4);out.mkdir(parents=True,exist_ok=True)
    code=['expanded_forecast.py','expanded_experiment.py','offline_forecast.py','spatiotemporal.py']
    inputs={city:(history_dir if i<2 else panel_dir)/(city+'.npz') for i,city in enumerate(CITIES)}
    config=dict(stage='12_four_city_expansion',cities=list(CITIES),groups=['all','property','assault_robbery'],methods=list(METHODS),
        seeds=list(seeds),epochs=epochs,scenarios=SCENARIOS,
        splits={k:[str(pd.date_range('2019-01-01','2026-10-01',freq='6h',inclusive='left')[v][i]) for i in (0,-1)] for k,v in chronological_masks(pd.date_range('2019-01-01','2026-10-01',freq='6h',inclusive='left')).items()},
        target='R11: N >= ceil(primary adaptive background + max(2,.5*primary adaptive background))',
        selection='per city/group maximum validation AP; tied by method name; baseline chosen on validation AP; no test tuning',
        alarm='regional 5% negative-cell FPR thresholds on separate past calibration; no guarantee on future cells',
        probability_calibration='monotone Platt on July2024 only; alarm calibration August1-December3',
        train_target_availability='known by2024-01-01; validation by2024-07-01; calibration by2025-01-01; 28day maturity exclusions',
        nyc_pretest_amendment='quarterly source: report-quarter end+45days assumed release; stress+90days; observable history ends150days ago, stress210days; 2019Sep training warmup; all labels known at actual fit2025-01-01; calendar uses pre2024 events only; alarm calibration Aug-Sep2024 only',
        neural='local vs city-balanced pooled; same per-city sampling budget; 3 seeds; Poisson residual count plus BCE risk',
        weather='quality-filtered NOAA observations delayed5days; stress10days; no actual future weather; historical publication times assumed',
        trees='240000 city-balanced training-only rows, no class reweighting; 160 iterations, leaves15/31, lr.05, L2=10, no internal shuffled validation',
        practical_screen=dict(minimum_precision=.30,minimum_recall=.10,maximum_fpr=.05,ap_beats_simple_baseline=True,brier_improvement_ci_high_below_zero=True),
        novelty='NYC/Seattle district outcomes not previously used in R10/R11; Chicago/SF holdout periods already inspected; all retrospective,2026preliminary',
        groups_dependent='all includes both exclusive groups; these are distinct targets, not independent replications',
        panel_sha256={city:sha(path) for city,path in inputs.items()},weather_sha256={city:sha(panel_dir/(city+'_weather.npz')) for city in CITIES},
        code_sha256={name:sha(Path(__file__).parent/name) for name in code},preparation_sha256=sha(Path('scripts/prepare_expanded_districts.py')),
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__,sklearn=sklearn.__version__))
    protocol=out/'protocol.json'
    if protocol.exists():
        if json.loads(protocol.read_text(encoding='utf-8'))!=safe(config):raise ValueError('protocol/input changed; choose a new output directory')
        if (out/'complete.json').exists():return pd.read_csv(out/'headline.csv')
    else:write_json(protocol,safe(config))
    data,blocks,calendars=assemble(panel_dir,history_dir);models=train_models(data,out,epochs,seeds)
    prediction=predict_models(data,models);cities,validation=fit_selection(data,blocks,prediction)
    validation.to_csv(out/'validation_metrics.csv',index=False)
    frozen=dict(protocol_sha256=sha(protocol),cities=cities,calendars=calendars,
        model_sha256={path.relative_to(out).as_posix():sha(path) for path in (out/'models').iterdir() if path.suffix in ('.pt','.joblib')})
    write_json(out/'frozen_plan.json',safe(frozen))
    primary_cutoffs={b['city']:b['cutoff'].reshape(b['shape']).copy() for b in blocks}
    entries=freeze_forecasts(data,blocks,prediction,cities,out,'primary',primary_cutoffs)
    del data,blocks,prediction
    data,blocks,_=assemble(panel_dir,history_dir,'delayed_stress',calendars);prediction=predict_models(data,models)
    entries+=freeze_forecasts(data,blocks,prediction,cities,out,'delayed_stress',primary_cutoffs)
    write_json(out/'forecast_journal.json',dict(contains_outcomes=False,frozen_plan_sha256=sha(out/'frozen_plan.json'),entries=entries))
    write_json(out/'test_started.json',dict(forecast_journal_sha256=sha(out/'forecast_journal.json'),frozen_plan_sha256=sha(out/'frozen_plan.json')))
    del data,blocks,prediction
    headline=evaluate(out,panel_dir,history_dir)
    artifacts={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='complete.json'}
    write_json(out/'complete.json',dict(protocol_sha256=sha(protocol),artifact_sha256=artifacts,successful_execution=True))
    print(headline[(headline.scenario=='primary')&(headline.group=='all')][['city','split','method','precision','recall','fpr','practical_screen_passed']].to_string(index=False),flush=True)
    return headline


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--panels',type=Path,default=Path('data/processed/expanded_districts_20261119'))
    parser.add_argument('--history',type=Path,default=Path('data/processed/offline_20261110'))
    parser.add_argument('--out',type=Path,default=Path('results/expanded_district_detector'));parser.add_argument('--epochs',type=int,default=120)
    args=parser.parse_args();run(args.panels,args.history,args.out,args.epochs)


if __name__=='__main__':main()
