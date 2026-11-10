from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from .n_accuracy import empirical_auc
from .selected_training import sha,write_json
from .spatiotemporal import (
    load_panel,district_graph,split_masks,fit_calendar,calendar_prediction,forecast_features,recent_calendar,
    exponential_history,fit_hawkes_count,fit_dispersion,poisson_nll,negative_binomial_nll,spike_probability,
    average_precision,fit_count_network,predict_count_network,counts_available_by,observed_memory,
)


METHODS=('seasonal','recent_calendar','temporal_hawkes','spatial_hawkes','neural_ensemble')


def safe(value):
    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    if isinstance(value,np.generic):value=value.item()
    return None if isinstance(value,float) and not np.isfinite(value) else value


def alarm_threshold(scores,labels,alpha=.05):
    values=np.sort(np.asarray(scores)[~np.asarray(labels,dtype=bool)])
    if not len(values):raise ValueError('calibration requires negative cells')
    rank=math.ceil((len(values)+1)*(1-alpha))
    return float(np.nextafter(values[rank-1],np.inf)) if rank<=len(values) else float(np.nextafter(1.,np.inf))


def evaluate_counts(counts,means,cutoffs,alpha,threshold):
    scores=spike_probability(means,cutoffs,alpha);labels=counts>cutoffs;alarms=scores>=threshold
    positive,negative=int(labels.sum()),int((~labels).sum());tp=int((labels&alarms).sum());fp=int((~labels&alarms).sum())
    district_false=np.divide((~labels&alarms).sum(0),(~labels).sum(0),out=np.zeros(counts.shape[1],dtype=float),where=(~labels).sum(0)>0)
    r=1/alpha;lower=nbinom.ppf(.025,r,r/(r+means));upper=nbinom.ppf(.975,r,r/(r+means))
    capacity=max(1,math.ceil(.2*counts.shape[1]));top=np.argsort(-means,axis=1)[:,:capacity]
    caught=np.take_along_axis(counts,top,axis=1).sum()
    return safe(dict(cells=counts.size,observed_events=int(counts.sum()),mean_observed_count=float(counts.mean()),mean_predicted_count=float(means.mean()),
        mae=float(np.abs(counts-means).mean()),rmse=float(np.sqrt(((counts-means)**2).mean())),poisson_nll=float(poisson_nll(counts,means).mean()),
        nb_nll=float(negative_binomial_nll(counts,means,alpha).mean()),nb_dispersion=alpha,interval95_coverage=float(((counts>=lower)&(counts<=upper)).mean()),
        interval95_mean_width=float((upper-lower).mean()),spike_prevalence=positive/counts.size,spike_positives=positive,spike_true_positive=tp,spike_false_positive=fp,
        spike_recall=tp/positive if positive else None,spike_false_positive_rate=fp/negative if negative else None,
        maximum_district_false_positive_rate=float(district_false.max()),spike_precision=tp/(tp+fp) if tp+fp else None,
        spike_brier=float(((scores-labels)**2).mean()),spike_average_precision=average_precision(labels,scores),
        spike_auc=float(empirical_auc(scores[~labels],scores[labels])),alarm_threshold=threshold,
        top_district_count=capacity,total_districts=counts.shape[1],top_district_event_capture=float(caught/counts.sum()) if counts.sum() else None))


def paired_block_improvement(counts,primary,baseline,bins_day,seed=173,bootstrap=1000):
    delta=(np.abs(counts-primary)-np.abs(counts-baseline)).sum(axis=1);weight=np.full(len(delta),counts.shape[1]);block=bins_day*14
    sums=np.array([delta[i:i+block].sum() for i in range(0,len(delta),block)])
    weights=np.array([weight[i:i+block].sum() for i in range(0,len(delta),block)])
    rng=np.random.default_rng(seed);draw=[]
    for _ in range(bootstrap):
        index=rng.integers(0,len(sums),len(sums));draw.append(sums[index].sum()/weights[index].sum())
    low,high=np.quantile(draw,[.025,.975])
    return dict(primary_minus_seasonal_mae=float(delta.sum()/weight.sum()),block_bootstrap_ci95_lower=float(low),block_bootstrap_ci95_upper=float(high),
        block_days=14,bootstrap_samples=bootstrap,method='resample complete14day blocks with all districts jointly, conditional on model selection')


def prepare_predictions(panel,output,lag,config):
    import torch
    dates=panel.dates;masks=split_masks(dates);bins_day=24//panel.bin_hours
    calendar=fit_calendar(panel);base,cutoffs=calendar_prediction(calendar,dates);graph=district_graph(panel.centroids)
    training_counts=counts_available_by(panel,'2024-01-01');validation_counts=counts_available_by(panel,'2024-07-01');calibration_counts=counts_available_by(panel,'2025-01-01')
    learning_counts=training_counts.copy();learning_counts[masks['validation']]=validation_counts[masks['validation']]
    features=forecast_features(panel,base,graph,lag);predictions={'seasonal':base,'recent_calendar':recent_calendar(panel,base,bins_day,lag)}
    specifications={'seasonal':dict(calendar=calendar),'recent_calendar':dict(window_days=28,prior_days=7,ratio_clip=[.25,4.])};candidates=[]
    for spatial,method in ((False,'temporal_hawkes'),(True,'spatial_hawkes')):
        choices=[]
        for tau in sorted(set((1,bins_day,3*bins_day))):
            memory=observed_memory(panel,tau,lag*bins_day)
            mean,spec=fit_hawkes_count(training_counts,base,memory,graph,masks['train'],spatial)
            score=float(poisson_nll(validation_counts[masks['validation']],mean[masks['validation']]).mean())
            candidates.append(dict(method=method,tau_bins=tau,validation_poisson_nll=score,**spec));choices.append((score,tau,mean,spec))
        chosen=min(choices,key=lambda item:(item[0],item[1]));predictions[method]=chosen[2];specifications[method]=dict(tau_bins=chosen[1],**chosen[3])
    checkpoint_paths=[]
    for seed in config['model_seeds']:
        path=output/'models'/f'lag{lag}_seed{seed}.pt'
        if not path.exists():
            record,history=fit_count_network(features,learning_counts,base,masks['train'],masks['validation'],seed,config['epochs'])
            record.update(information_lag_days=lag,bin_hours=panel.bin_hours,district_ids=panel.districts.tolist())
            torch.save(record,path);history.to_csv(path.with_suffix('.csv'),index=False)
        checkpoint_paths.append(path)
    predictions['neural_ensemble']=np.mean([predict_count_network(features,base,path) for path in checkpoint_paths],axis=0)
    specifications['neural_ensemble']=dict(model_paths=[path.name for path in checkpoint_paths],epochs=config['epochs'],seeds=config['model_seeds'])
    validation_rows=[];thresholds={};dispersions={}
    for method,mean in predictions.items():
        dispersion=fit_dispersion(training_counts[masks['train']],mean[masks['train']]);dispersions[method]=dispersion
        score=spike_probability(mean[masks['calibration']],cutoffs[masks['calibration']],dispersion)
        thresholds[method]=alarm_threshold(score,calibration_counts[masks['calibration']]>cutoffs[masks['calibration']])
        row=evaluate_counts(validation_counts[masks['validation']],mean[masks['validation']],cutoffs[masks['validation']],dispersion,thresholds[method])
        # Model selection uses count likelihood/AP, independent of later calibration threshold.
        validation_rows.append(dict(method=method,**row))
    table=pd.DataFrame(validation_rows)
    count_primary=table.sort_values(['nb_nll','method']).iloc[0]['method']
    spike_primary=table.sort_values(['spike_average_precision','method'],ascending=[False,True]).iloc[0]['method']
    return predictions,dict(calendar=calendar,graph=graph.tolist(),specifications=specifications,validation=validation_rows,
        count_primary=count_primary,spike_primary=spike_primary,thresholds=thresholds,dispersions=dispersions,
        hawkes_candidates=candidates),cutoffs


def run_spatial_experiment(panel_dir='data/processed/spatiotemporal_20261110',out='results/spatiotemporal_forecast',epochs=100,
        model_seeds=(137,149,163),information_lags=(0,7),bin_hours=6,bootstrap_samples=1000):
    import torch
    torch.set_num_threads(min(4,torch.get_num_threads()));panel_dir,out=Path(panel_dir),Path(out)
    city_names=('chicago','san_francisco')
    code_files=('spatiotemporal.py','spatial_experiment.py','spatial_report.py')
    protocol=dict(stage='10_district_time_count_forecast',bin_hours=bin_hours,city_names=list(city_names),model_seeds=list(model_seeds),epochs=epochs,
        information_lags=list(information_lags),primary_information_lag_days=7,bootstrap_samples=bootstrap_samples,
        split={'train':'2019-03-02..2023-12-31','validation':'2024-01-01..2024-06-30','calibration':'2024-07-01..2024-12-31',
            'test2025':'2025-01-01..2025-12-31','test2026':'2026-01-01..2026-09-30'},
        target='count of selected registered crime cases per police district in next interval; spike is strict exceedance of training95% weekday-slot district quantile',
        primary='per-city and information-lag model selected on validation NB NLL for counts and AP for spikes; thresholds only calibration',
        alarm_control='pooled negative district-time-cell FPR target5% on calibration, not whole-trajectory or simultaneous95% guarantee',
        spatial_graph='three nearest training-event district centroids, exponential distance weights, row normalized; fixed before validation',
        models=list(METHODS),hawkes='nonnegative delayed discrete count Hawkes regression with free background multiplier and coefficient sum<=.99; not inferred continuous-time n',
        neural='embedding6+MLP32-16, dropout.1, Poisson NLL, AdamW lr.002 wd.001, validation every10epochs; ensemble all3seeds',
        uncertainty='working NB dispersion fit on train per model; observed interval coverage reported; paired14day-block bootstrap for test MAE differences',
        data_availability='SF report-time-aware histories and fit-date-censored train/validation/calibration labels; historical edits unavailable. Chicago incident-date archive with lag7d approximation, not guaranteed real-time availability',
        panel_sha256={city:sha(panel_dir/(city+'.npz')) for city in city_names},
        source_metadata_sha256={city:sha(panel_dir/(city+'.json')) for city in city_names},
        code_sha256={name:sha(Path(__file__).parent/name) for name in code_files})
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text(encoding='utf-8'))!=protocol:raise ValueError('different spatial protocol; choose another output directory')
    else:write_json(out/'protocol.json',protocol)
    if (out/'complete.json').exists():
        complete=json.loads((out/'complete.json').read_text(encoding='utf-8'))
        for name,digest in complete['artifact_sha256'].items():
            if sha(out/name)!=digest:raise ValueError('spatial artifact changed: '+name)
        return pd.read_csv(out/'test_metrics.csv')
    prepared={};frozen={'protocol_sha256':sha(out/'protocol.json'),'cities':{},'test_used_for_selection':False};models={}
    for city in city_names:
        panel=load_panel(panel_dir/(city+'.npz'),bin_hours);prepared[city]=panel
        directory=out/city;directory.mkdir(exist_ok=True);(directory/'models').mkdir(exist_ok=True);models[city]={};frozen['cities'][city]={}
        for lag in information_lags:
            predictions,plan,cutoffs=prepare_predictions(panel,directory,lag,protocol)
            models[city][lag]=(predictions,cutoffs);frozen['cities'][city][str(lag)]=plan
            pd.DataFrame(plan['validation']).to_csv(directory/f'validation_lag{lag}.csv',index=False)
            pd.DataFrame(plan['hawkes_candidates']).to_csv(directory/f'hawkes_candidates_lag{lag}.csv',index=False)
            print(f'Spatial validation {city}/lag{lag}: count={plan["count_primary"]}, spike={plan["spike_primary"]}',flush=True)
    frozen['model_sha256']={str(path.relative_to(out)).replace('\\','/'):sha(path) for path in out.rglob('*.pt')}
    frozen_path=out/'frozen_plan.json'
    if frozen_path.exists() and json.loads(frozen_path.read_text(encoding='utf-8'))!=safe(frozen):raise ValueError('spatial plan already frozen')
    write_json(frozen_path,safe(frozen));write_json(out/'test_started.json',dict(frozen_plan_sha256=sha(frozen_path)))
    rows=[];district_rows=[];effects=[];example_rows=[]
    for city,panel in prepared.items():
        masks=split_masks(panel.dates)
        for lag in information_lags:
            predictions,cutoffs=models[city][lag];plan=frozen['cities'][city][str(lag)]
            arrays=dict(counts=panel.counts,time_ns=panel.time_ns,districts=panel.districts,spike_cutoffs=cutoffs)
            for method,mean in predictions.items():arrays[method]=mean
            np.savez_compressed(out/city/f'predictions_lag{lag}.npz',**arrays)
            for split in ('test2025','test2026'):
                use=masks[split];target=panel.counts[use];dates=panel.dates[use]
                for method,mean in predictions.items():
                    metric=evaluate_counts(target,mean[use],cutoffs[use],plan['dispersions'][method],plan['thresholds'][method])
                    rows.append(dict(city=city,information_lag_days=lag,split=split,method=method,count_primary=method==plan['count_primary'],spike_primary=method==plan['spike_primary'],**metric))
                    for district_index,district in enumerate(panel.districts):
                        one=evaluate_counts(target[:,district_index:district_index+1],mean[use,district_index:district_index+1],cutoffs[use,district_index:district_index+1],plan['dispersions'][method],plan['thresholds'][method])
                        district_rows.append(dict(city=city,district=district,information_lag_days=lag,split=split,method=method,**one))
                    # Fixed first seven days of each held-out period: example selection never follows errors.
                    for time_index in range(min(len(dates),7*(24//bin_hours))):
                        for district_index,district in enumerate(panel.districts):
                            mu=float(mean[use][time_index,district_index]);cut=int(cutoffs[use][time_index,district_index]);actual=int(target[time_index,district_index])
                            probability=float(spike_probability(np.array(mu),np.array(cut),plan['dispersions'][method]))
                            example_rows.append(dict(city=city,district=district,information_lag_days=lag,split=split,method=method,forecast_start=str(dates[time_index]),
                                forecast_end=str(dates[time_index]+pd.Timedelta(hours=bin_hours)),expected_count=mu,observed_count=actual,spike_cutoff=cut,spike_probability=probability,
                                alarm=probability>=plan['thresholds'][method],observed_spike=actual>cut))
                primary=plan['count_primary'];effect=paired_block_improvement(target,predictions[primary][use],predictions['seasonal'][use],24//bin_hours,bootstrap=bootstrap_samples)
                effects.append(dict(city=city,information_lag_days=lag,split=split,primary=primary,**effect))
    metrics=pd.DataFrame(rows);metrics.to_csv(out/'test_metrics.csv',index=False)
    pd.DataFrame(district_rows).to_csv(out/'district_metrics.csv',index=False);pd.DataFrame(effects).to_csv(out/'paired_count_errors.csv',index=False)
    pd.DataFrame(example_rows).to_csv(out/'forecast_examples.csv',index=False)
    from .spatial_report import write_spatial_report
    write_spatial_report(out,panel_dir)
    write_json(out/'complete.json',dict(artifact_sha256={str(path.relative_to(out)).replace('\\','/'):sha(path) for path in out.rglob('*') if path.is_file() and path.name!='complete.json'}))
    return metrics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--panel-dir',default='data/processed/spatiotemporal_20261110');parser.add_argument('--out',default='results/spatiotemporal_forecast')
    parser.add_argument('--hours',type=int,choices=[6,24],default=6);parser.add_argument('--epochs',type=int,default=100)
    args=parser.parse_args();run_spatial_experiment(panel_dir=args.panel_dir,out=args.out,bin_hours=args.hours,epochs=args.epochs)


if __name__=='__main__':main()
