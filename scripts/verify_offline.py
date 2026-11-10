"""Independent raw-count audit and frozen forecast replay for R11."""
from __future__ import annotations
import argparse,json,sqlite3,tempfile
from contextlib import closing
from pathlib import Path
import numpy as np
import pandas as pd
from critical_processes.selected_training import sha
from critical_processes.offline_forecast import (load_categories,causal_inputs,vintage_masks,available_targets,
    calibrate_probability,apply_calibration,regional_thresholds,rise_cutoff)
from critical_processes.offline_experiment import replay_vintage,ledger_entries,evaluate_ledger


def verify_counts(city,panel,meta):
    parts=[];known=set();raw_rows=0
    for name,digest in meta['source_sha256'].items():
        path=Path(name)
        if sha(path)!=digest:raise ValueError('source modified: '+name)
        frame=pd.read_csv(path,dtype=str);raw_rows+=len(frame);sf=city=='san_francisco'
        identifier='incident_id' if sf else 'case_number';date='incident_datetime' if sf else 'date';district='police_district' if sf else 'district'
        category='incident_category' if sf else 'primary_type'
        if '_202610' in path.name:frame=frame[~frame[identifier].isin(known)].copy()
        else:known.update(frame[identifier].dropna())
        region=frame[district].str.upper().str.strip()
        if not sf:region=pd.to_numeric(region,errors='coerce').astype('Int64').astype('string').str.zfill(2)
        parts.append(pd.DataFrame(dict(identifier=frame[identifier],time=pd.to_datetime(frame[date],format='mixed',errors='coerce'),region=region,
            violent=frame[category].isin(['ASSAULT','BATTERY','ROBBERY','Assault','Robbery']),
            report=pd.to_datetime(frame.report_datetime,format='mixed',errors='coerce') if sf else pd.NaT)))
    combined=pd.concat(parts,ignore_index=True).dropna(subset=['identifier','time'])
    combined=combined[combined.region.isin(panel.districts)].copy();conflicts=combined.groupby('identifier').region.nunique()
    combined=combined[~combined.identifier.isin(conflicts[conflicts>1].index)].copy()
    combined['violent']=combined.groupby('identifier').violent.transform('max');combined['report']=combined.groupby('identifier').report.transform('min')
    combined=combined.sort_values('time',kind='stable').drop_duplicates('identifier')
    origin=pd.Timestamp(panel.dates[0]);event=((combined.time.to_numpy(dtype='datetime64[ns]').astype('int64')-origin.value)//(6*3600*10**9)).astype(int)
    area=combined.region.map({d:i for i,d in enumerate(panel.districts)}).to_numpy(dtype=int);group=np.where(combined.violent,2,1)
    expected=np.zeros_like(panel.counts);np.add.at(expected,(event,area,np.zeros(len(event),dtype=int)),1);np.add.at(expected,(event,area,group),1)
    np.testing.assert_array_equal(expected,panel.counts);np.testing.assert_array_equal(event,panel.event_bins)
    np.testing.assert_array_equal(area,panel.area_indices);np.testing.assert_array_equal(group,panel.group_indices)
    if city=='san_francisco':
        report=(combined.report.to_numpy(dtype='datetime64[ns]').astype('int64')-origin.value)//(6*3600*10**9)
        report[combined.report.isna().to_numpy()]=len(panel.dates)+10000;np.testing.assert_array_equal(report,panel.report_bins)
    with np.load(meta['centroid_source']) as old:
        np.testing.assert_array_equal(panel.counts[:len(old['counts']),:,0],old['counts'])
        np.testing.assert_array_equal(panel.centroids,old['centroids'])
    assert int(panel.counts[:,:,0].sum())==meta['cases'] and raw_rows==meta['raw_rows']
    return raw_rows


def main():
    import torch
    torch.set_num_threads(min(4,torch.get_num_threads()))
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path('results/offline_readiness'))
    parser.add_argument('--panels',type=Path,default=Path('data/processed/offline_20261110'));args=parser.parse_args();out=args.out
    config=json.loads((out/'protocol.json').read_text(encoding='utf-8'));frozen=json.loads((out/'frozen_plan.json').read_text(encoding='utf-8'))
    complete=json.loads((out/'complete.json').read_text(encoding='utf-8'))
    for name,digest in complete['artifact_sha256'].items():assert sha(out/name)==digest,name
    for name,digest in config['code_sha256'].items():assert sha(Path('src/critical_processes')/name)==digest,name
    assert not frozen['selection_used_october'];assert frozen['protocol_sha256']==sha(out/'protocol.json')
    for name,digest in frozen['model_sha256'].items():assert sha(out/name)==digest,name
    journal=json.loads((out/'forecast_journal.json').read_text(encoding='utf-8'));started=json.loads((out/'evaluation_started.json').read_text(encoding='utf-8'))
    assert not journal['contains_outcomes'] and journal['frozen_plan_sha256']==sha(out/'frozen_plan.json')
    assert started['forecast_journal_sha256']==sha(out/'forecast_journal.json')
    entries=list(ledger_entries(out));source_rows=0;forecast_files=0
    for city,vintages in frozen['cities'].items():
        panel=load_categories(args.panels/(city+'.npz'));meta=json.loads((args.panels/(city+'.json')).read_text(encoding='utf-8'))
        assert sha(args.panels/(city+'.npz'))==config['panel_sha256'][city]
        assert sha(args.panels/(city+'.json'))==config['metadata_sha256'][city]
        source_rows+=verify_counts(city,panel,meta)
        for asof,plan in vintages.items():
            masks=vintage_masks(panel.dates,asof);features,seasonal,adaptive,calendars=causal_inputs(panel,masks['train_end'])
            assert calendars==plan['calendars']
            train=features[masks['train']].reshape(-1,features.shape[-1]);mean=train.mean(0);scale=train.std(0);scale[scale<1e-6]=1.
            for name in plan['model_paths']:
                path=out/'models'/city/asof/name;record=torch.load(path,map_location='cpu',weights_only=False);history=pd.read_csv(path.with_suffix('.csv'))
                np.testing.assert_array_equal(mean,record['mean']);np.testing.assert_array_equal(scale,record['scale'])
                valid=history.dropna(subset=['validation_loss']);best=valid.sort_values(['validation_loss','epoch']).iloc[0]
                assert record['selected_epoch']==int(best.epoch);np.testing.assert_allclose(record['validation_loss'],best.validation_loss,atol=1e-12)
                assert record['asof']==asof and record['district_ids']==panel.districts.tolist()
            table=pd.DataFrame(plan['validation'])
            for group in panel.groups:
                selected=table[table.group==group]
                assert plan['count_primary'][str(group)]==selected[selected.method!='neural_direct'].sort_values(['nb_nll','method']).iloc[0]['method']
                assert plan['rise_primary'][str(group)]==selected.sort_values(['ap','method'],ascending=[False,True]).iloc[0]['method']
            for scenario in config['scenarios']:
                means,scores,raw,cutoff=replay_vintage(panel,plan,out/'models'/city/asof,scenario)
                if scenario=='primary':
                    calibration_y=available_targets(panel,asof)
                    for method in config['methods']:
                        recalibrated=np.empty_like(raw[method])
                        for g in range(len(panel.groups)):
                            use=masks['probability_calibration'];fit=calibrate_probability(raw[method][use,:,g],calibration_y[use,:,g]>=cutoff[use,:,g])
                            np.testing.assert_allclose([fit['slope'],fit['intercept']],[plan['probability_calibration'][method][g]['slope'],plan['probability_calibration'][method][g]['intercept']],atol=1e-8)
                            recalibrated[:,:,g]=apply_calibration(raw[method][:,:,g],fit)
                        use=masks['alarm_calibration'];threshold,notes=regional_thresholds(recalibrated[use],calibration_y[use]>=cutoff[use])
                        np.testing.assert_allclose(threshold,plan['thresholds'][method],atol=1e-8);assert notes==plan['threshold_notes'][method]
                for entry in entries:
                    if (entry['city'],entry['asof'],entry['scenario'])!=(city,asof,scenario):continue
                    with np.load(out/entry['file']) as saved:
                        assert 'counts' not in saved.files and 'observed_count' not in saved.files
                        index=np.searchsorted(panel.time_ns,saved['time_ns']);np.testing.assert_array_equal(cutoff[index],saved['cutoff'])
                        for method in config['methods']:
                            np.testing.assert_allclose(means[method][index],saved['mean_'+method],atol=2e-7,rtol=2e-6)
                            np.testing.assert_allclose(scores[method][index],saved['risk_'+method],atol=2e-7,rtol=2e-6)
                    forecast_files+=1
    with tempfile.TemporaryDirectory() as directory:
        evaluate_ledger(out,args.panels,config['bootstrap_samples'],destination=directory)
        for name in ('metrics.csv','monthly_metrics.csv','district_metrics.csv','paired_differences.csv','october_examples.csv'):
            pd.testing.assert_frame_equal(pd.read_csv(out/name),pd.read_csv(Path(directory)/name),check_dtype=False,rtol=1e-10,atol=1e-10)
    assert forecast_files==len(entries)
    readiness=json.loads((out/'readiness.json').read_text(encoding='utf-8'));assert not readiness['real_pilot_performed'] and not readiness['historical_publication_times_verified']
    if (args.panels/'observations.sqlite').exists():
        with closing(sqlite3.connect(args.panels/'observations.sqlite')) as db:
            for city in frozen['cities']:
                earliest=db.execute('SELECT MIN(first_seen) FROM observations WHERE source=?',(city,)).fetchone()[0]
                recorded=db.execute('SELECT MIN(observed_at) FROM snapshots WHERE source=?',(city,)).fetchone()[0]
                assert earliest and earliest==recorded
    print(f'Verified {source_rows:,} raw rows, exact category grids and unchanged R10 history, all weights/splits/calibrations, {forecast_files} monthly forecast files and all result tables.')


if __name__=='__main__':main()
