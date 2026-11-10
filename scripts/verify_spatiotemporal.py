"""Independent replay of district counts, availability, frozen models and metrics."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from critical_processes.selected_training import sha
from critical_processes.spatiotemporal import (
    load_panel,district_graph,split_masks,fit_calendar,calendar_prediction,forecast_features,recent_calendar,
    observed_memory,counts_available_by,fit_dispersion,poisson_nll,spike_probability,predict_count_network,
)
from critical_processes.spatial_experiment import METHODS,evaluate_counts,alarm_threshold,paired_block_improvement


def main():
    import torch
    torch.set_num_threads(min(4,torch.get_num_threads()))
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path('results/spatiotemporal_forecast'))
    parser.add_argument('--panels',type=Path,default=Path('data/processed/spatiotemporal_20261110'));args=parser.parse_args();out=args.out
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'));frozen=json.loads((out/'frozen_plan.json').read_text(encoding='utf-8'))
    complete=json.loads((out/'complete.json').read_text(encoding='utf-8'))
    for name,digest in complete['artifact_sha256'].items():assert sha(out/name)==digest,name
    for name,digest in protocol['code_sha256'].items():assert sha(Path('src/critical_processes')/name)==digest,name
    assert frozen['protocol_sha256']==sha(out/'protocol.json') and not frozen['test_used_for_selection']
    assert json.loads((out/'test_started.json').read_text(encoding='utf-8'))['frozen_plan_sha256']==sha(out/'frozen_plan.json')
    for name,digest in frozen['model_sha256'].items():assert sha(out/name)==digest
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from prepare_spatiotemporal_data import SOURCES
    rows=[];effects=[];raw_rows=0
    for city in protocol['city_names']:
        panel=load_panel(args.panels/(city+'.npz'),protocol['bin_hours']);meta=json.loads((args.panels/(city+'.json')).read_text(encoding='utf-8'))
        assert sha(args.panels/(city+'.npz'))==protocol['panel_sha256'][city]
        assert sha(args.panels/(city+'.json'))==protocol['source_metadata_sha256'][city]
        pieces=[];source=SOURCES[city]
        for name,digest in meta['source_sha256'].items():
            assert sha(name)==digest
            frame=pd.read_csv(name,dtype=str);raw_rows+=len(frame)
            assert frame[source['category']].isin(source['categories']).all()
            dates=pd.to_datetime(frame[source['date']],format='mixed',errors='coerce')
            district=frame[source['district']].str.upper().str.strip()
            if city=='chicago':district=pd.to_numeric(district,errors='coerce').astype('Int64').astype('string').str.zfill(2)
            report=pd.to_datetime(frame['report_datetime'],format='mixed',errors='coerce') if city=='san_francisco' else pd.Series(pd.NaT,index=frame.index)
            pieces.append(pd.DataFrame(dict(id=frame[source['identifier']],time=dates,district=district,report=report)))
        frame=pd.concat(pieces,ignore_index=True).dropna(subset=['id','time']);frame=frame[frame.district.isin(source['districts'])]
        conflict=frame.groupby('id').district.nunique();bad=conflict[conflict>1].index
        frame=frame[~frame.id.isin(bad)].sort_values('time',kind='stable').drop_duplicates('id')
        left,right=pd.Timestamp(meta['start']),pd.Timestamp(meta['end_exclusive']);frame=frame[(frame.time>=left)&(frame.time<right)]
        event_bins=((frame.time.to_numpy(dtype='datetime64[ns]').astype(np.int64)-left.value)//(protocol['bin_hours']*3600*10**9)).astype(int)
        area=frame.district.map({value:i for i,value in enumerate(panel.districts)}).to_numpy(dtype=int)
        expected=np.zeros_like(panel.counts);np.add.at(expected,(event_bins,area),1);np.testing.assert_array_equal(panel.counts,expected)
        assert int(expected.sum())==meta['incidents']
        if city=='san_francisco':
            reports=(frame.report.to_numpy(dtype='datetime64[ns]').astype(np.int64)-left.value)//(protocol['bin_hours']*3600*10**9)
            reports[frame.report.isna().to_numpy()]=len(panel.counts)+1
            np.testing.assert_array_equal(panel.event_bins,event_bins);np.testing.assert_array_equal(panel.area_indices,area);np.testing.assert_array_equal(panel.report_bins,reports)
        masks=split_masks(panel.dates);calendar=fit_calendar(panel);base,cutoffs=calendar_prediction(calendar,panel.dates);graph=district_graph(panel.centroids)
        train_y=counts_available_by(panel,'2024-01-01');val_y=counts_available_by(panel,'2024-07-01');cal_y=counts_available_by(panel,'2025-01-01')
        for lag in protocol['information_lags']:
            plan=frozen['cities'][city][str(lag)];assert calendar==plan['calendar'];np.testing.assert_allclose(graph,plan['graph'])
            features=forecast_features(panel,base,graph,lag);predictions={'seasonal':base,'recent_calendar':recent_calendar(panel,base,24//panel.bin_hours,lag)}
            for method in ('temporal_hawkes','spatial_hawkes'):
                spec=plan['specifications'][method];memory=observed_memory(panel,spec['tau_bins'],lag*(24//panel.bin_hours))
                scale=np.asarray(spec['normalizing_rates']);adjusted=scale[:,None]*graph/scale[None,:]
                design=np.stack([base,memory,memory@adjusted.T],axis=-1);theta=np.asarray(spec['coefficients'])
                assert (theta>=0).all() and theta[1:].sum()<=.990001
                predictions[method]=np.maximum(design[:,:,:len(theta)]@theta,1e-8)
                candidates=[row for row in plan['hawkes_candidates'] if row['method']==method]
                best=min(candidates,key=lambda row:(row['validation_poisson_nll'],row['tau_bins']))
                assert spec['tau_bins']==best['tau_bins']
                val_score=float(poisson_nll(val_y[masks['validation']],predictions[method][masks['validation']]).mean())
                np.testing.assert_allclose(val_score,best['validation_poisson_nll'],rtol=1e-10,atol=1e-10)
            flat=features[masks['train']].reshape(-1,features.shape[-1]);mean,scale=flat.mean(0),flat.std(0);scale[scale<1e-6]=1.
            members=[]
            for seed in protocol['model_seeds']:
                path=out/city/'models'/f'lag{lag}_seed{seed}.pt';checkpoint=torch.load(path,map_location='cpu',weights_only=False)
                np.testing.assert_allclose(checkpoint['mean'],mean,rtol=1e-12,atol=1e-12);np.testing.assert_allclose(checkpoint['scale'],scale,rtol=1e-12,atol=1e-12)
                history=pd.read_csv(path.with_suffix('.csv'));choice=history.dropna(subset=['validation_loss']).sort_values(['validation_loss','epoch']).iloc[0]
                assert checkpoint['selected_epoch']==choice.epoch
                members.append(predict_count_network(features,base,path))
            predictions['neural_ensemble']=np.mean(members,axis=0)
            validation=[]
            with np.load(out/city/f'predictions_lag{lag}.npz') as saved:
                np.testing.assert_array_equal(saved['counts'],panel.counts);np.testing.assert_array_equal(saved['spike_cutoffs'],cutoffs)
                for method,mean in predictions.items():
                    np.testing.assert_allclose(saved[method],mean,rtol=1e-6,atol=1e-6)
                    dispersion=fit_dispersion(train_y[masks['train']],mean[masks['train']]);np.testing.assert_allclose(dispersion,plan['dispersions'][method],rtol=1e-6,atol=1e-8)
                    scores=spike_probability(mean[masks['calibration']],cutoffs[masks['calibration']],plan['dispersions'][method])
                    threshold=alarm_threshold(scores,cal_y[masks['calibration']]>cutoffs[masks['calibration']])
                    np.testing.assert_allclose(threshold,plan['thresholds'][method],rtol=1e-6,atol=1e-8)
                    validation.append(dict(method=method,**evaluate_counts(val_y[masks['validation']],mean[masks['validation']],cutoffs[masks['validation']],plan['dispersions'][method],plan['thresholds'][method])))
                    for split in ('test2025','test2026'):
                        use=masks[split];metric=evaluate_counts(panel.counts[use],mean[use],cutoffs[use],plan['dispersions'][method],plan['thresholds'][method])
                        rows.append(dict(city=city,information_lag_days=lag,split=split,method=method,count_primary=method==plan['count_primary'],spike_primary=method==plan['spike_primary'],**metric))
            validation=pd.DataFrame(validation)
            assert validation.sort_values(['nb_nll','method']).iloc[0]['method']==plan['count_primary']
            assert validation.sort_values(['spike_average_precision','method'],ascending=[False,True]).iloc[0]['method']==plan['spike_primary']
            for split in ('test2025','test2026'):
                use=masks[split];effect=paired_block_improvement(panel.counts[use],predictions[plan['count_primary']][use],base[use],24//panel.bin_hours,bootstrap=protocol['bootstrap_samples'])
                effects.append(dict(city=city,information_lag_days=lag,split=split,primary=plan['count_primary'],**effect))
    expected=pd.DataFrame(rows).sort_values(['city','information_lag_days','split','method']).reset_index(drop=True)
    saved=pd.read_csv(out/'test_metrics.csv').sort_values(['city','information_lag_days','split','method']).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected,saved,check_dtype=False,check_exact=False,rtol=1e-6,atol=1e-6)
    expected_effects=pd.DataFrame(effects).sort_values(['city','information_lag_days','split']).reset_index(drop=True)
    saved_effects=pd.read_csv(out/'paired_count_errors.csv').sort_values(['city','information_lag_days','split']).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected_effects,saved_effects,check_dtype=False,check_exact=False,rtol=1e-6,atol=1e-6)
    print(f'Verified {raw_rows:,} raw district rows, both exact count grids and SF availability arrays, frozen choices, all {len(saved)} test comparisons and paired block intervals.')


if __name__=='__main__':main()
