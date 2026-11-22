"""Verify R12 inputs, all frozen predictions and replayed evaluation tables."""
from __future__ import annotations
import argparse,json,tempfile,time
from pathlib import Path
import numpy as np
import pandas as pd
from critical_processes.selected_training import sha,write_json
from critical_processes.expanded_experiment import CITIES,SEEDS,assemble,predict_models,freeze_forecasts,evaluate
from critical_processes.offline_forecast import load_categories
from prepare_expanded_districts import parse_weather,STATIONS


def main():
    import joblib,torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4);threadpool_limits(4)
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path('results/expanded_district_detector'))
    parser.add_argument('--panels',type=Path,default=Path('data/processed/expanded_districts_20261119'))
    parser.add_argument('--history',type=Path,default=Path('data/processed/offline_20261110'))
    parser.add_argument('--raw',type=Path,default=Path('data/raw/expanded_districts_20261119'));args=parser.parse_args()
    out=args.out;config=json.loads((out/'protocol.json').read_text(encoding='utf-8'));frozen=json.loads((out/'frozen_plan.json').read_text(encoding='utf-8'))
    for name,value in config['code_sha256'].items():assert sha(Path('src/critical_processes')/name)==value,name
    assert sha(Path('scripts/prepare_expanded_districts.py'))==config['preparation_sha256']
    assert frozen['protocol_sha256']==sha(out/'protocol.json')
    for relative,value in frozen['model_sha256'].items():assert sha(out/relative)==value,relative
    cases={};districts={}
    for c,city in enumerate(CITIES):
        path=(args.history if c<2 else args.panels)/(city+'.npz');assert sha(path)==config['panel_sha256'][city]
        panel=load_categories(path);cases[city]=int(panel.counts[:,:,0].sum());districts[city]=len(panel.districts)
        if c>=2:
            metadata=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            for name,value in metadata['raw_sha256'].items():assert sha(Path(name))==value
            reconstructed=np.zeros_like(panel.counts)
            np.add.at(reconstructed,(panel.event_bins,panel.area_indices,np.zeros(len(panel.event_bins),dtype=int)),1)
            np.add.at(reconstructed,(panel.event_bins,panel.area_indices,panel.group_indices),1)
            np.testing.assert_array_equal(panel.counts,reconstructed)
            assert cases[city]==metadata['cases']
        weather_path=args.panels/(city+'_weather.npz');assert sha(weather_path)==config['weather_sha256'][city]
        with np.load(weather_path) as w:
            parsed=parse_weather((args.raw/(STATIONS[city]+'.dly')).read_text(encoding='ascii'),STATIONS[city])
            parsed=parsed.reindex(pd.DatetimeIndex(w['day_ns'].astype('datetime64[ns]')).strftime('%Y-%m-%d'))
            np.testing.assert_allclose(w['values'],parsed.to_numpy(dtype='float32'),equal_nan=True)
    models={}
    for name in ['local_'+city for city in CITIES]+['pooled','weather']:
        models[name]=[torch.load(out/'models'/f'{name}_s{seed}.pt',map_location='cpu',weights_only=False) for seed in config['seeds']]
    for leaves in (15,31):models[f'boosted{leaves}']=joblib.load(out/'models'/f'boosted{leaves}.joblib')
    journal=json.loads((out/'forecast_journal.json').read_text(encoding='utf-8'))
    assert not journal['contains_outcomes'];assert journal['frozen_plan_sha256']==sha(out/'frozen_plan.json')
    marker=json.loads((out/'test_started.json').read_text(encoding='utf-8'));assert marker['forecast_journal_sha256']==sha(out/'forecast_journal.json')
    root=Path('runs').resolve();root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='verify_expanded_',dir=root) as temporary:
        target=Path(temporary);assert target.resolve().is_relative_to(root)
        primary_cutoffs={}
        for scenario in config['scenarios']:
            data,blocks,_=assemble(args.panels,args.history,scenario,frozen['calendars'])
            # All past inputs are constructed first. Only held-period rows need
            # a model replay because those are the forecasts in the journal.
            reduced=[];position=0
            for block in blocks:
                take=np.asarray(block['dates']>=pd.Timestamp('2025-01-01'));old_shape=block['shape'];item=dict(block)
                for key in ('x','category','adaptive','seasonal','cutoff'):
                    array=block[key];tail=array.shape[1:]
                    item[key]=array.reshape((*old_shape,*tail))[take].reshape((-1,*tail))
                item['dates']=block['dates'][take];item['shape']=(int(take.sum()),*old_shape[1:])
                item['slice']=slice(position,position+len(item['x']));position+=len(item['x']);reduced.append(item)
            del data,blocks
            blocks=reduced;data={key:np.concatenate([b[key] for b in blocks]) for key in ('x','category','adaptive')}
            if scenario=='primary':primary_cutoffs={b['city']:b['cutoff'].reshape(b['shape']).copy() for b in blocks}
            predictions=predict_models(data,models)
            entries=freeze_forecasts(data,blocks,predictions,frozen['cities'],target,scenario,primary_cutoffs)
            for entry in entries:
                original=next(e for e in journal['entries'] if e['city']==entry['city'] and e['scenario']==scenario)
                with np.load(target/entry['file']) as replay,np.load(out/original['file']) as saved:
                    assert set(replay.files)==set(saved.files)
                    assert not any(k in saved.files for k in ('counts','target','labels','observed'))
                    for key in saved.files:np.testing.assert_allclose(replay[key],saved[key],rtol=2e-6,atol=2e-7)
            del data,blocks,predictions
        print('All frozen R12 forecasts replayed; waiting for the original evaluation if still running.',flush=True)
        deadline=time.monotonic()+1200
        while not (out/'complete.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('original evaluation has not completed')
            time.sleep(2)
        complete=json.loads((out/'complete.json').read_text(encoding='utf-8'))
        for relative,value in complete['artifact_sha256'].items():assert sha(out/relative)==value,relative
        evaluate(out,args.panels,args.history,destination=target)
        for name in ('metrics','district_metrics','headline','monthly_metrics'):
            pd.testing.assert_frame_equal(pd.read_csv(out/(name+'.csv')),pd.read_csv(target/(name+'.csv')),check_exact=False,rtol=1e-10,atol=1e-10)
        assert json.loads((out/'assessment.json').read_text(encoding='utf-8'))==json.loads((target/'assessment.json').read_text(encoding='utf-8'))
    result=dict(successful=True,protocol_sha256=sha(out/'protocol.json'),complete_sha256=sha(out/'complete.json'),
        verifier_sha256=sha(Path(__file__)),cases=cases,districts=districts,
        frozen_forecast_files=len(journal['entries']),checks=['source and input hashes','new-city event-grid reconstruction',
        'quality-filtered NOAA reparse','all neural and tree forecast replay','outcome-free prediction journal',
        'all metrics and district/month tables replay'],
        limits='retrospective source snapshots do not establish historical publication times or prospective usefulness')
    write_json(out/'verification.json',result)
    print(f'Verified R12: {sum(districts.values())} districts, {sum(cases.values()):,} source-panel cases, all 8 forecasts and all evaluation tables.')


if __name__=='__main__':main()
