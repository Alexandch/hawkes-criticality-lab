"""Independently replay R14 preparation, frozen selection and yearly forecasts."""
from __future__ import annotations
import contextlib,io,json,math,tempfile
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from critical_processes.selected_training import sha,write_json
from critical_processes.spatial_experiment import safe
from critical_processes.offline_forecast import load_categories
from critical_processes.delay_forecast import fit_reporting_cdf,build_inputs
from critical_processes.delay_forecast_experiment import (
    METHODS,NEURAL,SEEDS,SCENARIOS,preparer,score_models,replay_policy,choose,evaluate)
from critical_processes.adaptive_alarm import alarm_metrics


def close(left,right):
    if isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:close(left[key],right[key])
    elif isinstance(left,list):
        assert len(left)==len(right)
        for a,b in zip(left,right):close(a,b)
    elif isinstance(left,float):assert math.isclose(left,float(right),rel_tol=2e-6,abs_tol=2e-7),(left,right)
    else:assert left==right,(left,right)


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    import torch
    torch.set_num_threads(4);root=Path(__file__).resolve().parents[1]
    out=root/'results/delay_forecast_detector';raw=root/'data/raw/delay_forecast_20261122'
    dev=root/'data/processed/delay_forecast_20261122/development';final=root/'data/processed/delay_forecast_20261122/final'
    protocol=read(out/'protocol.json');frozen=read(out/'frozen_plan.json');bound=read(out/'new_input.json')
    for name,value in protocol['code_sha256'].items():assert sha(root/'src/critical_processes'/name)==value,name
    assert sha(root/'scripts/prepare_delay_forecast_data.py')==protocol['preparation_sha256']
    assert frozen['protocol_sha256']==sha(out/'protocol.json')==read(out/'complete.json')['protocol_sha256']
    for name,value in read(out/'complete.json')['artifact_sha256'].items():assert sha(out/name)==value,name
    assert sha(dev/'austin.npz')==protocol['development_panel_sha256']
    assert sha(dev/'austin_weather.npz')==protocol['weather_sha256']
    assert sha(final/'austin.npz')==bound['panel_sha256']
    assert sha(final/'austin_weather.npz')==bound['weather_sha256']
    assert sha(final/'austin.json')==bound['metadata_sha256']
    frozen_sha=sha(out/'frozen_plan.json');assert frozen_sha==bound['frozen_plan_sha256']
    for name in ('austin_2025.csv.gz','austin_weather_2025.json'):
        manifest=raw/(name+'.manifest.json');record=read(manifest)
        assert sha(raw/name)==record['sha256'];assert record['frozen_plan_sha256']==frozen_sha
        assert datetime.fromisoformat(frozen['frozen_utc'])<=datetime.fromisoformat(record['retrieved_utc'])
        assert sha(manifest)==bound['case_manifest_sha256' if name.endswith('gz') else 'weather_manifest_sha256']
    for path in raw.glob('*.manifest.json'):
        record=read(path);source=path.with_name(path.name.removesuffix('.manifest.json'))
        assert sha(source)==record['sha256'],source
    for directory in (dev,final):
        for name in ('austin.json','austin_weather.json'):
            for path,value in read(directory/name)['raw_sha256'].items():assert sha(root/path)==value,path
    panel=load_categories(dev/'austin.npz');full=load_categories(final/'austin.npz')
    with np.load(dev/'austin_weather.npz') as data:weather=data['features'].copy()
    with np.load(final/'austin_weather.npz') as data:full_weather=data['features'].copy()
    np.testing.assert_array_equal(full.counts[:len(panel.counts)],panel.counts)
    np.testing.assert_array_equal(full_weather[:len(weather)],weather)
    prep=preparer();runs=root/'runs';runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='verify_r14_data_',dir=runs) as temporary:
        destination=Path(temporary);assert destination.resolve().is_relative_to(runs.resolve())
        for original,years,history in ((dev,range(2021,2025),None),(final,[2025],dev/'austin.npz')):
            with contextlib.redirect_stdout(io.StringIO()):prep.prepare(raw,destination,years,None if history is None else frozen_sha,history)
            for name in ('austin.npz','austin_weather.npz'):
                with np.load(original/name) as saved,np.load(destination/name) as replay:
                    assert set(saved.files)==set(replay.files)
                    for key in saved.files:np.testing.assert_array_equal(saved[key],replay[key])
            for name in ('austin.json','austin_weather.json'):
                left=read(original/name);right=read(destination/name)
                for key in left:
                    if key not in ('raw_sha256','panel_sha256'):close(left[key],right[key])
    curve,profile=fit_reporting_cdf(panel)
    with np.load(out/'reporting_cdf.npz') as data:np.testing.assert_array_equal(curve,data['curve'])
    close(profile,read(out/'reporting_profile.json'));assert sha(out/'reporting_cdf.npz')==frozen['reporting_cdf_sha256']
    for name,value in frozen['model_sha256'].items():assert sha(out/name)==value,name
    models={name:[torch.load(out/'models'/f'{name}_s{seed}.pt',map_location='cpu',weights_only=False) for seed in SEEDS] for name in NEURAL}
    inputs=build_inputs(panel,weather,curve);close(safe(inputs['calendars']),frozen['calendars']);cutoff=inputs['cutoff']
    _,scores=score_models(inputs,models,frozen['alphas'],cutoff);rows=[];policies={}
    for method in METHODS:
        for alpha in (.025,.05):
            probability,alarm,journal=replay_policy(panel,scores,cutoff,method,alpha,'primary','2024-07-01','2024-12-01')
            row=dict(id=f'{method}_a{int(alpha*1000):03}',method=method,alpha=alpha)
            for split,start,end in [('selection','2024-07-01','2024-10-01'),('confirmation','2024-10-01','2024-12-01')]:
                take=(panel.dates>=pd.Timestamp(start))&(panel.dates<pd.Timestamp(end))
                row[split]=alarm_metrics(panel.counts[take,:,0],cutoff[take,:,0],probability[take,:,0],alarm[take,:,0])
            rows.append(row);policies[row['id']]=journal
    close(safe(rows),read(out/'development_candidates.json'));close(safe(policies),read(out/'development_policy_journal.json'))
    for role,allowed in [('selected',None),('neural',NEURAL),('baseline',METHODS[:3])]:assert choose(rows,allowed)==frozen['roles'][role]
    journal=read(out/'forecast_journal.json');assert not journal['contains_outcomes']
    assert read(out/'test_started.json')['forecast_journal_sha256']==sha(out/'forecast_journal.json')
    primary_cutoff=None
    for item in journal['entries']:
        scenario=item['scenario'];inputs=build_inputs(full,full_weather,curve,frozen['calendars'],**SCENARIOS[scenario])
        if scenario=='primary':primary_cutoff=inputs['cutoff']
        _,scores=score_models(inputs,models,frozen['alphas'],primary_cutoff);take=full.dates>=pd.Timestamp('2025-01-01')
        with np.load(out/item['file']) as saved:
            assert not any(key in saved.files for key in ('counts','target','observed','labels'))
            np.testing.assert_array_equal(saved['time_ns'],full.time_ns[take]);np.testing.assert_array_equal(saved['cutoff'],primary_cutoff[take])
            policies={}
            for role,selection in frozen['roles'].items():
                probability,alarm,policy=replay_policy(full,scores,primary_cutoff,selection['method'],selection['alpha'],scenario,'2025-01-01','2026-01-01')
                np.testing.assert_allclose(probability[take],saved['probability_'+role],rtol=2e-6,atol=2e-7)
                np.testing.assert_array_equal(alarm[take],saved['alarm_'+role]);policies[role]=policy
            close(safe(policies),read(out/f'{scenario}_policy_journal.json'))
    with tempfile.TemporaryDirectory(prefix='verify_r14_metrics_',dir=runs) as temporary:
        destination=Path(temporary);evaluate(out,final/'austin.npz',destination)
        for name in ('metrics.csv','district_metrics.csv','monthly_metrics.csv'):
            pd.testing.assert_frame_equal(pd.read_csv(out/name),pd.read_csv(destination/name),check_exact=False,rtol=1e-10,atol=1e-10)
        close(read(out/'assessment.json'),read(destination/'assessment.json'))
    receipt=dict(successful=True,protocol_sha256=sha(out/'protocol.json'),complete_sha256=sha(out/'complete.json'),verifier_sha256=sha(Path(__file__)),
        frozen_before_new_download=True,development_cases=int(panel.counts[:,:,0].sum()),new_test_cases=int(full.counts[take,:,0].sum()),
        development_candidates=len(rows),models=sum(map(len,models.values())),forecast_files=len(journal['entries']),
        metric_rows=30,district_rows=270,monthly_rows=360,time_blocks=27,
        checks=['raw snapshot and source hashes','exact case and weather re-preparation','unchanged historical prefix',
            'mature-cohort reporting CDF','all12selection candidates and confirmation metrics','nine frozen models inference',
            'all monthly causal policies in both scenarios','two sealed forecast files','all30/270/360result rows'],
        limitation='retrospective registered reports; assumed publication delay and nominal forecast availability; not prospective deployment')
    write_json(out/'verification.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
