"""Replay R13 data preparation, selection, causal calibration and new forecasts."""
from __future__ import annotations
import contextlib,io,json,math,tempfile
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from critical_processes.selected_training import sha,write_json
from critical_processes.offline_forecast import load_categories,available_targets
from critical_processes.adaptive_alarm import choose_candidate,fit_policy,apply_policy
from critical_processes.adaptive_alarm_experiment import (infer,raw_scores,static_forecast,develop,append_frame,evaluate,
    check_r12_replay,SCENARIOS,METHODS)


def close(left,right):
    if isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:close(left[key],right[key])
    elif isinstance(left,list):
        assert len(left)==len(right)
        for a,b in zip(left,right):close(a,b)
    elif isinstance(left,float):assert math.isclose(left,float(right),rel_tol=2e-6,abs_tol=2e-7),(left,right)
    else:assert left==right,(left,right)


def main():
    import torch
    torch.set_num_threads(4);root=Path(__file__).resolve().parents[1];out=root/'results/adaptive_seattle_detector';old=root/'results/expanded_district_detector'
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'));frozen=json.loads((out/'frozen_plan.json').read_text(encoding='utf-8'))
    complete=json.loads((out/'complete.json').read_text(encoding='utf-8'))
    for name,value in protocol['code_sha256'].items():assert sha(root/'src/critical_processes'/name)==value,name
    for name,value in complete['artifact_sha256'].items():assert sha(out/name)==value,name
    assert frozen['protocol_sha256']==sha(out/'protocol.json')
    assert sha(old/'complete.json')==protocol['r12_complete_sha256']
    history_path=root/'data/processed/expanded_districts_20261119/seattle.npz';assert sha(history_path)==protocol['history_sha256']
    panel_path=root/'data/processed/adaptive_seattle_20261122/seattle.npz';bound=json.loads((out/'new_input.json').read_text(encoding='utf-8'))
    assert sha(panel_path)==bound['panel_sha256']
    meta=json.loads((panel_path.with_suffix('.json')).read_text(encoding='utf-8'))
    raw_path=root/'data/raw/adaptive_seattle_20261122/seattle_october.csv.gz';raw_meta=json.loads(raw_path.with_name(raw_path.name+'.manifest.json').read_text(encoding='utf-8'))
    assert sha(raw_path)==raw_meta['sha256']==meta['raw_sha256']
    assert sha(out/'frozen_plan.json')==raw_meta['frozen_plan_sha256']==bound['frozen_plan_sha256']
    assert datetime.fromisoformat(frozen['frozen_utc'])<=datetime.fromisoformat(raw_meta['retrieved_utc'])
    history=load_categories(history_path);panel=load_categories(panel_path);ids=set()
    raw_history=root/'data/raw/expanded_districts_20261119'
    for year in range(2019,2027):
        file=raw_history/f'seattle_{year}.csv.gz';record=json.loads(file.with_name(file.name+'.manifest.json').read_text(encoding='utf-8'))
        assert sha(file)==record['sha256'];ids.update(pd.read_csv(file,usecols=['report_number'],dtype=str).report_number.dropna())
    reconstructed,quality=append_frame(history,pd.read_csv(raw_path,dtype=str),ids)
    for key,value in vars(panel).items():np.testing.assert_array_equal(value,getattr(reconstructed,key))
    for key,value in quality.items():assert meta[key]==value,key
    old_plan=json.loads((old/'frozen_plan.json').read_text(encoding='utf-8'))['cities']['seattle'];historical={};cutoff=None
    for scenario in SCENARIOS:
        result=infer(history,old,scenario)
        if scenario=='primary':cutoff=result['cutoff']
        raw=raw_scores(result,cutoff,old_plan);check_r12_replay(history,result,raw,old,old_plan,scenario);historical[scenario]=raw
    with contextlib.redirect_stdout(io.StringIO()):rows,policies=develop(history,historical['primary'],cutoff)
    close(rows,json.loads((out/'development_candidates.json').read_text(encoding='utf-8')))
    close(policies,json.loads((out/'development_policy_journal.json').read_text(encoding='utf-8')))
    for role,methods in [('selected',None),('rolling_neural',['local_neural']),('rolling_baseline',['recent_calendar','seasonal'])]:
        assert choose_candidate(rows,methods)==frozen['roles'][role]
    journal=json.loads((out/'forecast_journal.json').read_text(encoding='utf-8'));marker=json.loads((out/'test_started.json').read_text(encoding='utf-8'))
    assert not journal['contains_outcomes'];assert marker['forecast_journal_sha256']==sha(out/'forecast_journal.json')
    full_cutoff=None
    for item in journal['entries']:
        scenario=item['scenario'];known=available_targets(history,'2026-10-01',SCENARIOS[scenario]['publication_days'])
        for role,selection in frozen['roles'].items():
            if role.startswith('static'):continue
            policy=fit_policy(historical[scenario][selection['method']],cutoff,known,history.dates,'2026-10-01',selection['window_days'],selection['alpha'])
            close(policy,frozen['test_policies'][scenario][role])
        result=infer(panel,old,scenario)
        if scenario=='primary':full_cutoff=result['cutoff']
        raw=raw_scores(result,full_cutoff,old_plan);take=panel.dates>=pd.Timestamp('2026-10-01')
        assert sha(out/item['file'])==item['sha256']
        with np.load(out/item['file']) as saved:
            assert not any(key in saved.files for key in ('counts','target','observed','labels'))
            np.testing.assert_array_equal(saved['time_ns'],panel.time_ns[take]);np.testing.assert_array_equal(saved['cutoff'],full_cutoff[take])
            for role,selection in frozen['roles'].items():
                if role.startswith('static'):probability,alarm=static_forecast(raw[selection['method']][take],selection['method'],old_plan)
                else:probability,alarm=apply_policy(raw[selection['method']][take],frozen['test_policies'][scenario][role])
                np.testing.assert_allclose(probability,saved['probability_'+role],rtol=2e-6,atol=2e-7)
                np.testing.assert_array_equal(alarm,saved['alarm_'+role])
    runs=root/'runs';runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='verify_r13_',dir=runs) as temporary:
        destination=Path(temporary);assert destination.resolve().is_relative_to(runs.resolve())
        evaluate(out,panel_path,destination)
        for name in ('metrics.csv','district_metrics.csv'):
            pd.testing.assert_frame_equal(pd.read_csv(out/name),pd.read_csv(destination/name),check_exact=False,rtol=1e-10,atol=1e-10)
        assert json.loads((out/'assessment.json').read_text(encoding='utf-8'))==json.loads((destination/'assessment.json').read_text(encoding='utf-8'))
    receipt=dict(successful=True,protocol_sha256=sha(out/'protocol.json'),complete_sha256=sha(out/'complete.json'),verifier_sha256=sha(Path(__file__)),
        frozen_before_new_download=True,history_cases=quality['historical_cases'],new_october_cases=quality['new_cases'],
        development_candidates=len(rows),forecast_files=len(journal['entries']),
        checks=['unchanged R12 inputs and forecast replay','new CSV case preparation and exact historical prefix',
            'all27operating candidates and monthly past-only policies','selected configuration frozen before new download',
            'both October scenario forecasts without outcomes','all metrics and regional tables'],
        limitation='October has only2-3time blocks; retrospective and preliminary, no operational/prospective validation')
    write_json(out/'verification.json',receipt);print('Verified R13: 27 candidates, causal rolling policies, 3564 new cases, both forecast files and all result tables.')


if __name__=='__main__':main()
