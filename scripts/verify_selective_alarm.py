"""R15 replay: new data, sector calibration, policy selection and sealed forecasts."""
from __future__ import annotations
import contextlib,io,json,tempfile
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from critical_processes.selected_training import sha,write_json
from critical_processes.spatial_experiment import safe
from critical_processes.offline_forecast import load_categories
from critical_processes.selective_alarm import choose,replay,SCENARIOS
from critical_processes.selective_experiment import infer,develop,diagnosis,preparer,evaluate
from verify_delay_forecast import close


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    import torch
    torch.set_num_threads(4);root=Path(__file__).resolve().parents[1];out=root/'results/selective_austin_detector';old=root/'results/delay_forecast_detector'
    history_path=root/'data/processed/delay_forecast_20261122/final/austin.npz';panel_path=root/'data/processed/selective_austin_20261122/austin.npz'
    protocol=read(out/'protocol.json');frozen=read(out/'frozen_plan.json');complete=read(out/'complete.json');bound=read(out/'new_input.json')
    for name,value in protocol['code_sha256'].items():assert sha(root/'src/critical_processes'/name)==value,name
    assert sha(root/'scripts/prepare_selective_data.py')==protocol['preparation_sha256']
    assert sha(root/'scripts/prepare_delay_forecast_data.py')==protocol['r14_preparation_sha256']
    assert protocol['r14_complete_sha256']==sha(old/'complete.json');assert protocol['r14_plan_sha256']==sha(old/'frozen_plan.json')
    assert protocol['history_sha256']==sha(history_path)
    assert frozen['protocol_sha256']==complete['protocol_sha256']==sha(out/'protocol.json')
    for name,value in complete['artifact_sha256'].items():assert sha(out/name)==value,name
    for name,value in frozen['model_sha256'].items():assert sha(old/name)==value,name
    for name,value in read(old/'complete.json')['artifact_sha256'].items():assert sha(old/name)==value,name
    assert bound['panel_sha256']==sha(panel_path);assert bound['metadata_sha256']==sha(panel_path.with_suffix('.json'))
    raw=root/'data/raw/selective_austin_20261122';source=raw/'austin_2026_jan_aug.csv.gz';manifest=read(source.with_name(source.name+'.manifest.json'))
    assert sha(source)==manifest['sha256']==bound['raw_sha256'];assert sha(out/'frozen_plan.json')==manifest['frozen_plan_sha256']==bound['frozen_plan_sha256']
    assert datetime.fromisoformat(frozen['frozen_utc'])<=datetime.fromisoformat(manifest['retrieved_utc'])
    history=load_categories(history_path);panel=load_categories(panel_path);np.testing.assert_array_equal(history.counts,panel.counts[:len(history.counts)])
    for key in ('event_bins','area_indices','group_indices','report_bins'):
        np.testing.assert_array_equal(getattr(history,key),getattr(panel,key)[:len(history.event_bins)])
    runs=root/'runs';runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='verify_r15_data_',dir=runs) as temporary:
        destination=Path(temporary);assert destination.resolve().is_relative_to(runs.resolve())
        with contextlib.redirect_stdout(io.StringIO()):quality=preparer().prepare(raw,destination,history_path,root/'data/raw/delay_forecast_20261122',out/'frozen_plan.json')
        reconstructed=load_categories(destination/'austin.npz')
        for key in vars(panel):np.testing.assert_array_equal(getattr(panel,key),getattr(reconstructed,key))
        for key in quality:
            if key!='panel_sha256':close(quality[key],bound[key])
    scores,cutoff=infer(history,'primary',old)
    pd.testing.assert_frame_equal(diagnosis(history,cutoff),pd.read_csv(out/'maturity_diagnosis.csv'),check_dtype=False,rtol=1e-12,atol=1e-12)
    with contextlib.redirect_stdout(io.StringIO()):rows,policies=develop(history,scores,cutoff)
    close(safe(rows),read(out/'development_candidates.json'));close(safe(policies),read(out/'development_policy_journal.json'))
    for role,method in [('selected',None),('neural','nb_plain'),('baseline','delay_nb')]:assert choose(rows,method)==frozen['roles'][role]
    probability,alarm,_=replay(history,scores,cutoff,dict(method='nb_plain',policy='monthly'),'primary','2025-01-01','2026-01-01')
    with np.load(old/'forecasts/primary.npz') as saved:
        np.testing.assert_allclose(probability[-1460:],saved['probability_weather_ablation'],rtol=2e-6,atol=2e-7)
        np.testing.assert_array_equal(alarm[-1460:],saved['alarm_weather_ablation'])
    journal=read(out/'forecast_journal.json');assert not journal['contains_outcomes']
    assert sha(out/'forecast_journal.json')==read(out/'test_started.json')['forecast_journal_sha256']
    for item in journal['entries']:
        scenario=item['scenario'];scores,cutoff=infer(panel,scenario,old);take=panel.dates>=pd.Timestamp('2026-01-01');policies={}
        with np.load(out/item['file']) as saved:
            assert not any(k in saved.files for k in ('counts','target','labels','observed'))
            np.testing.assert_array_equal(saved['time_ns'],panel.time_ns[take]);np.testing.assert_array_equal(saved['cutoff'],cutoff[take])
            for role,config in frozen['roles'].items():
                probability,alarm,records=replay(panel,scores,cutoff,config,scenario,'2026-01-01','2026-09-01')
                np.testing.assert_allclose(probability[take],saved['probability_'+role],rtol=2e-6,atol=2e-7)
                np.testing.assert_array_equal(alarm[take],saved['alarm_'+role]);policies[role]=records
                for record in records:assert pd.Timestamp(record['last_label_time'])<pd.Timestamp(record['asof'])-pd.Timedelta(days=record['maturity_days'])
            close(safe(policies),read(out/f'{scenario}_policy_journal.json'))
    with tempfile.TemporaryDirectory(prefix='verify_r15_metrics_',dir=runs) as temporary:
        destination=Path(temporary);evaluate(out,panel_path,destination)
        for name in ('metrics.csv','district_metrics.csv','monthly_metrics.csv'):
            pd.testing.assert_frame_equal(pd.read_csv(out/name),pd.read_csv(destination/name),check_exact=False,rtol=1e-10,atol=1e-10)
        close(read(out/'assessment.json'),read(destination/'assessment.json'))
    receipt=dict(successful=True,protocol_sha256=sha(out/'protocol.json'),complete_sha256=sha(out/'complete.json'),verifier_sha256=sha(Path(__file__)),
        frozen_before_new_download=True,historical_cases=int(history.counts[:,:,0].sum()),new_test_cases=bound['new_cases'],development_candidates=len(rows),
        frozen_forecast_files=2,metric_rows=30,district_rows=270,monthly_rows=240,
        checks=['unchanged R14 source and weights','exact new CSV preparation and historical prefix','all6candidates and quarterly choice',
            'maturity diagnostic','exact R14 monthly2025forecast replay','causal sector calibrations in both scenarios','both sealed forecasts and all metrics'],
        limitation='retrospective and provisional labels; calibrated risk floor is not a precision guarantee')
    write_json(out/'verification.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
