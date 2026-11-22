"""R16 independent calculation replay; explicitly no independent outcome test."""
from __future__ import annotations
import contextlib,io,json,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from critical_processes.selected_training import sha,write_json
from critical_processes.spatial_experiment import safe
from critical_processes.offline_forecast import load_categories,available_targets
from critical_processes.delay_forecast import FIT_ASOF,predict_network
from critical_processes.delay_forecast_experiment import replay_policy
from critical_processes.cad_forecast import inputs,scores,SCENARIOS
from critical_processes.cad_experiment import evaluate,load_models,METHODS
from prepare_cad_data import prepare
from verify_delay_forecast import close


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    import torch
    torch.set_num_threads(4);root=Path(__file__).resolve().parents[1];out=root/'results/cad_context_detector';old=root/'results/delay_forecast_detector'
    panel_path=root/'data/processed/selective_austin_20261122/austin.npz';cad_path=root/'data/processed/cad_austin_20261122/cad.npz'
    protocol=read(out/'protocol.json');frozen=read(out/'frozen_plan.json');complete=read(out/'complete.json')
    assert protocol['independent_test'] is False and frozen['independent_test'] is False
    for name,value in protocol['code_sha256'].items():assert sha(root/'src/critical_processes'/name)==value,name
    assert sha(root/'scripts/prepare_cad_data.py')==protocol['preparation_sha256']
    assert sha(panel_path)==protocol['panel_sha256'];assert sha(cad_path)==protocol['cad_sha256']
    assert sha(cad_path.with_suffix('.json'))==protocol['cad_metadata_sha256']
    assert sha(old/'frozen_plan.json')==protocol['r14_plan_sha256']
    assert sha(out/'protocol.json')==frozen['protocol_sha256']==complete['protocol_sha256']
    for name,value in complete['artifact_sha256'].items():assert sha(out/name)==value,name
    for name,value in frozen['model_sha256'].items():assert sha(out/name)==value,name
    meta=read(cad_path.with_suffix('.json'))
    for name,value in meta['raw_sha256'].items():
        path=root/name;assert sha(path)==value
        record=read(path.with_name(path.name+'.manifest.json'));assert record['sha256']==value
        frame=pd.read_csv(path);assert int(frame.n.sum())==record['dispatched_incidents'] and len(frame)==record['aggregate_rows']
    with np.load(cad_path) as data:cad={k:data[k].copy() for k in data.files}
    runs=root/'runs';runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='verify_r16_cad_',dir=runs) as temporary:
        destination=Path(temporary);assert destination.resolve().is_relative_to(runs.resolve())
        with contextlib.redirect_stdout(io.StringIO()):record=prepare(root/'data/raw/cad_austin_20261122',destination)
        with np.load(destination/'cad.npz') as data:
            for key,value in cad.items():np.testing.assert_array_equal(value,data[key])
        for key in meta:
            if key not in ('panel_sha256','raw_sha256'):close(meta[key],record[key])
    panel=load_categories(panel_path);models=load_models(out);alphas=read(old/'frozen_plan.json')['alphas']['delay_nb']
    view=inputs(panel,cad,'primary',old);train=(panel.dates>=pd.Timestamp('2021-10-01'))&(panel.dates<pd.Timestamp('2024-01-01'))
    valid=(panel.dates>=pd.Timestamp('2024-02-01'))&(panel.dates<pd.Timestamp('2024-04-01'));known=available_targets(panel,FIT_ASOF,7)
    for method,records in models.items():
        width=view[method].shape[-1];flat=view[method][train].reshape(-1,width);mean=flat.mean(0);scale=flat.std(0);scale[scale<1e-6]=1.
        for record in records:
            np.testing.assert_array_equal(record['mean'],mean);np.testing.assert_array_equal(record['scale'],scale)
            log=pd.read_csv(out/'models'/f'{method}_s{record["seed"]}.csv')
            assert record['selected_epoch']==int(log.loc[log.validation_loss.idxmin(),'epoch'])
            mu,alpha=predict_network(view[method][valid],view['adaptive'][valid],record)
            loss=-nbinom.logpmf(known[valid],1/alpha,1/(1+alpha*mu)).mean()
            assert np.isclose(loss,record['validation_loss'],rtol=2e-6,atol=2e-6),(loss,record['validation_loss'])
    journal=read(out/'forecast_journal.json');assert not journal['contains_outcomes'] and not journal['independent_test']
    assert journal['frozen_plan_sha256']==sha(out/'frozen_plan.json')
    for item in journal['entries']:
        scenario=item['scenario'];view=inputs(panel,cad,scenario,old);raw,means=scores(view,models,alphas);take=panel.dates>=pd.Timestamp('2025-01-01');policies={}
        with np.load(out/item['file']) as saved:
            assert not any(key in saved.files for key in ('counts','target','labels','observed'))
            np.testing.assert_array_equal(saved['time_ns'],panel.time_ns[take]);np.testing.assert_array_equal(saved['cutoff'],view['cutoff'][take])
            for method in METHODS:
                probability,alarm,records=replay_policy(panel,raw,view['cutoff'],method,.05,scenario,'2025-01-01','2026-09-01')
                np.testing.assert_allclose(probability[take],saved['probability_'+method],rtol=2e-6,atol=2e-7)
                np.testing.assert_array_equal(alarm[take],saved['alarm_'+method]);policies[method]=records
        close(safe(policies),read(out/f'{scenario}_policy_journal.json'))
    with tempfile.TemporaryDirectory(prefix='verify_r16_metrics_',dir=runs) as temporary:
        destination=Path(temporary);evaluate(out,panel_path,destination)
        for name in ('metrics.csv','district_metrics.csv','monthly_metrics.csv'):
            pd.testing.assert_frame_equal(pd.read_csv(out/name),pd.read_csv(destination/name),check_exact=False,rtol=1e-10,atol=1e-10)
        close(read(out/'assessment.json'),read(destination/'assessment.json'))
    receipt=dict(successful=True,independent_test=False,protocol_sha256=sha(out/'protocol.json'),complete_sha256=sha(out/'complete.json'),verifier_sha256=sha(Path(__file__)),
        included_cad_incidents=meta['included_incidents'],models=sum(map(len,models.values())),forecast_files=2,metric_rows=36,district_rows=324,monthly_rows=360,
        checks=['public dispatch aggregate counts and hashes','exact CAD preparation','six models train-only scaling and checkpoint NLL',
            'both scenario inference and causal monthly policies','two forecast files without outcomes','all result tables','explicit exploratory status'],
        limitation='reused outcomes from R14/R15; this replay does not constitute new independent validation')
    write_json(out/'verification.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
