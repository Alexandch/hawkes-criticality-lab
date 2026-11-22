"""R16 exploratory paired retraining: additional CAD information on reused labels."""
from __future__ import annotations
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from .selected_training import sha,write_json
from .spatial_experiment import safe
from .offline_forecast import load_categories,available_targets
from .delay_forecast import FIT_ASOF,fit_network
from .delay_forecast_experiment import replay_policy
from .adaptive_alarm import alarm_metrics,preliminary_screen
from .expanded_forecast import operating_intervals
from .cad_forecast import SCENARIOS,inputs,scores

OUT=Path('results/cad_context_detector')
PANEL=Path('data/processed/selective_austin_20261122/austin.npz')
CAD=Path('data/processed/cad_austin_20261122/cad.npz')
OLD=Path('results/delay_forecast_detector')
SEEDS=(701,709,719)
METHODS=('base_nb','cad_nb','delay_nb')


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def load_models(out):
    import torch
    return {name:[torch.load(out/'models'/f'{name}_s{seed}.pt',map_location='cpu',weights_only=False) for seed in SEEDS] for name in METHODS[:2]}


def evaluate(out,panel_path=PANEL,destination=None):
    panel=load_categories(panel_path);rows=[];regional=[];monthly=[]
    for item in read(out/'forecast_journal.json')['entries']:
        assert sha(out/item['file'])==item['sha256']
        with np.load(out/item['file']) as data:saved={key:data[key].copy() for key in data.files}
        index=np.searchsorted(panel.time_ns,saved['time_ns']);dates=panel.dates[index];target=panel.counts[index]
        for split,start,end in [('development2025','2025-01-01','2026-01-01'),('reused2026','2026-01-01','2026-09-01')]:
            take=np.asarray((dates>=pd.Timestamp(start))&(dates<pd.Timestamp(end)))
            for method in METHODS:
                for g,group in enumerate(panel.groups):
                    y=target[take,:,g];cut=saved['cutoff'][take,:,g];p=saved['probability_'+method][take,:,g];alarm=saved['alarm_'+method][take,:,g]
                    base=saved['probability_base_nb'][take,:,g];simple=saved['probability_delay_nb'][take,:,g]
                    ref=alarm_metrics(y,cut,base,saved['alarm_base_nb'][take,:,g]);simpleref=alarm_metrics(y,cut,simple,saved['alarm_delay_nb'][take,:,g])
                    row=dict(split=split,scenario=item['scenario'],method=method,group=str(group),**alarm_metrics(y,cut,p,alarm),
                        baseline_ap=ref['ap'],baseline_brier=ref['brier'],simple_ap=simpleref['ap'],simple_brier=simpleref['brier'],
                        **operating_intervals(y>=cut,alarm,p,base))
                    screen=preliminary_screen(row);row.update({k:v for k,v in screen.items() if k!='conditions'});row['independent_test']=False;rows.append(row)
                    for area,district in enumerate(panel.districts):regional.append(dict(split=split,scenario=item['scenario'],method=method,group=str(group),district=str(district),
                        **alarm_metrics(y[:,area:area+1],cut[:,area:area+1],p[:,area:area+1],alarm[:,area:area+1])))
                    for month in dates[take].to_period('M').unique():
                        local=np.asarray(dates[take].to_period('M')==month);monthly.append(dict(split=split,scenario=item['scenario'],method=method,group=str(group),month=str(month),
                            **alarm_metrics(y[local],cut[local],p[local],alarm[local])))
    destination=out if destination is None else destination;destination.mkdir(exist_ok=True,parents=True);table=pd.DataFrame(rows)
    table.to_csv(destination/'metrics.csv',index=False);pd.DataFrame(regional).to_csv(destination/'district_metrics.csv',index=False);pd.DataFrame(monthly).to_csv(destination/'monthly_metrics.csv',index=False)
    primary=table.query("split=='reused2026' and scenario=='primary' and method=='cad_nb' and group=='all'").iloc[0]
    write_json(destination/'assessment.json',safe(dict(status='exploratory_reused_outcomes_no_independent_validation',independent_test=False,
        new_information='monthly dispatched CAD context, initial categories only',point_screen_passed=bool(primary.alarm_point_passed),
        paired_screen_passed=bool(primary.full_screen_passed),deployment_supported=False,
        limitation='2025and2026labels already inspected in R14/R15; any promising effect needs an untouched future period or geography')))
    return table


def run(out=OUT,epochs=100):
    import torch
    torch.set_num_threads(4);root=Path(__file__).resolve().parents[2];out.mkdir(exist_ok=True,parents=True)
    modules=('cad_forecast.py','cad_experiment.py','delay_forecast.py','delay_forecast_experiment.py','adaptive_alarm.py','offline_forecast.py','spatiotemporal.py','expanded_forecast.py')
    protocol=dict(stage='16_new_monthly_dispatch_information',independent_test=False,outcome_status='2025and2026Jan-Aug already inspected; exploratory reuse only',
        code_sha256={name:sha(Path(__file__).parent/name) for name in modules},preparation_sha256=sha(root/'scripts/prepare_cad_data.py'),
        panel_sha256=sha(PANEL),cad_sha256=sha(CAD),cad_metadata_sha256=sha(CAD.with_suffix('.json')),r14_plan_sha256=sha(OLD/'frozen_plan.json'),
        methods=list(METHODS),seeds=list(SEEDS),epochs=epochs,scenarios=SCENARIOS,
        features='13CAD context features: log daily rates over1/3published response months, ratio, age; only dispatched calls closed in an available month',
        publication='CAD closure-month end plus15days primary or45days stress; current archive revisions and first-seen unknown',
        training='same R14 NB2 architecture and schedule; paired base/CAD retraining2021Oct-2023; checkpoint2024Feb-Mar; targets known2024Jul1',
        operating_rule='fixed R14monthly180day calibration,maturity28,alpha.05; no post-outcome model/threshold selection',
        comparison='CAD against identically scheduled base network and simple delay-corrected background; unchanged R14reference target',
        primary_method='cad_nb, all categories, primary scenario; recorded before any new ablation metrics',
        readiness='no operational or independent validation claim, regardless of point screen')
    path=out/'protocol.json'
    if path.exists():assert read(path)==safe(protocol),'R16 code/input/protocol changed'
    else:write_json(path,safe(protocol))
    if (out/'complete.json').exists():
        for name,value in read(out/'complete.json')['artifact_sha256'].items():assert sha(out/name)==value,name
        return
    panel=load_categories(PANEL)
    with np.load(CAD) as file:cad={key:file[key].copy() for key in file.files}
    directory=out/'models';directory.mkdir(exist_ok=True);view=inputs(panel,cad,'primary');target=available_targets(panel,FIT_ASOF,7);summary=[]
    for method in METHODS[:2]:
        for seed in SEEDS:
            model=directory/f'{method}_s{seed}.pt'
            if model.exists():record=torch.load(model,map_location='cpu',weights_only=False)
            else:
                record,history=fit_network(view[method],view['adaptive'],target,view['cutoff'],panel.dates,'nb',seed,epochs)
                torch.save(record,model);history.to_csv(model.with_suffix('.csv'),index=False)
            summary.append(dict(method=method,seed=seed,selected_epoch=record['selected_epoch'],validation_loss=record['validation_loss']))
            print(f'R16 fitted {method}/s{seed}: epoch{record["selected_epoch"]}',flush=True)
    pd.DataFrame(summary).to_csv(out/'training_summary.csv',index=False)
    frozen=dict(protocol_sha256=sha(out/'protocol.json'),frozen_utc=datetime.now(timezone.utc).isoformat(),independent_test=False,
        primary_method='cad_nb',model_sha256={p.relative_to(out).as_posix():sha(p) for p in directory.glob('*.pt')},
        note='frozen before new ablation metrics, but outcomes already known from previous experiments')
    write_json(out/'frozen_plan.json',frozen);models=load_models(out);alphas=read(OLD/'frozen_plan.json')['alphas']['delay_nb']
    entries=[];(out/'forecasts').mkdir(exist_ok=True)
    for scenario in SCENARIOS:
        view=inputs(panel,cad,scenario);raw,means=scores(view,models,alphas);take=panel.dates>=pd.Timestamp('2025-01-01')
        arrays=dict(time_ns=panel.time_ns[take],cutoff=view['cutoff'][take]);policies={}
        for method in METHODS:
            probability,alarm,journal=replay_policy(panel,raw,view['cutoff'],method,.05,scenario,'2025-01-01','2026-09-01')
            arrays['probability_'+method]=probability[take];arrays['alarm_'+method]=alarm[take];policies[method]=journal
        write_json(out/f'{scenario}_policy_journal.json',safe(policies));path=out/'forecasts'/f'{scenario}.npz';np.savez_compressed(path,**arrays)
        entries.append(dict(file=path.relative_to(out).as_posix(),scenario=scenario,sha256=sha(path),contains_outcomes=False))
    write_json(out/'forecast_journal.json',dict(independent_test=False,contains_outcomes=False,frozen_plan_sha256=sha(out/'frozen_plan.json'),entries=entries))
    table=evaluate(out)
    write_json(out/'complete.json',dict(successful_execution=True,protocol_sha256=sha(out/'protocol.json'),artifact_sha256={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='complete.json'}))
    print(table.query("scenario=='primary' and group=='all'")[["split","method","precision","recall","fpr","ap","brier"]].to_string(index=False),flush=True)


if __name__=='__main__':run()
