"""R15: a sealed2026 test of selective decisions with frozen R14 count models."""
from __future__ import annotations
import importlib.util,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from .selected_training import sha,write_json
from .spatial_experiment import safe
from .offline_forecast import load_categories,available_targets
from .delay_forecast import build_inputs,predict_network
from .adaptive_alarm import alarm_metrics,preliminary_screen
from .expanded_forecast import operating_intervals
from .selective_alarm import CONFIGS,SCENARIOS,replay,choose

HISTORY=Path('data/processed/delay_forecast_20261122/final/austin.npz')
OLD=Path('results/delay_forecast_detector')
OUT=Path('results/selective_austin_detector')


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def preparer():
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('r15_preparer',root/'scripts/prepare_selective_data.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def infer(panel,scenario,old=OLD):
    import torch
    frozen=read(old/'frozen_plan.json')
    with np.load(old/'reporting_cdf.npz') as data:curve=data['curve'].copy()
    inputs=build_inputs(panel,np.zeros((len(panel.counts),3),dtype='float32'),curve,frozen['calendars'],**SCENARIOS[scenario])
    primary=inputs if scenario=='primary' else build_inputs(panel,np.zeros((len(panel.counts),3),dtype='float32'),curve,frozen['calendars'],**SCENARIOS['primary'])
    cutoff=primary['cutoff'];raw={}
    alpha=np.asarray(frozen['alphas']['delay_nb']);mu=inputs['adaptive'];raw['delay_nb']=nbinom.sf(cutoff-1,1/alpha,1/(1+alpha*mu))
    values=[]
    for seed in (601,613,619):
        record=torch.load(old/'models'/f'nb_plain_s{seed}.pt',map_location='cpu',weights_only=False)
        mean,dispersion=predict_network(inputs['features'][...,:-3],mu,record);values.append(nbinom.sf(cutoff-1,1/dispersion,1/(1+dispersion*mean)))
    raw['nb_plain']=np.mean(values,axis=0)
    return raw,cutoff


def develop(panel,scores,cutoff):
    rows=[];policies={};take=np.asarray((panel.dates>=pd.Timestamp('2025-01-01'))&(panel.dates<pd.Timestamp('2026-01-01')))
    for config in CONFIGS:
        probability,alarm,journal=replay(panel,scores,cutoff,config,'primary','2025-01-01','2026-01-01')
        row=dict(**config,overall=alarm_metrics(panel.counts[take,:,0],cutoff[take,:,0],probability[take,:,0],alarm[take,:,0]),quarters=[])
        for quarter in range(1,5):
            use=take&(panel.dates.quarter==quarter)
            row['quarters'].append(alarm_metrics(panel.counts[use,:,0],cutoff[use,:,0],probability[use,:,0],alarm[use,:,0]))
        rows.append(row);policies[config['id']]=journal
        print(f'R15 development {config["id"]}: {row["overall"]["precision"]}, recall={row["overall"]["recall"]:.3f}, FPR={row["overall"]["fpr"]:.3f}',flush=True)
    return rows,policies


def diagnosis(panel,cutoff):
    rows=[]
    for asof in pd.date_range('2025-01-01','2025-12-01',freq='MS'):
        known=available_targets(panel,asof,7)
        for maturity in (28,56,90):
            end=asof-pd.Timedelta(days=maturity);take=np.asarray((panel.dates>=end-pd.Timedelta(days=90))&(panel.dates<end))
            label=panel.counts[take,:,0]>=cutoff[take,:,0];visible=known[take,:,0]>=cutoff[take,:,0]
            rows.append(dict(asof=str(asof.date()),maturity_days=maturity,cells=label.size,true_positives=int(label.sum()),
                unrecognized_positives=int((label&~visible).sum()),missing_positive_fraction=float((label&~visible).sum()/max(1,label.sum()))))
    return pd.DataFrame(rows)


def evaluate(out,panel_path,destination=None):
    panel=load_categories(panel_path);frozen=read(out/'frozen_plan.json');rows=[];monthly=[];regional=[]
    for item in read(out/'forecast_journal.json')['entries']:
        assert sha(out/item['file'])==item['sha256']
        with np.load(out/item['file']) as data:saved={key:data[key].copy() for key in data.files}
        index=np.searchsorted(panel.time_ns,saved['time_ns']);dates=panel.dates[index];y=panel.counts[index]
        for role in frozen['roles']:
            for g,group in enumerate(panel.groups):
                prob=saved['probability_'+role][:,:,g];alarm=saved['alarm_'+role][:,:,g];cut=saved['cutoff'][:,:,g];base=saved['probability_baseline'][:,:,g]
                ref=alarm_metrics(y[:,:,g],cut,base,saved['alarm_baseline'][:,:,g]);metric=alarm_metrics(y[:,:,g],cut,prob,alarm)
                row=dict(scenario=item['scenario'],role=role,group=str(group),**metric,baseline_ap=ref['ap'],baseline_brier=ref['brier'],
                    **operating_intervals(y[:,:,g]>=cut,alarm,prob,base));screen=preliminary_screen(row)
                row.update({k:v for k,v in screen.items() if k!='conditions'});row['failed_conditions']=';'.join(k for k,v in screen['conditions'].items() if not v);rows.append(row)
                for d,district in enumerate(panel.districts):regional.append(dict(scenario=item['scenario'],role=role,group=str(group),district=str(district),
                    **alarm_metrics(y[:,d:d+1,g],cut[:,d:d+1],prob[:,d:d+1],alarm[:,d:d+1])))
                for month in dates.to_period('M').unique():
                    use=dates.to_period('M')==month;monthly.append(dict(scenario=item['scenario'],role=role,group=str(group),month=str(month),
                        **alarm_metrics(y[use,:,g],cut[use],prob[use],alarm[use])))
    destination=out if destination is None else destination;destination.mkdir(exist_ok=True,parents=True);table=pd.DataFrame(rows)
    table.to_csv(destination/'metrics.csv',index=False);pd.DataFrame(regional).to_csv(destination/'district_metrics.csv',index=False);pd.DataFrame(monthly).to_csv(destination/'monthly_metrics.csv',index=False)
    main=table.query("scenario=='primary' and role=='selected' and group=='all'").iloc[0]
    write_json(destination/'assessment.json',safe(dict(status='retrospective_selective_alarm_research',selected=frozen['roles']['selected'],
        primary_point_passed=bool(main.alarm_point_passed),primary_full_screen_passed=bool(main.full_screen_passed),blocks=int(main.blocks),
        prospective=False,publication_times_verified=False,new_period='Austin2026Jan-Aug;2025nowdevelopment',
        limitation='risk floor is a decision rule, not a guaranteed precision or FPR bound; late2026labels provisional')))
    return table


def run(out=OUT):
    import torch
    torch.set_num_threads(4);out.mkdir(exist_ok=True,parents=True);root=Path(__file__).resolve().parents[2]
    code=('selective_alarm.py','selective_experiment.py','delay_forecast.py','adaptive_alarm.py','offline_forecast.py','spatiotemporal.py','expanded_forecast.py')
    protocol=dict(stage='15_sector_calibration_selective_weekly',history_sha256=sha(HISTORY),r14_complete_sha256=sha(OLD/'complete.json'),r14_plan_sha256=sha(OLD/'frozen_plan.json'),
        code_sha256={name:sha(Path(__file__).parent/name) for name in code},preparation_sha256=sha(root/'scripts/prepare_selective_data.py'),
        r14_preparation_sha256=sha(root/'scripts/prepare_delay_forecast_data.py'),configs=list(CONFIGS),scenarios=SCENARIOS,
        new_approach='weekly selective alarms; maturity56days; positive shared logit slope with shrunk sector intercepts; calibrated risk floor30pct',
        calibration='180days split90/90, chronological probability and negative raw-score quantile; alpha.05; hierarchical penalty.005 fixed',
        model='frozen R14 NB2 no-weather three-seed ensemble; same features and corrected reference target; no retraining',
        selection='all2025nowdevelopment; prefer FPR<=5pct and recall>=10pct in all four quarters; maximize overall precision; predeclared safeF0.5 fallback',
        new_test='Austin2026Jan-Aug, first download only after freeze; 2026Sep-Nov excluded for label maturity',
        target='same R14 background and exceedance; no true crime or criticality labels',criteria=dict(precision=.30,recall=.10,fpr=.05,ap_beats_baseline=True,brier_ci_high_below_zero=True))
    path=out/'protocol.json'
    if path.exists():assert read(path)==safe(protocol),'R15 code/input/protocol changed'
    else:write_json(path,safe(protocol))
    if (out/'complete.json').exists():
        for name,value in read(out/'complete.json')['artifact_sha256'].items():assert sha(out/name)==value,name
        return
    panel=load_categories(HISTORY)
    if not (out/'frozen_plan.json').exists():
        scores,cutoff=infer(panel,'primary');diagnosis(panel,cutoff).to_csv(out/'maturity_diagnosis.csv',index=False)
        rows,policies=develop(panel,scores,cutoff);write_json(out/'development_candidates.json',safe(rows));write_json(out/'development_policy_journal.json',safe(policies))
        pd.DataFrame([dict(id=r['id'],method=r['method'],policy=r['policy'],**r['overall']) for r in rows]).to_csv(out/'development_metrics.csv',index=False)
        roles=dict(selected=choose(rows),neural=choose(rows,'nb_plain'),baseline=choose(rows,'delay_nb'),
            monthly_neural=dict(id='nb_plain_monthly',method='nb_plain',policy='monthly'),sector_neural=dict(id='nb_plain_weekly_sector',method='nb_plain',policy='weekly_sector'))
        frozen=dict(protocol_sha256=sha(path),frozen_utc=datetime.now(timezone.utc).isoformat(),roles=roles,new_2026_cases_downloaded=False,
            model_sha256={f'models/nb_plain_s{s}.pt':sha(OLD/'models'/f'nb_plain_s{s}.pt') for s in (601,613,619)})
        write_json(out/'frozen_plan.json',safe(frozen))
    prep=preparer();raw=Path('data/raw/selective_austin_20261122');final=Path('data/processed/selective_austin_20261122')
    prep.download(raw,out/'frozen_plan.json');quality=prep.prepare(raw,final,HISTORY,Path('data/raw/delay_forecast_20261122'),out/'frozen_plan.json')
    frozen=read(out/'frozen_plan.json');panel=load_categories(final/'austin.npz');write_json(out/'new_input.json',dict(**quality,metadata_sha256=sha(final/'austin.json')))
    entries=[];(out/'forecasts').mkdir(exist_ok=True)
    for scenario in SCENARIOS:
        scores,cutoff=infer(panel,scenario);take=panel.dates>=pd.Timestamp('2026-01-01');arrays=dict(time_ns=panel.time_ns[take],cutoff=cutoff[take]);policies={}
        for role,config in frozen['roles'].items():
            probability,alarm,journal=replay(panel,scores,cutoff,config,scenario,'2026-01-01','2026-09-01')
            arrays['probability_'+role]=probability[take];arrays['alarm_'+role]=alarm[take];policies[role]=journal
        write_json(out/f'{scenario}_policy_journal.json',safe(policies));path=out/'forecasts'/f'{scenario}.npz';np.savez_compressed(path,**arrays)
        entries.append(dict(file=path.relative_to(out).as_posix(),scenario=scenario,sha256=sha(path),contains_outcomes=False))
    write_json(out/'forecast_journal.json',dict(contains_outcomes=False,frozen_plan_sha256=sha(out/'frozen_plan.json'),entries=entries))
    write_json(out/'test_started.json',dict(forecast_journal_sha256=sha(out/'forecast_journal.json')))
    table=evaluate(out,final/'austin.npz')
    write_json(out/'complete.json',dict(successful_execution=True,protocol_sha256=sha(out/'protocol.json'),artifact_sha256={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='complete.json'}))
    print(table.query("scenario=='primary' and group=='all'")[["role","precision","recall","fpr","full_screen_passed"]].to_string(index=False),flush=True)


if __name__=='__main__':run()
