"""R15: sector-aware probability calibration and selective weekly alarms."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit
from .offline_forecast import calibrate_probability,apply_calibration,regional_thresholds,available_targets
from .adaptive_alarm import fit_policy,apply_policy,alarm_metrics

CONFIGS=tuple(dict(id=f'{method}_{policy}',method=method,policy=policy) for method in ('nb_plain','delay_nb')
    for policy in ('monthly','weekly_global','weekly_sector'))
SCENARIOS={'primary':dict(lag_days=7,publication_days=7),'delayed_stress':dict(lag_days=14,publication_days=14)}


def fit_sector_calibration(scores,labels,penalty=.005):
    """Shared positive logit slope; regularized sector intercepts, no future labels."""
    scores=np.asarray(scores);labels=np.asarray(labels,dtype=float)
    if scores.ndim!=2 or labels.shape!=scores.shape or not len(scores):raise ValueError('expected nonempty time x sector scores')
    x=logit(np.clip(scores,1e-6,1-1e-6));areas=x.shape[1]
    if np.unique(labels).size<2:
        return dict(slope=.05,intercept=float(logit((labels.sum()+1)/(labels.size+2))),offset=[0.]*areas,fallback=True)
    def objective(theta):
        slope,intercept=theta[:2];delta=theta[2:];z=slope*x+intercept+delta
        residual=expit(z)-labels
        loss=np.mean(np.logaddexp(0,z)-labels*z)+.001*((slope-1)**2+intercept**2)+penalty*np.mean(delta**2)
        gradient=np.r_[np.mean(residual*x)+.002*(slope-1),residual.mean()+.002*intercept,
            residual.sum(0)/labels.size+2*penalty*delta/areas]
        return loss,gradient
    result=minimize(objective,np.r_[1.,0.,np.zeros(areas)],jac=True,method='L-BFGS-B',
        bounds=[(.05,5.),(-8.,8.)]+[(-4.,4.)]*areas)
    if not result.success:raise RuntimeError('sector calibration failed: '+str(result.message))
    return dict(slope=float(result.x[0]),intercept=float(result.x[1]),offset=result.x[2:].tolist(),fallback=False)


def apply_sector_calibration(scores,record):
    return expit(record['slope']*logit(np.clip(scores,1e-6,1-1e-6))+record['intercept']+np.asarray(record['offset']))


def fit_selective_policy(raw,cutoff,known,dates,asof,sector=True,window_days=180,maturity_days=56,risk_floor=.30,alpha=.05):
    end=pd.Timestamp(asof)-pd.Timedelta(days=maturity_days);start=end-pd.Timedelta(days=window_days);middle=start+pd.Timedelta(days=window_days/2)
    first=np.asarray((dates>=start)&(dates<middle));second=np.asarray((dates>=middle)&(dates<end))
    if not first.any() or not second.any():raise ValueError('empty calibration interval')
    label=known>=cutoff;records=[]
    for g in range(raw.shape[2]):
        records.append((fit_sector_calibration if sector else calibrate_probability)(raw[first,:,g],label[first,:,g]))
    threshold,notes=regional_thresholds(raw[second],label[second],alpha)
    return dict(kind='sector' if sector else 'global',probability=records,raw_threshold=threshold.tolist(),risk_floor=risk_floor,
        alpha=alpha,window_days=window_days,maturity_days=maturity_days,asof=str(pd.Timestamp(asof)),
        probability_start=str(start),probability_end=str(middle),threshold_end_exclusive=str(end),
        last_label_time=str(dates[second][-1]),threshold_notes=notes)


def apply_selective_policy(raw,policy):
    apply=apply_sector_calibration if policy['kind']=='sector' else apply_calibration
    probability=np.stack([apply(raw[:,:,g],record) for g,record in enumerate(policy['probability'])],axis=2)
    alarm=(raw>=np.asarray(policy['raw_threshold']))&(probability>=policy['risk_floor'])
    return probability,alarm


def replay(panel,scores,cutoff,config,scenario,start,end):
    dates=panel.dates;raw=scores[config['method']];probability=np.zeros_like(raw);alarm=np.zeros_like(raw,dtype=bool);journal=[]
    dates_update=pd.date_range(start,pd.Timestamp(end)-pd.Timedelta(days=1),freq='MS' if config['policy']=='monthly' else '7D')
    for i,asof in enumerate(dates_update):
        stop=dates_update[i+1] if i+1<len(dates_update) else pd.Timestamp(end)
        take=np.asarray((dates>=asof)&(dates<stop));known=available_targets(panel,asof,SCENARIOS[scenario]['publication_days'])
        if config['policy']=='monthly':
            policy=fit_policy(raw,cutoff,known,dates,asof,180,.05,28);probability[take],alarm[take]=apply_policy(raw[take],policy)
        else:
            policy=fit_selective_policy(raw,cutoff,known,dates,asof,sector=config['policy']=='weekly_sector')
            probability[take],alarm[take]=apply_selective_policy(raw[take],policy)
        journal.append(policy)
    return probability,alarm,journal


def choose(rows,method=None):
    eligible=[r for r in rows if method is None or r['method']==method]
    def rank(row):
        parts=row['quarters'];maximum=max(p['fpr'] for p in parts);minimum=min(p['recall'] or 0 for p in parts)
        safe=maximum<=.05;feasible=safe and minimum>=.10
        return (0 if safe else 1,0 if feasible else 1, -(row['overall']['precision'] or 0) if feasible else
            (-row['overall']['fbeta_half'] if safe else maximum),-row['overall']['fbeta_half'],row['id'])
    selected=min(eligible,key=rank)
    return dict(id=selected['id'],method=selected['method'],policy=selected['policy'],
        selection_status='quarterwise_feasible' if all(p['fpr']<=.05 and (p['recall'] or 0)>=.10 for p in selected['quarters']) else 'predeclared_fallback')
