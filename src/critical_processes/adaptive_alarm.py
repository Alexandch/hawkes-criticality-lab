"""R13: mature-past calibration and explicit operating-point selection."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .offline_forecast import calibrate_probability,apply_calibration,regional_thresholds
from .spatiotemporal import average_precision


def calibration_masks(dates,asof,window_days,maturity_days=28):
    if window_days<60 or maturity_days<1:raise ValueError('invalid calibration/maturity window')
    end=pd.Timestamp(asof)-pd.Timedelta(days=maturity_days);start=end-pd.Timedelta(days=window_days)
    middle=start+(end-start)/2
    return dict(probability=np.asarray((dates>=start)&(dates<middle)),
        threshold=np.asarray((dates>=middle)&(dates<end)),start=str(start),middle=str(middle),end_exclusive=str(end),asof=str(pd.Timestamp(asof)))


def fit_policy(raw,cutoff,known_targets,dates,asof,window_days,alpha,maturity_days=28):
    masks=calibration_masks(dates,asof,window_days,maturity_days)
    if not masks['probability'].any() or not masks['threshold'].any():raise ValueError('empty causal calibration block')
    label=known_targets>=cutoff;records=[];calibrated=np.empty((int(masks['threshold'].sum()),*raw.shape[1:]))
    for group in range(raw.shape[2]):
        take=masks['probability'];record=calibrate_probability(raw[take,:,group],label[take,:,group]);records.append(record)
        calibrated[:,:,group]=apply_calibration(raw[masks['threshold'],:,group],record)
    threshold,notes=regional_thresholds(calibrated,label[masks['threshold']],alpha=alpha)
    return dict(probability=records,threshold=threshold.tolist(),threshold_notes=notes,window_days=window_days,alpha=alpha,
        maturity_days=maturity_days,asof=masks['asof'],probability_start=masks['start'],probability_end=masks['middle'],
        threshold_end_exclusive=masks['end_exclusive'],last_label_time=str(dates[masks['threshold']][-1]),
        label_counts=dict(probability_cells=int(masks['probability'].sum())*raw.shape[1]*raw.shape[2],
            threshold_cells=int(masks['threshold'].sum())*raw.shape[1]*raw.shape[2]))


def apply_policy(raw,policy):
    probability=np.stack([apply_calibration(raw[:,:,g],record) for g,record in enumerate(policy['probability'])],axis=2)
    return probability,probability>=np.asarray(policy['threshold'])


def alarm_metrics(target,cutoff,probability,alarm):
    label=np.asarray(target)>=cutoff;alarm=np.asarray(alarm,dtype=bool)
    if label.ndim!=2 or label.shape!=alarm.shape or probability.shape!=label.shape:raise ValueError('expected time x district cells')
    positive=int(label.sum());negative=int((~label).sum());tp=int((label&alarm).sum());fp=int((~label&alarm).sum())
    precision=tp/(tp+fp) if tp+fp else None;recall=tp/positive if positive else None
    fbeta=1.25*precision*recall/(.25*precision+recall) if precision and recall else 0.
    regional=np.divide((~label&alarm).sum(0),(~label).sum(0),out=np.zeros(label.shape[1]),where=(~label).sum(0)>0)
    return dict(cells=label.size,positives=positive,negatives=negative,true_positives=tp,false_positives=fp,alarms=tp+fp,
        precision=precision,recall=recall,fpr=fp/negative if negative else None,fbeta_half=fbeta,
        prevalence=positive/label.size,ap=average_precision(label,probability),brier=float(((probability-label)**2).mean()),
        maximum_district_fpr=float(regional.max()))


def choose_candidate(rows,allowed_methods=None):
    """Joint choice using both previously inspected development periods, never October."""
    candidates=[]
    for row in rows:
        if allowed_methods is not None and row['method'] not in allowed_methods:continue
        periods=[row['development2025'],row['development2026']]
        max_fpr=max(float(r['fpr']) for r in periods)
        min_recall=min(float(r['recall'] or 0.) for r in periods)
        worst_precision=min(float(r['precision'] or 0.) for r in periods)
        safe=max_fpr<=.05;feasible=safe and min_recall>=.10
        # FPR safety is first. A feasible point is preferred, then robust precision.
        rank=(0,0 if feasible else 1,-(worst_precision if feasible else row['combined']['fbeta_half']),
            -row['combined']['fbeta_half'],max_fpr,row['id']) if safe else (1,max_fpr,-row['combined']['fbeta_half'],0.,max_fpr,row['id'])
        candidates.append((rank,row))
    if not candidates:raise ValueError('no eligible candidate methods')
    _,chosen=min(candidates,key=lambda item:item[0])
    periods=[chosen['development2025'],chosen['development2026']]
    status='fpr_and_recall_feasible' if all(r['fpr']<=.05 and (r['recall'] or 0.)>=.10 for r in periods) else 'fallback_no_joint_feasible_point'
    return dict(candidate=chosen['id'],method=chosen['method'],window_days=chosen['window_days'],alpha=chosen['alpha'],selection_status=status)


def preliminary_screen(row):
    first_three=dict(precision_at_least_30pct=(row['precision'] or 0.)>=.30,
        recall_at_least_10pct=(row['recall'] or 0.)>=.10,fpr_at_most_5pct=row['fpr'] is not None and row['fpr']<=.05)
    improvement=dict(ap_beats_baseline=row['ap']>row['baseline_ap'],brier_beats_baseline=row['brier']<row['baseline_brier'])
    sufficient=row['blocks']>=4
    full=all(first_three.values()) and all(improvement.values()) and sufficient and row['brier_difference_ci_high'] is not None and row['brier_difference_ci_high']<0
    return dict(alarm_point_passed=all(first_three.values()),point_with_baseline_improvement_passed=all(first_three.values()) and all(improvement.values()),
        full_screen_passed=full,statistical_status='conditional_block_interval' if sufficient else 'preliminary_insufficient_time_blocks',
        conditions={**first_three,**improvement},sufficient_time_blocks=sufficient)
