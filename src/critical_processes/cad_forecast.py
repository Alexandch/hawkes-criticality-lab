"""R16: monthly public dispatch context with explicit closure/publication guards."""
from __future__ import annotations
import calendar,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from .delay_forecast import build_inputs,predict_network

SCENARIOS={'primary':dict(lag_days=7,publication_days=7,cad_publication_days=15),
    'delayed_stress':dict(lag_days=14,publication_days=14,cad_publication_days=45)}
OLD=Path('results/delay_forecast_detector')


def cad_features(data,dates,publication_days=15):
    """Only calls closed in a released past month; no event in a future month."""
    response=data['response_month'];closed=data['closed_month'];areas=data['areas'];counts=data['counts'];flags=data['channel_flags']
    if publication_days<0 or (closed<response).any() or (counts<0).any():raise ValueError('invalid CAD timing/counts')
    maximum=int(max(response.max(),closed.max()))+1
    if maximum>120:raise ValueError('unsupported CAD closure month')
    cube=np.zeros((maximum,maximum,9,4),dtype=float)
    for channel in range(4):
        use=flags[:,channel];np.add.at(cube,(closed[use],response[use],areas[use],np.full(use.sum(),channel)),counts[use])
    cube=np.cumsum(cube,axis=0);month=pd.date_range('2021-01-01',periods=maximum+1,freq='MS')
    release=(month[1:]+pd.Timedelta(days=publication_days)).as_unit('ns');last=np.searchsorted(release.asi8,dates.as_unit('ns').asi8,side='left')-1
    result=np.zeros((len(dates),9,13),dtype='float32')
    result[:,:,-1]=2.
    for m in np.unique(last[last>=0]):
        take=last==m;one=cube[m,m];start=max(0,m-2);length=sum(calendar.monthrange(d.year,d.month)[1] for d in month[start:m+1])
        rate1=one/calendar.monthrange(month[m].year,month[m].month)[1];rate3=cube[m,start:m+1].sum(0)/length
        numeric=np.concatenate([np.log1p(rate1),np.log1p(rate3),np.log((rate1+1)/(rate3+1))],axis=1)
        result[take,:,:12]=numeric[None]
        age=np.asarray((dates[take]-month[m+1])/pd.Timedelta(days=60),dtype=float)
        result[take,:,-1]=np.clip(age,0,2)[:,None]
    return result


def inputs(panel,cad,scenario,old=OLD):
    frozen=json.loads((old/'frozen_plan.json').read_text(encoding='utf-8'));settings=SCENARIOS[scenario]
    with np.load(old/'reporting_cdf.npz') as file:curve=file['curve'].copy()
    view=build_inputs(panel,np.zeros((len(panel.counts),3),dtype='float32'),curve,frozen['calendars'],
        lag_days=settings['lag_days'],publication_days=settings['publication_days'])
    cutoff=view['cutoff'] if scenario=='primary' else build_inputs(panel,np.zeros((len(panel.counts),3),dtype='float32'),curve,frozen['calendars'])['cutoff']
    dispatch=cad_features(cad,panel.dates,settings['cad_publication_days'])
    basic=view['features'][...,:-3];extra=np.broadcast_to(dispatch[:,:,None,:],(*basic.shape[:-1],dispatch.shape[-1]))
    return dict(base_nb=basic,cad_nb=np.concatenate([basic,extra],axis=-1),adaptive=view['adaptive'],cutoff=cutoff)


def scores(view,models,alphas):
    cutoff=view['cutoff'];raw={};means={}
    for method,records in models.items():
        outputs=[predict_network(view[method],view['adaptive'],record) for record in records]
        means[method]=np.mean([o[0] for o in outputs],axis=0)
        raw[method]=np.mean([nbinom.sf(cutoff-1,1/alpha,1/(1+alpha*mu)) for mu,alpha in outputs],axis=0)
    mu=view['adaptive'];alpha=np.asarray(alphas);raw['delay_nb']=nbinom.sf(cutoff-1,1/alpha,1/(1+alpha*mu));means['delay_nb']=mu
    return raw,means
