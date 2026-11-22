"""R14 delay-aware exposure and heteroskedastic NB2 count forecasts."""
from __future__ import annotations
import copy
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy.signal import lfilter
from .offline_forecast import available_targets,rise_cutoff
from .spatiotemporal import DistrictPanel,fit_calendar,calendar_prediction,observed_rolling,past_rolling

FIT_ASOF='2024-07-01'
EVENT_TRAIN_END='2024-01-01'


def fit_reporting_cdf(panel,max_age=4380,publication_days=7,shrinkage=200):
    """Mature pre2024 cohorts only; report/publication delay is strictly observed."""
    fit=int(np.searchsorted(panel.time_ns,pd.Timestamp(FIT_ASOF).value))
    cohort_end=int(np.searchsorted(panel.time_ns,(pd.Timestamp(EVENT_TRAIN_END)-pd.Timedelta(days=180)).value))
    published=panel.report_bins+publication_days*4;delay=np.maximum(0,published-panel.event_bins)
    eligible=(panel.event_bins<cohort_end)&(published<fit);ages=np.arange(max_age+1)
    result=np.empty((len(panel.districts),3,max_age+1));counts=np.zeros((len(panel.districts),3),dtype=int)
    for group in range(3):
        use=eligible if group==0 else eligible&(panel.group_indices==group)
        pooled=np.sort(delay[use])
        if not len(pooled):raise ValueError('no mature reporting cohorts')
        prior=np.searchsorted(pooled,ages,side='left')/len(pooled)
        for area in range(len(panel.districts)):
            sample=np.sort(delay[use&(panel.area_indices==area)]);counts[area,group]=len(sample)
            result[area,group]=(np.searchsorted(sample,ages,side='left')+shrinkage*prior)/(len(sample)+shrinkage)
    return result,dict(fit_asof=FIT_ASOF,event_cohort_end=str(panel.dates[cohort_end]),publication_days=publication_days,
        mature_cohort_days=180,counts=counts.tolist(),shrinkage=shrinkage,max_age_bins=max_age,
        assumption='past registered-case delay CDF transports to future cohorts; never-registered crimes are not identified')


def effective_exposure(base,curve,length,delay=28,publication_shift=0):
    """Expected already-public counts in a past window, not its complete counts."""
    if length<1 or delay<0 or publication_shift<0:raise ValueError('invalid exposure window')
    base=np.asarray(base,dtype=float);result=np.zeros_like(base);ages=np.arange(delay+1,delay+length+1)
    for area in range(base.shape[1]):
        index=np.clip(ages-publication_shift,0,curve.shape[1]-1)
        weights=curve[area,index].copy();weights[ages<=publication_shift]=0.
        kernel=np.zeros(delay+length+1);kernel[ages]=weights
        result[:,area]=lfilter(kernel,[1.],base[:,area])
    return np.maximum(result,0.)


def corrected_ratio(observed,exposure,prior):
    observed,exposure,prior=np.broadcast_arrays(observed,exposure,prior)
    if (exposure<0).any() or (prior<=0).any():raise ValueError('invalid exposure/prior')
    return np.clip((observed+prior)/(exposure+prior),.25,4.)


def build_inputs(panel,weather,curve,calendars=None,lag_days=7,publication_days=7):
    dates=panel.dates;features=[];seasonal=[];naive=[];corrected=[];fitted=[]
    # The static calendar is fit to mature pre2024 event counts known by July2024.
    known=available_targets(panel,FIT_ASOF,7) if calendars is None else None
    holiday=USFederalHolidayCalendar().holidays(start=dates[0]-pd.Timedelta(days=1),end=dates[-1]+pd.Timedelta(days=1))
    for g in range(3):
        if calendars is None:
            train_panel=DistrictPanel(known[:,:,g],panel.time_ns,panel.districts,panel.centroids,6)
            calendar=fit_calendar(train_panel,EVENT_TRAIN_END)
        else:calendar=calendars[g]
        base,_=calendar_prediction(calendar,dates);view=panel.view(g,publication_days);parts=[];delay=lag_days*4
        for length in (4,28,112,336):
            observed=observed_rolling(view,length,delay);expected=effective_exposure(base,curve[:,g],length,delay,(publication_days-7)*4)
            prior=7*4*base;ratio=corrected_ratio(observed,expected,prior)
            complete=past_rolling(base,length,delay)
            fraction=np.divide(expected,complete,out=np.zeros_like(expected),where=complete>1e-8)
            parts+=[np.log(ratio),np.clip(fraction,0.,1.)]
            city=corrected_ratio(observed.sum(1),expected.sum(1),prior.sum(1))
            parts.append(np.broadcast_to(np.log(city)[:,None],base.shape))
            if length==112:
                adaptive=base*ratio;uncorrected=base*corrected_ratio(observed,complete,prior)
        parts += [np.log1p(base),np.log(np.maximum(adaptive,.02))]
        for phase in (2*np.pi*np.asarray(dates.dayofweek)/7,2*np.pi*np.asarray(dates.hour)/24,2*np.pi*np.asarray(dates.dayofyear)/365.25):
            parts += [np.broadcast_to(np.sin(phase)[:,None],base.shape),np.broadcast_to(np.cos(phase)[:,None],base.shape)]
        parts.append(np.broadcast_to(dates.normalize().isin(holiday)[:,None],base.shape))
        numeric=np.stack(parts,axis=-1)
        forecast=np.broadcast_to(weather[:,None,:],(*base.shape,weather.shape[-1]))
        features.append(np.concatenate([numeric,forecast],axis=-1).astype('float32'))
        seasonal.append(base);naive.append(uncorrected);corrected.append(adaptive);fitted.append(calendar)
    return dict(features=np.stack(features,axis=2),seasonal=np.stack(seasonal,axis=2),naive=np.stack(naive,axis=2),
        adaptive=np.stack(corrected,axis=2),calendars=fitted,cutoff=rise_cutoff(np.stack(corrected,axis=2)))


def build_network(width,areas,kind):
    import torch
    class DistributionNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__();self.area=torch.nn.Embedding(areas,6);self.group=torch.nn.Embedding(3,4)
            self.body=torch.nn.Sequential(torch.nn.Linear(width+10,48),torch.nn.ReLU(),torch.nn.Dropout(.1),
                torch.nn.Linear(48,24),torch.nn.ReLU(),torch.nn.Linear(24,2))
        def forward(self,x,area,group,offset):
            raw=self.body(torch.cat([x,self.area(area),self.group(group)],1))
            logmean=offset+3*torch.tanh(raw[:,0]/3)
            second=.001+5*torch.sigmoid(raw[:,1]) if kind=='nb' else raw[:,1]
            return logmean,second
    return DistributionNetwork()


def nb2_nll(y,logmean,alpha):
    import torch
    r=1/alpha;mu=torch.exp(logmean)
    return torch.lgamma(r)+torch.lgamma(y+1)-torch.lgamma(y+r)+r*torch.log1p(mu/r)+y*torch.log1p(r/mu)


def fit_network(features,adaptive,target,cutoff,dates,kind,seed,epochs=100):
    import torch
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    train=np.asarray((dates>=pd.Timestamp('2021-10-01'))&(dates<pd.Timestamp(EVENT_TRAIN_END)))
    validation=np.asarray((dates>=pd.Timestamp('2024-02-01'))&(dates<pd.Timestamp('2024-04-01')))
    width=features.shape[-1];areas=adaptive.shape[1];mean=features[train].reshape(-1,width).mean(0);scale=features[train].reshape(-1,width).std(0);scale[scale<1e-6]=1.
    network=build_network(width,areas,kind);optimizer=torch.optim.AdamW(network.parameters(),lr=.001,weight_decay=.001)
    x=((features-mean)/scale).reshape(-1,width);shape=adaptive.shape
    area=np.broadcast_to(np.arange(areas)[None,:,None],shape).ravel();group=np.broadcast_to(np.arange(3)[None,None,:],shape).ravel()
    eligible=np.flatnonzero(np.broadcast_to(train[:,None,None],shape).ravel());valid=np.flatnonzero(np.broadcast_to(validation[:,None,None],shape).ravel())
    normalizer=max(float(target[train].mean()),.1)
    def tensors(index):return (torch.tensor(x[index]),torch.tensor(area[index]),torch.tensor(group[index]),
        torch.tensor(np.log(adaptive.ravel()[index]),dtype=torch.float32),torch.tensor(target.ravel()[index],dtype=torch.float32),
        torch.tensor((target.ravel()[index]>=cutoff.ravel()[index]).astype('float32')))
    def loss(batch):
        xx,aa,gg,offset,y,label=batch;mu,second=network(xx,aa,gg,offset)
        if kind=='nb':return nb2_nll(y,mu,second).mean()
        return (torch.exp(mu)-y*mu).mean()/normalizer+torch.nn.functional.binary_cross_entropy_with_logits(second,label)
    validation_parts=[tensors(p) for p in np.array_split(valid,max(1,int(np.ceil(len(valid)/8192))))]
    best=np.inf;state=None;selected=None;history=[]
    for epoch in range(1,epochs+1):
        network.train();draw=rng.choice(eligible,size=32768,replace=True);training=0.
        for part in np.array_split(draw,4):
            optimizer.zero_grad();value=loss(tensors(part));value.backward();optimizer.step();training+=float(value.detach())/4
        score=None
        if epoch%10==0 or epoch==epochs:
            network.eval()
            with torch.no_grad():score=sum(float(loss(batch))*len(batch[0]) for batch in validation_parts)/len(valid)
            if score<best:best=score;selected=epoch;state=copy.deepcopy(network.state_dict())
        history.append(dict(epoch=epoch,training_loss=training,validation_loss=score))
    return dict(state_dict=state,mean=mean,scale=scale,width=width,areas=areas,kind=kind,seed=seed,
        selected_epoch=selected,validation_loss=best,epochs=epochs),pd.DataFrame(history)


def predict_network(features,adaptive,record):
    import torch
    network=build_network(record['width'],record['areas'],record['kind']);network.load_state_dict(record['state_dict']);network.eval()
    shape=adaptive.shape;width=features.shape[-1];x=((features-record['mean'])/record['scale']).reshape(-1,width)
    areas=np.broadcast_to(np.arange(shape[1])[None,:,None],shape).ravel();groups=np.broadcast_to(np.arange(3)[None,None,:],shape).ravel()
    means=np.empty(len(x));second=np.empty(len(x))
    with torch.no_grad():
        for start in range(0,len(x),8192):
            sl=slice(start,start+8192);mu,tail=network(torch.tensor(x[sl]),torch.tensor(areas[sl]),torch.tensor(groups[sl]),torch.tensor(np.log(adaptive.ravel()[sl]),dtype=torch.float32))
            means[sl]=torch.exp(mu).numpy();second[sl]=tail.numpy()
    return means.reshape(shape),second.reshape(shape)
