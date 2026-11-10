"""R11 causal category forecasts with separate count and excess-event heads."""
from __future__ import annotations
import copy
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy.optimize import minimize
from scipy.special import expit,logit
from scipy.stats import nbinom
from .n_accuracy import empirical_auc
from .spatiotemporal import (DistrictPanel,district_graph,fit_calendar,calendar_prediction,forecast_features,
    recent_calendar,observed_rolling,past_rolling,counts_available_by,negative_binomial_nll,average_precision)


@dataclass
class CategoryPanel:
    counts: np.ndarray
    time_ns: np.ndarray
    districts: np.ndarray
    groups: np.ndarray
    centroids: np.ndarray
    event_bins: np.ndarray
    area_indices: np.ndarray
    group_indices: np.ndarray
    report_bins: np.ndarray | None = None

    @property
    def dates(self):return pd.DatetimeIndex(self.time_ns.astype('datetime64[ns]'))

    def view(self,group,publication_days=3):
        use=np.ones(len(self.event_bins),dtype=bool) if group==0 else self.group_indices==group
        return DistrictPanel(self.counts[:,:,group],self.time_ns,self.districts,self.centroids,6,
            self.event_bins[use] if self.report_bins is not None else None,
            self.report_bins[use]+4*publication_days if self.report_bins is not None else None,
            self.area_indices[use] if self.report_bins is not None else None)


def load_categories(path):
    with np.load(path) as data:panel=CategoryPanel(**{key:data[key].copy() for key in data.files})
    if panel.counts.shape!=(len(panel.time_ns),len(panel.districts),len(panel.groups)):raise ValueError('invalid category dimensions')
    if (panel.counts<0).any() or not (np.diff(panel.time_ns)==6*3600*10**9).all():raise ValueError('invalid counts/time grid')
    if not np.array_equal(panel.counts[:,:,0],panel.counts[:,:,1:].sum(2)):raise ValueError('groups must partition the total')
    return panel


def vintage_masks(dates,asof,maturity_days=28):
    end=pd.Timestamp(asof);train_end=end-pd.DateOffset(months=6);validation_end=end-pd.DateOffset(months=3)
    train_start=max(pd.Timestamp('2019-03-02'),train_end-pd.DateOffset(years=3));calibration_start=validation_end
    def mask(start,stop):return np.asarray((dates>=start)&(dates<stop-pd.Timedelta(days=maturity_days)))
    return dict(train=mask(train_start,train_end),validation=mask(train_end,validation_end),
        probability_calibration=np.asarray((dates>=calibration_start)&(dates<calibration_start+pd.Timedelta(days=30))),
        alarm_calibration=np.asarray((dates>=calibration_start+pd.Timedelta(days=30))&(dates<end-pd.Timedelta(days=maturity_days))),
        train_end=str(train_end),validation_end=str(validation_end),calibration_end=str(end))


def available_targets(panel,cutoff,publication_days=3):
    return np.stack([counts_available_by(panel.view(group,publication_days),cutoff) for group in range(len(panel.groups))],axis=2)


def rise_cutoff(adaptive):
    return np.ceil(adaptive+np.maximum(2.,.5*adaptive)).astype(np.int32)


def causal_inputs(panel,train_end,lag_days=7,publication_days=3,calendars=None):
    dates=panel.dates;graph=district_graph(panel.centroids);features=[];seasonal=[];adaptive=[];fitted=[]
    holiday=USFederalHolidayCalendar().holidays(start=dates[0]-pd.Timedelta(days=2),end=dates[-1]+pd.Timedelta(days=2))
    day=dates.normalize();calendar_flags=[day.isin(holiday), (day+pd.Timedelta(days=1)).isin(holiday), (day-pd.Timedelta(days=1)).isin(holiday),
        np.asarray(dates.is_month_start),np.asarray(dates.is_month_end)]
    for group in range(len(panel.groups)):
        view=panel.view(group,publication_days);calendar=fit_calendar(view,train_end) if calendars is None else calendars[group]
        base,_=calendar_prediction(calendar,dates);recent=recent_calendar(view,base,4,lag_days);parts=[forecast_features(view,base,graph,lag_days)]
        for length in (56*4,84*4):
            observed=observed_rolling(view,length,lag_days*4);expected=past_rolling(base,length,lag_days*4)
            ratio=(observed+28*4*base)/(expected+28*4*base);parts.append(np.log(np.clip(ratio,.05,20.))[:,:,None])
        parts.append(np.log(np.maximum(recent,.02))[:,:,None])
        for flag in calendar_flags:parts.append(np.broadcast_to(np.asarray(flag)[:,None,None],(*base.shape,1)))
        features.append(np.concatenate(parts,axis=2).astype('float32'));seasonal.append(base);adaptive.append(recent);fitted.append(calendar)
    return np.stack(features,axis=2),np.stack(seasonal,axis=2),np.stack(adaptive,axis=2),fitted


def build_network(features,districts,groups):
    import torch
    class ForecastNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__();self.district=torch.nn.Embedding(districts,6);self.group=torch.nn.Embedding(groups,4)
            self.body=torch.nn.Sequential(torch.nn.Linear(features+10,48),torch.nn.ReLU(),torch.nn.Dropout(.1),torch.nn.Linear(48,24),torch.nn.ReLU())
            self.head=torch.nn.Linear(24,2)
        def forward(self,x,district,group,offset):
            result=self.head(self.body(torch.cat([x,self.district(district),self.group(group)],dim=-1)))
            return offset+4*torch.tanh(result[:,0]/4),result[:,1]
    return ForecastNetwork()


def fit_network(features,target,adaptive,train,validation,seed,epochs=60):
    import torch
    torch.manual_seed(seed);torch.set_num_threads(min(4,torch.get_num_threads()))
    width=features.shape[-1];districts,groups=target.shape[1:]
    mean=features[train].reshape(-1,width).mean(0);scale=features[train].reshape(-1,width).std(0);scale[scale<1e-6]=1.
    network=build_network(width,districts,groups);optimizer=torch.optim.AdamW(network.parameters(),lr=.002,weight_decay=.001)
    normalizer=max(float(target[train].mean()),.1)
    def tensors(mask):
        length=int(mask.sum());shape=(length,districts,groups)
        return (torch.tensor(((features[mask]-mean)/scale).reshape(-1,width),dtype=torch.float32),
            torch.arange(districts).repeat_interleave(groups).repeat(length),torch.arange(groups).repeat(length*districts),
            torch.tensor(np.log(adaptive[mask]).ravel(),dtype=torch.float32),torch.tensor(target[mask].ravel(),dtype=torch.float32),
            torch.tensor((target[mask]>=rise_cutoff(adaptive[mask])).ravel(),dtype=torch.float32))
    tx,td,tg,to,ty,tl=tensors(train);vx,vd,vg,vo,vy,vl=tensors(validation)
    best_loss=np.inf;state=None;best_epoch=None;history=[]
    for epoch in range(1,epochs+1):
        network.train();optimizer.zero_grad();logmean,score=network(tx,td,tg,to)
        loss=(torch.exp(logmean)-ty*logmean).mean()/normalizer+torch.nn.functional.binary_cross_entropy_with_logits(score,tl)
        loss.backward();optimizer.step();value=None
        if epoch%10==0 or epoch==epochs:
            network.eval()
            with torch.no_grad():
                vm,vs=network(vx,vd,vg,vo);value=float((torch.exp(vm)-vy*vm).mean()/normalizer+torch.nn.functional.binary_cross_entropy_with_logits(vs,vl))
            if value<best_loss:best_loss=value;best_epoch=epoch;state=copy.deepcopy(network.state_dict())
        history.append(dict(epoch=epoch,training_loss=float(loss.detach()),validation_loss=value))
    return dict(state_dict=state,mean=mean,scale=scale,features=width,districts=districts,groups=groups,seed=seed,selected_epoch=best_epoch,
        validation_loss=best_loss,epochs=epochs,count_normalizer=normalizer,architecture='district6/group4 + MLP48-24, Poisson residual and excess BCE heads'),pd.DataFrame(history)


def predict_network(features,adaptive,record):
    import torch
    network=build_network(record['features'],record['districts'],record['groups']);network.load_state_dict(record['state_dict']);network.eval()
    x=((features-record['mean'])/record['scale']).reshape(-1,features.shape[-1]);districts,groups=adaptive.shape[1:]
    areas=np.tile(np.repeat(np.arange(districts),groups),len(adaptive));classes=np.tile(np.arange(groups),len(adaptive)*districts)
    offsets=np.log(adaptive.ravel());means=[];probabilities=[]
    with torch.no_grad():
        for start in range(0,len(x),8192):
            sl=slice(start,start+8192);mu,score=network(torch.tensor(x[sl],dtype=torch.float32),torch.tensor(areas[sl]),torch.tensor(classes[sl]),torch.tensor(offsets[sl],dtype=torch.float32))
            means.append(torch.exp(mu).numpy());probabilities.append(torch.sigmoid(score).numpy())
    return np.concatenate(means).reshape(adaptive.shape).astype(float),np.concatenate(probabilities).reshape(adaptive.shape).astype(float)


def calibrate_probability(scores,labels):
    x=logit(np.clip(np.asarray(scores).ravel(),1e-6,1-1e-6));y=np.asarray(labels,dtype=float).ravel()
    if not len(y):raise ValueError('empty probability calibration')
    if len(np.unique(y))<2:
        return dict(slope=0.,intercept=float(logit((y.sum()+1)/(len(y)+2))),fallback=True)
    def objective(theta):
        z=theta[0]*x+theta[1];residual=expit(z)-y
        loss=np.mean(np.logaddexp(0,z)-y*z)+.001*((theta[0]-1)**2+theta[1]**2)
        gradient=np.array([np.mean(residual*x),residual.mean()])+.002*np.array([theta[0]-1,theta[1]])
        return loss,gradient
    result=minimize(objective,[1.,0.],jac=True,bounds=[(0.,5.),(-8.,8.)],method='L-BFGS-B')
    if not result.success:raise RuntimeError('probability calibration failed: '+str(result.message))
    return dict(slope=float(result.x[0]),intercept=float(result.x[1]),fallback=False)


def apply_calibration(scores,record):
    return expit(record['slope']*logit(np.clip(scores,1e-6,1-1e-6))+record['intercept'])


def regional_thresholds(scores,labels,alpha=.05,minimum=40):
    def threshold(negative):
        ordered=np.sort(negative);rank=int(np.ceil((len(ordered)+1)*(1-alpha)))
        return float(np.nextafter(ordered[rank-1],np.inf)) if rank<=len(ordered) else float(np.nextafter(1.,np.inf))
    values=np.empty(scores.shape[1:]);fallback=np.zeros_like(values,dtype=bool);negative_counts=np.zeros_like(values,dtype=int)
    for group in range(scores.shape[2]):
        pooled=scores[:,:,group][~labels[:,:,group]]
        if not len(pooled):raise ValueError('no negative calibration cells')
        for district in range(scores.shape[1]):
            negative=scores[:,district,group][~labels[:,district,group]];negative_counts[district,group]=len(negative)
            fallback[district,group]=len(negative)<minimum;values[district,group]=threshold(pooled if fallback[district,group] else negative)
    return values,dict(fallback=fallback.tolist(),negative_counts=negative_counts.tolist(),unit='district-group negative cells',target_fpr=alpha)


def evaluate_forecast(y,mean,probability,cutoff,threshold,dispersion):
    label=y>=cutoff;alarm=probability>=threshold;positive=int(label.sum());negative=int((~label).sum())
    tp=int((label&alarm).sum());fp=int((~label&alarm).sum());r=1/dispersion
    lower=nbinom.ppf(.025,r,r/(r+mean));upper=nbinom.ppf(.975,r,r/(r+mean))
    regional=np.divide((~label&alarm).sum(0),(~label).sum(0),out=np.zeros(y.shape[1]),where=(~label).sum(0)>0)
    return dict(cells=int(y.size),events=int(y.sum()),mean_observed=float(y.mean()),mean_predicted=float(mean.mean()),mae=float(np.abs(y-mean).mean()),
        rmse=float(np.sqrt(((y-mean)**2).mean())),nb_nll=float(negative_binomial_nll(y,mean,dispersion).mean()),
        interval95_coverage=float(((y>=lower)&(y<=upper)).mean()),interval95_mean_width=float((upper-lower).mean()),
        prevalence=positive/y.size,positives=positive,true_positives=tp,false_positives=fp,alarms=tp+fp,
        recall=tp/positive if positive else None,precision=tp/(tp+fp) if tp+fp else None,fpr=fp/negative if negative else None,
        maximum_district_fpr=float(regional.max()),ap=average_precision(label,probability),
        auc=empirical_auc(probability[~label],probability[label]),brier=float(((probability-label)**2).mean()))


def paired_blocks(y,mean,reference,probability,reference_probability,cutoff,bootstrap=1000,seed=233):
    label=y>=cutoff;differences=np.column_stack([(np.abs(y-mean)-np.abs(y-reference)).sum(1),((probability-label)**2-(reference_probability-label)**2).sum(1)])
    block=56;sums=np.array([differences[i:i+block].sum(0) for i in range(0,len(y),block)]);weights=np.array([min(block,len(y)-i)*y.shape[1] for i in range(0,len(y),block)])
    rng=np.random.default_rng(seed);draw=[]
    for _ in range(bootstrap):
        chosen=rng.integers(0,len(sums),len(sums));draw.append(sums[chosen].sum(0)/weights[chosen].sum())
    low,high=np.quantile(draw,[.025,.975],axis=0);point=sums.sum(0)/weights.sum()
    return dict(mae_difference=float(point[0]),mae_ci_low=float(low[0]),mae_ci_high=float(high[0]),brier_difference=float(point[1]),
        brier_ci_low=float(low[1]),brier_ci_high=float(high[1]),blocks=len(sums),bootstrap_samples=bootstrap,
        interpretation='paired14day blocks, all districts jointly; fewer than4blocks is insufficient for a useful interval')
