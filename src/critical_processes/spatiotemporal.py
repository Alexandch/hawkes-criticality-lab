"""District-time count forecasting; distinct from latent-n warning targets."""
from __future__ import annotations
import copy
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize,minimize_scalar
from scipy.special import gammaln
from scipy.stats import nbinom


@dataclass
class DistrictPanel:
    counts: np.ndarray
    time_ns: np.ndarray
    districts: np.ndarray
    centroids: np.ndarray
    bin_hours: int
    event_bins: np.ndarray | None = None
    report_bins: np.ndarray | None = None
    area_indices: np.ndarray | None = None

    @property
    def dates(self):return pd.DatetimeIndex(self.time_ns.astype('datetime64[ns]'))


def load_panel(path,hours=6):
    with np.load(path) as data:panel=DistrictPanel(**{name:data[name].copy() for name in ('counts','time_ns','districts','centroids','event_bins','report_bins','area_indices') if name in data},bin_hours=hours)
    if panel.counts.shape!=(len(panel.time_ns),len(panel.districts)) or (panel.counts<0).any():raise ValueError('invalid district panel')
    if not (np.diff(panel.time_ns)==hours*3600*10**9).all():raise ValueError('time grid must be contiguous and equally spaced')
    return panel


def district_graph(centroids,neighbors=3):
    coordinates=np.asarray(centroids,dtype=float)
    if coordinates.ndim!=2 or coordinates.shape[1]!=2 or len(coordinates)<2 or not np.isfinite(coordinates).all():raise ValueError('need finite district centroids')
    latitude=np.deg2rad(coordinates[:,0]);longitude=np.deg2rad(coordinates[:,1]);mean=float(latitude.mean())
    xy=np.column_stack([longitude*np.cos(mean),latitude])*6371.
    distance=np.sqrt(((xy[:,None]-xy[None])**2).sum(-1));np.fill_diagonal(distance,np.inf)
    graph=np.zeros_like(distance)
    for row in range(len(graph)):
        chosen=np.argsort(distance[row])[:min(neighbors,len(graph)-1)]
        scale=max(float(np.median(distance[row,chosen])),.01)
        weights=np.exp(-distance[row,chosen]/scale);graph[row,chosen]=weights/weights.sum()
    return graph


def split_masks(dates):
    return {name:np.asarray((dates>=pd.Timestamp(start))&(dates<pd.Timestamp(end))) for name,start,end in (
        ('train','2019-03-02','2024-01-01'),('validation','2024-01-01','2024-07-01'),
        ('calibration','2024-07-01','2025-01-01'),('test2025','2025-01-01','2026-01-01'),('test2026','2026-01-01','2026-10-01'))}


def calendar_indices(dates,hours):
    return np.asarray(dates.dayofweek*(24//hours)+dates.hour//hours,dtype=int)


def fit_calendar(panel,train_end='2024-01-01',shrinkage=10.):
    dates=panel.dates;last=(dates>=pd.Timestamp(train_end)-pd.Timedelta(days=365))&(dates<pd.Timestamp(train_end))
    all_train=dates<pd.Timestamp(train_end);slots=7*(24//panel.bin_hours);keys=calendar_indices(dates,panel.bin_hours)
    known=counts_available_by(panel,train_end)
    means=np.maximum(known[last].mean(0),.02);table=np.zeros((slots,len(means)))
    for key in range(slots):
        values=known[last&(keys==key)];table[key]=(values.sum(0)+shrinkage*means)/(len(values)+shrinkage)
    base=table[keys];month=np.asarray(dates.month)-1;ratios=[]
    for m in range(12):
        use=all_train&(month==m);ratios.append((known[use].sum()+1000)/(base[use].sum()+1000))
    ratios=np.asarray(ratios);normalizer=float((base[last]*ratios[month[last],None]).sum()/base[last].sum());ratios/=normalizer
    # Spikes are defined before validation/test, relative to local weekday/time history.
    thresholds=np.stack([np.quantile(known[all_train&(keys==key)],.95,axis=0,method='higher') for key in range(slots)]).astype(int)
    return dict(table=table.tolist(),month_ratio=ratios.tolist(),mean_district=means.tolist(),spike_cutoffs=thresholds.tolist(),
        bin_hours=panel.bin_hours,fit_end=train_end,definition='last365d shrunk district-weekday-slot means, full-training shared month effect; strict exceedance of training95% cell quantile')


def calendar_prediction(calendar,dates):
    keys=calendar_indices(dates,calendar['bin_hours']);month=np.asarray(dates.month)-1
    means=np.asarray(calendar['table'])[keys]*np.asarray(calendar['month_ratio'])[month,None]
    cutoffs=np.asarray(calendar['spike_cutoffs'])[keys]
    return np.maximum(means,.02),cutoffs


def past_rolling(values,length,delay=0):
    values=np.asarray(values,dtype=float)
    prefix=np.vstack([np.zeros((1,values.shape[1])),np.cumsum(values,axis=0)])
    end=np.maximum(0,np.arange(len(values))-delay);start=np.maximum(0,end-length)
    return prefix[end]-prefix[start]


def counts_available_by(panel,cutoff):
    """Training/selection targets may only include reports available by that fit date."""
    if panel.event_bins is None:return panel.counts.copy()
    stop=int(np.searchsorted(panel.time_ns,pd.Timestamp(cutoff).value))
    use=(panel.report_bins<stop)&(panel.event_bins<stop)
    counts=np.zeros_like(panel.counts)
    np.add.at(counts,(panel.event_bins[use],panel.area_indices[use]),1)
    return counts


def observed_rolling(panel,length,delay=0):
    if panel.event_bins is None:return past_rolling(panel.counts,length,delay)
    first=np.maximum(panel.event_bins+delay+1,panel.report_bins+1)
    end=panel.event_bins+delay+length+1
    use=(first<end)&(first<len(panel.counts))
    delta=np.zeros((len(panel.counts)+1,panel.counts.shape[1]))
    np.add.at(delta,(first[use],panel.area_indices[use]),1.)
    np.add.at(delta,(np.minimum(end[use],len(panel.counts)),panel.area_indices[use]),-1.)
    return np.cumsum(delta,axis=0)[:-1]


def observed_memory(panel,tau_bins,delay_bins=0):
    if panel.event_bins is None:return exponential_history(panel.counts,tau_bins,delay_bins)
    rho=np.exp(-1/tau_bins);first=np.maximum(panel.event_bins+delay_bins+1,panel.report_bins+1)
    use=first<len(panel.counts);impulses=np.zeros_like(panel.counts,dtype=float)
    weights=(1-rho)*np.power(rho,first[use]-delay_bins-1-panel.event_bins[use])
    np.add.at(impulses,(first[use],panel.area_indices[use]),weights)
    state=np.zeros(panel.counts.shape[1]);history=np.zeros_like(impulses)
    for index in range(len(history)):
        state=rho*state+impulses[index];history[index]=state
    return history


def exponential_history(counts,tau_bins,delay_bins=0):
    if tau_bins<=0 or delay_bins<0:raise ValueError('invalid memory or information delay')
    counts=np.asarray(counts,dtype=float);rho=np.exp(-1/tau_bins);state=np.zeros(counts.shape[1]);result=np.zeros_like(counts)
    for t in range(len(counts)):
        visible=t-delay_bins-1
        state=rho*state+(1-rho)*(counts[visible] if visible>=0 else 0.)
        result[t]=state
    return result


def forecast_features(panel,base,graph,delay_days=0):
    bins_day=24//panel.bin_hours;delay=delay_days*bins_day;features=[];ratios=[]
    for window in (1,bins_day,7*bins_day,28*bins_day):
        observed=observed_rolling(panel,window,delay);expected=past_rolling(base,window,delay)
        prior=(7*bins_day if window>bins_day else 2)*base
        ratio=np.clip((observed+prior)/(expected+prior),.05,20.)
        ratios.append(ratio);features.append(np.log(ratio))
    features += [np.log(np.maximum(ratio@graph.T,.05)) for ratio in ratios]
    features.append(np.log1p(base))
    dates=panel.dates
    for phase in (2*np.pi*np.asarray(dates.dayofweek)/7,2*np.pi*np.asarray(dates.hour)/24,2*np.pi*np.asarray(dates.dayofyear)/365.25):
        features += [np.broadcast_to(np.sin(phase)[:,None],base.shape),np.broadcast_to(np.cos(phase)[:,None],base.shape)]
    for window in (7*bins_day,28*bins_day):
        observed=observed_rolling(panel,window,delay).sum(1);expected=past_rolling(base,window,delay).sum(1)
        ratio=(observed+7*bins_day*base.sum(1))/(expected+7*bins_day*base.sum(1))
        features.append(np.broadcast_to(np.log(np.clip(ratio,.05,20.))[:,None],base.shape))
    return np.stack(features,axis=-1)


def recent_calendar(panel,base,bins_day,delay_days):
    observed=observed_rolling(panel,28*bins_day,delay_days*bins_day);expected=past_rolling(base,28*bins_day,delay_days*bins_day)
    ratio=np.clip((observed+7*bins_day*base)/(expected+7*bins_day*base),.25,4.)
    return base*ratio


def poisson_nll(y,mean):
    mean=np.maximum(mean,1e-8)
    return mean-y*np.log(mean)+gammaln(y+1)


def fit_hawkes_count(counts,base,memory,graph,mask,spatial):
    scale=np.maximum(counts[mask].mean(0),.02)
    # Similarity transform preserves graph spectral radius while normalizing district volume.
    adjusted=scale[:,None]*graph/scale[None,:]
    neighbor=memory@adjusted.T
    design=np.stack([base,memory,neighbor],axis=-1)[mask].reshape(-1,3);target=counts[mask].reshape(-1)
    if not spatial:design=design[:,:2]
    def objective(theta):
        mean=np.maximum(design@theta,1e-8)
        return float(np.mean(mean-target*np.log(mean))),design.T@(1-target/mean)/len(target)
    result=minimize(objective,np.array([.8,.1,.05]) if spatial else np.array([.8,.1]),jac=True,method='SLSQP',
        bounds=[(1e-6,5.)]+[(0.,.99)]*(2 if spatial else 1),
        constraints=[{'type':'ineq','fun':lambda theta:.99-theta[1:].sum(),'jac':lambda theta:np.r_[0.,-np.ones(len(theta)-1)]}],
        options={'maxiter':250,'ftol':1e-10})
    if not result.success:raise RuntimeError('count Hawkes MLE did not converge: '+str(result.message))
    theta=result.x
    all_design=np.stack([base,memory,neighbor],axis=-1)
    mean=np.maximum(all_design[:,:,:len(theta)]@theta,1e-8)
    return mean,dict(coefficients=theta.tolist(),spatial=bool(spatial),spectral_radius_proxy=float(theta[1:].sum()),
        optimizer_success=True,normalizing_rates=scale.tolist(),graph=graph.tolist())


def fit_dispersion(y,mean):
    def loss(log_alpha):return float(negative_binomial_nll(y,mean,float(np.exp(log_alpha))).mean())
    result=minimize_scalar(loss,bounds=(np.log(1e-4),np.log(3.)),method='bounded',options={'xatol':1e-5})
    if not result.success:raise RuntimeError('dispersion fit failed')
    return float(np.exp(result.x))


def negative_binomial_nll(y,mean,alpha):
    r=1/alpha;mean=np.maximum(mean,1e-8)
    return gammaln(r)+gammaln(y+1)-gammaln(r+y)+r*np.log1p(mean/r)+y*np.log1p(r/mean)


def spike_probability(mean,cutoffs,alpha):
    r=1/alpha
    return nbinom.sf(cutoffs,r,r/(r+np.maximum(mean,1e-8)))


def average_precision(labels,scores):
    y=np.asarray(labels,dtype=bool).ravel();s=np.asarray(scores).ravel()
    if not y.any():return float('nan')
    order=np.argsort(-s,kind='stable');y,s=y[order],s[order]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1])+1,len(s)];tp=np.cumsum(y)[ends-1]
    return float(np.sum(np.diff(np.r_[0,tp])/y.sum()*tp/ends))


def build_count_network(features,districts):
    import torch
    class CountNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__();self.embedding=torch.nn.Embedding(districts,6)
            self.network=torch.nn.Sequential(torch.nn.Linear(features+6,32),torch.nn.ReLU(),torch.nn.Dropout(.1),torch.nn.Linear(32,16),torch.nn.ReLU(),torch.nn.Linear(16,1))
        def forward(self,x,area,offset):
            residual=self.network(torch.cat([x,self.embedding(area)],dim=-1)).squeeze(-1)
            return offset+4*torch.tanh(residual/4)
    return CountNetwork()


def fit_count_network(features,counts,base,train,validation,seed,epochs=100):
    import torch
    torch.manual_seed(seed);torch.set_num_threads(min(4,torch.get_num_threads()))
    mean=features[train].reshape(-1,features.shape[-1]).mean(0);scale=features[train].reshape(-1,features.shape[-1]).std(0);scale[scale<1e-6]=1.
    network=build_count_network(features.shape[-1],counts.shape[1]);optimizer=torch.optim.AdamW(network.parameters(),lr=.002,weight_decay=.001)
    def dataset(mask):
        x=torch.tensor(((features[mask]-mean)/scale).reshape(-1,len(mean)),dtype=torch.float32)
        area=torch.arange(counts.shape[1]).repeat(mask.sum());offset=torch.tensor(np.log(base[mask].ravel()),dtype=torch.float32)
        y=torch.tensor(counts[mask].ravel(),dtype=torch.float32)
        return x,area,offset,y
    tx,ta,to,ty=dataset(train);vx,va,vo,vy=dataset(validation)
    best_loss=np.inf;best_state=None;best_epoch=None;history=[]
    for epoch in range(1,epochs+1):
        network.train();optimizer.zero_grad();logmean=network(tx,ta,to)
        loss=(torch.exp(logmean)-ty*logmean).mean();loss.backward();optimizer.step();val=None
        if epoch%10==0 or epoch==epochs:
            network.eval()
            with torch.no_grad():logmean=network(vx,va,vo);val=float((torch.exp(logmean)-vy*logmean).mean())
            if val<best_loss:best_loss,best_state,best_epoch=val,copy.deepcopy(network.state_dict()),epoch
        history.append(dict(epoch=epoch,training_loss=float(loss.detach()),validation_loss=val))
    return dict(state_dict=best_state,mean=mean,scale=scale,features=features.shape[-1],districts=counts.shape[1],seed=seed,selected_epoch=best_epoch,
        validation_loss=best_loss,architecture='district embedding6 + MLP32-16 Poisson log-mean residual',epochs=epochs),pd.DataFrame(history)


def predict_count_network(features,base,path):
    import torch
    record=torch.load(path,map_location='cpu',weights_only=False);network=build_count_network(record['features'],record['districts']);network.load_state_dict(record['state_dict']);network.eval()
    x=((features-record['mean'])/record['scale']).reshape(-1,features.shape[-1]);areas=np.tile(np.arange(base.shape[1]),len(base));offset=np.log(base.ravel());predictions=[]
    with torch.no_grad():
        for start in range(0,len(x),8192):
            sl=slice(start,start+8192);logmean=network(torch.tensor(x[sl],dtype=torch.float32),torch.tensor(areas[sl],dtype=torch.long),torch.tensor(offset[sl],dtype=torch.float32))
            predictions.append(torch.exp(logmean).numpy())
    return np.concatenate(predictions).reshape(base.shape).astype(float)
