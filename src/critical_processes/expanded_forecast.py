"""R12 helpers: chronological splits, delayed weather and pooled neural risk."""
from __future__ import annotations
import copy
import numpy as np
import pandas as pd
from scipy.special import expit


def chronological_masks(dates,city=None):
    intervals={'train':('2019-03-02','2023-12-04'),
        'validation':('2024-01-01','2024-06-03'),
        'probability_calibration':('2024-07-01','2024-08-01'),
        'alarm_calibration':('2024-08-01','2024-12-04'),
        'test2025':('2025-01-01','2026-01-01'),
        'test2026':('2026-01-01','2026-10-01')}
    if city=='nyc':
        intervals['train']=('2019-09-01','2023-12-04')
        intervals['alarm_calibration']=('2024-08-01','2024-10-01')
    return {name:np.asarray((dates>=pd.Timestamp(start))&(dates<pd.Timestamp(stop)))
        for name,(start,stop) in intervals.items()}


def quarterly_release_bins(report_bins,origin_ns,release_days=45):
    """Assumed publication after the report quarter, not a recovered vintage.

    NYC input report dates were conservatively placed at the following midnight.
    Subtract one nanosecond to recover the correct quarter of the recorded day.
    """
    if release_days<0:raise ValueError('negative publication delay')
    dates=pd.DatetimeIndex((origin_ns+np.asarray(report_bins,dtype='int64')*(6*3600*10**9)-1).astype('datetime64[ns]'))
    release=dates.to_period('Q').end_time.normalize()+pd.Timedelta(days=release_days+1)
    return ((release.as_unit('ns').asi8-origin_ns)//(6*3600*10**9)).astype('int32')


def delayed_weather(dates,day_ns,values,lag_days=5):
    """A calendar-day lag, with missing flags and no backward fill from the future."""
    if lag_days<1:raise ValueError('weather must precede the forecast day')
    day_ns=np.asarray(day_ns);values=np.asarray(values,dtype=float)
    if values.shape!=(len(day_ns),4) or (np.diff(day_ns)<=0).any():raise ValueError('invalid weather grid')
    wanted=dates.normalize().as_unit('ns').asi8-pd.Timedelta(days=lag_days).value
    indices=np.searchsorted(day_ns,wanted)
    good=(indices<len(day_ns))
    good[good]&=day_ns[indices[good]]==wanted[good]
    observed=np.full((len(dates),4),np.nan);observed[good]=values[indices[good]]
    # Temperature remains in degrees C; precipitation/snow are nonnegative amounts.
    observed[:,2:]=np.log1p(np.maximum(observed[:,2:],0))
    missing=~np.isfinite(observed)
    # Zero is a fixed neutral value. The explicit mask avoids fitted future imputation.
    return np.concatenate([np.where(missing,0.,observed),missing.astype(float)],axis=1).astype('float32')


def build_network(width,districts,cities=4,groups=3):
    import torch
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__();self.district=torch.nn.Embedding(districts,8)
            self.city=torch.nn.Embedding(cities,3);self.group=torch.nn.Embedding(groups,3)
            self.body=torch.nn.Sequential(torch.nn.Linear(width+14,64),torch.nn.ReLU(),
                torch.nn.Dropout(.1),torch.nn.Linear(64,32),torch.nn.ReLU(),torch.nn.Linear(32,2))
        def forward(self,x,category,offset):
            h=self.body(torch.cat([x,self.district(category[:,0]),self.city(category[:,1]),self.group(category[:,2])],1))
            return offset+3*torch.tanh(h[:,0]/3),h[:,1]
    return Network()


def fit_neural(x,category,adaptive,target,train,validation,seed,epochs=120,local_city=None):
    import torch
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    eligible=train.copy();valid=validation.copy()
    if local_city is not None:eligible&=category[:,1]==local_city;valid&=category[:,1]==local_city
    if not eligible.any() or not valid.any():raise ValueError('empty neural split')
    mean=x[eligible].mean(0);scale=x[eligible].std(0);scale[scale<1e-5]=1.
    network=build_network(x.shape[1],int(category[:,0].max()+1));optimizer=torch.optim.AdamW(network.parameters(),lr=.001,weight_decay=.001)
    train_indices=np.flatnonzero(eligible);city_pools=[train_indices[category[train_indices,1]==c] for c in np.unique(category[train_indices,1])]
    # Each city has equal expected contribution in the shared model; local training
    # receives the same expected number of examples per city and epoch.
    size_per_city=16384
    normalizer=np.ones((4,3),dtype='float32')
    for c in np.unique(category[train_indices,1]):
        for g in range(3):normalizer[c,g]=max(float(target[eligible&(category[:,1]==c)&(category[:,2]==g)].mean()),.1)
    def tensors(indices):
        cat=category[indices]
        return (torch.tensor((x[indices]-mean)/scale),torch.tensor(cat,dtype=torch.long),
            torch.tensor(np.log(np.maximum(adaptive[indices],.02)),dtype=torch.float32),
            torch.tensor(target[indices],dtype=torch.float32),
            torch.tensor(target[indices]>=np.ceil(adaptive[indices]+np.maximum(2.,.5*adaptive[indices])),dtype=torch.float32),
            torch.tensor(normalizer[cat[:,1],cat[:,2]]))
    def loss(tensors):
        xx,cc,off,y,label,norm=tensors;mu,score=network(xx,cc,off)
        return ((torch.exp(mu)-y*mu)/norm).mean()+torch.nn.functional.binary_cross_entropy_with_logits(score,label)
    valid_indices=np.flatnonzero(valid);validation_parts=[tensors(part) for part in np.array_split(valid_indices,max(1,int(np.ceil(len(valid_indices)/32768))))]
    best=np.inf;state=None;selected=0;history=[]
    for epoch in range(1,epochs+1):
        chosen=np.concatenate([rng.choice(pool,size=size_per_city,replace=True) for pool in city_pools]);rng.shuffle(chosen)
        network.train();training_loss=0.
        for part in np.array_split(chosen,8):
            optimizer.zero_grad();value=loss(tensors(part));value.backward();optimizer.step();training_loss+=float(value.detach())/8
        value=None
        if epoch%10==0 or epoch==epochs:
            network.eval()
            with torch.no_grad():
                value=sum(float(loss(batch))*len(batch[0]) for batch in validation_parts)/len(valid_indices)
            if value<best:best=value;state=copy.deepcopy(network.state_dict());selected=epoch
        history.append(dict(epoch=epoch,training_loss=training_loss,validation_loss=value))
    record=dict(state_dict=state,mean=mean,scale=scale,width=x.shape[1],districts=int(category[:,0].max()+1),
        seed=seed,epochs=epochs,selected_epoch=selected,validation_loss=best,local_city=local_city,
        architecture='district8/city3/group3 + MLP64-32; residual Poisson count and unweighted BCE risk',
        examples_per_city_per_epoch=size_per_city)
    return record,pd.DataFrame(history)


def predict_neural(x,category,adaptive,record):
    import torch
    network=build_network(record['width'],record['districts']);network.load_state_dict(record['state_dict']);network.eval()
    means=np.empty(len(x));risks=np.empty(len(x))
    with torch.no_grad():
        for start in range(0,len(x),16384):
            sl=slice(start,start+16384)
            mu,score=network(torch.tensor((x[sl]-record['mean'])/record['scale']),torch.tensor(category[sl],dtype=torch.long),
                torch.tensor(np.log(np.maximum(adaptive[sl],.02)),dtype=torch.float32))
            means[sl]=torch.exp(mu).numpy();risks[sl]=torch.sigmoid(score).numpy()
    return means,risks


def operating_intervals(labels,alarms,probability,reference,bootstrap=1000,seed=359):
    """Paired 14-day blocks, including all districts of a city in each block."""
    labels=np.asarray(labels,dtype=bool);alarms=np.asarray(alarms,dtype=bool)
    if labels.ndim!=2 or labels.shape!=alarms.shape:raise ValueError('need time x district cells')
    blocks=[]
    for start in range(0,len(labels),56):
        y=labels[start:start+56];a=alarms[start:start+56];p=probability[start:start+56];r=reference[start:start+56]
        blocks.append([np.sum(y&a),np.sum(~y&a),y.sum(),(~y).sum(),y.size,np.sum((p-y)**2-(r-y)**2)])
    values=np.asarray(blocks,dtype=float);rng=np.random.default_rng(seed)
    draw=values[rng.integers(0,len(values),size=(bootstrap,len(values)))].sum(1)
    def ratio(a,b):return np.divide(a,b,out=np.full_like(a,np.nan),where=b>0)
    quantities={'precision':ratio(draw[:,0],draw[:,0]+draw[:,1]),'recall':ratio(draw[:,0],draw[:,2]),
        'fpr':ratio(draw[:,1],draw[:,3]),'brier_difference':ratio(draw[:,5],draw[:,4])}
    result={'blocks':len(values),'bootstrap_samples':bootstrap}
    for name,value in quantities.items():
        finite=value[np.isfinite(value)]
        result[name+'_ci_low']=float(np.quantile(finite,.025)) if len(finite) and len(values)>=4 else None
        result[name+'_ci_high']=float(np.quantile(finite,.975)) if len(finite) and len(values)>=4 else None
    return result


def practical_screen(row):
    """Predeclared research screen; no claim of operational acceptance or impact."""
    requirements={'precision_at_least_30pct':row['precision'] is not None and row['precision']>=.30,
        'recall_at_least_10pct':row['recall'] is not None and row['recall']>=.10,
        'fpr_at_most_5pct':row['fpr'] is not None and row['fpr']<=.05,
        'ap_beats_simple_baseline':row['ap']>row['baseline_ap'],
        'brier_improvement_ci_below_zero':row['brier_difference_ci_high'] is not None and row['brier_difference_ci_high']<0}
    return {'passed':all(requirements.values()),'requirements':requirements}
