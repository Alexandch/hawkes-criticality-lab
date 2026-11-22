"""R15 appends only an unseen2026 period to the byte-frozen R14 history."""
from __future__ import annotations
import gzip,json,shutil,urllib.parse
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from download_expanded_data import get_json,open_url,digest,write_json
from prepare_delay_forecast_data import BASE,SECTORS,ALIASES,CATEGORIES,local_clock
from critical_processes.offline_forecast import load_categories


def download(raw,plan):
    if not plan.is_file():raise ValueError('a real frozen plan is required')
    frozen=json.loads(plan.read_text(encoding='utf-8'));assert frozen['new_2026_cases_downloaded'] is False
    value=digest(plan);raw.mkdir(parents=True,exist_ok=True);path=raw/'austin_2026_jan_aug.csv.gz';meta=path.with_name(path.name+'.manifest.json')
    if meta.exists():
        saved=json.loads(meta.read_text(encoding='utf-8'))
        assert saved['sha256']==digest(path) and saved['frozen_plan_sha256']==value
        return path
    where="occ_date >= '2026-01-01T00:00:00' AND occ_date < '2026-09-01T00:00:00' AND category_description in("+','.join("'"+v+"'" for v in CATEGORIES)+')'
    count_url=BASE+'.json?'+urllib.parse.urlencode({'$select':'count(*) as n','$where':where})
    url=BASE+'.csv?'+urllib.parse.urlencode({'$select':'incident_report_number,occ_date,occ_time,rep_date,rep_time,sector,category_description',
        '$where':where,'$order':'occ_date,incident_report_number','$limit':100000})
    before=int(get_json(count_url)[0]['n']);assert 0<before<100000
    partial=path.with_name(path.name+'.partial')
    with open_url(url) as response,gzip.open(partial,'wb') as output:shutil.copyfileobj(response,output)
    actual=len(pd.read_csv(partial,compression='gzip',dtype=str));after=int(get_json(count_url)[0]['n']);assert before==after==actual
    partial.replace(path);write_json(meta,dict(rows=actual,url=url,count_url=count_url,sha256=digest(path),frozen_plan_sha256=value,
        retrieved_utc=datetime.now(timezone.utc).isoformat(),scope='Austin2026Jan-Aug, predefined sectors and five categories',
        warning='registered written reports, historical publication times unknown; late2026labels remain provisional'))
    print(f'R15 new source: {actual} rows',flush=True);return path


def prepare(raw,out,history,old_raw,plan):
    source=raw/'austin_2026_jan_aug.csv.gz';frame=pd.read_csv(source,dtype=str);count=len(frame);past=load_categories(history)
    ids=set()
    for year in range(2021,2026):ids.update(pd.read_csv(old_raw/f'austin_{year}.csv.gz',usecols=['incident_report_number'],dtype=str).incident_report_number.dropna())
    repeated=frame.incident_report_number.isin(ids);repeat=int(repeated.sum());frame=frame[~repeated].copy()
    frame=pd.DataFrame(dict(event_id=frame.incident_report_number,time=local_clock(frame.occ_date,frame.occ_time),report=local_clock(frame.rep_date,frame.rep_time),
        district=frame.sector.str.strip().str.upper().replace(ALIASES),group=np.where(frame.category_description.isin(['Aggravated Assault','Robbery']),2,1)))
    invalid=int((frame.event_id.isna()|frame.time.isna()).sum());frame=frame.dropna(subset=['event_id','time'])
    unknown=int((~frame.district.isin(SECTORS)).sum());frame=frame[frame.district.isin(SECTORS)].copy()
    conflicts=frame.groupby('event_id').district.nunique();bad=conflicts[conflicts>1].index;frame=frame[~frame.event_id.isin(bad)].copy()
    frame['group']=frame.groupby('event_id')['group'].transform('max');frame['report']=frame.groupby('event_id')['report'].transform('min')
    frame=frame.sort_values('time',kind='stable');duplicates=int(frame.event_id.duplicated().sum());frame=frame.drop_duplicates('event_id')
    assert ((frame.time>=pd.Timestamp('2026-01-01'))&(frame.time<pd.Timestamp('2026-09-01'))).all()
    dates=pd.date_range(past.dates[0],'2026-09-01',freq='6h',inclusive='left').as_unit('ns');assert dates[len(past.counts)]==pd.Timestamp('2026-01-01')
    events=((frame.time.to_numpy(dtype='datetime64[ns]').astype('int64')-dates[0].value)//(6*3600*10**9)).astype('int32')
    areas=frame.district.map({s:i for i,s in enumerate(SECTORS)}).to_numpy(dtype='int16');groups=frame.group.to_numpy(dtype='int8')
    reports=(frame.report.to_numpy(dtype='datetime64[ns]').astype('int64')-dates[0].value)//(6*3600*10**9)
    missing=frame.report.isna().to_numpy();reports[missing]=len(dates)+10000;reports=np.maximum(events,reports).astype('int32')
    counts=np.zeros((len(dates),9,3),dtype='int32');counts[:len(past.counts)]=past.counts
    np.add.at(counts,(events,areas,np.zeros(len(events),dtype=int)),1);np.add.at(counts,(events,areas,groups),1)
    for sector in SECTORS:assert set(pd.period_range('2026-01','2026-08',freq='M'))<=set(frame.loc[frame.district==sector,'time'].dt.to_period('M'))
    out.mkdir(parents=True,exist_ok=True);target=out/'austin.npz'
    np.savez_compressed(target,counts=counts,time_ns=dates.asi8,districts=past.districts,groups=past.groups,centroids=past.centroids,
        event_bins=np.r_[past.event_bins,events],area_indices=np.r_[past.area_indices,areas],group_indices=np.r_[past.group_indices,groups],report_bins=np.r_[past.report_bins,reports])
    record=dict(raw_rows=count,new_cases=len(frame),historical_cases=int(past.counts[:,:,0].sum()),all_cases=int(counts[:,:,0].sum()),
        invalid_rows=invalid,unknown_sector_rows=unknown,conflicting_cases=len(bad),duplicate_rows=duplicates,excluded_history_rows=repeat,missing_report_time=int(missing.sum()),
        raw_sha256=digest(source),history_sha256=digest(history),panel_sha256=digest(target),frozen_plan_sha256=digest(plan),
        start=str(dates[0]),end_exclusive='2026-09-01',sectors=SECTORS,warning='same nine coarse sectors; late labels provisional, real publication times unknown')
    write_json(out/'austin.json',record);print(f'R15 prepared: {len(frame)} new cases',flush=True);return record
