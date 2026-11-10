"""Create a separate category panel and a previously unused October snapshot."""
from __future__ import annotations
import argparse,csv,gzip,json,shutil,urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from download_expanded_data import open_url,get_json,digest,write_json
from prepare_spatiotemporal_data import SOURCES,category_filter


GROUPS=['all','property','assault_robbery']


def download_october(city,directory):
    source=SOURCES[city];path=directory/(city+'_202610.csv.gz');manifest=path.with_name(path.name+'.manifest.json')
    if manifest.exists():
        record=json.loads(manifest.read_text(encoding='utf-8'))
        if not path.exists():raise FileNotFoundError('external raw CSV absent; choose a new --raw directory')
        if digest(path)!=record['sha256']:raise ValueError('raw snapshot changed')
        return path
    where=f"{source['date']} >= '2026-10-01T00:00:00' AND {source['date']} < '2026-11-01T00:00:00' AND "+category_filter(source)
    base=f"https://{source['host']}/resource/{source['dataset']}"
    count_url=base+'.json?'+urllib.parse.urlencode({'$select':'count(*) as rows','$where':where})
    url=base+'.csv?'+urllib.parse.urlencode({'$select':source['columns'],'$where':where,'$order':source['date']+','+source['identifier'],'$limit':1000000})
    before=int(get_json(count_url)[0]['rows'])
    if not 0<before<1000000:raise ValueError('invalid October source count')
    temporary=path.with_name(path.name+'.partial')
    with open_url(url) as response,gzip.open(temporary,'wb') as output:shutil.copyfileobj(response,output)
    with gzip.open(temporary,'rt',encoding='utf-8-sig',newline='') as handle:
        reader=csv.reader(handle);columns=next(reader);actual=sum(1 for _ in reader)
    after=int(get_json(count_url)[0]['rows'])
    if before!=actual or actual!=after:raise ValueError('source changed during download; keep partial and retry in a new directory')
    temporary.replace(path)
    write_json(manifest,dict(city=city,start='2026-10-01',end_exclusive='2026-11-01',rows=actual,columns=columns,url=url,
        sha256=digest(path),retrieved_utc=datetime.now(timezone.utc).isoformat(),label_status='preliminary snapshot, later reports and corrections remain possible'))
    print(f'New October input {city}: {actual:,} rows',flush=True)
    return path


def prepare(city,history,october,out,centroids):
    source=SOURCES[city];files=[history/f'{city}_{year}.csv.gz' for year in range(2019,2027)]+[october]
    pieces=[]
    for path in files:
        frame=pd.read_csv(path,dtype=str)
        district=frame[source['district']].str.upper().str.strip()
        if city=='chicago':district=pd.to_numeric(district,errors='coerce').astype('Int64').astype('string').str.zfill(2)
        violent=frame[source['category']].isin(['ASSAULT','BATTERY','ROBBERY','Assault','Robbery'])
        pieces.append(pd.DataFrame(dict(event_id=frame[source['identifier']],time=pd.to_datetime(frame[source['date']],format='mixed',errors='coerce'),
            district=district,group=np.where(violent,2,1),is_new=path==october,report_time=pd.to_datetime(frame.report_datetime,format='mixed',errors='coerce') if city=='san_francisco' else pd.NaT)))
    frame=pd.concat(pieces,ignore_index=True);raw_rows=len(frame)
    # A newly retrieved duplicate may not rewrite the historical training group or report date.
    old_ids=set(frame.loc[~frame.is_new,'event_id'].dropna());repeat=frame.is_new&frame.event_id.isin(old_ids)
    excluded_new_repeats=int(repeat.sum());frame=frame[~repeat].copy()
    invalid=int((frame.event_id.isna()|frame.time.isna()).sum());frame=frame.dropna(subset=['event_id','time'])
    unknown=int((~frame.district.isin(source['districts'])).sum());frame=frame[frame.district.isin(source['districts'])].copy()
    conflicts=frame.groupby('event_id').district.nunique();conflict_ids=conflicts[conflicts>1].index
    frame=frame[~frame.event_id.isin(conflict_ids)].copy();frame['group']=frame.groupby('event_id')['group'].transform('max')
    frame['report_time']=frame.groupby('event_id').report_time.transform('min')
    frame=frame.sort_values('time',kind='stable');duplicates=int(frame.event_id.duplicated().sum());frame=frame.drop_duplicates('event_id')
    start,end=pd.Timestamp('2019-01-01'),pd.Timestamp('2026-11-01');frame=frame[(frame.time>=start)&(frame.time<end)].copy()
    dates=pd.date_range(start,end,freq='6h',inclusive='left');districts=source['districts'];mapping={d:i for i,d in enumerate(districts)}
    event_bins=((frame.time.to_numpy(dtype='datetime64[ns]').astype('int64')-start.value)//(6*3600*10**9)).astype('int32')
    area=frame.district.map(mapping).to_numpy(dtype='int16');group=frame['group'].to_numpy(dtype='int8')
    counts=np.zeros((len(dates),len(districts),3),dtype='int32')
    np.add.at(counts,(event_bins,area,np.zeros(len(area),dtype=int)),1);np.add.at(counts,(event_bins,area,group),1)
    if not np.array_equal(counts[:,:,0],counts[:,:,1:].sum(2)):raise ValueError('category partition failed')
    missing=[]
    for district in districts:
        seen=set(frame.loc[frame.district==district,'time'].dt.to_period('M'))
        missing.extend(f'{district}:{m}' for m in pd.period_range(start,end-pd.Timedelta(days=1),freq='M') if m not in seen)
    if missing:raise ValueError(f'missing district-month coverage: {missing[:10]}')
    with np.load(centroids/(city+'.npz')) as old:
        if old['districts'].tolist()!=districts:raise ValueError('district ordering differs from graph archive')
        centers=old['centroids'].copy()
    arrays=dict(counts=counts,time_ns=dates.to_numpy(dtype='datetime64[ns]').astype('int64'),districts=np.array(districts),groups=np.array(GROUPS),centroids=centers,
        event_bins=event_bins,area_indices=area,group_indices=group)
    if city=='san_francisco':
        report=(frame.report_time.to_numpy(dtype='datetime64[ns]').astype('int64')-start.value)//(6*3600*10**9)
        report[frame.report_time.isna().to_numpy()]=len(dates)+10000;arrays['report_bins']=report.astype('int32')
    path=out/(city+'.npz');np.savez_compressed(path,**arrays)
    write_json(out/(city+'.json'),dict(city=city,bin_hours=6,start=str(start),end_exclusive=str(end),raw_rows=raw_rows,cases=len(frame),
        group_counts={name:int(counts[:,:,i].sum()) for i,name in enumerate(GROUPS)},districts=districts,groups=GROUPS,
        invalid_rows=invalid,unknown_district_rows=unknown,conflicting_cases=len(conflict_ids),duplicate_rows=duplicates,new_rows_for_existing_historical_cases=excluded_new_repeats,
        source_sha256={str(f):digest(f) for f in files},output_sha256=digest(path),centroid_source=str(centroids/(city+'.npz')),
        centroid_source_sha256=digest(centroids/(city+'.npz')),availability='report time plus assumed publication delay for SF; Chicago incident lag only; historical publication vintages unknown',
        novel_control='October 2026, preliminary; primary first14days, supplementary full month'))
    print(f'Prepared offline panel {city}: {counts.shape}, {len(frame):,} cases',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--history',type=Path,default=Path('data/raw/spatiotemporal_20261110'))
    parser.add_argument('--raw',type=Path,default=Path('data/raw/offline_20261110'));parser.add_argument('--out',type=Path,default=Path('data/processed/offline_20261110'))
    parser.add_argument('--centroids',type=Path,default=Path('data/processed/spatiotemporal_20261110'));args=parser.parse_args()
    args.raw.mkdir(parents=True,exist_ok=True);args.out.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:paths=list(pool.map(lambda city:download_october(city,args.raw),SOURCES))
    for city,path in zip(SOURCES,paths):prepare(city,args.history,path,args.out,args.centroids)


if __name__=='__main__':main()
