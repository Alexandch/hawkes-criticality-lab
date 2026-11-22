"""R12: additional coarse public districts and quality-filtered NOAA weather.

Snapshots are immutable. Case identifiers are only used locally for deduplication;
individual coordinates and personal attributes are never downloaded.
"""
from __future__ import annotations
import argparse, calendar, csv, gzip, json, shutil, time, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from download_expanded_data import digest, get_json, open_url, write_json

NYC_DISTRICTS = [1,5,6,7,9,10,13,14,17,18,19,20,22,23,24,25,26,28,30,32,33,34,
    40,41,42,43,44,45,46,47,48,49,50,52,60,61,62,63,66,67,68,69,70,71,72,73,75,
    76,77,78,79,81,83,84,88,90,94,100,101,102,103,104,105,106,107,108,109,110,
    111,112,113,114,115,120,121,122,123]
SOURCES = {
    'nyc': dict(host='data.cityofnewyork.us',dataset='qgea-i56i',date='cmplnt_fr_dt',
        identifier='cmplnt_num',district='addr_pct_cd',category='ofns_desc',report='rpt_dt',
        columns='cmplnt_num,cmplnt_fr_dt,cmplnt_fr_tm,rpt_dt,addr_pct_cd,ofns_desc',
        property=['PETIT LARCENY','GRAND LARCENY','GRAND LARCENY OF MOTOR VEHICLE','BURGLARY'],
        violent=['FELONY ASSAULT','ASSAULT 3 & RELATED OFFENSES','ROBBERY'],
        districts=[str(n).zfill(3) for n in NYC_DISTRICTS]),
    'seattle': dict(host='data.seattle.gov',dataset='tazs-3rd5',date='offense_date',
        identifier='report_number',district='precinct',category='offense_sub_category',report='report_date_time',
        columns='report_number,offense_date,report_date_time,precinct,offense_sub_category',
        property=['LARCENY-THEFT','MOTOR VEHICLE THEFT','BURGLARY'],
        violent=['ASSAULT OFFENSES','AGGRAVATED ASSAULT','ROBBERY'],
        districts=['EAST','NORTH','SOUTH','SOUTHWEST','WEST'])}
STATIONS = {'chicago':'USW00094846','san_francisco':'USW00023234',
    'nyc':'USW00094728','seattle':'USW00024233'}
START, END = '2019-01-01', '2026-10-01'


def source_filter(source):
    return source['category']+' in('+','.join("'"+s.replace("'","''")+"'" for s in source['property']+source['violent'])+')'


def download_year_once(job):
    city,year,raw=job; source=SOURCES[city].copy()
    if city=='nyc' and year==2026: source['dataset']='5uac-w243'
    path=raw/f'{city}_{year}.csv.gz'; meta=path.with_name(path.name+'.manifest.json')
    if meta.exists():
        record=json.loads(meta.read_text(encoding='utf-8'))
        if not path.exists() or digest(path)!=record['sha256']: raise ValueError(f'snapshot mismatch: {path}')
        return record
    start=f'{year}-01-01T00:00:00'; end=min(f'{year+1}-01-01T00:00:00',END+'T00:00:00')
    base=f"https://{source['host']}/resource/{source['dataset']}"
    where=f"{source['date']} >= '{start}' AND {source['date']} < '{end}' AND "+source_filter(source)
    count_url=base+'.json?'+urllib.parse.urlencode({'$select':'count(*) as rows','$where':where})
    count=int(get_json(count_url)[0]['rows'])
    if not 0<count<1000000: raise ValueError(f'unsupported page count: {city}/{year}: {count}')
    url=base+'.csv?'+urllib.parse.urlencode({'$select':source['columns'],'$where':where,
        '$order':source['date']+','+source['identifier'],'$limit':1000000})
    partial=path.with_name(path.name+'.partial')
    with open_url(url) as response,gzip.open(partial,'wb') as handle: shutil.copyfileobj(response,handle)
    with gzip.open(partial,'rt',encoding='utf-8-sig',newline='') as handle:
        reader=csv.reader(handle); columns=next(reader); actual=sum(1 for _ in reader)
    after=int(get_json(count_url)[0]['rows'])
    if actual!=count or after!=count: raise ValueError(f'live source changed: {city}/{year}: {count}/{actual}/{after}')
    partial.replace(path)
    record=dict(city=city,year=year,rows=actual,url=url,count_url=count_url,columns=columns,
        sha256=digest(path),retrieved_utc=datetime.now(timezone.utc).isoformat(),start=start,end_exclusive=end)
    write_json(meta,record); print(f'Downloaded {city}/{year}: {actual:,} rows',flush=True)
    return record


def download_year(job):
    for attempt in range(4):
        try:return download_year_once(job)
        except (OSError,ValueError) as error:
            if attempt==3:raise
            print(f'Retrying {job[0]}/{job[1]} after {type(error).__name__}',flush=True)
            time.sleep(2**attempt)


def prepare_city(city,raw,out):
    source=SOURCES[city]; parts=[]; inputs={}
    for year in range(2019,2027):
        path=raw/f'{city}_{year}.csv.gz'; inputs[path.as_posix()]=digest(path)
        frame=pd.read_csv(path,dtype=str)
        time=pd.to_datetime(frame[source['date']],format='mixed',errors='coerce')
        report=pd.to_datetime(frame[source['report']],format='mixed',errors='coerce')
        district=frame[source['district']].str.strip().str.upper()
        if city=='nyc':
            duration=pd.to_timedelta(frame.cmplnt_fr_tm,errors='coerce')
            duration=duration.where((duration>=pd.Timedelta(0))&(duration<pd.Timedelta(days=1)))
            time=time.dt.normalize()+duration
            district=pd.to_numeric(district,errors='coerce').astype('Int64').astype('string').str.zfill(3)
            report=report.dt.normalize()+pd.Timedelta(days=1) # report date has no clock time
        parts.append(pd.DataFrame(dict(event_id=frame[source['identifier']],time=time,report=report,
            district=district,group=np.where(frame[source['category']].isin(source['violent']),2,1))))
    frame=pd.concat(parts,ignore_index=True); raw_rows=len(frame)
    invalid=int((frame.time.isna()|frame.event_id.isna()).sum()); frame=frame.dropna(subset=['time','event_id'])
    unknown=int((~frame.district.isin(source['districts'])).sum()); frame=frame[frame.district.isin(source['districts'])].copy()
    conflicts=frame.groupby('event_id').district.nunique(); bad=conflicts[conflicts>1].index
    frame=frame[~frame.event_id.isin(bad)].copy()
    frame['group']=frame.groupby('event_id')['group'].transform('max')
    frame['report']=frame.groupby('event_id')['report'].transform('min')
    frame=frame.sort_values('time',kind='stable'); duplicates=int(frame.event_id.duplicated().sum());frame=frame.drop_duplicates('event_id')
    start,end=pd.Timestamp(START),pd.Timestamp(END);frame=frame[(frame.time>=start)&(frame.time<end)].copy()
    dates=pd.date_range(start,end,freq='6h',inclusive='left'); districts=source['districts']
    events=((frame.time.to_numpy(dtype='datetime64[ns]').astype('int64')-start.value)//(6*3600*10**9)).astype('int32')
    areas=frame.district.map({d:i for i,d in enumerate(districts)}).to_numpy(dtype='int16'); groups=frame.group.to_numpy(dtype='int8')
    reports=((frame.report.to_numpy(dtype='datetime64[ns]').astype('int64')-start.value)//(6*3600*10**9))
    reports[frame.report.isna().to_numpy()]=len(dates)+10000
    reports=np.maximum(reports,events)
    counts=np.zeros((len(dates),len(districts),3),dtype='int32')
    np.add.at(counts,(events,areas,np.zeros(len(events),dtype=int)),1);np.add.at(counts,(events,areas,groups),1)
    monthly=[]
    for district in districts:
        seen=set(frame.loc[frame.district==district,'time'].dt.to_period('M'))
        monthly.extend(f'{district}:{m}' for m in pd.period_range(START,end-pd.Timedelta(days=1),freq='M') if m not in seen)
    if monthly: raise ValueError(f'missing district-month coverage: {monthly[:12]}')
    centroid_path=raw/f'{city}_training_centroids.json'
    if centroid_path.exists(): centers=json.loads(centroid_path.read_text(encoding='utf-8'))
    else:
        where=f"{source['date']} >= '2019-01-01T00:00:00' AND {source['date']} < '2019-02-01T00:00:00' AND {source['report']} < '2023-01-01T00:00:00' AND latitude is not null AND longitude is not null AND "+source_filter(source)
        select=source['district']+',avg(latitude) as lat,avg(longitude) as lon'
        if city=='seattle':
            select=source['district']+',avg(latitude::number) as lat,avg(longitude::number) as lon'
            where+=" AND latitude != 'REDACTED' AND longitude != 'REDACTED'"
        url=f"https://{source['host']}/resource/{source['dataset']}.json?"+urllib.parse.urlencode({'$select':select,'$where':where,'$group':source['district'],'$limit':200})
        centers=get_json(url); write_json(centroid_path,centers)
    lookup={}
    for row in centers:
        key=str(row.get(source['district'],'')).strip().upper()
        if city=='nyc' and key: key=str(int(float(key))).zfill(3)
        if key in districts: lookup[key]=[float(row['lat']),float(row['lon'])]
    coordinates=np.array([lookup[d] for d in districts])
    path=out/(city+'.npz')
    np.savez_compressed(path,counts=counts,time_ns=dates.as_unit('ns').asi8,districts=np.array(districts),
        groups=np.array(['all','property','assault_robbery']),centroids=coordinates,
        event_bins=events,area_indices=areas,group_indices=groups,report_bins=reports.astype('int32'))
    delay=(frame.report-frame.time).dt.total_seconds()/3600
    record=dict(city=city,raw_rows=raw_rows,cases=len(frame),districts=districts,start=START,end_exclusive=END,
        invalid_rows=invalid,unknown_district_rows=unknown,conflicting_cases=len(bad),duplicate_offense_rows=duplicates,
        report_missing=int(frame.report.isna().sum()),report_delay_over7d=float((delay.dropna()>168).mean()),
        raw_sha256=inputs,centroid_sha256=digest(centroid_path),centroid_scope='January2019 events reported before2023; coarse graph only',panel_sha256=digest(path),
        scope=source['property']+source['violent'],report_dates_are_publication_times=False,
        report_date_precision='conservative end of recorded day' if city=='nyc' else 'recorded report timestamp',
        notes='selected registered cases; violence priority for mixed cases; no personal attributes; retrospective revised snapshot')
    write_json(out/(city+'.json'),record); print(f'Prepared {city}: {len(frame):,} cases, {len(districts)} districts',flush=True)


def parse_weather(text,station,start=START,end=END):
    rows=[]
    for line in text.splitlines():
        if len(line)<269 or line[:11]!=station: continue
        year,month=int(line[11:15]),int(line[15:17]);element=line[17:21]
        if element not in ('TMAX','TMIN','PRCP','SNOW'):continue
        if not int(start[:4])<=year<=int(end[:4]):continue
        for day in range(1,calendar.monthrange(year,month)[1]+1):
            field=line[21+8*(day-1):29+8*(day-1)]; value=int(field[:5]);quality=field[6]
            date=f'{year:04}-{month:02}-{day:02}'
            if start<=date<end and value!=-9999 and quality==' ':
                rows.append((date,element,value/(10. if element!='SNOW' else 1.)))
    table=pd.DataFrame(rows,columns=['date','element','value'])
    if table.duplicated(['date','element']).any():raise ValueError('duplicate station element/date')
    return table.pivot(index='date',columns='element',values='value').reindex(columns=['TMAX','TMIN','PRCP','SNOW'])


def prepare_weather(city,raw,out):
    station=STATIONS[city]; path=raw/(station+'.dly'); meta=path.with_name(path.name+'.manifest.json')
    url='https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/'+station+'.dly'
    if meta.exists():
        if digest(path)!=json.loads(meta.read_text(encoding='utf-8'))['sha256']:raise ValueError('weather snapshot changed')
    else:
        with open_url(url) as response:content=response.read()
        path.write_bytes(content);write_json(meta,dict(station=station,url=url,sha256=digest(path),
            retrieved_utc=datetime.now(timezone.utc).isoformat(),source='NOAA GHCN-Daily',
            documentation='https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt',
            warning='current revised observations; historical publication times not provided'))
    table=parse_weather(path.read_text(encoding='ascii'),station)
    index=pd.date_range(START,pd.Timestamp(END)-pd.Timedelta(days=1),freq='D');table=table.reindex(index.strftime('%Y-%m-%d'))
    target=out/(city+'_weather.npz');np.savez_compressed(target,day_ns=index.as_unit('ns').asi8,values=table.to_numpy(dtype='float32'))
    record=dict(city=city,station=station,start=START,end_exclusive=END,elements=list(table.columns),
        units=['degrees C','degrees C','mm','mm'],missing_fraction=table.isna().mean().to_dict(),
        input_sha256=digest(path),panel_sha256=digest(target),
        availability='daily observation with assumed 5-day lag; not target-interval weather or verified historical release')
    write_json(out/(city+'_weather.json'),record);print(f'Weather {city}: {table.notna().sum().to_dict()}',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,default=Path('data/raw/expanded_districts_20261119'))
    parser.add_argument('--out',type=Path,default=Path('data/processed/expanded_districts_20261119'));args=parser.parse_args()
    args.raw.mkdir(parents=True,exist_ok=True);args.out.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download_year,[(city,year,args.raw) for city in SOURCES for year in range(2019,2027)]))
    for city in SOURCES:prepare_city(city,args.raw,args.out)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda city:prepare_weather(city,args.raw,args.out),STATIONS))


if __name__=='__main__':main()
