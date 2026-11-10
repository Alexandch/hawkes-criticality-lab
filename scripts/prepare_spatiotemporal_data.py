"""Download coarse-district event fields; no individual coordinates are stored."""
from __future__ import annotations
import argparse,csv,gzip,json,shutil,time,urllib.parse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from download_expanded_data import get_json,open_url,digest,write_json


SOURCES={
 'chicago':dict(host='data.cityofchicago.org',dataset='ijzp-q8t2',date='date',identifier='case_number',district='district',category='primary_type',
     columns='case_number,date,district,primary_type',categories=['ASSAULT','BATTERY','BURGLARY','ROBBERY','THEFT','MOTOR VEHICLE THEFT'],
     districts=[f'{n:02}' for n in [1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,22,24,25]]),
 'san_francisco':dict(host='data.sfgov.org',dataset='wg3w-h783',date='incident_datetime',identifier='incident_id',district='police_district',category='incident_category',
     columns='incident_id,incident_datetime,report_datetime,police_district,incident_category',categories=['Assault','Burglary','Robbery','Larceny Theft','Motor Vehicle Theft'],
     districts=['BAYVIEW','CENTRAL','INGLESIDE','MISSION','NORTHERN','PARK','RICHMOND','SOUTHERN','TARAVAL','TENDERLOIN'])}


def category_filter(source):
    values=','.join("'"+value+"'" for value in source['categories'])
    return source['category']+' in('+values+')'


def download_year(task):
    city,year,directory=task;s=SOURCES[city];start=f'{year}-01-01T00:00:00';end=f'{year+1}-01-01T00:00:00' if year<2026 else '2026-10-01T00:00:00'
    path=directory/f'{city}_{year}.csv.gz';manifest=path.with_name(path.name+'.manifest.json')
    if manifest.exists():
        if not path.exists():
            raise FileNotFoundError(f'Archived manifest exists without its external CSV: {path}. Download a new snapshot with --raw data/raw/spatial_local --out data/processed/spatial_local; do not overwrite archived manifests.')
        record=json.loads(manifest.read_text(encoding='utf-8'));assert digest(path)==record['sha256'];return record
    where=f"{s['date']} >= '{start}' AND {s['date']} < '{end}' AND "+category_filter(s)
    base=f"https://{s['host']}/resource/{s['dataset']}"
    count_url=base+'.json?'+urllib.parse.urlencode({'$select':'count(*) as rows','$where':where})
    url=base+'.csv?'+urllib.parse.urlencode({'$select':s['columns'],'$where':where,'$order':s['date']+','+s['identifier'],'$limit':1000000})
    for attempt in range(4):
        try:
            count=int(get_json(count_url)[0]['rows']);assert count<1000000
            temporary=path.with_name(path.name+'.partial')
            with open_url(url) as response,gzip.open(temporary,'wb') as output:shutil.copyfileobj(response,output)
            with gzip.open(temporary,'rt',encoding='utf-8-sig',newline='') as handle:
                reader=csv.reader(handle);columns=next(reader);actual=sum(1 for _ in reader)
            after=int(get_json(count_url)[0]['rows']);assert actual==count==after
            temporary.replace(path)
            record=dict(city=city,year=year,url=url,count_url=count_url,rows=actual,columns=columns,sha256=digest(path),
                retrieved_utc=datetime.now(timezone.utc).isoformat(),end_exclusive=end,scope=s['categories'])
            write_json(manifest,record);print(f'Spatial fields {city}/{year}: {actual:,} rows',flush=True);return record
        except Exception:
            if attempt==3:raise
            time.sleep(2**attempt)


def prepare_city(city,raw,out,hours):
    source=SOURCES[city];parts=[];raw_files=[]
    for year in range(2019,2027):
        path=raw/f'{city}_{year}.csv.gz';raw_files.append(path);frame=pd.read_csv(path,dtype=str)
        dates=pd.to_datetime(frame[source['date']],format='mixed',errors='coerce')
        district=frame[source['district']].str.upper().str.strip()
        if city=='chicago':district=pd.to_numeric(district,errors='coerce').astype('Int64').astype('string').str.zfill(2)
        report=pd.to_datetime(frame['report_datetime'],format='mixed',errors='coerce') if city=='san_francisco' else pd.Series(pd.NaT,index=frame.index)
        parts.append(pd.DataFrame(dict(event_id=frame[source['identifier']],time=dates,district=district,report_time=report)))
    frame=pd.concat(parts,ignore_index=True);raw_rows=len(frame)
    invalid=int((frame.time.isna()|frame.event_id.isna()).sum());frame=frame.dropna(subset=['time','event_id'])
    unknown=int((~frame.district.isin(source['districts'])).sum());frame=frame[frame.district.isin(source['districts'])].copy()
    conflicts=frame.groupby('event_id').district.nunique();bad=conflicts[conflicts>1].index
    frame=frame[~frame.event_id.isin(bad)].sort_values('time',kind='stable')
    duplicates=int(frame.event_id.duplicated().sum());frame=frame.drop_duplicates('event_id')
    start,end=pd.Timestamp('2019-01-01'),pd.Timestamp('2026-10-01')
    frame=frame[(frame.time>=start)&(frame.time<end)].copy()
    index=pd.date_range(start,end,freq=f'{hours}h',inclusive='left');districts=source['districts'];mapping={value:i for i,value in enumerate(districts)}
    bin_index=((frame.time.to_numpy(dtype='datetime64[ns]').astype(np.int64)-start.value)//(hours*3600*10**9)).astype(int)
    area_index=frame.district.map(mapping).to_numpy(dtype=int)
    counts=np.zeros((len(index),len(districts)),dtype=np.int32);np.add.at(counts,(bin_index,area_index),1)
    missing_months=[]
    for district in districts:
        observed=set(frame.loc[frame.district==district,'time'].dt.to_period('M').astype(str))
        missing_months += [district+':'+str(month) for month in pd.period_range(start,end-pd.Timedelta(days=1),freq='M') if str(month) not in observed]
    if missing_months:raise ValueError(f'{city}: missing district-months: {missing_months[:12]}')
    # Training-only event centroids: coarse public district graph, not household locations.
    where=f"{source['date']} >= '2019-01-01T00:00:00' AND {source['date']} < '2024-01-01T00:00:00' AND "+category_filter(source)+" AND latitude is not null AND longitude is not null"
    if city=='san_francisco':where+=" AND report_datetime < '2024-01-01T00:00:00'"
    query={'$select':source['district']+',avg(latitude) as center_latitude,avg(longitude) as center_longitude','$where':where,'$group':source['district'],'$limit':100}
    centroid_url=f"https://{source['host']}/resource/{source['dataset']}.json?"+urllib.parse.urlencode(query)
    centroid_cache=raw/(f'{city}_asof20240101_centroids.json' if city=='san_francisco' else f'{city}_training_centroids.json')
    if centroid_cache.exists():centroids=json.loads(centroid_cache.read_text(encoding='utf-8'))
    else:centroids=get_json(centroid_url);write_json(centroid_cache,centroids)
    lookup={}
    for row in centroids:
        key=row[source['district']].upper().strip()
        if city=='chicago':key=str(int(key)).zfill(2)
        lookup[key]=(float(row['center_latitude']),float(row['center_longitude']))
    coordinates=np.array([lookup[district] for district in districts])
    out.mkdir(parents=True,exist_ok=True)
    path=out/(city+'.npz')
    arrays=dict(counts=counts,time_ns=index.to_numpy(dtype='datetime64[ns]').astype(np.int64),districts=np.array(districts),centroids=coordinates)
    if city=='san_francisco':
        report_ns=frame.report_time.to_numpy(dtype='datetime64[ns]').astype(np.int64)
        report_bins=(report_ns-start.value)//(hours*3600*10**9)
        report_bins[frame.report_time.isna().to_numpy()]=len(index)+1
        arrays.update(event_bins=bin_index.astype(np.int32),report_bins=report_bins.astype(np.int32),area_indices=area_index.astype(np.int16))
    np.savez_compressed(path,**arrays)
    delay=(frame.report_time-frame.time).dt.total_seconds()/3600
    meta=dict(city=city,bin_hours=hours,districts=districts,start=str(start),end_exclusive=str(end),raw_rows=raw_rows,
        invalid_rows=invalid,unknown_district_rows=unknown,conflicting_district_cases=len(bad),duplicate_offense_rows=duplicates,
        incidents=len(frame),grid_cells=int(counts.size),fraction_zero_cells=float((counts==0).mean()),
        fraction_midnight=float((frame.time.dt.normalize()==frame.time).mean()),
        report_delay_available=city=='san_francisco',report_delay_fraction_over6h=float((delay[delay.notna()]>6).mean()) if delay.notna().any() else None,
        report_delay_fraction_over7days=float((delay[delay.notna()]>168).mean()) if delay.notna().any() else None,
        report_time_missing_fraction=float(frame.report_time.isna().mean()) if city=='san_francisco' else None,
        category_scope=source['categories'],source_sha256={str(path):digest(path) for path in raw_files},output_sha256=digest(path),
        centroid_url=centroid_url,centroids=coordinates.tolist(),centroid_definition='mean training-event coordinate per district, 2019-2023 only; not boundary adjacency',
        caveat='retrospective incident-date snapshot; historical reporting vintages are unavailable, particularly for Chicago')
    write_json(out/(city+'.json'),meta);print(f'Prepared {city}: {counts.shape}, {len(frame):,} incidents',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,default=Path('data/raw/spatiotemporal_20261110'))
    parser.add_argument('--out',type=Path,default=Path('data/processed/spatiotemporal_20261110'));parser.add_argument('--hours',type=int,choices=[6,24],default=6)
    args=parser.parse_args();args.raw.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending=[pool.submit(download_year,(city,year,args.raw)) for city in SOURCES for year in range(2019,2027)]
        records=[future.result() for future in as_completed(pending)]
    write_json(args.raw/'download_summary.json',dict(records=records,scope=SOURCES))
    for city in SOURCES:prepare_city(city,args.raw,args.out,args.hours)


if __name__=='__main__':main()
