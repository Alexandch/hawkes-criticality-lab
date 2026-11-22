"""R14 Austin coarse case fields and genuine archived 24h temperature forecasts.

Development downloads exclude2025. A final-year download requires a pretest plan
hash. No addresses, individual coordinates or personal attributes are requested.
"""
from __future__ import annotations
import argparse,gzip,json,shutil,urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from download_expanded_data import get_json,open_url,digest,write_json

SECTORS=['AD','BA','CH','DA','ED','FR','GE','HE','ID']
ALIASES={'ADAM':'AD','DAVID':'DA','DAVD':'DA','CHAR':'CH','CHARLIE':'CH','FRNK':'FR','EDWD':'ED','EDWARD':'ED','HENR':'HE','HENRY':'HE'}
CATEGORIES=['Theft','Auto Theft','Burglary','Aggravated Assault','Robbery']
BASE='https://data.austintexas.gov/resource/fdj4-gpfu'
START='2021-04-01'


def download_cases(year,raw,frozen_plan_sha=None):
    if year>=2025 and frozen_plan_sha is None:raise ValueError('final-year cases require a frozen plan')
    path=raw/f'austin_{year}.csv.gz';meta=path.with_name(path.name+'.manifest.json')
    if meta.exists():
        record=json.loads(meta.read_text(encoding='utf-8'))
        if digest(path)!=record['sha256'] or record.get('frozen_plan_sha256')!=frozen_plan_sha:raise ValueError('case snapshot/plan changed')
        return path
    where=f"occ_date >= '{year}-01-01T00:00:00' AND occ_date < '{year+1}-01-01T00:00:00' AND category_description in("+','.join("'"+v+"'" for v in CATEGORIES)+')'
    count_url=BASE+'.json?'+urllib.parse.urlencode({'$select':'count(*) as n','$where':where})
    url=BASE+'.csv?'+urllib.parse.urlencode({'$select':'incident_report_number,occ_date,occ_time,rep_date,rep_time,sector,category_description',
        '$where':where,'$order':'occ_date,incident_report_number','$limit':100000})
    before=int(get_json(count_url)[0]['n'])
    if not 0<before<100000:raise ValueError('unsupported yearly source count')
    partial=path.with_name(path.name+'.partial')
    with open_url(url) as response,gzip.open(partial,'wb') as output:shutil.copyfileobj(response,output)
    actual=len(pd.read_csv(partial,compression='gzip',dtype=str));after=int(get_json(count_url)[0]['n'])
    if before!=actual or after!=actual:raise ValueError('source changed during case download')
    partial.replace(path)
    write_json(meta,dict(year=year,rows=actual,sha256=digest(path),url=url,count_url=count_url,
        retrieved_utc=datetime.now(timezone.utc).isoformat(),frozen_plan_sha256=frozen_plan_sha,
        warning='registered written police reports; weekly public updates; historical publication times unknown'))
    print(f'R14 cases {year}: {actual:,} rows',flush=True);return path


def download_weather(year,raw,frozen_plan_sha=None):
    if year>=2025 and frozen_plan_sha is None:raise ValueError('final-year weather requires a frozen plan')
    path=raw/f'austin_weather_{year}.json';meta=path.with_name(path.name+'.manifest.json')
    if meta.exists():
        record=json.loads(meta.read_text(encoding='utf-8'))
        if digest(path)!=record['sha256'] or record.get('frozen_plan_sha256')!=frozen_plan_sha:raise ValueError('weather snapshot/plan changed')
        return path
    url='https://previous-runs-api.open-meteo.com/v1/forecast?'+urllib.parse.urlencode(dict(latitude=30.2672,longitude=-97.7431,
        start_date=f'{year}-01-01',end_date=f'{year}-12-31',hourly='temperature_2m_previous_day1',models='gfs_global',timezone='UTC'))
    data=get_json(url)
    if 'hourly' not in data:raise ValueError('forecast archive unavailable')
    write_json(path,data)
    values=data['hourly']['temperature_2m_previous_day1'];valid=sum(v is not None for v in values)
    if year>=2022 and valid/len(values)<.90:raise ValueError('incomplete temperature forecast archive')
    write_json(meta,dict(year=year,rows=len(values),non_missing=valid,url=url,sha256=digest(path),
        retrieved_utc=datetime.now(timezone.utc).isoformat(),frozen_plan_sha256=frozen_plan_sha,
        source='Open-Meteo Previous Runs, GFS Global, nominal24h lead',
        documentation='https://open-meteo.com/en/docs/previous-runs-api',
        warning='archived operational forecasts, not reanalysis; model versions change; exact delivery timestamps not supplied'))
    print(f'R14 forecast weather {year}: {valid:,}/{len(values):,} values',flush=True);return path


def local_clock(date,time):
    clock=pd.to_numeric(time,errors='coerce');hour=clock//100;minute=clock%100
    valid=clock.notna()&(hour>=0)&(hour<24)&(minute>=0)&(minute<60)&(clock==np.floor(clock))
    result=pd.to_datetime(date,format='mixed',errors='coerce').dt.normalize()+pd.to_timedelta(hour.fillna(0),unit='h')+pd.to_timedelta(minute.fillna(0),unit='m')
    return result.where(valid)


def weather_features(series,dates):
    local=series.index.tz_convert('America/Chicago').tz_localize(None);bins=local.floor('6h')
    rows=[];latest=[]
    for date in dates:
        use=bins==date;values=series.to_numpy()[use];hours=series.index[use];valid=np.isfinite(values)
        rows.append([float(values[valid].mean()) if valid.any() else 0.,float(values[valid].max()-values[valid].min()) if valid.any() else 0.,float(1-valid.mean()) if len(valid) else 1.])
        issue=int((hours.max()-pd.Timedelta(hours=24)).value) if len(hours) else np.iinfo('int64').min
        origin=date.tz_localize('America/Chicago').tz_convert('UTC').value
        if len(hours) and issue>=origin-pd.Timedelta(hours=12).value:raise ValueError('forecast issue too late for the target interval')
        latest.append(issue)
    return np.asarray(rows,dtype='float32'),np.asarray(latest,dtype='int64')


def prepare(raw,out,years,frozen_plan_sha=None,history=None):
    from critical_processes.offline_forecast import CategoryPanel,load_categories
    pieces=[];inputs={}
    for year in years:
        path=raw/f'austin_{year}.csv.gz';inputs[path.as_posix()]=digest(path);frame=pd.read_csv(path,dtype=str)
        pieces.append(pd.DataFrame(dict(event_id=frame.incident_report_number,time=local_clock(frame.occ_date,frame.occ_time),
            report=local_clock(frame.rep_date,frame.rep_time),district=frame.sector.str.strip().str.upper().replace(ALIASES),
            group=np.where(frame.category_description.isin(['Aggravated Assault','Robbery']),2,1))))
    frame=pd.concat(pieces,ignore_index=True);raw_rows=len(frame);excluded_history=0
    if history is not None:
        old_ids=set()
        for year in range(2021,2025):old_ids.update(pd.read_csv(raw/f'austin_{year}.csv.gz',usecols=['incident_report_number'],dtype=str).incident_report_number.dropna())
        repeated=frame.event_id.isin(old_ids);excluded_history=int(repeated.sum());frame=frame[~repeated].copy()
    invalid=int((frame.event_id.isna()|frame.time.isna()).sum());frame=frame.dropna(subset=['event_id','time'])
    unknown=int((~frame.district.isin(SECTORS)).sum());frame=frame[frame.district.isin(SECTORS)].copy()
    conflicts=frame.groupby('event_id').district.nunique();bad=conflicts[conflicts>1].index;frame=frame[~frame.event_id.isin(bad)].copy()
    frame['group']=frame.groupby('event_id')['group'].transform('max');frame['report']=frame.groupby('event_id')['report'].transform('min')
    frame=frame.sort_values('time',kind='stable');duplicate=int(frame.event_id.duplicated().sum());frame=frame.drop_duplicates('event_id')
    end=f'{max(years)+1}-01-01';dates=pd.date_range(START,end,freq='6h',inclusive='left').as_unit('ns')
    frame=frame[(frame.time>=pd.Timestamp('2025-01-01') if history is not None else frame.time>=pd.Timestamp(START))&(frame.time<pd.Timestamp(end))].copy()
    events=((frame.time.to_numpy(dtype='datetime64[ns]').astype('int64')-dates[0].value)//(6*3600*10**9)).astype('int32')
    areas=frame.district.map({s:i for i,s in enumerate(SECTORS)}).to_numpy(dtype='int16');groups=frame.group.to_numpy(dtype='int8')
    reports=((frame.report.to_numpy(dtype='datetime64[ns]').astype('int64')-dates[0].value)//(6*3600*10**9))
    missing=frame.report.isna().to_numpy();reports[missing]=len(dates)+10000;reports=np.maximum(reports,events).astype('int32')
    counts=np.zeros((len(dates),len(SECTORS),3),dtype='int32')
    if history is not None:
        past=load_categories(history);counts[:len(past.counts)]=past.counts
    np.add.at(counts,(events,areas,np.zeros(len(events),dtype=int)),1);np.add.at(counts,(events,areas,groups),1)
    arrays=dict(counts=counts,time_ns=dates.asi8,districts=np.array(SECTORS),groups=np.array(['all','property','assault_robbery']),
        centroids=np.zeros((len(SECTORS),2)),event_bins=events,area_indices=areas,group_indices=groups,report_bins=reports)
    if history is not None:
        for key in ('event_bins','area_indices','group_indices','report_bins'):arrays[key]=np.concatenate([getattr(past,key),arrays[key]])
        np.testing.assert_array_equal(counts[:len(past.counts)],past.counts)
    # Require genuine observations in every sector-month of this downloaded scope.
    for district in SECTORS:
        seen=set(frame.loc[frame.district==district,'time'].dt.to_period('M'))
        expected=pd.period_range('2025-01-01' if history is not None else START,pd.Timestamp(end)-pd.Timedelta(days=1),freq='M')
        if set(expected)-seen:raise ValueError(f'missing sector-month: {district}')
    target=out/'austin.npz';np.savez_compressed(target,**arrays)
    write_json(out/'austin.json',dict(start=START,end_exclusive=end,raw_rows=raw_rows,new_cases=len(frame),all_cases=int(counts[:,:,0].sum()),
        invalid_rows=invalid,unknown_sector_rows=unknown,conflicting_cases=len(bad),duplicate_rows=duplicate,excluded_history_rows=excluded_history,
        missing_report_time=int(missing.sum()),raw_sha256=inputs,panel_sha256=digest(target),frozen_plan_sha256=frozen_plan_sha,
        timestamp='local naive America/Chicago wall-clock bins; exact source publication history unknown',
        sectors=SECTORS,scope=CATEGORIES,warning='coarse recorded sectors; original historical boundaries not reconstructed'))
    weather=[];weather_inputs={}
    for year in (range(2021,max(years)+1)):
        path=raw/f'austin_weather_{year}.json';weather_inputs[path.as_posix()]=digest(path)
        data=json.loads(path.read_text(encoding='utf-8'));weather.append(pd.Series(data['hourly']['temperature_2m_previous_day1'],index=pd.to_datetime(data['hourly']['time'],utc=True),dtype=float))
    # UTC Jan1 hours cover the final local December31 evening. These are prior
    # forecasts for old local-date targets, not next-year crime observations.
    boundary=raw/f'austin_weather_boundary_{max(years)}.json';boundary_meta=boundary.with_name(boundary.name+'.manifest.json')
    if not boundary_meta.exists():
        url='https://previous-runs-api.open-meteo.com/v1/forecast?'+urllib.parse.urlencode(dict(latitude=30.2672,longitude=-97.7431,
            start_date=f'{max(years)+1}-01-01',end_date=f'{max(years)+1}-01-01',hourly='temperature_2m_previous_day1',models='gfs_global',timezone='UTC'))
        write_json(boundary,get_json(url));write_json(boundary_meta,dict(url=url,sha256=digest(boundary),retrieved_utc=datetime.now(timezone.utc).isoformat(),
            purpose='UTC padding for the last old-year local6h bin',frozen_plan_sha256=frozen_plan_sha))
    if digest(boundary)!=json.loads(boundary_meta.read_text(encoding='utf-8'))['sha256']:raise ValueError('weather boundary snapshot changed')
    data=json.loads(boundary.read_text(encoding='utf-8'));weather_inputs[boundary.as_posix()]=digest(boundary)
    weather.append(pd.Series(data['hourly']['temperature_2m_previous_day1'],index=pd.to_datetime(data['hourly']['time'],utc=True),dtype=float))
    series=pd.concat(weather).sort_index()
    if series.index.duplicated().any():raise ValueError('duplicate forecast valid hours')
    rows,latest=weather_features(series,dates)
    target=out/'austin_weather.npz';np.savez_compressed(target,time_ns=dates.asi8,features=rows,latest_nominal_issue_utc_ns=latest)
    write_json(out/'austin_weather.json',dict(features=['forecast24h_temperature_mean','forecast24h_temperature_range','missing_fraction'],
        raw_sha256=weather_inputs,panel_sha256=digest(target),frozen_plan_sha256=frozen_plan_sha,
        nominal_latest_issue_before_bin_hours=12,model='gfs_global',not_observed_future_weather=True,
        warning='nominal lead-time archive; actual dissemination times not provided'))
    print(f'R14 prepared Austin: {int(counts[:,:,0].sum()):,} cases, {len(SECTORS)} sectors',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,default=Path('data/raw/delay_forecast_20261122'))
    parser.add_argument('--out',type=Path,default=Path('data/processed/delay_forecast_20261122/development'));args=parser.parse_args()
    args.raw.mkdir(parents=True,exist_ok=True);args.out.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks=[pool.submit(download_cases,year,args.raw) for year in range(2021,2025)]+[pool.submit(download_weather,year,args.raw) for year in range(2021,2025)]
        for task in tasks:task.result()
    prepare(args.raw,args.out,range(2021,2025))


if __name__=='__main__':main()
