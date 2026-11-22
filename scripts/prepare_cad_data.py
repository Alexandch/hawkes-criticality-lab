"""R16: public monthly CAD aggregates, no addresses or individual attributes."""
from __future__ import annotations
import argparse,gzip,json,shutil,urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from download_expanded_data import get_json,open_url,digest,write_json
from prepare_delay_forecast_data import SECTORS

BASE='https://data.austintexas.gov/resource/22de-7rzg'
RAW=Path('data/raw/cad_austin_20261122')
OUT=Path('data/processed/cad_austin_20261122')
ALIASES={name:code for name,code in zip(['ADAM','BAKER','CHARLIE','DAVID','EDWARD','FRANK','GEORGE','HENRY','IDA'],SECTORS)}
CHANNELS=['all_dispatched','disturbance','alarm','traffic']


def download_year(year,raw):
    path=raw/f'cad_{year}.csv.gz';meta=path.with_name(path.name+'.manifest.json')
    if meta.exists():
        saved=json.loads(meta.read_text(encoding='utf-8'));assert digest(path)==saved['sha256'];return saved
    end=min(f'{year+1}-01-01','2026-09-01')
    where=f"response_datetime >= '{year}-01-01T00:00:00' AND response_datetime < '{end}T00:00:00' AND incident_type='Dispatched Incident'"
    params={'$where':where,'$select':'count(*) as n'};count_url=BASE+'.json?'+urllib.parse.urlencode(params)
    count=int(get_json(count_url)[0]['n'])
    if count<=0:raise ValueError('empty CAD source year')
    params['$select']='date_trunc_ym(response_datetime) as response_month,date_trunc_ym(call_closed_datetime) as closed_month,sector,initial_problem_category,count(*) as n'
    params.update({'$group':'response_month,closed_month,sector,initial_problem_category','$order':'response_month,closed_month,sector,initial_problem_category','$limit':100000})
    url=BASE+'.csv?'+urllib.parse.urlencode(params);partial=path.with_name(path.name+'.partial')
    with open_url(url) as response,gzip.open(partial,'wb') as output:shutil.copyfileobj(response,output)
    frame=pd.read_csv(partial,compression='gzip',dtype=str)
    if len(frame)>=100000 or int(pd.to_numeric(frame.n).sum())!=count:raise ValueError('CAD aggregate truncation or changing source')
    partial.replace(path);record=dict(year=year,aggregate_rows=len(frame),dispatched_incidents=count,url=url,count_url=count_url,
        sha256=digest(path),retrieved_utc=datetime.now(timezone.utc).isoformat(),
        source='APD Computer Aided Dispatch Incidents; monthly publication; counts by response/closure month, sector, initial category',
        note='only dispatched incidents; no officer-initiated events, personal data, outcomes or final classifications')
    write_json(meta,record);print(f'R16 CAD{year}: {count} incidents in {len(frame)} aggregate rows',flush=True);return record


def prepare(raw,out):
    pieces=[];inputs={};total=0
    for year in range(2021,2027):
        path=raw/f'cad_{year}.csv.gz';meta=json.loads(path.with_name(path.name+'.manifest.json').read_text(encoding='utf-8'))
        assert digest(path)==meta['sha256'];frame=pd.read_csv(path,dtype=str);pieces.append(frame);inputs[path.as_posix()]=digest(path);total+=meta['dispatched_incidents']
    frame=pd.concat(pieces,ignore_index=True);number=pd.to_numeric(frame.n).to_numpy(dtype='int64')
    sector=frame.sector.str.strip().str.upper().replace(ALIASES);areas=sector.map({s:i for i,s in enumerate(SECTORS)})
    response=pd.to_datetime(frame.response_month,format='mixed',errors='coerce');closed=pd.to_datetime(frame.closed_month,format='mixed',errors='coerce')
    bad=areas.isna()|response.isna()|closed.isna();invalid_count=int(number[bad].sum())
    frame=frame[~bad].copy();number=number[~bad];areas=areas[~bad].to_numpy(dtype='int16');response=response[~bad];closed=closed[~bad]
    # A call may cross month boundaries. It never becomes available before closure.
    closed=closed.where(closed>=response,response)
    response_month=((response.dt.year-2021)*12+response.dt.month-1).to_numpy(dtype='int16')
    closed_month=((closed.dt.year-2021)*12+closed.dt.month-1).to_numpy(dtype='int16')
    category=frame.initial_problem_category.fillna('').str.lower()
    flags=np.stack([np.ones(len(frame),dtype=bool),category.str.contains('disturb',regex=False),
        category.str.contains('alarm',regex=False),category.str.contains('traffic',regex=False)],axis=1)
    for month in range(68):
        for area in range(9):
            if not ((response_month==month)&(areas==area)).any():raise ValueError(f'missing CAD sector-month {area}/{month}')
    out.mkdir(exist_ok=True,parents=True);target=out/'cad.npz'
    np.savez_compressed(target,response_month=response_month,closed_month=closed_month,areas=areas,counts=number,channel_flags=flags)
    record=dict(source_incidents=total,included_incidents=int(number.sum()),excluded_incidents=invalid_count,
        aggregate_rows=len(frame),channels=CHANNELS,sectors=SECTORS,raw_sha256=inputs,panel_sha256=digest(target),
        scope='2021Jan-2026Aug; monthly response and closure aggregates of dispatched incidents',
        publication='month following closure plus15days primary; plus45days stress; assumptions, not historical first-seen',
        limitation='monthly public source cannot support real-time911claims; current archive can contain retrospective revisions')
    write_json(out/'cad.json',record);print(f'R16 prepared {record["included_incidents"]} included incidents',flush=True);return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,default=RAW);parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args()
    args.raw.mkdir(parents=True,exist_ok=True)
    before=get_json('https://data.austintexas.gov/api/views/22de-7rzg.json')
    with ThreadPoolExecutor(max_workers=2) as pool:
        for record in pool.map(lambda year:download_year(year,args.raw),range(2021,2027)):pass
    after=get_json('https://data.austintexas.gov/api/views/22de-7rzg.json')
    if before.get('rowsUpdatedAt')!=after.get('rowsUpdatedAt'):raise ValueError('CAD source changed during preparation')
    write_json(args.raw/'source_metadata.json',dict(dataset='22de-7rzg',rows_updated_at=after.get('rowsUpdatedAt'),
        description=after['description'],columns=[dict(field=c['fieldName'],description=c.get('description','')) for c in after['columns'] if not c['fieldName'].startswith(':')],
        metadata=after.get('metadata',{}),retrieved_utc=datetime.now(timezone.utc).isoformat()))
    prepare(args.raw,args.out)


if __name__=='__main__':main()
