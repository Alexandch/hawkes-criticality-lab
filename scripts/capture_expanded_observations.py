"""Start genuine first-seen observations for the two new public city sources.

This manual snapshot does not issue crime forecasts or reconstruct past releases.
Missing rows in a bounded report-date query never imply deletion.
"""
from __future__ import annotations
import argparse,csv,gzip,json,shutil,urllib.parse
from datetime import datetime,timezone
from pathlib import Path
from critical_processes.observation_ledger import record_snapshot
from download_expanded_data import get_json,open_url,digest,write_json
from prepare_expanded_districts import SOURCES,source_filter


def capture(city,since,until,directory,ledger):
    source=SOURCES[city].copy()
    if city=='nyc':source['dataset']='5uac-w243'
    identifier='cmplnt_num' if city=='nyc' else 'offense_id'
    columns=source['columns'] if city=='nyc' else 'offense_id,'+source['columns']
    where=f"{source['report']} >= '{since}T00:00:00' AND {source['report']} < '{until}T00:00:00' AND "+source_filter(source)
    base=f"https://{source['host']}/resource/{source['dataset']}"
    count_url=base+'.json?'+urllib.parse.urlencode({'$select':'count(*) as rows','$where':where})
    count=int(get_json(count_url)[0]['rows'])
    if count==0:
        directory.mkdir(parents=True,exist_ok=True)
        write_json(directory/(city+'_empty_window.json'),dict(city=city,since=since,until_exclusive=until,
            count_url=count_url,observed_utc=datetime.now(timezone.utc).isoformat(),rows=0,
            interpretation='No published records in the requested report-date window; not evidence of no crimes',
            source_frequency='Quarterly' if city=='nyc' else 'Daily'))
        print(f'Observed {city}: no published records in requested report-date window',flush=True)
        return
    if count>=100000:raise ValueError('excessive observation window')
    url=base+'.csv?'+urllib.parse.urlencode({'$select':columns,'$where':where,'$order':identifier,'$limit':100000})
    directory.mkdir(parents=True,exist_ok=True);path=directory/(city+'.csv.gz');manifest=path.with_name(path.name+'.manifest.json')
    if manifest.exists():
        record=json.loads(manifest.read_text(encoding='utf-8'))
        if digest(path)!=record['sha256']:raise ValueError('snapshot modified')
        if record['since']!=since or record['until_exclusive']!=until:raise ValueError('choose a new output directory')
    else:
        with open_url(url) as response,gzip.open(path.with_suffix('.partial'),'wb') as target:shutil.copyfileobj(response,target)
        path.with_suffix('.partial').replace(path)
        with gzip.open(path,'rt',encoding='utf-8-sig',newline='') as handle:rows=list(csv.DictReader(handle))
        if len(rows)!=count or int(get_json(count_url)[0]['rows'])!=count:raise ValueError('live source changed during observation')
        record=dict(city=city,url=url,since=since,until_exclusive=until,rows=count,sha256=digest(path),
            observed_utc=datetime.now(timezone.utc).isoformat(),scope='partial report-date window; no inference from absence',
            not_training_or_test_input_for_R12=True)
        write_json(manifest,record)
    with gzip.open(path,'rt',encoding='utf-8-sig',newline='') as handle:rows=list(csv.DictReader(handle))
    if any(not row.get(identifier) for row in rows):raise ValueError('observation has no unique source key')
    ids=[row[identifier] for row in rows]
    if len(set(ids))!=len(ids):raise ValueError('source identifiers are not unique')
    result=record_snapshot(ledger,city,[(row[identifier],row) for row in rows],record['observed_utc'],record['scope'])
    write_json(directory/(city+'_ledger_summary.json'),dict(**result,source_manifest_sha256=digest(manifest),
        prospective_forecasts_issued=False,notes='first-seen begins now; recording alone does not validate prediction accuracy'))
    print(f'Observed {city}: {len(rows):,} public source rows; {result}',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--since',required=True);parser.add_argument('--until',required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--ledger',type=Path,default=Path('data/processed/public_observations_r12/rows.sqlite'))
    args=parser.parse_args();args.ledger.parent.mkdir(parents=True,exist_ok=True)
    if not args.since<args.until:raise ValueError('invalid observation date window')
    for city in SOURCES:capture(city,args.since,args.until,args.out,args.ledger)


if __name__=='__main__':main()
