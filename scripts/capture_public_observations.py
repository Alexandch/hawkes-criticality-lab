"""Manual public-source snapshots using row IDs and actual observation times."""
from __future__ import annotations
import argparse,csv,gzip,json,shutil,urllib.parse
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from download_expanded_data import get_json,open_url,digest,write_json
from prepare_spatiotemporal_data import SOURCES,category_filter
from critical_processes.observation_ledger import record_snapshot


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--city',choices=list(SOURCES),required=True)
    parser.add_argument('--since',required=True);parser.add_argument('--until',required=True,help='exclusive end of source update/report window, local naive date/time')
    parser.add_argument('--out',type=Path,required=True,help='new snapshot directory, never an archived directory')
    parser.add_argument('--ledger',type=Path,default=Path('data/processed/public_observations/rows.sqlite'));args=parser.parse_args()
    start,end=pd.Timestamp(args.since),pd.Timestamp(args.until)
    if start.tzinfo is not None or end.tzinfo is not None or start>=end:raise ValueError('use increasing local naive timestamps')
    source=SOURCES[args.city];unique='id' if args.city=='chicago' else 'row_id';changed='updated_on' if args.city=='chicago' else 'report_datetime'
    columns=unique+','+source['columns']+(','+changed if changed not in source['columns'].split(',') else '')
    where=f"{changed} >= '{start.isoformat()}' AND {changed} < '{end.isoformat()}' AND "+category_filter(source)
    base=f"https://{source['host']}/resource/{source['dataset']}"
    count_url=base+'.json?'+urllib.parse.urlencode({'$select':'count(*) as rows','$where':where})
    count=int(get_json(count_url)[0]['rows'])
    if count>=1000000:raise ValueError('split the update window into smaller snapshots')
    args.out.mkdir(parents=True,exist_ok=True);path=args.out/(args.city+'.csv.gz')
    if path.exists() or path.with_name(path.name+'.manifest.json').exists():raise ValueError('snapshot output already exists; choose a new --out')
    url=base+'.csv?'+urllib.parse.urlencode({'$select':columns,'$where':where,'$order':unique,'$limit':1000000})
    temporary=path.with_name(path.name+'.partial')
    with open_url(url) as response,gzip.open(temporary,'wb') as output:shutil.copyfileobj(response,output)
    frame=pd.read_csv(temporary,compression='gzip',dtype=str).fillna('')
    if len(frame)!=count or int(get_json(count_url)[0]['rows'])!=count:raise ValueError('source changed while reading; discard partial snapshot using normal file management and retry')
    if frame[unique].duplicated().any():raise ValueError('source row identifiers are not unique')
    temporary.replace(path);observed=datetime.now(timezone.utc).isoformat()
    result=record_snapshot(args.ledger,args.city+':source_rows',((row[unique],row) for row in frame.to_dict('records')),observed,f'{changed}:{start.isoformat()}..{end.isoformat()}')
    write_json(path.with_name(path.name+'.manifest.json'),dict(city=args.city,rows=len(frame),url=url,sha256=digest(path),observed_at=observed,
        filter_column=changed,ledger=str(args.ledger),ledger_receipt=result,
        limitations='SF report window does not capture all later edits to older reports; Chicago update window depends on source updated_on. Absence from a partial window is not deletion.'))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
