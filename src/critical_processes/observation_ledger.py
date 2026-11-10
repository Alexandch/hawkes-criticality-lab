"""Record actual snapshot observation times; never infer them from incident dates."""
from __future__ import annotations
import argparse,hashlib,json,sqlite3
from contextlib import closing
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd


def record_snapshot(ledger,source,records,observed_at,scope='unspecified'):
    moment=datetime.fromisoformat(observed_at.replace('Z','+00:00'))
    if moment.tzinfo is None:raise ValueError('observation time requires an explicit timezone')
    moment=moment.astimezone(timezone.utc).isoformat();normalized={}
    for key,value in records:
        if not key:raise ValueError('empty record identifier')
        payload=json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
        normalized.setdefault(str(key),set()).add(payload)
    hashes={key:hashlib.sha256('\n'.join(sorted(values)).encode('utf-8')).hexdigest() for key,values in normalized.items()}
    snapshot_hash=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    path=Path(ledger);path.parent.mkdir(parents=True,exist_ok=True)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute('CREATE TABLE IF NOT EXISTS snapshots(source TEXT, observed_at TEXT, scope TEXT, sha256 TEXT, records INTEGER, PRIMARY KEY(source,observed_at,scope))')
        db.execute('CREATE TABLE IF NOT EXISTS observations(source TEXT, record_id TEXT, first_seen TEXT, last_seen TEXT, payload_sha256 TEXT, PRIMARY KEY(source,record_id))')
        db.execute('CREATE TABLE IF NOT EXISTS revisions(source TEXT, record_id TEXT, observed_at TEXT, payload_sha256 TEXT, PRIMARY KEY(source,record_id,observed_at))')
        latest=db.execute('SELECT MAX(observed_at) FROM snapshots WHERE source=?',(source,)).fetchone()[0]
        existing=db.execute('SELECT sha256 FROM snapshots WHERE source=? AND observed_at=? AND scope=?',(source,moment,scope)).fetchone()
        if existing:
            if existing[0]!=snapshot_hash:raise ValueError('same snapshot identity has different content')
            return dict(records=len(hashes),new=0,changed=0,cached=True,observed_at=moment,sha256=snapshot_hash)
        if latest and moment<latest:raise ValueError('cannot backdate an observation snapshot')
        new=changed=0
        for key,value in hashes.items():
            previous=db.execute('SELECT payload_sha256 FROM observations WHERE source=? AND record_id=?',(source,key)).fetchone()
            if previous is None:
                new+=1;db.execute('INSERT INTO observations VALUES(?,?,?,?,?)',(source,key,moment,moment,value))
            else:
                if previous[0]!=value:
                    changed+=1;db.execute('INSERT INTO revisions VALUES(?,?,?,?)',(source,key,moment,value))
                db.execute('UPDATE observations SET last_seen=?,payload_sha256=? WHERE source=? AND record_id=?',(moment,value,source,key))
        db.execute('INSERT INTO snapshots VALUES(?,?,?,?,?)',(source,moment,scope,snapshot_hash,len(hashes)))
    return dict(records=len(hashes),new=new,changed=changed,cached=False,observed_at=moment,sha256=snapshot_hash,
        caveat='absence from this scope does not imply deletion; first_seen begins at actual snapshot observation, not historical incident time')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--csv',type=Path,required=True);parser.add_argument('--id-column',required=True)
    parser.add_argument('--source',required=True);parser.add_argument('--ledger',type=Path,default=Path('runs/observations.sqlite'))
    parser.add_argument('--observed-at',required=True,help='actual snapshot observation timestamp with timezone');parser.add_argument('--scope',default='unspecified')
    args=parser.parse_args();frame=pd.read_csv(args.csv,dtype=str).fillna('')
    if args.id_column not in frame:raise ValueError('missing identifier column')
    print(json.dumps(record_snapshot(args.ledger,args.source,((row[args.id_column],row) for row in frame.to_dict('records')),args.observed_at,args.scope),indent=2))


if __name__=='__main__':main()
