"""Initialize first-seen evidence only from the actual new October snapshots."""
import json
from pathlib import Path
import pandas as pd
from critical_processes.observation_ledger import record_snapshot
from critical_processes.selected_training import write_json


def main():
    raw=Path('data/raw/offline_20261110');out=Path('data/processed/offline_20261110');records=[]
    for city,identifier in [('chicago','case_number'),('san_francisco','incident_id')]:
        path=raw/(city+'_202610.csv.gz');meta=json.loads(path.with_name(path.name+'.manifest.json').read_text(encoding='utf-8'))
        frame=pd.read_csv(path,dtype=str).fillna('')
        result=record_snapshot(out/'observations.sqlite',city,((row[identifier],row) for row in frame.to_dict('records')),meta['retrieved_utc'],'incident_date_2026_10')
        records.append(dict(city=city,source_sha256=meta['sha256'],**result))
    write_json(out/'observation_ledger_summary.json',dict(snapshots=records,note='Only October-query scope is observed. No historic first-seen times are invented. No claims about deletion outside this scope.'))
    print('Initialized actual first-seen ledger for both new source snapshots.')


if __name__=='__main__':main()
