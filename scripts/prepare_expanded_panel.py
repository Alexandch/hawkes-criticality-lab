"""Prepare auditable event streams, preserving source units and timestamp ties."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from critical_processes.selected_training import sha, write_json
from prepare_code_subset import DEFAULT_CITIES
from download_expanded_data import SOURCES


def prepare_stream(frame,start,end,path,source,raw_rows,source_files):
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    frame=frame.copy()
    parsed=pd.to_datetime(frame["event_time"],errors="coerce",format="mixed")
    missing_id=frame.event_id.isna()|frame.event_id.eq("")
    invalid=parsed.isna()
    outside=parsed.notna()&((parsed<start)|(parsed>=end))
    frame=frame.loc[~missing_id&~invalid&~outside].copy()
    frame["event_time"]=parsed.loc[frame.index]
    frame=frame.sort_values("event_time",kind="stable")
    duplicates=int(frame.event_id.duplicated().sum())
    # Identifier can repeat with different offense times; earliest observed time defines an incident.
    frame=frame.drop_duplicates("event_id",keep="first")
    path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False,compression="gzip")
    dates=frame.event_time.to_numpy(dtype="datetime64[ns]")
    np.savez_compressed(path.with_suffix(".npz"),event_ns=dates.astype(np.int64),event_ids=frame.event_id.to_numpy(dtype=str))
    index=pd.date_range(start,end-pd.Timedelta(days=1),freq="D")
    daily=frame.groupby(frame.event_time.dt.normalize()).size().reindex(index,fill_value=0)
    daily.rename("events").rename_axis("date").to_csv(path.with_name(path.name.replace(".csv.gz","_daily.csv")))
    actual_months=set(frame.event_time.dt.to_period("M").astype(str))
    months=pd.period_range(start,end-pd.Timedelta(days=1),freq="M").astype(str)
    missing_months=[str(m) for m in months if m not in actual_months]
    stats=[]
    for year in range(start.year,end.year+1):
        left,right=max(start,pd.Timestamp(year,1,1)),min(end,pd.Timestamp(year+1,1,1))
        if left>=right:continue
        exposure=(right-left).days
        count=int(((frame.event_time>=left)&(frame.event_time<right)).sum())
        stats.append(dict(year=year,start=str(left.date()),end_exclusive=str(right.date()),events=count,
            exposure_days=exposure,mean_daily_rate=count/exposure,full_calendar_year=left==pd.Timestamp(year,1,1) and right==pd.Timestamp(year+1,1,1)))
    manifest=dict(source=source,start=str(start.date()),end_exclusive=str(end.date()),raw_rows=raw_rows,
        invalid_timestamps=int(invalid.sum()),missing_identifiers=int(missing_id.sum()),outside_requested_range=int(outside.sum()),
        duplicate_identifiers_removed=duplicates,events=len(frame),unique_timestamps=int(frame.event_time.nunique()),
        fraction_tied_records=float(1-frame.event_time.nunique()/len(frame)) if len(frame) else None,
        fraction_midnight=float((frame.event_time.dt.normalize()==frame.event_time).mean()),
        source_files_sha256={str(p):sha(p) for p in source_files},output_sha256=sha(path),events_npz_sha256=sha(path.with_suffix(".npz")),
        missing_calendar_months=missing_months,eligible_for_full_history=not missing_months,
        yearly=stats,unit="registered offense record, uid namespaced by source year" if source.startswith("CODE") else "case/report, earliest observed time",
        timestamp_notes="source local naive time; no UTC conversion or random jitter; equal timestamps stay equal; recorded date can be estimated")
    write_json(path.with_name(path.name+".manifest.json"),manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--raw",type=Path,default=Path("data/raw/expanded_20261110"))
    parser.add_argument("--out",type=Path,default=Path("data/processed/expanded_20261110"))
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    manifests=[]
    for city,source in SOURCES.items():
        path=args.out/f"portal_{city}.csv.gz"
        if path.with_name(path.name+".manifest.json").exists():
            saved=json.loads(path.with_name(path.name+".manifest.json").read_text(encoding="utf-8"))
            assert sha(path)==saved["output_sha256"];manifests.append(saved);continue
        parts,files=[],[]
        for year in range(2019,2027):
            raw=args.raw/"cities"/f"{city}_{year}.csv.gz";files.append(raw)
            frame=pd.read_csv(raw,dtype=str)
            stamp=frame[source["date"]]
            if city=="nyc":stamp=stamp.str[:10]+" "+frame["cmplnt_fr_tm"]
            parts.append(pd.DataFrame(dict(event_id=frame[source["id"]],event_time=stamp)))
        joined=pd.concat(parts,ignore_index=True)
        saved=prepare_stream(joined,"2019-01-01","2026-10-01",path,"portal:"+city,len(joined),files)
        manifests.append(saved);print(f"Prepared {city}: {saved['events']:,} cases/reports, {len(saved['missing_calendar_months'])} missing months",flush=True)
    code_parts={city:[] for city in DEFAULT_CITIES};code_files=[]
    completed=all((args.out/f"code_{city.lower().replace(' ','_')}.csv.gz.manifest.json").exists() for city in DEFAULT_CITIES)
    if not completed:
        for year in range(2018,2023):
            path=(args.raw if year in (2018,2019,2022) else Path("data/raw"))/f"crime_open_database_core_{year}.csv.gz"
            code_files.append(path)
            for chunk in pd.read_csv(path,usecols=["uid","city_name","date_single"],dtype=str,chunksize=250000):
                selected=chunk[chunk.city_name.isin(DEFAULT_CITIES)]
                for city,group in selected.groupby("city_name",sort=False):
                    # CODE uid is an annual row identifier, not a global case identifier.
                    code_parts[city].append(pd.DataFrame(dict(event_id=str(year)+":"+group.uid,event_time=group.date_single)))
            print(f"Read CODE/{year}",flush=True)
        for city,parts in code_parts.items():
            path=args.out/f"code_{city.lower().replace(' ','_')}.csv.gz"
            if path.with_name(path.name+".manifest.json").exists():continue
            joined=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=["event_id","event_time"])
            saved=prepare_stream(joined,"2018-01-01","2023-01-01",path,"CODE:"+city,len(joined),code_files)
            print(f"Prepared CODE/{city}: {saved['events']:,} records; {len(saved['missing_calendar_months'])} missing months",flush=True)
    for city in DEFAULT_CITIES:
        path=args.out/f"code_{city.lower().replace(' ','_')}.csv.gz.manifest.json"
        manifests.append(json.loads(path.read_text(encoding="utf-8")))
    write_json(args.out/"panel_manifest.json",dict(streams=manifests,raw_download_summary_sha256=sha(args.raw/"download_summary.json"),
        code_latest_annual_file_at_discovery=2022,city_period="2019-01-01..2026-09-30",code_period="2018-01-01..2022-12-31"))
    print(f"Prepared {len(manifests)} event streams with source units, ties and provenance",flush=True)


if __name__=="__main__":main()
