"""Read-only public downloads with row counts, retries, checksums and caching."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import shutil
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


SOURCES = {
    "chicago": dict(host="data.cityofchicago.org", dataset="ijzp-q8t2", date="date", id="case_number", columns="case_number,date"),
    "nyc": dict(host="data.cityofnewyork.us", dataset="qgea-i56i", date="cmplnt_fr_dt", id="cmplnt_num", columns="cmplnt_num,cmplnt_fr_dt,cmplnt_fr_tm"),
    "san_francisco": dict(host="data.sfgov.org", dataset="wg3w-h783", date="incident_datetime", id="incident_id", columns="incident_id,incident_datetime"),
    "seattle": dict(host="data.seattle.gov", dataset="tazs-3rd5", date="offense_date", id="report_number", columns="report_number,offense_date"),
}
OSF = "https://api.osf.io/v2/nodes/zyaqn/files/osfstorage/5b2ceceed65eaa0011d95f95/?page[size]=100"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def open_url(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "CriticalProcessesResearch/1.0", "Accept-Encoding": "identity"}), timeout=120)


def get_json(url):
    for attempt in range(4):
        try:
            with open_url(url) as response:
                return json.load(response)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def download_code(item, target):
    name, url = item["attributes"]["name"], item["links"]["download"]
    path, manifest = target/name, target/(name+".manifest.json")
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if digest(path) != saved["sha256"]:
            raise ValueError(f"cached file checksum mismatch: {path}")
        return saved
    if path.exists():
        raise FileExistsError(f"existing archive without this downloader's manifest: {path}")
    for attempt in range(4):
        try:
            partial = path.with_name(path.name+".partial")
            with open_url(url) as response, partial.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024*1024)
            with gzip.open(partial, "rt", encoding="utf-8-sig") as handle:
                columns = next(csv.reader(handle))
                if "date_single" not in columns or "city_name" not in columns:
                    raise ValueError("expected CODE core fields not present")
            partial.replace(path)
            saved = {"source": "CODE core", "url": url, "name": name, "columns": columns,
                "downloaded_utc": datetime.now(timezone.utc).isoformat(), "bytes": path.stat().st_size, "sha256": digest(path)}
            write_json(manifest, saved)
            print(f"Downloaded {name}: {saved['bytes']:,} bytes", flush=True)
            return saved
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def download_city(task):
    city, year, target, end = task
    source = SOURCES[city].copy()
    if city == "nyc" and year == 2026:
        source["dataset"] = "5uac-w243"
    start = f"{year}-01-01T00:00:00"
    stop = min(f"{year+1}-01-01T00:00:00", end+"T00:00:00")
    path = target/f"{city}_{year}.csv.gz"
    manifest = path.with_name(path.name+".manifest.json")
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if digest(path) != saved["sha256"] or saved["start"] != start or saved["end_exclusive"] != stop:
            raise ValueError(f"cached download does not match request: {path}")
        return saved
    where = f"{source['date']} >= '{start}' AND {source['date']} < '{stop}'"
    base = f"https://{source['host']}/resource/{source['dataset']}"
    count_url = base+".json?"+urllib.parse.urlencode({"$select": "count(*) as rows", "$where": where})
    url = base+".csv?"+urllib.parse.urlencode({"$select": source["columns"], "$where": where,
        "$order": f"{source['date']},{source['id']}", "$limit": 1000000})
    for attempt in range(4):
        try:
            expected = int(get_json(count_url)[0]["rows"])
            if expected >= 1000000:
                raise ValueError("year exceeds fixed download page limit")
            partial = path.with_name(path.name+".partial")
            with open_url(url) as response, gzip.open(partial, "wb", compresslevel=6) as handle:
                shutil.copyfileobj(response, handle, length=1024*1024)
            with gzip.open(partial, "rt", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                columns = next(reader)
                actual = sum(1 for _ in reader)
            after = int(get_json(count_url)[0]["rows"])
            if actual != expected or after != expected:
                raise ValueError(f"live source changed/count mismatch: before={expected}, rows={actual}, after={after}")
            partial.replace(path)
            saved = {"city": city, **source, "url": url, "count_url": count_url, "columns_returned": columns,
                "start": start, "end_exclusive": stop, "rows": actual, "count_before": expected, "count_after": after,
                "downloaded_utc": datetime.now(timezone.utc).isoformat(), "sha256": digest(path), "bytes": path.stat().st_size,
                "notes": "registered incidents; one case may have multiple offense rows; local timestamps; deduplicate by case/report identifier during preparation"}
            write_json(manifest, saved)
            print(f"Downloaded {city}/{year}: {actual:,} rows", flush=True)
            return saved
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/raw/expanded_20261110"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    discovery = get_json(OSF)
    write_json(args.out/"code_discovery.json", {"retrieved_utc": datetime.now(timezone.utc).isoformat(), "url": OSF, "response": discovery})
    core = {entry["attributes"]["name"]: entry for entry in discovery["data"] if entry["attributes"]["name"].startswith("crime_open_database_core_")}
    (args.out/"cities").mkdir(exist_ok=True)
    tasks = [(city, year, args.out/"cities", "2026-10-01") for city in SOURCES for year in range(2019, 2027)]
    records, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(download_city, task): f"{task[0]}/{task[1]}" for task in tasks}
        for year in (2018, 2019, 2022):
            item = core[f"crime_open_database_core_{year}.csv.gz"]
            pending[pool.submit(download_code, item, args.out)] = f"CODE/{year}"
        for future in as_completed(pending):
            try:
                records.append(future.result())
            except Exception as error:
                errors.append({"task": pending[future], "error": str(error)})
                print(f"FAILED {pending[future]}: {error}", flush=True)
    write_json(args.out/"download_summary.json", {"records": records, "errors": errors})
    if errors:
        raise RuntimeError(f"{len(errors)} downloads failed; successful cached files are retained for retry")
    print(f"Complete: {len(records)} downloads, {sum(v.get('rows',0) for v in records):,} city offense rows", flush=True)


if __name__ == "__main__":
    main()
