from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize the downloaded NYPD extract for project analysis.")
    parser.add_argument("--input", default="data/raw/nypd_complaints_2024_2025_verified.csv")
    parser.add_argument("--output", default="data/processed/nypd_2024_2025_verified.csv")
    parser.add_argument("--report", default="data/processed/nypd_2024_2025_verified_quality.json")
    args = parser.parse_args()

    source = pd.read_csv(args.input, dtype=str)
    required = {"cmplnt_num", "cmplnt_fr_dt", "cmplnt_fr_tm", "ofns_desc"}
    missing_columns = required - set(source.columns)
    if missing_columns:
        raise ValueError(f"Input is missing columns: {sorted(missing_columns)}")

    duplicate_exact_rows = int(source.duplicated().sum())
    source = source.drop_duplicates()
    timestamp = pd.to_datetime(
        source["cmplnt_fr_dt"].str.slice(0, 10) + " " + source["cmplnt_fr_tm"],
        errors="coerce",
    )
    quality = {
        "source_rows": int(len(source)),
        "missing_or_invalid_date_time": int(timestamp.isna().sum()),
        "missing_complaint_id": int(source["cmplnt_num"].isna().sum()),
        "duplicate_exact_rows": duplicate_exact_rows,
        "duplicate_complaint_id_rows_after_exact_dedup": int(source["cmplnt_num"].duplicated(keep=False).sum()),
        "timestamps_have_minute_precision": bool(source["cmplnt_fr_tm"].str.endswith(":00").all()),
        "year_counts": {
            str(year): int((timestamp.dt.year == year).sum()) for year in (2024, 2025)
        },
    }

    normalized = pd.DataFrame(
        {
            "date": timestamp,
            "category": source["ofns_desc"].str.strip().str.upper(),
        }
    )
    normalized = normalized.loc[timestamp.notna()].sort_values("date")
    quality["normalized_events"] = int(len(normalized))
    quality["category_count"] = int(normalized["category"].nunique())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output, index=False, date_format="%Y-%m-%d %H:%M:%S")
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(quality, ensure_ascii=False, indent=2))
    print(f"Saved normalized events to {output.resolve()}")


if __name__ == "__main__":
    main()
