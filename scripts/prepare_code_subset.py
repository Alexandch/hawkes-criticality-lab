from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


DEFAULT_CITIES = (
    "Austin", "Boston", "Charlotte", "Chicago", "Colorado Springs", "Detroit",
    "Houston", "Kansas City", "Los Angeles", "Louisville", "Memphis", "Mesa",
    "Minneapolis", "Nashville", "New York", "San Francisco", "Seattle",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter annual CODE core CSV.GZ files to a comparable city panel")
    parser.add_argument("--input", nargs="+", required=True, help="annual CODE core CSV.GZ files")
    parser.add_argument("--out", required=True, help="output CSV.GZ path")
    parser.add_argument("--cities", nargs="+", default=list(DEFAULT_CITIES))
    args = parser.parse_args()

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {output}")

    requested = {name.casefold(): name for name in args.cities}
    counts: Counter[tuple[int, str]] = Counter()
    wrote_header = False
    total_rows = 0
    try:
        for source in sorted(map(Path, args.input)):
            match = re.search(r"20\d{2}", source.name)
            if match is None:
                raise ValueError(f"cannot infer year from input filename: {source.name}")
            year = int(match.group())
            for chunk in pd.read_csv(source, compression="gzip", chunksize=250_000, low_memory=False):
                if "city_name" not in chunk or "date_single" not in chunk:
                    raise ValueError(f"expected CODE core columns missing in {source.name}")
                city_keys = chunk["city_name"].astype("string").str.casefold()
                selected = chunk.loc[city_keys.isin(requested)].copy()
                if selected.empty:
                    continue
                selected.insert(0, "source_year", year)
                for city, count in selected["city_name"].value_counts().items():
                    counts[(year, str(city))] += int(count)
                selected.to_csv(
                    output,
                    index=False,
                    mode="a" if wrote_header else "w",
                    header=not wrote_header,
                    compression="gzip",
                )
                wrote_header = True
                total_rows += len(selected)
        if not wrote_header:
            raise ValueError("no rows matched the requested cities")
    except Exception:
        output.unlink(missing_ok=True)
        raise

    manifest = {
        "source": "Crime Open Database (CODE), core annual files",
        "source_url": "https://osf.io/zyaqn/",
        "years": sorted({year for year, _ in counts}),
        "cities_requested": list(args.cities),
        "cities_present": sorted({city for _, city in counts}),
        "rows_total": total_rows,
        "rows_by_year_city": [
            {"year": year, "city": city, "rows": count}
            for (year, city), count in sorted(counts.items())
        ],
        "output": str(output),
        "date_field": "date_single",
        "city_field": "city_name",
        "notes": [
            "CODE contains police-recorded offenses and may have city- and year-specific coverage changes.",
            "date_single may be an estimated date/time rather than the exact event time.",
            "The added source_year field records the annual source file used.",
        ],
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "rows": total_rows, "cities": manifest["cities_present"], "years": manifest["years"]}, indent=2))


if __name__ == "__main__":
    main()
