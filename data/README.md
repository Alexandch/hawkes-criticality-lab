# CODE data

## Offline readiness R11 — 2026-11-10

Separate [October manifests](raw/offline_20261110/) extend the unchanged R10 history by 12,913 Chicago rows and 2,773 SF rows. After validation there are 1,189,263 Chicago cases and 426,309 SF reports in [the category panels](processed/offline_20261110/). Historical case IDs repeated in the October query cannot rewrite their earlier group or report date. Exact R10 historical total grids are retained.

The NPZ adds three groups: `all,property,assault_robbery`. A selected case with any assault/robbery category belongs to that group; otherwise it belongs to property. The two exclusive groups sum to `all`. This is a technical partition of selected categories, not all crime or a legal taxonomy. Per-case `event_bins,area_indices,group_indices` and SF `report_bins` support chronological training labels. October labels remain preliminary. Files are external to Git, checksums and scope are retained.

A separate actual observation journal starts on November 10. [Chicago source-row snapshot](raw/public_observed_20261110/chicago/chicago.csv.gz.manifest.json) contains 4,304 rows filtered by source `updated_on`; [SF source-row snapshot](raw/public_observed_20261110/san_francisco/san_francisco.csv.gz.manifest.json) contains 568 rows filtered by `report_datetime`. Unique `id` / `row_id` keys avoid conflating different offenses in the same case. The SQLite ledger stores actual first/last observation times and payload hashes. Partial-window absence is not deletion; SF report-date windows cannot detect every later edit. No historical publication dates are invented. [Instructions](../report/OFFLINE_GUIDE.md).

## District × 6-hour snapshot R10 — 2026-11-10

This additional task uses separate portal extracts with district fields: [raw manifests](raw/spatiotemporal_20261110/), [Chicago metadata](processed/spatiotemporal_20261110/chicago.json), [San Francisco metadata](processed/spatiotemporal_20261110/san_francisco.json). The 16 CSV.GZ files cover 2019–2025 and January–September 2026. They contain 1,621,231 selected offense rows, reduced to 1,176,356 Chicago cases and 423,703 SF reports after validation and case deduplication. The grid has 11,320 six-hour intervals, 22 Chicago police districts and 10 SF police districts.

Chicago fields: `case_number,date,district,primary_type`; categories ASSAULT, BATTERY, BURGLARY, ROBBERY, THEFT, MOTOR VEHICLE THEFT. SF fields: `incident_id,incident_datetime,report_datetime,police_district,incident_category`; categories Assault, Burglary, Robbery, Larceny Theft, Motor Vehicle Theft. These are selected categories, not all recorded crime; city definitions are not identical. Unknown districts, invalid dates and conflicting district case IDs are explicitly counted and excluded. All district-months must be present. Local source timestamps are binned without jitter; DST repeated hours cannot be reconstructed from naive timestamps.

Prepared NPZ arrays: `counts` (time × district), `time_ns` (local naive bin start), `districts`, `centroids`. SF also has per-case `event_bins,report_bins,area_indices` for report-aware histories and labels. No addresses or individual event coordinates are stored. Centroids are server-side mean coordinates of training events in 2019–2023; SF additionally restricts reports known before 2024. They define a coarse similarity graph, not boundary adjacency.

SF report delays exceed six hours for 54.974% and seven days for 11.446% of included reports. Chicago lacks initial registration timestamps. A seven-day information lag and report-aware SF history partially address availability; neither reconstructs all historical record revisions. This is a retrospective evaluation, not a live operational guarantee. [Protocol and splits](../report/SPATIOTEMPORAL_PROTOCOL.md), [metrics](../results/spatiotemporal_forecast/test_metrics.csv).

Large CSV.GZ/NPZ files are external to Git. For a new download use `python scripts/prepare_spatiotemporal_data.py --raw data/raw/spatial_local --out data/processed/spatial_local`; preserve archived manifests. New source vintages may differ.

## Final expanded snapshot — 2026-11-10

The OSF file list was checked again and now includes `core_2022`. Additional CODE core files for 2018, 2019 and 2022 are stored in `raw/expanded_20261110/`, alongside discovery JSON, download URLs and SHA-256 manifests. Existing 2020/2021 archives were preserved. The prepared 2018–2022 panel contains 9,543,104 offense records in 17 cities. Annual filenames do not guarantee complete coverage: Chicago lacks 2022 Q4; Houston and Minneapolis lack 2018. Those three streams are excluded from the fixed long-history neural application, leaving 14 eligible CODE cities.

The direct portals for Chicago, NYC, San Francisco and Seattle were downloaded for 2019–2025 and January–September 2026. The common exclusive end is 2026-10-01 because NYC's current YTD source ends in September. These are not full-year 2026 figures or an updated CODE release. There are 7,395,987 raw city rows and 7,146,403 prepared cases/reports. Source time and incident/report identifiers only were downloaded, without locations or personal attributes.

Files:

- [Raw expanded downloads](raw/expanded_20261110/) — 32 city-year CSV.GZ files plus 3 CODE archives, discovery and manifests.
- [Prepared event streams](processed/expanded_20261110/) — `portal_*.csv.gz`, `code_*.csv.gz`, daily counts, event NPZ caches and manifests.
- [Panel manifest](processed/expanded_20261110/panel_manifest.json) — coverage, parsing exclusions, units, annual counts and checksums.
- [Application results](../results/empirical_transfer/analysis.md) — frozen R4/R9 scores in 22 blocks, no invented criticality labels.

CODE `uid` is an annual record identifier; the prepared identifier is `source_year:uid`. Portal IDs are deduplicated across years, retaining the earliest observed timestamp. Equal timestamps remain equal and source-local naive time is preserved. CODE offenses and portal cases/reports are different units; category definitions are not harmonized for cross-source crime-risk comparisons.

```powershell
.\.venv\Scripts\python.exe scripts/download_expanded_data.py
.\.venv\Scripts\python.exe scripts/prepare_expanded_panel.py
.\.venv\Scripts\python.exe -m critical_processes.empirical_transfer
.\.venv\Scripts\python.exe scripts/verify_empirical_transfer.py
```

Completed downloads and analysis use cached snapshots; source URL/count/hash manifests permit audit without silently replacing prior files. The original 2020–2021 / 2024–2025 pilots below are retained as historical provenance.

## Original CODE snapshot

Source: Crime Open Database (CODE), annual `core` CSV.GZ files from the [OSF project](https://osf.io/zyaqn/). CODE is released under CC BY 4.0; cite the database and acknowledge its source in any report using these data.

Downloaded files:

- [`raw/crime_open_database_core_2020.csv.gz`](https://osf.io/ng49y/download)
- [`raw/crime_open_database_core_2021.csv.gz`](https://osf.io/gmkz4/download)

Downloaded on 2026-10-12. These are the latest two annual CODE core files listed by the project at download time. The harmonized panel `processed/code_core_2020_2021_17cities.csv.gz` contains **3,753,014 records** across **17 cities** present in both years. St Louis and Tucson were left out because the 2021 file has no records for them. It adds a `source_year` column and records city-year row counts in `processed/code_core_2020_2021_17cities.csv.manifest.json`.

Rebuild the panel with:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_code_subset.py --input data\raw\crime_open_database_core_2020.csv.gz data\raw\crime_open_database_core_2021.csv.gz --out data\processed\code_core_2020_2021_17cities.csv.gz
```

The two-year CODE panel supports empirical application and transfer checks; it has no known Hawkes branching-ratio labels and therefore is not supervised training data for the neural detector.

CODE records police-reported offenses. `date_single` can represent an estimated date/time, so continuous-time Hawkes estimates must be interpreted in light of the timestamp precision and reporting process.

## Chicago pilot (2024-2025)

Source: City of Chicago Data Portal, [Crimes - 2001 to Present](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2), downloaded through the Socrata API with a `year between 2024 and 2025` filter on 2026-10-12. The raw export is `raw/chicago_crimes_2024_2025.csv` (497,788 rows; 2024-01-01 through 2025-12-31). The `date` field contains timestamps, mostly to minute precision. There are 497,737 unique case numbers; use `--id-column case_number` to count incidents rather than offense/victim rows. The likelihood groups events with identical timestamps so their order within a recorded minute does not create instantaneous excitation.

Chicago's portal notes that records are reported incidents, may be preliminary, and may be revised. The dataset also represents reported crime, not all crime that occurred. These data are an empirical application, not ground truth for criticality. The dataset's documented exception for murders (one row per victim) is why incident-level deduplication should be enabled.

Analyze the two complete years with:

```powershell
.\.venv\Scripts\critical-processes.exe analyze-code data\raw\chicago_crimes_2024_2025.csv --date-column date --id-column case_number --start 2024-01-01 --end 2026-01-01 --split-date 2025-01-01 --out results\chicago_2024_2025
```

Run category and quarterly robustness checks with:

```powershell
.\.venv\Scripts\critical-processes.exe robustness-code data\raw\chicago_crimes_2024_2025.csv --date-column date --category-column primary_type --id-column case_number --start 2024-01-01 --end 2026-01-01 --split-date 2025-01-01 --out results\chicago_robustness
```
