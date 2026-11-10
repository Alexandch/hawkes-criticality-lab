from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_event_csv(
    path: str | Path,
    date_column: str,
    city_column: str | None = None,
    city: str | None = None,
    start: str | None = None,
    end: str | None = None,
    id_column: str | None = None,
    category_column: str | None = None,
    category: str | None = None,
) -> tuple[np.ndarray, float, pd.Series]:
    """Read CODE-like incident CSV, returning event times in days and daily counts."""
    columns = {date_column, city_column, id_column, category_column} - {None}
    frame = pd.read_csv(path, usecols=lambda name: name in columns)
    if date_column not in frame:
        raise ValueError(f"date column {date_column!r} not found")
    if id_column is not None and id_column not in frame:
        raise ValueError(f"ID column {id_column!r} not found")
    if city is not None:
        if city_column is None or city_column not in frame:
            raise ValueError("--city requires a valid --city-column")
        frame = frame[frame[city_column].astype(str).str.casefold() == city.casefold()]
    if category is not None:
        if category_column is None or category_column not in frame:
            raise ValueError("--category requires a valid --category-column")
        frame = frame[frame[category_column].astype(str).str.casefold() == category.casefold()]
    if id_column is not None:
        identified = frame[id_column].notna()
        frame = pd.concat(
            [frame.loc[identified].drop_duplicates(id_column), frame.loc[~identified]],
            ignore_index=True,
        )
    dates = pd.to_datetime(frame[date_column], errors="coerce", utc=True).dropna().dt.tz_convert(None)
    if start:
        dates = dates[dates >= pd.Timestamp(start)]
    if end:
        dates = dates[dates < pd.Timestamp(end)]
    dates = dates.sort_values().reset_index(drop=True)
    if dates.empty:
        raise ValueError("no valid incident dates remain after filtering")

    left = pd.Timestamp(start) if start else dates.iloc[0].normalize()
    right = pd.Timestamp(end) if end else dates.iloc[-1].normalize() + pd.Timedelta(days=1)
    horizon = (right - left).total_seconds() / 86400.0
    if horizon <= 0:
        raise ValueError("end must be later than start")
    seconds = (dates - left).dt.total_seconds().to_numpy()
    event_times = seconds / 86400.0

    # Date-only source files contain ties. Uniform within-day jitter enables continuous-time likelihood fitting.
    has_intraday = bool(((dates - dates.dt.normalize()).dt.total_seconds() != 0).any())
    if not has_intraday:
        rng = np.random.default_rng(731)
        event_times += rng.uniform(0.0, 1.0, size=len(event_times))
    event_times = np.sort(event_times[(event_times >= 0) & (event_times < horizon)])
    daily = pd.Series(1, index=dates.dt.normalize()).groupby(level=0).sum()
    daily_start = left.normalize()
    daily_end = (right - pd.Timedelta(nanoseconds=1)).normalize()
    daily = daily.reindex(pd.date_range(daily_start, daily_end, freq="D"), fill_value=0)
    daily.name = "incident_count"
    daily.attrs["has_intraday_timestamps"] = has_intraday
    return event_times, float(horizon), daily
