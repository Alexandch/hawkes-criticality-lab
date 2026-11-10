from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from .background_models import fit_background_forecast
from .detector import transform_detector_features
from .fast_window_mle import fit_fast_window
from .indicators import summarize_counts
from .negative_controls import poisson_control_rates, stage_features_from_counts
from .simulation import simulate_staged_hawkes_counts
from .task_audit import AUDIT_FAMILIES
from .window_features import prepare_hawkes_history


FEATURE_GROUPS = {"counts": 16, "constant_mle": 39, "calendar_mle": 64}


def selection_feature_names():
    counts = [f"lag{lag}_{name}" for lag in (2, 1, 0) for name in ("log_rate", "fano", "acf1", "cluster_per_count")]
    counts += [f"rolling_{name}" for name in ("log_rate", "fano", "acf1", "cluster_per_count")]
    fits = []
    for variant in ("constant", "calendar"):
        fits += [f"{variant}_{block}_{name}" for block in ("lag2", "lag1", "lag0", "rolling")
                 for name in ("n", "curvature_se", "baseline_fraction", "gain", "valid")]
        fits += [f"{variant}_{name}" for name in ("slope", "projected", "rolling_minus_recent")]
    return counts + fits + ["forecast_shape_cv", "forecast_valid"]


@dataclass
class SelectionBatch:
    ids: np.ndarray
    families: np.ndarray
    rates: np.ndarray
    ratios: np.ndarray
    counts: np.ndarray
    matrix: np.ndarray
    fits: np.ndarray
    parameter_seeds: np.ndarray
    simulator_seeds: np.ndarray


def selection_tasks(seed, split, size, config, reserved=()):
    rng, used, tasks = np.random.default_rng(seed), set(int(s) for s in reserved), []
    def next_seed():
        value = int(rng.integers(0, 2**32, dtype=np.uint64))
        while value in used:
            value = int(rng.integers(0, 2**32, dtype=np.uint64))
        used.add(value)
        return value
    for family in AUDIT_FAMILIES:
        for replicate in range(size):
            tasks.append((f"{split}/{family}/{replicate}", family, next_seed(), next_seed(), config))
    return tasks


def simulate_selection_trajectory(task):
    _, family, parameter_seed, simulator_seed, config = task
    rng, noise = np.random.default_rng(parameter_seed), np.random.default_rng(simulator_seed)
    stages, width, prefix, critical = [config[key] for key in ("stages", "stage_days", "prefix_days", "critical_level")]
    rate = float(np.exp(rng.uniform(np.log(config["rate_min"]), np.log(config["rate_max"]))))
    anchors = [v for v in (2., 20., 50.) if config["rate_min"] <= v <= config["rate_max"]]
    if anchors and rng.random() < .3:
        rate = float(rng.choice(anchors))
    if family in ("hawkes_rising", "hawkes_abrupt"):
        crossing = int(rng.integers(config["lookback"] + config["forecast_stages"] + 1, stages - 1))
        initial = rng.uniform(.15, .5)
        before = np.linspace(initial, rng.uniform(critical - .10, critical - .005), crossing) if family == "hawkes_rising" else np.full(crossing, initial)
        after = np.linspace(rng.uniform(critical + .005, min(critical + .04, .985)), .985, stages - crossing)
        ratios = np.r_[before, after]
    elif family == "hawkes_stable":
        ratios = np.full(stages, rng.uniform(.05, critical - .01))
    elif family == "hawkes_subthreshold":
        ratios = np.linspace(rng.uniform(.15, .5), rng.uniform(critical - .15, critical - .005), stages)
    elif family == "hawkes_high":
        ratios = np.full(stages, rng.uniform(critical + .01, .98))
    elif family == "hawkes_near_threshold":
        ratios = np.full(stages, rng.uniform(critical - .04, critical - .001))
    elif family == "hawkes_falling":
        change = int(rng.integers(config["lookback"] + 1, stages - 2))
        ratios = np.r_[np.full(change, rng.uniform(critical + .01, .98)), np.linspace(critical - .04, .4, stages - change)]
    else:
        ratios = np.zeros(stages)
    horizon = prefix + stages * width
    if family.startswith("hawkes_"):
        daily_n = np.r_[np.full(prefix, ratios[0]), np.repeat(ratios, width)]
        counts, events = simulate_staged_hawkes_counts(rate * (1 - daily_n), daily_n, 1, 1., simulator_seed,
            burn_in=8 / (1 - ratios[0]), record_events=True)
    else:
        if family in ("step_up", "step_down"):
            change = prefix + int(rng.integers(width * config["lookback"], width * (stages - 1)))
            weights = np.where(np.arange(horizon) < change, .5, 2.)
            if family == "step_down":
                weights = 2.5 - weights
            background = rate * weights / weights.mean()
        else:
            background = poisson_control_rates(family, rate, horizon, rng)
        counts = noise.poisson(background)
        days = np.repeat(np.arange(horizon), counts)
        events = np.sort(days + noise.random(len(days)))
    return counts, events, ratios, rate


def build_selection_features(counts, events, windows, prefix_days=730, stage_days=90, lookback=3, forecast_stages=2):
    """Encode observations only; the calendar shape is frozen before regime monitoring."""
    counts, events = np.asarray(counts), np.asarray(events)
    if lookback != 3 or len(counts) <= prefix_days or (len(counts) - prefix_days) % stage_days:
        raise ValueError("need three lag blocks and complete monitored stages after the prefix")
    if min(windows) < 3 or max(windows) > prefix_days + stage_days:
        raise ValueError("windows must fit the observed history at the first stage")
    if not np.array_equal(np.bincount(np.floor(events).astype(int), minlength=len(counts)), counts):
        raise ValueError("event times must agree with counts")
    forecast, forecast_valid = fit_background_forecast(counts[:prefix_days], len(counts))
    history = prepare_hawkes_history(events, 1.)
    monitored = counts[prefix_days:]
    stage_raw = stage_features_from_counts(monitored, stage_days)
    stages, cache = len(stage_raw), {}
    def fit(start, end, variant):
        key = (start, end, variant)
        if key not in cache:
            cache[key] = fit_fast_window(history, start, end, forecast if variant else None)
        return cache[key]
    short = np.stack([[fit(prefix_days + i * stage_days, prefix_days + (i + 1) * stage_days, variant)
                       for i in range(stages)] for variant in (0, 1)])
    matrix, rolling = [], []
    centered = np.array([-1., 0., 1.])
    for window in windows:
        rows, diagnostics = [], []
        for i in range(lookback - 1, stages):
            end = prefix_days + (i + 1) * stage_days
            base = np.r_[stage_raw[i-2:i+1].ravel(), summarize_counts(counts[end-window:end], 1.)]
            row = list(transform_detector_features(base[None], version=2)[0])
            fit_pair = []
            for variant in (0, 1):
                recent = short[variant, i-2:i+1]
                current = fit(end-window, end, variant)
                slope = float(np.dot(centered, recent[:, 0]) / 2)
                row.extend(np.r_[recent.ravel(), current, slope, recent[-1, 0] + forecast_stages * slope, current[0] - recent[-1, 0]])
                fit_pair.append(np.r_[current, current[0] + forecast_stages * slope, recent[:, 4].min()])
            profile = forecast[end-window:end]
            cv = float(profile.std() / profile.mean()) if forecast_valid else 0.
            row.extend([cv, float(forecast_valid)])
            rows.append(row)
            diagnostics.append(fit_pair)
        matrix.append(rows)
        rolling.append(diagnostics)
    return np.asarray(matrix), np.asarray(rolling)


def _selection_record(task):
    counts, events, ratios, rate = simulate_selection_trajectory(task)
    config = task[-1]
    matrix, fits = build_selection_features(counts, events, config["windows"], config["prefix_days"],
        config["stage_days"], config["lookback"], config["forecast_stages"])
    return counts.astype(np.int32), ratios, rate, matrix, fits


def generate_selection_batch(tasks, workers=2):
    records = []
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        iterator = pool.map(_selection_record, tasks, chunksize=4) if pool else map(_selection_record, tasks)
        for i, record in enumerate(iterator, 1):
            records.append(record)
            if i % 150 == 0 or i == len(tasks):
                print(f"Selected training {tasks[0][0].split('/')[0]}: {i}/{len(tasks)} trajectories", flush=True)
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
    counts, ratios, rates, matrix, fits = zip(*records)
    return SelectionBatch(np.asarray([t[0] for t in tasks]), np.asarray([t[1] for t in tasks]), np.asarray(rates),
        np.stack(ratios), np.stack(counts), np.stack(matrix), np.stack(fits),
        np.asarray([t[2] for t in tasks]), np.asarray([t[3] for t in tasks]))


def save_selection_batch(batch, path):
    np.savez_compressed(path, **batch.__dict__)


def load_selection_batch(path):
    with np.load(path) as data:
        return SelectionBatch(**{key: data[key].copy() for key in SelectionBatch.__dataclass_fields__})
