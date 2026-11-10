from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None


def _staged_hawkes_counts_impl(baselines, ratios, stage_days, decay, burn_in, seed, max_events, record_events):
    np.random.seed(seed)
    counts = np.zeros(len(ratios) * stage_days, dtype=np.int64)
    events = [0.0]
    time, excitation, event_count = 0.0, 0.0, 0
    # The first segment warms up the initial regime. Excitation is retained at
    # every boundary; only the baseline and future offspring amplitudes change.
    for segment in range(-1, len(ratios)):
        parameter_index = max(segment, 0)
        end = burn_in if segment < 0 else burn_in + (segment + 1) * stage_days
        baseline, ratio = baselines[parameter_index], ratios[parameter_index]
        while time < end:
            upper = baseline + excitation
            candidate = time + np.random.exponential(1.0 / upper)
            if candidate >= end:
                excitation *= np.exp(-decay * (end - time))
                time = end
                break
            excitation *= np.exp(-decay * (candidate - time))
            time = candidate
            if np.random.random() * upper <= baseline + excitation:
                event_count += 1
                if event_count > max_events:
                    raise RuntimeError("staged Hawkes event limit reached")
                if segment >= 0:
                    day = min(int(time - burn_in), len(counts) - 1)
                    counts[day] += 1
                    if record_events:
                        events.append(time - burn_in)
                excitation += ratio * decay
    return counts, np.asarray(events[1:])


_staged_hawkes_counts = njit(cache=True)(_staged_hawkes_counts_impl) if njit is not None else None


def simulate_staged_hawkes_counts(
    baselines: np.ndarray,
    branching_ratios: np.ndarray,
    stage_days: int,
    decay: float,
    seed: int,
    burn_in: float = 0.0,
    max_events: int = 10_000_000,
    record_events: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Ogata simulation with continuous excitation and parent-time kernel amplitudes.

    An event born in stage k contributes n[k]*decay*exp(-decay*age), including
    after later boundaries. Thus n[k] is its expected direct offspring count.
    """
    baselines = np.asarray(baselines, dtype=float)
    ratios = np.asarray(branching_ratios, dtype=float)
    if baselines.ndim != 1 or ratios.shape != baselines.shape or len(ratios) == 0:
        raise ValueError("baselines and branching_ratios must be nonempty matching vectors")
    if not np.isfinite(baselines).all() or not np.isfinite(ratios).all() or (baselines <= 0).any() or ((ratios < 0) | (ratios >= 1)).any():
        raise ValueError("baselines must be positive and finite, branching ratios in [0, 1)")
    if stage_days < 1 or int(stage_days) != stage_days or not np.isfinite(decay) or decay <= 0 or not np.isfinite(burn_in) or burn_in < 0:
        raise ValueError("stage_days must be a positive integer, decay positive, burn_in nonnegative")
    if not 0 <= seed <= 2**32 - 1 or max_events < 1:
        raise ValueError("seed must fit uint32 and max_events must be positive")
    if _staged_hawkes_counts is None:
        raise RuntimeError("staged Hawkes simulation requires: pip install -e '.[fast]'")
    counts, events = _staged_hawkes_counts(baselines, ratios, int(stage_days), decay, burn_in, seed, max_events, record_events)
    return (counts, events) if record_events else counts


def simulate_poisson(rate: float, horizon: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate a homogeneous Poisson process by exponential waiting times."""
    if rate <= 0 or horizon <= 0:
        raise ValueError("rate and horizon must be positive")
    events: list[float] = []
    time = 0.0
    while True:
        time += float(rng.exponential(1.0 / rate))
        if time >= horizon:
            break
        events.append(time)
    return np.asarray(events, dtype=float)


def simulate_hawkes(
    baseline: float,
    branching_ratio: float,
    decay: float,
    horizon: float,
    rng: np.random.Generator,
    max_events: int = 1_000_000,
    burn_in: float = 0.0,
) -> np.ndarray:
    """Ogata thinning for a stationary exponential Hawkes process (branching < 1)."""
    if baseline <= 0 or decay <= 0 or horizon <= 0 or burn_in < 0:
        raise ValueError("baseline, decay, and horizon must be positive; burn_in cannot be negative")
    if not 0 <= branching_ratio < 1:
        raise ValueError("branching_ratio must be in [0, 1)")

    events: list[float] = []
    time = 0.0
    last_time = 0.0
    excitation = 0.0
    total_horizon = horizon + burn_in
    while time < total_horizon:
        intensity_upper = baseline + excitation
        time += float(rng.exponential(1.0 / intensity_upper))
        if time >= total_horizon:
            break
        excitation *= np.exp(-decay * (time - last_time))
        intensity = baseline + excitation
        if rng.random() * intensity_upper <= intensity:
            events.append(time)
            excitation += branching_ratio * decay
            if len(events) >= max_events:
                raise RuntimeError("event limit reached; reduce branching ratio or horizon")
        last_time = time
    retained = np.asarray(events, dtype=float)
    return retained[retained >= burn_in] - burn_in
