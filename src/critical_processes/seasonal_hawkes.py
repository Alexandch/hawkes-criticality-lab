from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize

from .baselines import calendar_design, fit_calendar_poisson

try:
    from numba import njit
except ImportError:
    njit = None


def _negative_log_likelihood_and_gradient_impl(
    parameters, event_times, event_counts, event_days, daily_design, daily_exposure, score_start, score_end
):
    feature_count = daily_design.shape[1]
    branching_ratio = parameters[feature_count]
    decay = np.exp(parameters[feature_count + 1])
    daily_rates = np.exp(np.clip(daily_design @ parameters[:feature_count], -30.0, 30.0))
    gradient = np.zeros(feature_count + 2, dtype=np.float64)

    log_likelihood = 0.0
    for day in range(len(daily_rates)):
        exposure = daily_exposure[day]
        if exposure != 0.0:
            log_likelihood -= daily_rates[day] * exposure
            for feature in range(feature_count):
                gradient[feature] -= daily_rates[day] * exposure * daily_design[day, feature]

    memory = 0.0
    memory_derivative = 0.0
    previous_time = 0.0
    compensator_n_derivative = 0.0
    compensator_decay_derivative = 0.0

    for i in range(len(event_times)):
        time = event_times[i]
        count = event_counts[i]
        gap = time - previous_time
        decay_factor = np.exp(-decay * gap)
        prior_memory = memory
        prior_memory_derivative = memory_derivative
        memory = decay_factor * prior_memory
        memory_derivative = decay_factor * (prior_memory_derivative - gap * prior_memory)

        if score_start <= time < score_end:
            day = event_days[i]
            baseline = daily_rates[day]
            intensity = baseline + branching_ratio * decay * memory
            log_likelihood += count * np.log(intensity)
            baseline_weight = count * baseline / intensity
            for feature in range(feature_count):
                gradient[feature] += baseline_weight * daily_design[day, feature]
            gradient[feature_count] += count * decay * memory / intensity
            gradient[feature_count + 1] += count * branching_ratio * (memory + decay * memory_derivative) / intensity

        if time < score_end:
            lower_gap = max(score_start - time, 0.0)
            upper_gap = score_end - time
            lower_tail = np.exp(-decay * lower_gap)
            upper_tail = np.exp(-decay * upper_gap)
            excitation_integral = lower_tail - upper_tail
            log_likelihood -= branching_ratio * count * excitation_integral
            compensator_n_derivative += count * excitation_integral
            compensator_decay_derivative += branching_ratio * count * (
                -lower_gap * lower_tail + upper_gap * upper_tail
            )

        memory += count
        previous_time = time

    gradient[feature_count] -= compensator_n_derivative
    gradient[feature_count + 1] -= compensator_decay_derivative
    gradient[feature_count + 1] *= decay
    return -log_likelihood, -gradient


if njit is not None:
    _negative_log_likelihood_and_gradient = njit(cache=True)(_negative_log_likelihood_and_gradient_impl)
else:
    def _negative_log_likelihood_and_gradient(*args):
        raise RuntimeError("seasonal Hawkes fitting requires the fast extra: pip install -e '.[fast]'")


def seasonal_hawkes_backtest(
    events: np.ndarray,
    daily_counts: pd.Series,
    split_date: str,
    quarterly: bool = False,
    hour_of_day: bool = False,
) -> dict[str, object]:
    """Fit a calendar-baseline Hawkes model and score a later conditional holdout.

    With hour_of_day, the baseline is piecewise constant on hourly bins. All
    times, rates and compensators still use days as their unit.
    """
    dates = daily_counts.index
    if len(dates) == 0 or not dates.equals(pd.date_range(dates[0].normalize(), periods=len(dates), freq="D")):
        raise ValueError("daily_counts must have a contiguous midnight-aligned daily index")
    if hour_of_day and daily_counts.attrs.get("has_intraday_timestamps") is False:
        raise ValueError("hour-of-day fitting requires observed intraday timestamps, not date-only jitter")
    split = pd.Timestamp(split_date)
    if split != split.normalize():
        raise ValueError("split_date must be aligned to midnight")
    train_mask = dates < split
    test_mask = dates >= split
    if not train_mask.any() or not test_mask.any():
        raise ValueError("split_date must leave at least one training day and one test day")

    split_day = float((split - dates[0]).total_seconds() / 86400.0)
    end_day = float(len(dates))
    all_events = np.sort(np.asarray(events, dtype=float))
    if not np.isfinite(all_events).all() or (all_events < 0).any() or (all_events >= end_day).any():
        raise ValueError("events must be finite and inside the daily_counts observation interval")
    observed_daily = np.bincount(np.floor(all_events).astype(np.int64), minlength=len(dates))
    if not np.array_equal(observed_daily, daily_counts.to_numpy()):
        raise ValueError("daily_counts must agree with the supplied event timestamps")
    training_events = all_events[all_events < split_day]
    test_events = all_events[(all_events >= split_day) & (all_events < end_day)]
    if len(training_events) < 3 or len(test_events) == 0:
        raise ValueError("both the training and test intervals need event data")

    bins_per_day = 24 if hour_of_day else 1
    interval_days = 1.0 / bins_per_day
    grid_dates = pd.date_range(dates[0], periods=len(dates) * bins_per_day, freq="h" if hour_of_day else "D")
    grid_train_mask = grid_dates < split
    grid_test_mask = ~grid_train_mask
    daily_design = calendar_design(grid_dates, hour_of_day=hour_of_day)
    # Minute-rounded observations can be a few ulps below an exact hour boundary.
    all_event_bins = np.floor(all_events * bins_per_day + 1e-9).astype(np.int64)
    interval_counts = np.bincount(all_event_bins, minlength=len(grid_dates))
    train_counts = pd.Series(interval_counts[grid_train_mask], index=grid_dates[grid_train_mask])
    calendar_parameters = fit_calendar_poisson(train_counts, interval_days, hour_of_day=hour_of_day)
    feature_count = daily_design.shape[1]
    grouped_times, grouped_counts = np.unique(training_events, return_counts=True)
    grouped_days = np.floor(grouped_times * bins_per_day + 1e-9).astype(np.int64)
    exposure = grid_train_mask.astype(float) * interval_days

    bounds = [(-20.0, 20.0)] * feature_count + [(0.0, 0.995), (-8.0, 8.0)]
    fits = []
    starts = ((0.0, 1.0), (0.8, np.exp(-8.0)), (0.4, 0.03), (0.25, 1.0), (0.55, 5.0), (0.8, 15.0))
    for branching_ratio, decay in starts:
        initial = calendar_parameters.copy()
        history_fraction = 1.0 + np.expm1(-decay * split_day) / (decay * split_day)
        initial[0] += np.log(1.0 - branching_ratio * history_fraction)
        start = np.concatenate([initial, [branching_ratio, np.log(decay)]])

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            return _negative_log_likelihood_and_gradient(
                parameters,
                grouped_times,
                grouped_counts,
                grouped_days,
                daily_design,
                exposure,
                0.0,
                split_day,
            )

        fits.append(minimize(
            objective, start, jac=True, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-11},
        ))
    valid_fits = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
    if not valid_fits:
        raise ValueError("seasonal Hawkes likelihood optimization failed")
    best = min(valid_fits, key=lambda fit: fit.fun)

    profile_cache: dict[float, float] = {}

    def profile_log_likelihood(fixed_branching_ratio: float) -> float:
        if fixed_branching_ratio in profile_cache:
            return profile_cache[fixed_branching_ratio]
        if fixed_branching_ratio == 0.0:
            training_rates = np.exp(daily_design[grid_train_mask] @ calendar_parameters)
            value = float(np.sum(train_counts.to_numpy() * np.log(training_rates) - training_rates * interval_days))
            profile_cache[fixed_branching_ratio] = value
            return value

        def profile_objective(nuisance: np.ndarray) -> tuple[float, np.ndarray]:
            full_parameters = np.concatenate(
                [nuisance[:feature_count], [fixed_branching_ratio, nuisance[-1]]]
            )
            loss, gradient = _negative_log_likelihood_and_gradient(
                full_parameters,
                grouped_times,
                grouped_counts,
                grouped_days,
                daily_design,
                exposure,
                0.0,
                split_day,
            )
            return float(loss), np.concatenate([gradient[:feature_count], [gradient[-1]]])

        nuisance_fits = []
        for log_decay in np.unique([best.x[-1], -8.0, np.log(0.03), np.log(15.0)]):
            start = np.concatenate([calendar_parameters.copy(), [log_decay]])
            decay = np.exp(log_decay)
            history_fraction = 1.0 + np.expm1(-decay * split_day) / (decay * split_day)
            start[0] += np.log(1.0 - fixed_branching_ratio * history_fraction)
            nuisance_fits.append(minimize(
                profile_objective, start, jac=True, method="L-BFGS-B",
                bounds=[(-20.0, 20.0)] * feature_count + [(-8.0, 8.0)],
                options={"maxiter": 500, "ftol": 1e-11},
            ))
        valid_nuisance = [fit for fit in nuisance_fits if fit.success and np.isfinite(fit.fun)]
        if not valid_nuisance:
            raise ValueError("profile likelihood optimization failed for every start")
        value = float(-min(fit.fun for fit in valid_nuisance))
        if value > -best.fun + 0.05:
            raise ValueError("profile likelihood found a better optimum; add Hawkes optimization starts")
        profile_cache[fixed_branching_ratio] = value
        return value

    estimate = float(best.x[feature_count])
    interval_cutoff = float(best.fun + 0.5 * 3.841458820694124)

    # Probe the full range: weakly identified slow-decay solutions can create a
    # second supported region far from the point estimate.
    profile_grid = np.unique([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.995, estimate])
    profile_nll = np.array([-profile_log_likelihood(value) for value in profile_grid])
    supported = np.flatnonzero(profile_nll <= interval_cutoff)
    first, last = int(supported[0]), int(supported[-1])
    if first == 0:
        lower_ci = 0.0
    else:
        lower_ci = float(
            brentq(
                lambda value: -profile_log_likelihood(value) - interval_cutoff,
                profile_grid[first - 1],
                profile_grid[first],
                xtol=0.001,
            )
        )
    if last == len(profile_grid) - 1:
        upper_ci = 0.995
    else:
        upper_ci = float(
            brentq(
                lambda value: -profile_log_likelihood(value) - interval_cutoff,
                profile_grid[last],
                profile_grid[last + 1],
                xtol=0.001,
            )
        )

    # Retain the training history when computing the conditional held-out score.
    score_times, score_counts = np.unique(all_events[all_events < end_day], return_counts=True)
    score_days = np.floor(score_times * bins_per_day + 1e-9).astype(np.int64)
    test_exposure = grid_test_mask.astype(float) * interval_days
    seasonal_test_nll, _ = _negative_log_likelihood_and_gradient(
        best.x,
        score_times,
        score_counts,
        score_days,
        daily_design,
        test_exposure,
        split_day,
        end_day,
    )
    coefficients = best.x[:feature_count]
    branching_ratio = float(best.x[feature_count])
    decay = float(np.exp(best.x[feature_count + 1]))

    calendar_test_rates = np.exp(np.clip(daily_design[grid_test_mask] @ calendar_parameters, -30.0, 30.0))
    calendar_poisson_ll = float(np.sum(
        interval_counts[grid_test_mask] * np.log(calendar_test_rates) - calendar_test_rates * interval_days
    ))
    train_rate = len(training_events) / split_day
    homogeneous_poisson_ll = float(len(test_events) * np.log(train_rate) - train_rate * (end_day - split_day))
    seasonal_hawkes_ll = float(-seasonal_test_nll)

    result: dict[str, object] = {
        "baseline_effects": "weekday+month+hour" if hour_of_day else "weekday+month",
        "baseline_interval_days": interval_days,
        "training_days": float(train_mask.sum()),
        "test_days": float(test_mask.sum()),
        "training_events": int(len(training_events)),
        "test_events": int(len(test_events)),
        "converged": bool(best.success),
        "optimizer_message": str(best.message),
        "branching_ratio": branching_ratio,
        "branching_ratio_ci95_lower": lower_ci,
        "branching_ratio_ci95_upper": upper_ci,
        "branching_ratio_ci_method": "profile likelihood, 1 df, 95% LR cutoff",
        "profile_interval_is_support_envelope": True,
        "profile_grid": [
            {"branching_ratio": float(value), "log_likelihood": float(-loss)}
            for value, loss in zip(profile_grid, profile_nll)
        ],
        "profile_reaches_parameter_bound": bool(first == 0 or last == len(profile_grid) - 1),
        "excitation_at_zero_boundary": bool(branching_ratio < 1e-6),
        "confidence_interval_caveat": (
            "LR support envelope with cutoff 3.8415; nominal 95% coverage is not guaranteed at n=0 "
            "or under weak identification. Decay is unidentified when n=0."
        ),
        "decay_per_day": decay,
        "mean_calendar_baseline_rate": float(np.mean(np.exp(daily_design[grid_train_mask] @ coefficients))),
        "train_log_likelihood": float(-best.fun),
        "test_log_likelihood": seasonal_hawkes_ll,
        "test_log_likelihood_per_event": seasonal_hawkes_ll / len(test_events),
        "calendar_poisson_test_log_likelihood": calendar_poisson_ll,
        "homogeneous_poisson_test_log_likelihood": homogeneous_poisson_ll,
        "test_ll_gain_vs_calendar_poisson": seasonal_hawkes_ll - calendar_poisson_ll,
        "test_ll_gain_per_event_vs_calendar_poisson": (seasonal_hawkes_ll - calendar_poisson_ll) / len(test_events),
        "test_ll_gain_vs_homogeneous_poisson": seasonal_hawkes_ll - homogeneous_poisson_ll,
    }
    if hour_of_day:
        for label, parameters in (("hawkes", coefficients), ("poisson", calendar_parameters)):
            hourly = np.exp(np.concatenate([[0.0], parameters[-23:]]))
            result[f"{label}_hour_of_day_multipliers"] = (hourly / hourly.mean()).tolist()
    if quarterly:
        quarter_scores: dict[str, dict[str, float]] = {}
        observation_end = dates[-1] + pd.Timedelta(days=1)
        boundaries = [split]
        boundaries.extend(date for date in pd.date_range(split, observation_end, freq="QS") if split < date < observation_end)
        boundaries.append(observation_end)
        for quarter_start, quarter_end in zip(boundaries[:-1], boundaries[1:]):
            q_start = float((quarter_start - dates[0]).total_seconds() / 86400.0)
            q_end = float((quarter_end - dates[0]).total_seconds() / 86400.0)
            q_mask = (grid_dates >= quarter_start) & (grid_dates < quarter_end)
            q_exposure = q_mask.astype(float) * interval_days
            if not q_exposure.any():
                continue
            q_times, q_counts = np.unique(all_events[all_events < q_end], return_counts=True)
            q_days = np.floor(q_times * bins_per_day + 1e-9).astype(np.int64)
            q_nll, _ = _negative_log_likelihood_and_gradient(
                best.x,
                q_times,
                q_counts,
                q_days,
                daily_design,
                q_exposure,
                q_start,
                q_end,
            )
            q_counts_by_bin = interval_counts[q_mask]
            q_rates = np.exp(np.clip(daily_design[q_mask] @ calendar_parameters, -30.0, 30.0))
            q_calendar_ll = float(np.sum(q_counts_by_bin * np.log(q_rates) - q_rates * interval_days))
            q_event_count = int(np.sum(q_counts_by_bin))
            q_days_count = float(np.sum(q_exposure))
            q_homogeneous_ll = float(q_event_count * np.log(train_rate) - train_rate * q_days_count)
            q_hawkes_ll = float(-q_nll)
            quarter_scores[quarter_start.strftime("%Y-Q") + str((quarter_start.month - 1) // 3 + 1)] = {
                "days": q_days_count,
                "events": float(q_event_count),
                "hawkes_log_likelihood": q_hawkes_ll,
                "calendar_poisson_log_likelihood": q_calendar_ll,
                "homogeneous_poisson_log_likelihood": q_homogeneous_ll,
                "hawkes_gain_vs_calendar_poisson": q_hawkes_ll - q_calendar_ll,
            }
        result["quarterly_test_windows"] = quarter_scores
    return result
