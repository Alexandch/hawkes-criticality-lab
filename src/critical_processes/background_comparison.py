from __future__ import annotations

import hashlib
import json
import platform
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from .background_models import fit_background_forecast, fit_background_window
from .negative_controls import poisson_control_rates, wilson_interval
from .simulation import simulate_staged_hawkes_counts
from .window_features import prepare_hawkes_history


SCENARIOS = {
    "homogeneous": "Однородный Пуассон", "seasonal": "Сезонный Пуассон",
    "trend": "Плавный рост", "step_up": "Скачок вверх", "step_down": "Скачок вниз",
    "seasonal_trend": "Сезонность и рост", "hawkes_n0p5": "Хоукс n=0.5", "hawkes_n0p9": "Хоукс n=0.9",
}
METHODS = {
    "constant_joint": "Постоянный фон, уровень свободен",
    "oracle_fixed": "Истинный фон целиком",
    "oracle_shape_joint": "Истинная форма, уровень свободен",
    "past_fixed": "Прогноз по прошлому целиком",
    "past_shape_joint": "Форма по прошлому, уровень свободен",
}
GROUP_COLUMNS = ["scenario", "true_n", "target_rate", "window_days", "method"]


def background_tasks(scenarios, rates, windows, replicates, seed, prefix_days, decay):
    rng = np.random.default_rng(seed)
    used, tasks = set(), []
    for scenario in scenarios:
        for rate_index, rate in enumerate(rates):
            for replicate in range(replicates):
                simulator_seed = int(rng.integers(0, 2**32, dtype=np.uint64))
                while simulator_seed in used:
                    simulator_seed = int(rng.integers(0, 2**32, dtype=np.uint64))
                used.add(simulator_seed)
                tasks.append((f"{scenario}/rate{rate_index}/rep{replicate}", scenario, rate, windows,
                              replicate, simulator_seed, prefix_days, decay))
    return tasks


def generate_background_trial(task):
    _, scenario, rate, windows, _, simulator_seed, prefix_days, decay = task
    rng = np.random.default_rng(simulator_seed)
    total_days = prefix_days + max(windows)
    n = {"hawkes_n0p5": 0.5, "hawkes_n0p9": 0.9}.get(scenario, 0.0)
    change_day = -1
    if n:
        baseline = np.full(total_days, rate * (1 - n))
        counts, events = simulate_staged_hawkes_counts(
            np.array([baseline[0]]), np.array([n]), total_days, decay, simulator_seed,
            burn_in=8 / (decay * (1 - n)), record_events=True,
        )
    else:
        if scenario in ("step_up", "step_down"):
            # A novel change after training; the prefix predictor cannot know it.
            offset = int(rng.integers(1, min(120, max(windows) - 1) + 1))
            change_day = prefix_days + offset
            weights = np.where(np.arange(total_days) < change_day, 0.5, 2.0)
            if scenario == "step_down":
                weights = 2.5 - weights
            baseline = rate * weights / weights.mean()
        else:
            baseline = poisson_control_rates(scenario, rate, total_days, rng)
        counts = rng.poisson(baseline)
        event_days = np.repeat(np.arange(total_days), counts)
        events = np.sort(event_days + rng.random(len(event_days)))
    return counts, events, baseline, n, change_day


def simulate_background_trial(task):
    identity, scenario, rate, windows, replicate, simulator_seed, prefix_days, decay = task
    counts, events, baseline, n, change_day = generate_background_trial(task)
    predicted, forecast_valid = fit_background_forecast(counts[:prefix_days], len(counts))
    history = prepare_hawkes_history(events, decay)
    records = []
    for window in windows:
        end = prefix_days + window
        truth = baseline[prefix_days:end]
        forecast = predicted[prefix_days:end]
        event_count = int(counts[prefix_days:end].sum())
        for method in METHODS:
            background = np.ones(len(counts)) if method == "constant_joint" else (
                baseline if method.startswith("oracle") else predicted
            )
            fit = fit_background_window(history, prefix_days, end, background, method.endswith("fixed"))
            records.append({
                "trajectory_id": identity, "scenario": scenario, "true_n": n, "target_rate": rate,
                "replicate": replicate, "simulator_seed": simulator_seed, "prefix_days": prefix_days,
                "window_days": window, "decay_per_day": decay, "change_day": change_day,
                "method": method, "event_count": event_count, "observed_rate": event_count / window,
                "true_baseline_mean": float(truth.mean()), "prefix_event_count": int(counts[:prefix_days].sum()),
                "forecast_valid": forecast_valid,
                "forecast_baseline_mean": float(forecast.mean()),
                "forecast_relative_mae": float(np.abs(forecast - truth).mean() / truth.mean()),
                "forecast_relative_bias": float((forecast - truth).mean() / truth.mean()), **fit,
            })
    example = pd.DataFrame()
    if replicate == 0:
        example = pd.DataFrame({
            "trajectory_id": identity, "scenario": scenario, "target_rate": rate, "true_n": n,
            "simulator_seed": simulator_seed, "day": np.arange(len(counts)), "event_count": counts,
            "true_baseline": baseline, "forecast_baseline": predicted,
            "training_prefix": np.arange(len(counts)) < prefix_days,
        })
    return records, example


def summarize_backgrounds(raw: pd.DataFrame) -> pd.DataFrame:
    records = []
    for keys, frame in raw.groupby(GROUP_COLUMNS, sort=True):
        valid = frame[frame.fit_valid]
        count, valid_count = len(frame), len(valid)
        error = valid.n_hat.to_numpy() - keys[1]
        mc_se = float(error.std(ddof=1) / np.sqrt(valid_count)) if valid_count >= 2 else np.nan
        critical = float(t.ppf(0.975, valid_count - 1)) if valid_count >= 2 else np.nan
        bias = float(error.mean()) if valid_count else np.nan
        row = dict(zip(GROUP_COLUMNS, keys))
        row.update({
            "replicates": count, "valid_estimates": valid_count, "valid_fraction": valid_count / count,
            "forecast_valid_fraction": float(frame.forecast_valid.mean()),
            "insufficient_events_fraction": float((frame.failure_reason == "insufficient_events").mean()),
            "optimizer_failure_fraction": float((frame.failure_reason == "optimization_failed").mean()),
            "mean_n_hat_valid": float(valid.n_hat.mean()), "median_n_hat_valid": float(valid.n_hat.median()),
            "n_hat_q05_valid": float(valid.n_hat.quantile(0.05)), "n_hat_q95_valid": float(valid.n_hat.quantile(0.95)),
            "bias_valid": bias, "bias_mc_ci95_lower": bias - critical * mc_se, "bias_mc_ci95_upper": bias + critical * mc_se,
            "mae_valid": float(np.abs(error).mean()) if valid_count else np.nan,
            "rmse_valid": float(np.sqrt(np.mean(error**2))) if valid_count else np.nan,
            "accuracy_within_0p10_all_trials": float(np.sum(np.abs(error) <= 0.1) / count),
            "mean_observed_rate": float(frame.observed_rate.mean()),
            "mean_forecast_relative_mae": float(frame.forecast_relative_mae.mean()),
            "mean_forecast_relative_bias": float(frame.forecast_relative_bias.mean()),
            "mean_baseline_relative_bias_valid": float(((valid.baseline_mean_hat - valid.true_baseline_mean) / valid.true_baseline_mean).mean()),
        })
        row["valid_ci95_lower"], row["valid_ci95_upper"] = wilson_interval(valid_count, count)
        for threshold, label in ((0.5, "0p5"), (0.9, "0p9")):
            high = int((valid.n_hat >= threshold).sum())
            row[f"n_ge_{label}_all_trials"] = high / count
            row[f"n_ge_{label}_valid_only"] = high / valid_count if valid_count else np.nan
            row[f"n_ge_{label}_ci95_lower"], row[f"n_ge_{label}_ci95_upper"] = wilson_interval(high, count)
        records.append(row)
    return pd.DataFrame(records)


def summarize_background_pairs(raw: pd.DataFrame) -> pd.DataFrame:
    records = []
    for keys, cell in raw.groupby(GROUP_COLUMNS[:-1], sort=True):
        reference = cell[cell.method == "constant_joint"].set_index("trajectory_id")
        for method in METHODS:
            if method == "constant_joint":
                continue
            other = cell[cell.method == method].set_index("trajectory_id").loc[reference.index]
            common = reference.fit_valid & other.fit_valid
            difference = (reference.loc[common, "n_hat"] - keys[1]).abs() - (other.loc[common, "n_hat"] - keys[1]).abs()
            count = len(difference)
            mean = float(difference.mean())
            se = float(difference.std(ddof=1) / np.sqrt(count)) if count >= 2 else np.nan
            critical = float(t.ppf(0.975, count - 1)) if count >= 2 else np.nan
            row = dict(zip(GROUP_COLUMNS[:-1], keys))
            row.update({
                "reference": "constant_joint", "method": method, "replicates": len(reference),
                "paired_valid_estimates": count, "paired_valid_fraction": count / len(reference),
                "mae_reduction_common_valid": mean, "mae_reduction_mc_ci95_lower": mean - critical * se,
                "mae_reduction_mc_ci95_upper": mean + critical * se,
            })
            records.append(row)
    return pd.DataFrame(records)


def run_background_comparison(
    out: str | Path = "results/background_comparison", scenarios=None, rates=None, windows=None,
    replicates: int = 300, seed: int = 20261115, prefix_days: int = 730, decay: float = 1.0, workers: int = 2,
) -> pd.DataFrame:
    scenarios = list(SCENARIOS) if scenarios is None else list(scenarios)
    rates = [0.2, 2.0, 20.0, 50.0] if rates is None else list(rates)
    windows = [30, 90, 180, 360] if windows is None else list(windows)
    if not scenarios or len(set(scenarios)) != len(scenarios) or any(s not in SCENARIOS for s in scenarios):
        raise ValueError("need distinct supported scenarios")
    if not rates or len(set(rates)) != len(rates) or any(not np.isfinite(r) or r <= 0 for r in rates):
        raise ValueError("need distinct finite positive rates")
    if not windows or len(set(windows)) != len(windows) or any(not np.isfinite(w) or w < 3 or w != int(w) for w in windows):
        raise ValueError("need distinct whole-day windows >= 3")
    if prefix_days < 30 or prefix_days != int(prefix_days) or replicates < 2 or workers < 1 or seed < 0 or not np.isfinite(decay) or decay <= 0:
        raise ValueError("need prefix >=30 days, replicates >=2, workers >=1, seed >=0 and decay >0")
    rates, windows = sorted(rates), sorted(int(w) for w in windows)
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    tasks = background_tasks(scenarios, rates, windows, replicates, seed, prefix_days, decay)
    records, examples = [], []
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        results = pool.map(simulate_background_trial, tasks, chunksize=8) if pool else map(simulate_background_trial, tasks)
        for index, (group, example) in enumerate(results, 1):
            records.extend(group)
            if not example.empty:
                examples.append(example)
            if index % replicates == 0:
                print(f"Background comparison: {index}/{len(tasks)} independent trajectories completed", flush=True)
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
    raw = pd.DataFrame(records)
    summary = summarize_backgrounds(raw)
    pairs = summarize_background_pairs(raw)
    raw.to_csv(output / "replicates.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    pairs.to_csv(output / "paired_comparison.csv", index=False)
    pd.concat(examples, ignore_index=True).to_csv(output / "baseline_examples.csv", index=False)
    config = {
        "stage": "2_background_misspecification", "scenarios": scenarios, "methods": METHODS,
        "target_rates": rates, "windows_days": windows, "prefix_days": prefix_days,
        "replicates_per_scenario_rate": replicates, "seed": seed, "workers": workers,
        "independent_trajectories": len(tasks), "window_fits": len(raw), "decay_per_day": decay,
        "decay_known_to_estimator": True, "n_bounds": [0, 0.995],
        "forecast_training": "only counts[0:prefix_days]; no scenario, true n, true baseline or future counts",
        "forecast_design": "Poisson GLM: intercept, 6 weekday dummies, annual Fourier harmonics 1 and 2, linear log trend",
        "forecast_penalty": "1e-5 L2 except intercept; intercept bounds [-20,20], others [-5,5]",
        "forecast_fixed_before_all_windows": True, "past_shape_joint_level_fitted_on_current_window": True,
        "oracle_is_privileged_diagnostic": True, "poisson_intensity_piecewise_constant_per_day": True,
        "poisson_time_average_rate_normalized_over_full_horizon": True,
        "hawkes_mean_rate": "stationary mu/(1-n)=target_rate, burn_in=8/(decay*(1-n))",
        "steps": "change after prefix at uniformly sampled integer offset 1..min(120,max_window-1)",
        "conditional_likelihood": "observed prefix history; normalized loss/N; beta fixed; amplitude/n concave likelihood",
        "joint_level_bounds_as_fraction_of_observed_rate": [1e-8, 5],
        "failed_estimates_are_missing": True, "high_n_fractions_are_not_calibrated_detector_false_alarm_rates": True,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
    }
    for filename in ("replicates.csv", "baseline_examples.csv"):
        config[filename.removesuffix(".csv") + "_sha256"] = hashlib.sha256((output / filename).read_bytes()).hexdigest()
    (output / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    from .background_report import write_background_report

    write_background_report(output)
    print(f"Saved background comparison to {output.resolve()}", flush=True)
    return summary
