from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_event_csv
from .baselines import calendar_poisson_backtest
from .detector import detector_config, score_feature_matrix, train_neural_detector
from .estimation import hawkes_mle, poisson_mle
from .indicators import FEATURE_NAMES, rolling_features, summarize_counts
from .negative_controls import evaluate_negative_controls
from .robust_training import train_robust_detector
from .window_benchmark import run_window_benchmark
from .n_accuracy import run_n_accuracy
from .background_comparison import SCENARIOS as BACKGROUND_SCENARIOS, run_background_comparison
from .task_audit import run_task_audit
from .selected_training import run_selected_training
from .sensitivity_diagnostics import run_sensitivity_diagnostics
from .seasonal_hawkes import seasonal_hawkes_backtest
from .simulation import simulate_hawkes, simulate_poisson


def run_simulation(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    ratios = np.array([0.2, 0.4, 0.6, 0.75, 0.85, 0.92, 0.97])
    records: list[dict[str, float | int | str]] = []
    for model in ("poisson", "hawkes"):
        levels = [0.0] if model == "poisson" else ratios
        for n in levels:
            for replicate in range(args.replicates):
                baseline = args.rate if model == "poisson" else args.rate * (1.0 - n)
                burn_in = args.burn_in_times / (args.decay * (1.0 - n)) if model == "hawkes" else 0.0
                events = simulate_poisson(args.rate, args.horizon, rng) if model == "poisson" else simulate_hawkes(baseline, float(n), args.decay, args.horizon, rng, burn_in=burn_in)
                counts = np.histogram(events, bins=np.arange(0, args.horizon + 1))[0]
                metrics = summarize_counts(counts, 1.0, cluster_gap=args.cluster_gap)
                row: dict[str, float | int | str] = {
                    "model": model,
                    "true_branching_ratio": float(n),
                    "replicate": replicate,
                    "event_count": len(events),
                    **dict(zip(FEATURE_NAMES, metrics.tolist())),
                }
                if model == "hawkes" and len(events) >= 3:
                    try:
                        row.update({f"mle_{key}": value for key, value in hawkes_mle(events, args.horizon).items()})
                    except (ValueError, FloatingPointError):
                        row["mle_failed"] = 1
                elif model == "poisson":
                    row.update({f"poisson_mle_{key}": value for key, value in poisson_mle(events, args.horizon).items()})
                records.append(row)
    results = pd.DataFrame.from_records(records)
    results.to_csv(out / "simulation_replicates.csv", index=False)
    if "mle_branching_ratio" not in results:
        results["mle_branching_ratio"] = np.nan
    summary = results.groupby(["model", "true_branching_ratio"], as_index=False).agg(
        replicates=("replicate", "count"),
        mean_rate=("mean_rate", "mean"),
        mean_rate_sd=("mean_rate", "std"),
        variance=("variance", "mean"),
        variance_sd=("variance", "std"),
        lag1_autocorrelation=("lag1_autocorrelation", "mean"),
        lag1_autocorrelation_sd=("lag1_autocorrelation", "std"),
        mean_cluster_size=("mean_cluster_size", "mean"),
        mean_cluster_size_sd=("mean_cluster_size", "std"),
        event_count=("event_count", "mean"),
        mle_branching_ratio=("mle_branching_ratio", "mean"),
        mle_branching_ratio_sd=("mle_branching_ratio", "std"),
    )
    for metric in ("mean_rate", "variance", "lag1_autocorrelation", "mean_cluster_size", "mle_branching_ratio"):
        summary[f"{metric}_ci95"] = 1.96 * summary[f"{metric}_sd"] / np.sqrt(summary["replicates"])
    summary.to_csv(out / "simulation_summary.csv", index=False)
    config = {key: value for key, value in vars(args).items() if key != "func"}
    (out / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, column in zip(axes.flat, FEATURE_NAMES):
        hawkes = summary[summary["model"] == "hawkes"]
        error = hawkes[f"{column}_ci95"] if f"{column}_ci95" in hawkes else None
        axis.errorbar(hawkes["true_branching_ratio"], hawkes[column], yerr=error, marker="o", capsize=3, label="Хоукс")
        poisson_value = summary.loc[summary["model"] == "poisson", column].mean()
        axis.axhline(poisson_value, color="gray", linestyle="--", label="Пуассон")
        axis.set(xlabel="Коэффициент ветвления n", ylabel=column)
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.savefig(out / "indicators.png", dpi=160)
    plt.close(fig)
    print(f"Сохранено в {out.resolve()}")


def run_code_analysis(args: argparse.Namespace) -> None:
    if (args.hour_of_day or args.quarterly) and not args.split_date:
        raise ValueError("--hour-of-day and --quarterly require --split-date")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    events, horizon, daily = load_event_csv(
        args.input,
        args.date_column,
        args.city_column,
        args.city,
        args.start,
        args.end,
        args.id_column,
        args.category_column,
        args.category,
    )
    daily.rename_axis("date").to_csv(out / "daily_counts.csv", header=True)
    if len(daily) >= args.window:
        features = rolling_features(daily.to_numpy(), args.window, 1.0)
        dates = daily.index[args.window - 1:]
        pd.DataFrame(features, columns=FEATURE_NAMES, index=dates).rename_axis("date").to_csv(out / "rolling_indicators.csv")
    poisson_fit = poisson_mle(events, horizon)
    estimates: dict[str, object] = {
        "input": str(Path(args.input)),
        "city": args.city,
        "events": int(len(events)),
        "horizon_days": horizon,
        "poisson": poisson_fit,
    }
    if args.split_date:
        calendar_metrics, forecasts = calendar_poisson_backtest(daily, args.split_date)
        estimates["calendar_poisson_backtest"] = calendar_metrics
        forecasts.to_csv(out / "calendar_backtest_predictions.csv", index=False)
    try:
        hawkes_fit = hawkes_mle(events, horizon)
        estimates["hawkes"] = hawkes_fit
        poisson_aic = 2 - 2 * poisson_fit["log_likelihood"]
        hawkes_aic = 6 - 2 * hawkes_fit["log_likelihood"]
        estimates["comparison"] = {
            "poisson_aic": poisson_aic,
            "hawkes_aic": hawkes_aic,
            "hawkes_minus_poisson_aic": hawkes_aic - poisson_aic,
        }
    except ValueError as exc:
        estimates["hawkes_error"] = str(exc)
    if args.split_date:
        estimates["seasonal_hawkes_backtest"] = seasonal_hawkes_backtest(
            events, daily, args.split_date, quarterly=args.quarterly, hour_of_day=args.hour_of_day
        )
    (out / "estimates.json").write_text(json.dumps(estimates, ensure_ascii=False, indent=2), encoding="utf-8")
    config = {key: value for key, value in vars(args).items() if key != "func"}
    (out / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(estimates, ensure_ascii=False, indent=2))


def run_code_scoring(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = detector_config(args.model)
    window = args.window or int(round(config["stage_horizon"]))
    lookback = args.lookback or int(config["lookback"])
    if window != int(round(config["stage_horizon"])):
        raise ValueError(f"--window must match the model stage horizon ({config['stage_horizon']} days)")
    if lookback != int(config["lookback"]):
        raise ValueError(f"--lookback must match the model ({config['lookback']})")
    stage_step = args.stage_step if args.stage_step is not None else window
    if stage_step <= 0:
        raise ValueError("--stage-step must be positive")
    _, _, daily = load_event_csv(
        args.input,
        args.date_column,
        args.city_column,
        args.city,
        args.start,
        args.end,
        args.id_column,
        args.category_column,
        args.category,
    )
    feature_matrix = rolling_features(daily.to_numpy(), window, 1.0)[::stage_step]
    scores = score_feature_matrix(feature_matrix, args.model)
    feature_dates = daily.index[window - 1::stage_step]
    dates = feature_dates[lookback - 1:lookback - 1 + len(scores)]
    pd.DataFrame({
        "date": dates,
        "early_warning_score": scores,
        "warning": scores >= float(config["threshold"]),
        "threshold": float(config["threshold"]),
    }).to_csv(out / "code_warning_scores.csv", index=False)
    print(f"Сохранено {len(scores)} оценок в {(out / 'code_warning_scores.csv').resolve()}")


def run_code_robustness(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    quarter_rows: list[dict[str, object]] = []
    for category in args.categories:
        events, _, daily = load_event_csv(
            args.input,
            args.date_column,
            start=args.start,
            end=args.end,
            id_column=args.id_column,
            category_column=args.category_column,
            category=category,
        )
        metrics = seasonal_hawkes_backtest(events, daily, args.split_date, quarterly=True, hour_of_day=args.hour_of_day)
        summary_rows.append({"category": category, **metrics})
        for period, values in metrics["quarterly_test_windows"].items():
            quarter_rows.append({"category": category, "period": period, **values})
        print(f"{category}: branching ratio {metrics['branching_ratio']:.4f}")

    summary = pd.DataFrame.from_records(summary_rows)
    summary.drop(columns=["quarterly_test_windows"], errors="ignore").to_csv(out / "category_robustness.csv", index=False)
    pd.DataFrame.from_records(quarter_rows).to_csv(out / "quarterly_robustness.csv", index=False)
    (out / "category_robustness.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved category and quarterly robustness results to {out.resolve()}")


def run_neural_training(args: argparse.Namespace) -> None:
    result = train_neural_detector(
        out=args.out,
        trajectories=args.trajectories,
        stages=args.stages,
        lookback=args.lookback,
        seed=args.seed,
        epochs=args.epochs,
        stage_horizon=args.stage_horizon,
        forecast_stages=args.forecast_stages,
        critical_level=args.critical_level,
        rate_min=args.rate_min,
        rate_max=args.rate_max,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_negative_controls(args: argparse.Namespace) -> None:
    evaluate_negative_controls(
        args.model, args.out, rates=args.rates, replicates=args.replicates,
        stages=args.stages, seed=args.seed, bootstrap_samples=args.bootstrap_samples,
    )
    print(f"Saved negative-control audit to {Path(args.out).resolve()}")


def run_robust_training(args: argparse.Namespace) -> None:
    train_robust_detector(
        args.out, args.baseline_model, train_per_family=args.train_per_family,
        calibration_per_family=args.calibration_per_family, test_per_family=args.test_per_family,
        epochs=args.epochs, seed=args.seed, target_false_alarm_probability=args.target_false_alarm_probability,
    )


def run_window_comparison(args: argparse.Namespace) -> None:
    run_window_benchmark(
        args.out, seed=args.seed, train_per_family=args.train_per_family,
        calibration_per_family=args.calibration_per_family, test_per_family=args.test_per_family,
        epochs=args.epochs, workers=args.workers,
        target_false_alarm_probability=args.target_false_alarm_probability,
    )


def run_n_accuracy_experiment(args: argparse.Namespace) -> None:
    run_n_accuracy(
        args.out, ratios=args.ratios, rates=args.rates, windows=args.windows,
        replicates=args.replicates, seed=args.seed, decay=args.decay,
        history_days=args.history_days, burn_in_times=args.burn_in_times,
        workers=args.workers, auc_bootstrap_samples=args.auc_bootstrap_samples,
    )


def run_background_experiment(args: argparse.Namespace) -> None:
    run_background_comparison(
        args.out, scenarios=args.scenarios, rates=args.rates, windows=args.windows,
        replicates=args.replicates, prefix_days=args.prefix_days, seed=args.seed, decay=args.decay, workers=args.workers,
    )


def run_state_warning_audit(args: argparse.Namespace) -> None:
    run_task_audit(
        args.out, args.model_dir, calibration_per_family=args.calibration_per_family,
        test_per_family=args.test_per_family, counterfactual_pairs=args.counterfactual_pairs,
        seed=args.seed, workers=args.workers, target_false_alarm_probability=args.target_false_alarm_probability,
        bootstrap_samples=args.bootstrap_samples,
    )


def run_selected_experiment(args: argparse.Namespace) -> None:
    run_selected_training(
        args.out, train_per_family=args.train_per_family,
        validation_calibration_per_family=args.validation_calibration_per_family,
        validation_per_family=args.validation_per_family, calibration_per_family=args.calibration_per_family,
        test_per_family=args.test_per_family, model_seeds=args.model_seeds, windows=args.windows,
        epochs=args.epochs, data_seed=args.data_seed, final_test_seed=args.final_test_seed,
        workers=args.workers, bootstrap_samples=args.bootstrap_samples,
    )


def run_sensitivity_experiment(args: argparse.Namespace) -> None:
    run_sensitivity_diagnostics(args.source, args.out, train_per_family=args.train_per_family,
        calibration_per_family=args.calibration_per_family, evaluation_per_family=args.evaluation_per_family,
        data_seed=args.data_seed, evaluation_seed=args.evaluation_seed, model_seed=args.model_seed,
        epochs=args.epochs, workers=args.workers, bootstrap_samples=args.bootstrap_samples)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="critical-processes", description="Симуляция и анализ критических режимов процессов событий")
    subparsers = parser.add_subparsers(dest="command", required=True)
    simulate = subparsers.add_parser("simulate", help="Монте-Карло сравнение Пуассона и Хоукса")
    simulate.add_argument("--out", default="results/simulation")
    simulate.add_argument("--replicates", type=int, default=30)
    simulate.add_argument("--horizon", type=float, default=365.0)
    simulate.add_argument("--rate", type=float, default=2.0, help="стационарная интенсивность для контроля масштаба")
    simulate.add_argument("--decay", type=float, default=1.0)
    simulate.add_argument("--burn-in-times", type=float, default=8.0, help="длина прогрева в единицах времени релаксации 1/(decay*(1-n))")
    simulate.add_argument("--cluster-gap", type=float, default=0.5, help="порог разрыва между активными временными бинами")
    simulate.add_argument("--seed", type=int, default=42)
    simulate.set_defaults(func=run_simulation)

    code = subparsers.add_parser("analyze-code", help="Анализ зарегистрированных событий из CSV")
    code.add_argument("input")
    code.add_argument("--date-column", required=True)
    code.add_argument("--city-column")
    code.add_argument("--city")
    code.add_argument("--id-column", help="deduplicate events by this incident identifier")
    code.add_argument("--category-column")
    code.add_argument("--category")
    code.add_argument("--start")
    code.add_argument("--end")
    code.add_argument("--window", type=int, default=30)
    code.add_argument("--split-date", help="score calendar Poisson baselines from this date onward")
    code.add_argument("--hour-of-day", action="store_true", help="add hourly calendar effects to both event-time models")
    code.add_argument("--quarterly", action="store_true", help="also score each held-out quarter without refitting")
    code.add_argument("--out", default="results/code")
    code.set_defaults(func=run_code_analysis)

    train = subparsers.add_parser("train-neural", help="Обучение MLP раннего предупреждения на симуляциях")
    train.add_argument("--out", default="results/neural_detector.pt")
    train.add_argument("--trajectories", type=int, default=120)
    train.add_argument("--stages", type=int, default=12)
    train.add_argument("--lookback", type=int, default=3)
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--stage-horizon", type=float, default=30.0)
    train.add_argument("--forecast-stages", type=int, default=2)
    train.add_argument("--critical-level", type=float, default=0.9)
    train.add_argument("--rate-min", type=float, default=0.2)
    train.add_argument("--rate-max", type=float, default=50.0)
    train.add_argument("--seed", type=int, default=42)
    train.set_defaults(func=run_neural_training)

    validate = subparsers.add_parser("validate-detector", help="audit a frozen neural detector on noncritical Poisson controls")
    validate.add_argument("--model", required=True)
    validate.add_argument("--rates", type=float, nargs="+", default=[2.0, 20.0, 50.0, 680.0, 1560.0])
    validate.add_argument("--replicates", type=int, default=300)
    validate.add_argument("--stages", type=int, default=12)
    validate.add_argument("--seed", type=int, default=20261110)
    validate.add_argument("--bootstrap-samples", type=int, default=1000)
    validate.add_argument("--out", default="results/detector_negative_controls")
    validate.set_defaults(func=run_negative_controls)

    robust_train = subparsers.add_parser("train-robust", help="train with mixed controls, calibrate trajectory alarms, test early warnings")
    robust_train.add_argument("--baseline-model", default="results/neural_detector_90d_multirate_500.pt")
    robust_train.add_argument("--out", default="results/neural_detector_robust_90d")
    robust_train.add_argument("--train-per-family", type=int, default=300)
    robust_train.add_argument("--calibration-per-family", type=int, default=200)
    robust_train.add_argument("--test-per-family", type=int, default=200)
    robust_train.add_argument("--epochs", type=int, default=200)
    robust_train.add_argument("--seed", type=int, default=20261111)
    robust_train.add_argument("--target-false-alarm-probability", type=float, default=0.05)
    robust_train.set_defaults(func=run_robust_training)

    window_comparison = subparsers.add_parser("compare-window-detectors", help="compare count MLP, event-time Hawkes-feature MLP and n-trend detector")
    window_comparison.add_argument("--out", default="results/window_detector_comparison")
    window_comparison.add_argument("--seed", type=int, default=20261113)
    window_comparison.add_argument("--train-per-family", type=int, default=200)
    window_comparison.add_argument("--calibration-per-family", type=int, default=200)
    window_comparison.add_argument("--test-per-family", type=int, default=200)
    window_comparison.add_argument("--epochs", type=int, default=200)
    window_comparison.add_argument("--workers", type=int, default=2)
    window_comparison.add_argument("--target-false-alarm-probability", type=float, default=0.05)
    window_comparison.set_defaults(func=run_window_comparison)

    accuracy = subparsers.add_parser("estimate-n-accuracy", help="Monte Carlo recovery of stationary branching ratios at equal mean event rates")
    accuracy.add_argument("--out", default="results/n_estimation_accuracy")
    accuracy.add_argument("--ratios", type=float, nargs="+", default=[0.0, 0.5, 0.8, 0.9, 0.95])
    accuracy.add_argument("--rates", type=float, nargs="+", default=[0.2, 2.0, 20.0, 50.0])
    accuracy.add_argument("--windows", type=int, nargs="+", default=[30, 90, 180, 360])
    accuracy.add_argument("--replicates", type=int, default=500)
    accuracy.add_argument("--seed", type=int, default=20261114)
    accuracy.add_argument("--decay", type=float, default=1.0)
    accuracy.add_argument("--history-days", type=int)
    accuracy.add_argument("--burn-in-times", type=float, default=8.0)
    accuracy.add_argument("--workers", type=int, default=2)
    accuracy.add_argument("--auc-bootstrap-samples", type=int, default=500)
    accuracy.set_defaults(func=run_n_accuracy_experiment)

    background = subparsers.add_parser("compare-backgrounds", help="compare known and past-only forecast backgrounds on Poisson and Hawkes controls")
    background.add_argument("--out", default="results/background_comparison")
    background.add_argument("--scenarios", nargs="+", choices=list(BACKGROUND_SCENARIOS), default=list(BACKGROUND_SCENARIOS))
    background.add_argument("--rates", type=float, nargs="+", default=[0.2, 2.0, 20.0, 50.0])
    background.add_argument("--windows", type=int, nargs="+", default=[30, 90, 180, 360])
    background.add_argument("--replicates", type=int, default=300)
    background.add_argument("--prefix-days", type=int, default=730)
    background.add_argument("--seed", type=int, default=20261115)
    background.add_argument("--decay", type=float, default=1.0)
    background.add_argument("--workers", type=int, default=2)
    background.set_defaults(func=run_background_experiment)

    task_audit = subparsers.add_parser("audit-state-warning", help="separately evaluate current high n and strictly future first-crossing warnings")
    task_audit.add_argument("--out", default="results/state_warning_audit")
    task_audit.add_argument("--model-dir", default="results/window_detector_comparison/models")
    task_audit.add_argument("--calibration-per-family", type=int, default=200)
    task_audit.add_argument("--test-per-family", type=int, default=300)
    task_audit.add_argument("--counterfactual-pairs", type=int, default=300)
    task_audit.add_argument("--seed", type=int, default=20261116)
    task_audit.add_argument("--workers", type=int, default=2)
    task_audit.add_argument("--target-false-alarm-probability", type=float, default=0.05)
    task_audit.add_argument("--bootstrap-samples", type=int, default=500)
    task_audit.set_defaults(func=run_state_warning_audit)

    selected = subparsers.add_parser("train-selected-detectors", help="select windows on validation, train multiple initializations, calibrate and freeze before one new final test")
    selected.add_argument("--out", default="results/selected_detectors")
    selected.add_argument("--train-per-family", type=int, default=150)
    selected.add_argument("--validation-calibration-per-family", type=int, default=150)
    selected.add_argument("--validation-per-family", type=int, default=150)
    selected.add_argument("--calibration-per-family", type=int, default=300)
    selected.add_argument("--test-per-family", type=int, default=300)
    selected.add_argument("--model-seeds", type=int, nargs="+", default=[17, 29, 41])
    selected.add_argument("--windows", type=int, nargs="+", default=[30, 90, 180, 360])
    selected.add_argument("--epochs", type=int, default=100)
    selected.add_argument("--data-seed", type=int, default=20261117)
    selected.add_argument("--final-test-seed", type=int, default=20261118)
    selected.add_argument("--workers", type=int, default=2)
    selected.add_argument("--bootstrap-samples", type=int, default=500)
    selected.set_defaults(func=run_selected_experiment)

    sensitivity = subparsers.add_parser("diagnose-sensitivity", help="diagnose n-estimation, threshold calibration and difficult controls on new data")
    sensitivity.add_argument("--source", default="results/selected_detectors")
    sensitivity.add_argument("--out", default="results/sensitivity_diagnostics")
    sensitivity.add_argument("--train-per-family", type=int, default=150)
    sensitivity.add_argument("--calibration-per-family", type=int, default=300)
    sensitivity.add_argument("--evaluation-per-family", type=int, default=300)
    sensitivity.add_argument("--data-seed", type=int, default=20261119)
    sensitivity.add_argument("--evaluation-seed", type=int, default=20261120)
    sensitivity.add_argument("--model-seed", type=int, default=53)
    sensitivity.add_argument("--epochs", type=int, default=100)
    sensitivity.add_argument("--workers", type=int, default=2)
    sensitivity.add_argument("--bootstrap-samples", type=int, default=500)
    sensitivity.set_defaults(func=run_sensitivity_experiment)

    score = subparsers.add_parser("score-code", help="Оценка нейросетевого сигнала на суточном ряде CODE")
    score.add_argument("input")
    score.add_argument("--model", required=True)
    score.add_argument("--date-column", required=True)
    score.add_argument("--city-column")
    score.add_argument("--city")
    score.add_argument("--id-column", help="deduplicate events by this incident identifier")
    score.add_argument("--category-column")
    score.add_argument("--category")
    score.add_argument("--start")
    score.add_argument("--end")
    score.add_argument("--window", type=int, help="по умолчанию берется длина стадии из файла модели")
    score.add_argument("--stage-step", type=int, help="шаг между входными окнами; по умолчанию равен --window")
    score.add_argument("--lookback", type=int, help="по умолчанию берется из файла модели")
    score.add_argument("--out", default="results/code_scores")
    score.set_defaults(func=run_code_scoring)

    robustness = subparsers.add_parser(
        "robustness-code", help="compare seasonal Hawkes fits by incident category and quarter"
    )
    robustness.add_argument("input")
    robustness.add_argument("--date-column", required=True)
    robustness.add_argument("--category-column", required=True)
    robustness.add_argument("--categories", nargs="+", default=["THEFT", "BATTERY", "CRIMINAL DAMAGE", "ASSAULT"])
    robustness.add_argument("--id-column")
    robustness.add_argument("--start", default="2024-01-01")
    robustness.add_argument("--end", default="2026-01-01")
    robustness.add_argument("--split-date", default="2025-01-01")
    robustness.add_argument("--hour-of-day", action="store_true", help="add hourly calendar effects to both event-time models")
    robustness.add_argument("--out", default="results/code_robustness")
    robustness.set_defaults(func=run_code_robustness)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
