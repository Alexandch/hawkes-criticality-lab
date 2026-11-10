from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .confident_calibration import calibrate_confident
from .dynamic_estimation import (
    DAMPING_GRID, Q_GRID, calibrate_trajectory_bands, choose_filter, coverage_rows,
    delay_rows, estimate_scores, extension_gate, filter_n_history, fit_measurement_noise,
    raw_estimate_path, stage_measurements, trajectory_error_summary, trim_path, validation_quality,
)
from .negative_controls import wilson_interval
from .selected_features import generate_selection_batch, load_selection_batch, save_selection_batch, selection_tasks
from .selected_training import selection_targets, sha, write_json
from .task_protocol import calibrate_task_threshold, evaluate_task_scores
from .window_benchmark import fit_window_network, score_window_matrix


METHODS = ("constant90", "calendar90", "calendar360", "dynamic_calendar")
FEATURE_NAMES = ["n_lag2", "n_lag1", "n_lag0", "slope_per_stage", "projection_two_stages",
                 "valid_lag2", "valid_lag1", "valid_lag0"]


def estimate_paths(batch, noise, selected, windows):
    return {
        "constant90": raw_estimate_path(batch, "constant"),
        "calendar90": raw_estimate_path(batch, "calendar"),
        "calendar360": raw_estimate_path(batch, "calendar", max(windows), windows),
        "dynamic_calendar": trim_path(filter_n_history(stage_measurements(batch), noise, selected["q"], selected["damping"])),
    }


def matched_features(batch, source, noise, selected):
    short = stage_measurements(batch)
    if source == "calendar90":
        n, valid = short[:, :, 0], short[:, :, 4]
        path = raw_estimate_path(batch)
    elif source == "dynamic_calendar":
        full = filter_n_history(short, noise, selected["q"], selected["damping"])
        n, valid, path = full.n, full.valid.astype(float), trim_path(full)
    else:
        raise ValueError("unknown matched feature source")
    values = np.lib.stride_tricks.sliding_window_view(n, 3, axis=1)
    flags = np.lib.stride_tricks.sliding_window_view(valid, 3, axis=1)
    return np.concatenate([values, path.slope[:, :, None], (path.n + 2*path.slope)[:, :, None], flags], axis=-1)


def score_batch(batch, paths, output, noise, selected, seeds):
    result = {}
    for task in ("state", "warning"):
        for method, path in paths.items():
            if method == "dynamic_calendar":
                trend_valid = path.valid
            else:
                short = stage_measurements(batch, "constant" if method == "constant90" else "calendar")
                trend_valid = np.lib.stride_tricks.sliding_window_view(short[:, :, 4] > 0, 3, axis=1).all(axis=-1)
            result[f"{task}__stat_{method}"] = estimate_scores(path, task, trend_valid)
        for source in ("calendar90", "dynamic_calendar"):
            features = matched_features(batch, source, noise, selected)
            members = []
            for seed in seeds:
                scores = score_window_matrix(features, output / "models" / f"{task}_{source}_s{seed}.pt")
                result[f"{task}__mlp_{source}_s{seed}"] = scores, np.ones_like(scores, dtype=bool)
                members.append(scores)
            ensemble = np.mean(members, axis=0)
            result[f"{task}__mlp_{source}_ensemble"] = ensemble, np.ones_like(ensemble, dtype=bool)
    return result


def save_scores(path, scores):
    np.savez_compressed(path, **{name: item for key, pair in scores.items()
                                for name, item in ((key, pair[0]), (key + "__valid", pair[1]))})


def old_snapshot(sources):
    """Read fingerprints of completed stages, never use their holdout outcomes."""
    files = ("protocol.json", "frozen_plan.json", "metrics.json", "run_config.json", "final_test_complete.json")
    result = {}
    for source in sources:
        for name in files:
            path = source / name
            if path.exists():
                result[str(path.resolve())] = sha(path)
        for path in (source / "data").glob("*.npz"):
            result[str(path.resolve())] = sha(path)
        for path in source.rglob("*.pt"):
            result[str(path.resolve())] = sha(path)
    return result


def reserved_seeds(sources):
    reserved = []
    for source in sources:
        for path in (source / "data").glob("*.npz"):
            with np.load(path) as data:
                for key in ("parameter_seeds", "simulator_seeds"):
                    if key in data:
                        reserved.extend(data[key].tolist())
    return reserved


def aggregate_errors(frame):
    return frame.groupby(["method", "family"], sort=False).agg(
        trajectories=("trajectory_id", "size"), valid_fraction=("valid_fraction", "mean"),
        mae=("mae", "mean"), bias=("bias", "mean"), slope_mae=("slope_mae", "mean")).reset_index()


def aggregate_coverage(frame):
    rows = []
    for (method, target, family), group in frame.groupby(["method", "target", "family"], sort=False):
        valid = group[group.valid_windows > 0]
        hits = int(valid.trajectory_covered.sum())
        lo, hi = wilson_interval(hits, len(valid)) if len(valid) else (np.nan, np.nan)
        rows.append({"method": method, "target": target, "family": family, "trajectories": len(group),
            "trajectories_with_valid_estimates": len(valid), "trajectory_covered": hits,
            "trajectory_coverage": hits/len(valid) if len(valid) else np.nan,
            "coverage_ci95_lower": lo, "coverage_ci95_upper": hi,
            "window_coverage": float(valid.covered_windows.sum()/valid.valid_windows.sum()) if len(valid) else np.nan,
            "mean_width": float(valid.mean_width.mean())})
    return pd.DataFrame(rows)


def paired_error_effects(errors, seed=21, samples=500):
    rng, rows = np.random.default_rng(seed), []
    for group_name, families in (("all", None), ("near_high", ["hawkes_high", "hawkes_near_threshold"]),
                                 ("abrupt", ["hawkes_abrupt"]), ("rising", ["hawkes_rising"]), ("falling", ["hawkes_falling"])):
        subset = errors if families is None else errors[errors.family.isin(families)]
        pair = subset.pivot(index="trajectory_id", columns="method", values="mae")
        for reference in ("calendar90", "calendar360"):
            delta = (pair.dynamic_calendar - pair[reference]).dropna().to_numpy()
            boot = [float(delta[rng.integers(0, len(delta), len(delta))].mean()) for _ in range(samples)]
            lo, hi = np.quantile(boot, [.025, .975])
            rows.append({"group": group_name, "reference": reference, "trajectories": len(delta),
                "dynamic_minus_reference_mae": float(delta.mean()), "paired_bootstrap_ci95_lower": lo,
                "paired_bootstrap_ci95_upper": hi})
    return pd.DataFrame(rows)


def _load_or_generate(output, split, size, seed, generator, reserved, workers):
    path = output / "data" / f"{split}.npz"
    tasks = selection_tasks(seed, f"dynamic_{split}", size, generator, reserved)
    if path.exists():
        batch = load_selection_batch(path)
        if batch.ids.tolist() != [t[0] for t in tasks] or batch.parameter_seeds.tolist() != [t[2] for t in tasks] or batch.simulator_seeds.tolist() != [t[3] for t in tasks]:
            raise ValueError("saved data do not match the fixed seed protocol")
    else:
        batch = generate_selection_batch(tasks, workers)
        save_selection_batch(batch, path)
    reserved.extend(batch.parameter_seeds.tolist() + batch.simulator_seeds.tolist())
    return batch


def run_dynamic_experiment(out="results/dynamic_estimation", train_per_family=150, validation_per_family=150,
        calibration_per_family=300, evaluation_per_family=300, data_seed=20261121, evaluation_seed=20261122,
        model_seeds=(61, 73, 89), epochs=100, workers=2, bootstrap_samples=500,
        prefix_days=730, stage_days=90, stages=12, rate_min=.2, rate_max=50., sources=None):
    import torch

    output = Path(out)
    sources = [Path(p) for p in (sources if sources is not None else
               ("results/selected_detectors", "results/sensitivity_diagnostics"))]
    if any(output.resolve() == p.resolve() for p in sources):
        raise ValueError("new experiment needs its own output directory")
    if min(train_per_family, validation_per_family, evaluation_per_family, workers, epochs, bootstrap_samples) < 1 or calibration_per_family < 19:
        raise ValueError("positive sizes and at least 19 calibration trajectories required")
    if stages < 8 or stage_days < 3 or prefix_days < 3*stage_days or not 0 < rate_min <= rate_max:
        raise ValueError("invalid monitoring dimensions or rates")
    if data_seed == evaluation_seed or not model_seeds or len(set(model_seeds)) != len(model_seeds):
        raise ValueError("independent data roots and distinct initialization seeds required")
    generator = dict(prefix_days=prefix_days, stage_days=stage_days, stages=stages, lookback=3, forecast_stages=2,
                     critical_level=.9, rate_min=rate_min, rate_max=rate_max, windows=[stage_days, 4*stage_days])
    source_hashes = old_snapshot(sources)
    protected = 2 * (4 + 2*(len(model_seeds)+1))
    code_names = ("dynamic_estimation.py", "dynamic_experiment.py", "dynamic_report.py", "selected_features.py", "simulation.py",
                  "fast_window_mle.py", "background_models.py", "task_protocol.py", "confident_calibration.py", "window_benchmark.py")
    protocol = {"stage": "6_causal_dynamic_n", "generator": generator,
        "sizes": dict(train=train_per_family, validation=validation_per_family, calibration=calibration_per_family, evaluation=evaluation_per_family),
        "data_seed": data_seed, "evaluation_seed": evaluation_seed, "model_seeds": list(model_seeds), "epochs": epochs,
        "workers": workers, "bootstrap_samples": bootstrap_samples, "q_grid": list(Q_GRID), "damping_grid": list(DAMPING_GRID),
        "measurement_noise": "train empirical MSE in 4x4 n_hat/SE bins, shrinkage100, variance floor0.0025; not raw curvature variance",
        "filter": "x=(n,slope), F=[[1,rho],[0,rho]], Q=q^2[[1/3,1/2],[1/2,1]]; Joseph update; clip n[0,.995], slope[-.5,.5]; at most2 missing blocks",
        "selection": "minimum validation macro family MAE; ties capped abrupt delay then larger q and damping",
        "extension_gate": "near/high MAE at least5% lower, macro MAE at most5% worse, worst family MAE at most0.02 worse, mean capped abrupt delay at most one stage worse; validation only",
        "slope_target": "causal secant (n[k]-n[k-2])/2 per stage; no derivative or future slope label in features",
        "bands": "nominal95% whole-trajectory separate n/slope bands; maximum family conformal rank ceil((m+1)*.95); insufficient valid calibration => full parameter domain; invalid decisions excluded and coverage reported",
        "detectors": "four statistical methods plus separately retrained matched8-feature calendar90/dynamic MLP; report all seeds and fixed ensemble, no winner selection on test",
        "matched_feature_names": FEATURE_NAMES, "protected_score_streams": protected,
        "policies": ["family_confident", "family_nominal_sensitivity"], "alpha": .05, "confidence_failure_probability": .05,
        "primary_comparison": "dynamic_calendar vs calendar90 on estimation; matched MLP ensembles are diagnostic, never replace R4 primary",
        "old_snapshot_sha256": source_hashes,
        "code_sha256": {name: sha(Path(__file__).parent/name) for name in code_names}}
    output.mkdir(parents=True, exist_ok=True)
    for name in ("data", "models"):
        (output / name).mkdir(exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("output belongs to a different protocol/code; use a separate directory")
    else:
        if (output / "data/evaluation.npz").exists():
            raise ValueError("cannot establish a protocol over an existing holdout")
        write_json(protocol_path, protocol)
    complete_path = output / "complete.json"
    if complete_path.exists():
        completed = json.loads(complete_path.read_text(encoding="utf-8"))
        for name, digest in completed["artifact_sha256"].items():
            if sha(output/name) != digest:
                raise ValueError(f"completed artifact changed: {name}")
        return json.loads((output/"metrics.json").read_text(encoding="utf-8"))
    reserved, batches, hashes = reserved_seeds(sources), {}, {}
    for index, split in enumerate(("train", "validation", "calibration")):
        seed = int(np.random.SeedSequence([data_seed, index]).generate_state(1)[0])
        batches[split] = _load_or_generate(output, split, protocol["sizes"][split], seed, generator, reserved, workers)
        hashes[split] = sha(output/"data"/f"{split}.npz")
    noise = fit_measurement_noise(stage_measurements(batches["train"]), batches["train"].ratios)
    write_json(output/"measurement_noise.json", noise)
    candidates = []
    for q in Q_GRID:
        for damping in DAMPING_GRID:
            path = trim_path(filter_n_history(stage_measurements(batches["validation"]), noise, q, damping))
            quality = validation_quality(path, batches["validation"], "dynamic_calendar", stage_days)
            candidates.append({**quality, "q": q, "damping": damping})
    candidates = pd.DataFrame(candidates)
    candidates.to_csv(output/"validation_candidates.csv", index=False)
    selected = choose_filter(candidates)
    baseline = validation_quality(raw_estimate_path(batches["validation"]), batches["validation"], "calendar90", stage_days)
    gate = extension_gate(selected, baseline, stage_days)
    print(f"Dynamic estimator selected q={selected['q']}, damping={selected['damping']}, validation gate={gate['passed']}", flush=True)
    paths = {split: estimate_paths(batch, noise, selected, generator["windows"]) for split, batch in batches.items()}
    targets = {split: selection_targets(batch, generator) for split, batch in batches.items()}
    for task in ("state", "warning"):
        for method in ("calendar90", "dynamic_calendar"):
            x = matched_features(batches["train"], method, noise, selected)
            for seed in model_seeds:
                model_path = output / "models" / f"{task}_{method}_s{seed}.pt"
                if not model_path.exists():
                    checkpoint, history = fit_window_network(x, targets["train"][task].labels,
                        targets["train"][task].eligible, list(range(8)), seed, epochs)
                    checkpoint.update({"window_feature_names": FEATURE_NAMES, "schema": "causal_dynamic_n_v1",
                        "task": task, "method": method, "seed": seed, "epochs": epochs,
                        "training_sha256": hashes["train"], "threshold": None})
                    torch.save(checkpoint, model_path)
                    history.to_csv(model_path.with_suffix(".csv"), index=False)
    cal_scores = score_batch(batches["calibration"], paths["calibration"], output, noise, selected, model_seeds)
    if len(cal_scores) != protected:
        raise RuntimeError("declared calibration scope does not match score streams")
    save_scores(output/"data/calibration_scores.npz", cal_scores)
    thresholds, cal_tables = {}, []
    for key, (scores, _) in cal_scores.items():
        task = key.split("__")[0]
        for policy in protocol["policies"]:
            if policy == "family_confident":
                cutoff, table = calibrate_confident(scores, targets["calibration"][task], batches["calibration"].families, .05, .05/protected)
            else:
                cutoff, table = calibrate_task_threshold(scores, targets["calibration"][task], batches["calibration"].families, .05)
            thresholds[f"{key}__{policy}"] = cutoff
            cal_tables.append(table.assign(stream=key, policy=policy))
    pd.concat(cal_tables, ignore_index=True).to_csv(output/"calibration.csv", index=False)
    bands, band_tables = {}, []
    for method in METHODS:
        bands[method], table = calibrate_trajectory_bands(paths["calibration"][method], batches["calibration"])
        band_tables.append(table.assign(method=method))
    pd.concat(band_tables, ignore_index=True).to_csv(output/"interval_calibration.csv", index=False)
    frozen = {"protocol_sha256": sha(protocol_path), "pretest_dataset_sha256": hashes, "selected_filter": selected,
        "validation_baseline": baseline, "validation_gate": gate, "noise_sha256": sha(output/"measurement_noise.json"),
        "thresholds": thresholds, "interval_multipliers": bands,
        "model_sha256": {str(p.relative_to(output)).replace("\\", "/"): sha(p) for p in (output/"models").glob("*.pt")},
        "calibration_scores_sha256": sha(output/"data/calibration_scores.npz"), "holdout_used_for_selection": False}
    frozen_path = output/"frozen_plan.json"
    if (output/"data/evaluation.npz").exists() and not frozen_path.exists():
        raise ValueError("an existing holdout requires its original frozen plan")
    if frozen_path.exists() and json.loads(frozen_path.read_text(encoding="utf-8")) != frozen:
        raise ValueError("cannot change a frozen plan after holdout access")
    write_json(frozen_path, frozen)
    # This is the only point where the final holdout can first be generated.
    write_json(output/"evaluation_started.json", {"frozen_plan_sha256": sha(frozen_path)})
    batch = _load_or_generate(output, "evaluation", evaluation_per_family, evaluation_seed, generator, reserved, workers)
    hashes["evaluation"] = sha(output/"data/evaluation.npz")
    eval_paths, target = estimate_paths(batch, noise, selected, generator["windows"]), selection_targets(batch, generator)
    eval_scores = score_batch(batch, eval_paths, output, noise, selected, model_seeds)
    save_scores(output/"data/evaluation_scores.npz", eval_scores)
    metrics, metric_rows, family_rows, trajectory_rows = {}, [], [], []
    for key, (score, valid) in eval_scores.items():
        task, method = key.split("__", 1)
        for policy in protocol["policies"]:
            name = f"{key}__{policy}"
            metric, family, trajectory, _ = evaluate_task_scores(score, valid, target[task], batch.families,
                batch.ids, batch.rates, thresholds[name], task, bootstrap_seed=data_seed+500, bootstrap_samples=bootstrap_samples)
            metrics[name] = metric
            metric_rows.append({"stream": key, "method": method, "policy": policy,
                **{k: v for k, v in metric.items() if not isinstance(v, (list, tuple))}})
            family_rows.append(family.assign(stream=key, policy=policy))
            trajectory_rows.append(trajectory.assign(stream=key, policy=policy))
    write_json(output/"metrics.json", metrics)
    pd.DataFrame(metric_rows).to_csv(output/"operating_metrics.csv", index=False)
    pd.concat(family_rows, ignore_index=True).to_csv(output/"family_metrics.csv", index=False)
    pd.concat(trajectory_rows, ignore_index=True).to_csv(output/"trajectory_metrics.csv", index=False)
    errors = pd.concat([trajectory_error_summary(path, batch, method, stage_days) for method, path in eval_paths.items()], ignore_index=True)
    errors.to_csv(output/"estimation_trajectories.csv", index=False)
    aggregate_errors(errors).to_csv(output/"estimation_summary.csv", index=False)
    paired_error_effects(errors, data_seed+501, bootstrap_samples).to_csv(output/"paired_errors.csv", index=False)
    delay = pd.concat([delay_rows(path, batch, method, stage_days) for method, path in eval_paths.items()], ignore_index=True)
    delay.to_csv(output/"abrupt_delay.csv", index=False)
    coverage = pd.concat([coverage_rows(path, batch, method, bands[method]) for method, path in eval_paths.items()], ignore_index=True)
    coverage.to_csv(output/"interval_trajectories.csv", index=False)
    aggregate_coverage(coverage).to_csv(output/"interval_coverage.csv", index=False)
    # Full paths permit verification and plots without resimulation.
    np.savez_compressed(output/"data/estimate_paths.npz", **{f"{method}__{name}": value
        for method, path in eval_paths.items() for name, value in path.__dict__.items()})
    if old_snapshot(sources) != source_hashes:
        raise RuntimeError("earlier completed experiments changed")
    from .dynamic_report import write_dynamic_report
    write_dynamic_report(output)
    names = [str(p.relative_to(output)).replace("\\", "/") for p in output.rglob("*") if p.is_file() and p.name != "complete.json"]
    write_json(complete_path, {"artifact_sha256": {name: sha(output/name) for name in names},
        "dataset_sha256": hashes, "old_experiments_unchanged": True, "numpy_version": np.__version__, "torch_version": torch.__version__})
    return json.loads(json.dumps(metrics, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description="Causal n/slope estimation with an independently frozen holdout")
    parser.add_argument("--out", default="results/dynamic_estimation")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--train-per-family", type=int, default=150)
    parser.add_argument("--validation-per-family", type=int, default=150)
    parser.add_argument("--calibration-per-family", type=int, default=300)
    parser.add_argument("--evaluation-per-family", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    args = parser.parse_args()
    run_dynamic_experiment(**vars(args))


if __name__ == "__main__":
    main()
