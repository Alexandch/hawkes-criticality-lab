from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from critical_processes.background_comparison import (
    GROUP_COLUMNS, METHODS, background_tasks, generate_background_trial, simulate_background_trial,
    summarize_background_pairs, summarize_backgrounds,
)
from critical_processes.background_models import fit_background_forecast


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct background diagnostics and replay first/last trajectories per cell.")
    parser.add_argument("--out", type=Path, default=Path("results/background_comparison"))
    output = parser.parse_args().out
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    for filename in ("replicates.csv", "baseline_examples.csv"):
        digest = hashlib.sha256((output / filename).read_bytes()).hexdigest()
        assert digest == config[filename.removesuffix(".csv") + "_sha256"]
    raw = pd.read_csv(output / "replicates.csv", float_precision="round_trip").fillna({"failure_reason": ""})
    examples = pd.read_csv(output / "baseline_examples.csv", float_precision="round_trip")
    assert len(raw) == config["window_fits"]
    assert raw.trajectory_id.nunique() == config["independent_trajectories"]
    assert raw.groupby("trajectory_id").simulator_seed.nunique().eq(1).all()
    assert raw.drop_duplicates("trajectory_id").simulator_seed.is_unique
    assert raw.groupby("trajectory_id").size().eq(len(METHODS) * len(config["windows_days"])).all()
    assert raw.groupby(GROUP_COLUMNS).size().eq(config["replicates_per_scenario_rate"]).all()
    assert raw.loc[~raw.fit_valid, "n_hat"].isna().all()
    assert raw.loc[raw.fit_valid, "n_hat"].between(0, 0.995).all()
    assert raw.groupby(["trajectory_id", "window_days"]).event_count.nunique().eq(1).all()
    steps = raw[raw.scenario.isin(["step_up", "step_down"])]
    assert (steps.change_day > steps.prefix_days).all()
    assert (steps.change_day < steps.prefix_days + max(config["windows_days"])).all()
    for summarize, filename in ((summarize_backgrounds, "summary.csv"), (summarize_background_pairs, "paired_comparison.csv")):
        reconstructed = summarize(raw)
        saved = pd.read_csv(output / filename, float_precision="round_trip")
        pd.testing.assert_frame_equal(reconstructed, saved, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    tasks = background_tasks(
        config["scenarios"], config["target_rates"], config["windows_days"],
        config["replicates_per_scenario_rate"], config["seed"], config["prefix_days"], config["decay_per_day"],
    )
    selected = [task for task in tasks if task[4] in (0, config["replicates_per_scenario_rate"] - 1)]
    for task in selected:
        replay, example = simulate_background_trial(task)
        original = raw[raw.trajectory_id == task[0]].reset_index(drop=True)
        pd.testing.assert_frame_equal(pd.DataFrame(replay), original, check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-9)
        if not example.empty:
            pd.testing.assert_frame_equal(example, examples[examples.trajectory_id == task[0]].reset_index(drop=True),
                                          check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-9)
            counts, events, baseline, n, _ = generate_background_trial(task)
            np.testing.assert_array_equal(np.bincount(np.floor(events).astype(int), minlength=len(counts)), counts)
            if n:
                np.testing.assert_allclose(baseline / (1 - n), task[2], atol=1e-12)
            else:
                np.testing.assert_allclose(baseline.mean(), task[2], atol=1e-12)
            prefix = config["prefix_days"]
            short, valid = fit_background_forecast(counts[:prefix], prefix + min(config["windows_days"]))
            long, long_valid = fit_background_forecast(counts[:prefix], len(counts))
            assert valid == long_valid
            np.testing.assert_allclose(short, long[:len(short)], rtol=1e-10, atol=1e-10, equal_nan=True)
    print(f"Verified {len(raw)} fits, all summaries and paired statistics, hashes, prefix-only forecasts and {len(selected)} replayed trajectories.")


if __name__ == "__main__":
    main()
