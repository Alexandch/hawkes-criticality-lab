from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from critical_processes.n_accuracy import (
    GROUP_COLUMNS, accuracy_tasks, simulate_accuracy_trial, summarize_accuracy,
    summarize_distinguishability,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct stationary-n accuracy statistics and replay simulated trajectories.")
    parser.add_argument("--out", type=Path, default=Path("results/n_estimation_accuracy"))
    output = parser.parse_args().out
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    raw_path = output / "replicates.csv"
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == config["replicates_sha256"]
    raw = pd.read_csv(raw_path, float_precision="round_trip").fillna({"failure_reason": ""})
    assert len(raw) == config["window_fits"]
    assert raw.trajectory_id.nunique() == config["independent_trajectories"]
    assert raw.groupby("trajectory_id").simulator_seed.nunique().eq(1).all()
    assert raw.drop_duplicates("trajectory_id").simulator_seed.is_unique
    assert raw.groupby("trajectory_id").size().eq(len(config["windows_days"])).all()
    assert raw.groupby(GROUP_COLUMNS).size().eq(config["replicates_per_parameter_pair"]).all()
    np.testing.assert_allclose(raw.true_baseline / (1 - raw.true_n), raw.target_rate, atol=1e-12)
    assert raw.loc[~raw.fit_valid, "n_hat"].isna().all()
    assert raw.loc[raw.fit_valid, "n_hat"].between(0, 0.995).all()
    reconstructed = summarize_accuracy(raw)
    saved = pd.read_csv(output / "summary.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(reconstructed, saved, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    pairs = summarize_distinguishability(raw, config["ratios"], config["seed"], config["auc_bootstrap_samples"])
    saved_pairs = pd.read_csv(output / "distinguishability.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(pairs, saved_pairs, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    tasks = accuracy_tasks(
        config["ratios"], config["target_rates"], config["windows_days"],
        config["replicates_per_parameter_pair"], config["seed"], config["decay_per_day"],
        config["history_days"], config["burn_in_relaxation_times"],
    )
    selected = [task for task in tasks if task[4] in (0, config["replicates_per_parameter_pair"] - 1)]
    for task in selected:
        replay = pd.DataFrame(simulate_accuracy_trial(task))
        original = raw[raw.trajectory_id == task[0]].reset_index(drop=True)
        pd.testing.assert_frame_equal(replay, original, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10)
    print(f"Verified {len(raw)} fits, all summary/AUC statistics, fixed expected rates and {len(selected)} replayed trajectories.")


if __name__ == "__main__":
    main()
