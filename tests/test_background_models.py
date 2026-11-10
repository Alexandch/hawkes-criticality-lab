from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import check_grad, minimize_scalar

from critical_processes.background_comparison import (
    background_tasks, generate_background_trial, run_background_comparison, simulate_background_trial,
    summarize_backgrounds,
)
from critical_processes.background_models import (
    fit_background_forecast, fit_background_window, fixed_background_loss, shaped_background_loss,
)
from critical_processes.window_features import conditional_window_data, fit_hawkes_window, prepare_hawkes_history


class BackgroundModelsTests(unittest.TestCase):
    def test_shaped_likelihood_gradient_and_day_integral(self) -> None:
        events = np.array([0.2, 1.2, 2.1, 2.1, 2.8, 3.4, 4.9, 5.2])
        history = prepare_hawkes_history(events, decay=0.7)
        memory, counts, total, integral, rate = conditional_window_data(history, 2, 5)
        background = np.array([2., 1., 3., 1., 2., 2.])
        selected = history.times[(history.times >= 2) & (history.times < 5)]
        shape = background[np.floor(selected).astype(int)] / background[2:5].mean()
        parameters = np.array([0.6, 0.4])
        args = (shape, memory, counts, total, integral)
        error = check_grad(lambda p: shaped_background_loss(p, *args)[0],
                           lambda p: shaped_background_loss(p, *args)[1], parameters)
        self.assertLess(error, 1e-6)
        # Independently integrate each prior exponential and evaluate strict-past intensities.
        prior = events[events < 5]
        exact_integral = sum(np.exp(-0.7 * max(0, 2 - s)) - np.exp(-0.7 * (5 - s)) for s in prior)
        self.assertAlmostEqual(integral, exact_integral)
        logs = []
        for s in events[(events >= 2) & (events < 5)]:
            past = events[events < s]
            intensity = parameters[0] * background[int(s)] / background[2:5].mean()
            intensity += parameters[1] * 0.7 * np.exp(-0.7 * (s - past)).sum() / rate
            logs.append(np.log(intensity))
        expected = (parameters[0] * total + parameters[1] * exact_integral - total - sum(logs)) / total
        self.assertAlmostEqual(shaped_background_loss(parameters, *args)[0], expected)

    def test_fixed_background_concave_solver_matches_independent_bounded_search(self) -> None:
        events = np.random.default_rng(52).uniform(0, 50, 150)
        history = prepare_hawkes_history(events)
        baseline = np.linspace(1, 3, 50)
        fit = fit_background_window(history, 10, 50, baseline, True)
        memory, counts, total, integral, rate = conditional_window_data(history, 10, 50)
        selected = history.times[(history.times >= 10) & (history.times < 50)]
        at_events = baseline[np.floor(selected).astype(int)] / rate
        args = (at_events, memory, counts, total, integral, baseline[10:50].sum())
        def objective(n):
            return fixed_background_loss(n, *args)[0]
        independent = minimize_scalar(objective, bounds=(0, 0.995), method="bounded")
        candidates = [0., 0.995, independent.x]
        best = min(candidates, key=objective)
        self.assertTrue(fit["fit_valid"])
        self.assertAlmostEqual(fit["n_hat"], best, places=5)
        n = 0.4
        numeric = (objective(n + 1e-6) - objective(n - 1e-6)) / 2e-6
        self.assertAlmostEqual(fixed_background_loss(n, *args)[1], numeric, places=7)

    def test_shape_rescaling_is_irrelevant_and_constant_matches_existing_mle(self) -> None:
        history = prepare_hawkes_history(np.random.default_rng(6).uniform(0, 100, 300))
        baseline = np.linspace(1, 3, 100)
        a = fit_background_window(history, 20, 100, baseline, False)
        b = fit_background_window(history, 20, 100, 50 * baseline, False)
        self.assertTrue(a["fit_valid"] and b["fit_valid"])
        self.assertAlmostEqual(a["n_hat"], b["n_hat"], places=7)
        constant = fit_background_window(history, 20, 100, np.ones(100), False)
        old = fit_hawkes_window(history, 20, 100)
        self.assertTrue(old[4] and constant["fit_valid"])
        self.assertAlmostEqual(constant["n_hat"], old[0], places=5)

    def test_forecast_horizon_and_future_events_do_not_change_past_fit(self) -> None:
        prefix = np.random.default_rng(9).poisson(4, 730)
        short, valid = fit_background_forecast(prefix, 760)
        long, other_valid = fit_background_forecast(prefix, 1090)
        self.assertTrue(valid and other_valid)
        np.testing.assert_allclose(short, long[:760], atol=1e-10, rtol=1e-10)
        events = np.arange(0.1, 60, 0.4)
        a = fit_background_window(prepare_hawkes_history(events), 20, 50, np.ones(100), False)
        b = fit_background_window(prepare_hawkes_history(np.append(events, np.arange(60.1, 99, 0.1))), 20, 50, np.ones(100), False)
        self.assertEqual(a, b)

    def test_step_occurs_after_prefix_and_seeds_replay(self) -> None:
        tasks = background_tasks(["step_up", "step_down"], [2.], [10, 30], 2, 314, 60, 1.)
        self.assertEqual(len({task[5] for task in tasks}), len(tasks))
        for task in tasks:
            counts, events, baseline, n, change = generate_background_trial(task)
            self.assertGreater(change, 60)
            self.assertLess(change, 90)
            self.assertEqual(n, 0)
            self.assertAlmostEqual(baseline.mean(), 2.)
            np.testing.assert_array_equal(np.bincount(np.floor(events).astype(int), minlength=90), counts)
            records, _ = simulate_background_trial(task)
            repeated, _ = simulate_background_trial(task)
            np.testing.assert_equal(records, repeated)

    def test_failed_fits_remain_missing_and_reduce_all_trial_accuracy(self) -> None:
        rows, _ = simulate_background_trial(background_tasks(["homogeneous"], [2.], [10], 2, 718, 60, 1.)[0])
        raw = pd.DataFrame([rows[0], {**rows[0], "n_hat": np.nan, "fit_valid": False,
                                     "failure_reason": "insufficient_events", "baseline_mean_hat": np.nan}])
        raw.loc[0, "n_hat"] = 0.05
        summary = summarize_backgrounds(raw).iloc[0]
        self.assertAlmostEqual(summary.mean_n_hat_valid, 0.05)
        self.assertEqual(summary.valid_fraction, 0.5)
        self.assertEqual(summary.accuracy_within_0p10_all_trials, 0.5)
        no_events = fit_background_window(prepare_hawkes_history([]), 0, 10, np.ones(10), False)
        self.assertFalse(no_events["fit_valid"])
        self.assertTrue(np.isnan(no_events["n_hat"]))

    def test_small_complete_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stdout(io.StringIO()):
                summary = run_background_comparison(directory, scenarios=["seasonal", "hawkes_n0p5"],
                    rates=[2.], windows=[10, 30], prefix_days=90, replicates=2, seed=135, workers=1)
            output = Path(directory)
            raw = pd.read_csv(output / "replicates.csv")
            config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(len(raw), 40)
            self.assertEqual(len(summary), 20)
            self.assertEqual(config["independent_trajectories"], 4)
            self.assertTrue(raw.loc[~raw.fit_valid, "n_hat"].isna().all())
            for name in ("analysis.md", "n_background_comparison.png", "baseline_forecasts.png", "paired_comparison.csv"):
                self.assertTrue((output / name).is_file())


if __name__ == "__main__":
    unittest.main()
