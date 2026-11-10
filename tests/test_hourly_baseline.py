from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from critical_processes.baselines import calendar_design, fit_calendar_poisson
from critical_processes.seasonal_hawkes import (
    _negative_log_likelihood_and_gradient,
    seasonal_hawkes_backtest,
)


class HourlyBaselineTests(unittest.TestCase):
    def test_poisson_exposure_and_hourly_rates(self) -> None:
        index = pd.date_range("2024-01-01", periods=24 * 28, freq="h")
        expected_counts = np.where((index.hour >= 10) & (index.hour < 20), 9.0, 2.0)
        counts = pd.Series(expected_counts, index=index)
        coefficients = fit_calendar_poisson(counts, exposure_days=1.0 / 24, hour_of_day=True)
        predicted_counts = np.exp(calendar_design(index, hour_of_day=True) @ coefficients) / 24
        np.testing.assert_allclose(predicted_counts, expected_counts, rtol=0.001)

    def test_zero_branching_matches_event_time_poisson_on_holdout(self) -> None:
        index = pd.date_range("2024-01-31", periods=48, freq="h")
        design = calendar_design(index, hour_of_day=True)
        parameters = np.zeros(design.shape[1] + 2)
        parameters[0] = np.log(12.0)
        parameters[18:] = np.linspace(-0.4, 0.6, len(parameters) - 18)
        parameters[-2:] = [0.0, np.log(3.0)]
        times = np.array([0.1, 0.8, 1.0, 1.125, 1.7])
        counts = np.array([1, 2, 2, 1, 1], dtype=np.int64)
        bins = np.floor(times * 24).astype(np.int64)
        exposure = np.concatenate([np.zeros(24), np.full(24, 1.0 / 24)])
        nll, _ = _negative_log_likelihood_and_gradient(
            parameters, times, counts, bins, design, exposure, 1.0, 2.0
        )
        rates = np.exp(design @ parameters[:-2])
        held_out = times >= 1.0
        expected_ll = np.sum(counts[held_out] * np.log(rates[bins[held_out]])) - np.dot(rates, exposure)
        self.assertAlmostEqual(-nll, expected_ll, places=10)

    def test_all_hourly_gradient_components_with_pre_holdout_history(self) -> None:
        index = pd.date_range("2024-01-31", periods=72, freq="h")
        design = calendar_design(index, hour_of_day=True)
        rng = np.random.default_rng(72)
        parameters = rng.normal(0.0, 0.1, design.shape[1] + 2)
        parameters[0] = np.log(8.0)
        parameters[-2:] = [0.45, np.log(4.0)]
        times = np.array([0.1, 0.8, 1.0, 1.125, 1.7, 2.3, 2.9])
        counts = np.array([1, 2, 2, 1, 1, 2, 1], dtype=np.int64)
        bins = np.floor(times * 24).astype(np.int64)
        exposure = np.concatenate([np.zeros(24), np.full(48, 1.0 / 24)])
        _, gradient = _negative_log_likelihood_and_gradient(
            parameters, times, counts, bins, design, exposure, 1.0, 3.0
        )
        epsilon = 1e-6
        numerical = np.zeros_like(gradient)
        for position in range(len(parameters)):
            plus, minus = parameters.copy(), parameters.copy()
            plus[position] += epsilon
            minus[position] -= epsilon
            upper = _negative_log_likelihood_and_gradient(plus, times, counts, bins, design, exposure, 1.0, 3.0)[0]
            lower = _negative_log_likelihood_and_gradient(minus, times, counts, bins, design, exposure, 1.0, 3.0)[0]
            numerical[position] = (upper - lower) / (2.0 * epsilon)
        np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-7)

    def test_rejects_date_only_jitter_for_hourly_analysis(self) -> None:
        daily = pd.Series([2, 2], index=pd.date_range("2024-01-01", periods=2))
        daily.attrs["has_intraday_timestamps"] = False
        with self.assertRaisesRegex(ValueError, "intraday timestamps"):
            seasonal_hawkes_backtest(np.array([0.1, 0.5, 1.1, 1.5]), daily, "2024-01-02", hour_of_day=True)

    def test_split_must_be_exactly_midnight_even_far_from_origin(self) -> None:
        daily = pd.Series(1, index=pd.date_range("2024-01-01", periods=731))
        with self.assertRaisesRegex(ValueError, "midnight"):
            seasonal_hawkes_backtest(np.arange(731) + 0.5, daily, "2025-01-01 00:01:00", hour_of_day=True)

    def test_quarterly_scores_add_up_and_holdout_does_not_change_fit(self) -> None:
        rng = np.random.default_rng(2026)
        dates = pd.date_range("2024-01-01", periods=120, freq="D")
        # A Poisson process with a strong deterministic daily rhythm and no excitation.
        counts = rng.poisson(np.tile(np.array([0.2] * 12 + [1.5] * 12), len(dates)))
        hours = np.repeat(np.arange(len(counts)), counts)
        events = np.sort((hours + rng.uniform(size=len(hours))) / 24)
        daily = pd.Series(counts.reshape(-1, 24).sum(axis=1), index=dates)
        result = seasonal_hawkes_backtest(events, daily, "2024-03-01", quarterly=True, hour_of_day=True)
        self.assertTrue(result["converged"])
        self.assertLess(result["branching_ratio"], 0.2)
        quarters = list(result["quarterly_test_windows"].values())
        self.assertEqual(sum(q["events"] for q in quarters), result["test_events"])
        self.assertAlmostEqual(sum(q["hawkes_log_likelihood"] for q in quarters), result["test_log_likelihood"], places=7)
        self.assertAlmostEqual(
            sum(q["calendar_poisson_log_likelihood"] for q in quarters),
            result["calendar_poisson_test_log_likelihood"], places=7,
        )
        changed = events.copy()
        test = changed >= 60
        changed[test] = 60 + rng.uniform(0, 60, size=test.sum())
        changed.sort()
        changed_daily = pd.Series(np.bincount(np.floor(changed).astype(int), minlength=120), index=dates)
        refit = seasonal_hawkes_backtest(changed, changed_daily, "2024-03-01", hour_of_day=True)
        self.assertAlmostEqual(refit["branching_ratio"], result["branching_ratio"], places=10)
        self.assertAlmostEqual(refit["train_log_likelihood"], result["train_log_likelihood"], places=10)


if __name__ == "__main__":
    unittest.main()
