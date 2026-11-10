from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from critical_processes.data import load_event_csv
from critical_processes.baselines import calendar_poisson_backtest
from critical_processes.estimation import hawkes_log_likelihood
from critical_processes.seasonal_hawkes import _negative_log_likelihood_and_gradient
from critical_processes.baselines import calendar_design


class EventDataTests(unittest.TestCase):
    def test_deduplicates_known_ids_but_keeps_rows_without_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            pd.DataFrame(
                {
                    "occurred_at": ["2024-01-01 01:00", "2024-01-01 01:00", "2024-01-02 02:00", "2024-01-03 03:00"],
                    "case_id": ["A", "A", None, None],
                    "kind": ["theft", "theft", "assault", "theft"],
                }
            ).to_csv(path, index=False)
            events, horizon, daily = load_event_csv(
                path,
                "occurred_at",
                start="2024-01-01",
                end="2024-01-04",
                id_column="case_id",
                category_column="kind",
                category="theft",
            )

        self.assertEqual(len(events), 2)
        self.assertEqual(horizon, 3.0)
        self.assertEqual(int(daily.sum()), 2)

    def test_tied_timestamps_do_not_trigger_each_other_instantly(self) -> None:
        result = hawkes_log_likelihood(np.array([0.1, 0.1, 0.4]), 1.0, 2.0, 0.5, 1.0)
        intensity_after_gap = 2.0 + np.exp(-0.3)
        expected = (
            2 * np.log(2.0)
            + np.log(intensity_after_gap)
            - 2.0
            - 0.5 * (2 * (1 - np.exp(-0.9)) + (1 - np.exp(-0.6)))
        )
        self.assertAlmostEqual(result, expected, places=10)

    def test_weekday_baseline_improves_forecast_for_weekly_pattern(self) -> None:
        index = pd.date_range("2024-01-01", "2025-12-31", freq="D")
        expected = np.array([10, 12, 14, 16, 18, 22, 25])[index.dayofweek]
        counts = pd.Series(expected, index=index, dtype=float)
        metrics, predictions = calendar_poisson_backtest(counts, "2025-01-01")

        self.assertLess(
            metrics["weekday"]["test_negative_log_likelihood"],
            metrics["homogeneous"]["test_negative_log_likelihood"],
        )
        self.assertEqual(len(predictions), 365)

    def test_seasonal_hawkes_gradient_matches_finite_difference(self) -> None:
        index = pd.date_range("2024-01-01", periods=4, freq="D")
        design = calendar_design(index)
        parameters = np.zeros(design.shape[1] + 2)
        parameters[0] = np.log(3.0)
        parameters[-2:] = [0.4, np.log(2.0)]
        event_times = np.array([0.1, 0.1, 0.4, 1.2])
        event_counts = np.array([2, 1, 1], dtype=np.int64)
        grouped_times = np.array([0.1, 0.4, 1.2])
        event_days = np.floor(grouped_times).astype(np.int64)
        exposure = np.array([1.0, 1.0, 0.0, 0.0])
        value, gradient = _negative_log_likelihood_and_gradient(
            parameters, grouped_times, event_counts, event_days, design, exposure, 0.0, 2.0
        )

        epsilon = 1e-6
        for position in (0, len(parameters) - 2, len(parameters) - 1):
            plus, minus = parameters.copy(), parameters.copy()
            plus[position] += epsilon
            minus[position] -= epsilon
            plus_value = _negative_log_likelihood_and_gradient(
                plus, grouped_times, event_counts, event_days, design, exposure, 0.0, 2.0
            )[0]
            minus_value = _negative_log_likelihood_and_gradient(
                minus, grouped_times, event_counts, event_days, design, exposure, 0.0, 2.0
            )[0]
            numerical = (plus_value - minus_value) / (2.0 * epsilon)
            self.assertAlmostEqual(gradient[position], numerical, places=5)


if __name__ == "__main__":
    unittest.main()
