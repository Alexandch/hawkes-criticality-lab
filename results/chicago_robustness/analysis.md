# Chicago robustness checks

Update 2026-11-10: these category fits do not control for hour of day. The [intraday control comparison](../intraday_comparison/analysis.md) removes the all-incidents held-out Hawkes advantage. Category-specific conclusions below still need that stronger baseline control.

Each category-specific weekday/month Hawkes model was fitted to 2024 and scored on 2025, with the pre-2025 event history carried into the conditional test likelihood. The branching-ratio intervals are profile-likelihood 95% intervals conditional on the seasonal exponential Hawkes model.

| Category | 2024 events | Branching ratio | Profile 95% interval | Decay per day | 2025 log-likelihood gain vs calendar Poisson |
| --- | ---: | ---: | ---: | ---: | ---: |
| All incidents | 259,626 | 0.4752 | 0.4607-0.4898 | 10.09 | 2,921.42 |
| Theft | 60,560 | 0.5863 | 0.5662-0.6063 | 11.12 | 1,966.03 |
| Battery | 46,188 | 0.4979 | 0.4715-0.5244 | 12.62 | 920.07 |
| Criminal damage | 28,557 | 0.1882 | 0.1504-0.2273 | 7.36 | 136.88 |
| Assault | 23,488 | 0.5308 | 0.5016-0.5600 | 12.73 | 690.65 |

The seasonal Hawkes test log-likelihood gain over the calendar Poisson was positive in every quarter of 2025 for each of the four categories. This is a repeatable predictive improvement within this city and split, not proof that each process is critical or that the fitted branching ratio is causal. Interval estimates are model-based and do not account for model misspecification, multiple category comparisons, crime reporting bias, or timestamp rounding. In particular, one city and one train/test split are not enough to claim generality.

`category_robustness.csv` contains the category-level estimates and intervals. `quarterly_robustness.csv` contains the quarter-by-quarter test scores. `category_robustness.json` preserves the full output, including optimizer details.

Reproduce from the project root:

```powershell
.\.venv\Scripts\critical-processes.exe robustness-code data\raw\chicago_crimes_2024_2025.csv --date-column date --category-column primary_type --id-column case_number --start 2024-01-01 --end 2026-01-01 --split-date 2025-01-01 --out results\chicago_robustness
```
