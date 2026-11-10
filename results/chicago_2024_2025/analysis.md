# Chicago pilot: 2024-2025

Update 2026-11-10: this is the earlier fit without an hour-of-day baseline. The [intraday control comparison](../intraday_comparison/analysis.md) shows that its held-out Hawkes gain does not survive the hourly-baseline control. Use that report for the current interpretation.

Source: City of Chicago, [Crimes - 2001 to Present](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2), queried on 2026-10-12 for calendar years 2024 and 2025.

The export has 497,788 rows. Deduplicating the 51 repeated case numbers leaves 497,737 events over 731 days. Timestamps are available mostly at minute resolution; the Hawkes likelihood groups events with the same timestamp, avoiding an arbitrary within-minute event order.

| Quantity | Estimate |
| --- | ---: |
| Mean daily count | 680.90 |
| Daily count variance | 6,432.33 |
| Poisson rate (events/day) | 680.90 |
| Hawkes baseline (events/day) | 309.98 |
| Hawkes branching ratio | 0.5453 |
| Hawkes decay (per day) | 9.8577 |
| Hawkes AIC minus Poisson AIC | -11,306.53 |

The fitted Hawkes model has a lower in-sample AIC than a homogeneous Poisson model. Its point estimate of the branching ratio is below one, but this is not evidence that the real crime process is near a critical regime. The homogeneous Poisson comparison does not control for daily/weekly seasonality, long-term trend, reporting changes, or spatial heterogeneity; these can all produce apparent clustering. The likelihood also treats minute-rounded timestamps as tied observations. Treat the fit as an empirical illustration, not a causal or criticality claim.

`daily_counts.csv`, `rolling_indicators.csv`, and `estimates.json` contain the reproducible outputs. Before drawing conclusions, compare against a nonhomogeneous Poisson baseline with calendar effects and check stability across crime categories and time windows.

## Calendar-only holdout

Calendar baselines were fitted on 2024 daily counts and evaluated on 2025. Lower negative log-likelihood (NLL) is better; each model is scored on the same 365 held-out days.

| Baseline | Test NLL | NLL per day | Improvement over homogeneous |
| --- | ---: | ---: | ---: |
| Homogeneous Poisson | 3,868.80 | 10.5994 | 0.00 |
| Weekday rates | 3,830.40 | 10.4943 | 38.40 |
| Month-of-year rates | 3,339.25 | 9.1486 | 529.55 |
| Weekday + month Poisson GLM | 3,299.67 | 9.0402 | 569.13 |

The joint calendar model reduces held-out NLL by about 14.7% versus the homogeneous baseline. Much of the gain comes from month-of-year effects; weekday effects add a smaller improvement. This makes the in-sample Hawkes-vs-homogeneous-Poisson AIC comparison insufficient to attribute clustering to self-excitation. The calendar GLM models daily counts, not event-time dependence, so the next statistical comparison should be a Hawkes model with a calendar-varying baseline evaluated on a time holdout.

Reproduce the holdout with `--split-date 2025-01-01`. Predictions are in `calendar_backtest_predictions.csv`.

## Calendar-baseline Hawkes holdout

A second fit uses the same weekday/month log-linear baseline for the Hawkes immigrant rate. It is trained on 2024 only; its 2025 score is conditional on the 2024 event history at the split boundary. Held-out log-likelihoods below use the same event-time likelihood convention (events rounded to the same minute are grouped) and are directly comparable.

| Model | 2025 test log-likelihood |
| --- | ---: |
| Homogeneous Poisson, rate estimated on 2024 | 1,304,130.66 |
| Weekday + month Poisson baseline | 1,304,699.78 |
| Weekday + month Hawkes baseline | 1,307,621.20 |

The seasonal Hawkes improves held-out log-likelihood by 2,921.42 over the calendar Poisson baseline (about 0.0123 log-likelihood units per test event) and by 3,490.55 over the homogeneous Poisson baseline. Its fitted branching ratio is 0.4752 (profile-likelihood 95% interval 0.4607-0.4898) and decay is 10.089 per day (an exponential half-life of about 1.65 hours). The fitted mean immigrant baseline is about 372 events/day; self-excitation raises the stationary mean implied by the model to roughly the observed training rate.

This is evidence that event-history dependence can improve this particular held-out fit after weekday/month adjustment. It is not evidence of criticality: the estimated branching ratio is well below one, and the result may still reflect omitted short-term structure, reporting artifacts, or the minute-rounded observation process. It should be checked across categories, time windows, and additional cities. The day-level calendar baseline also does not account for spatial heterogeneity or changes in reporting.
