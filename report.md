# P4-Routing: Neural Travel Times, CSP Van Assignment and Heuristic Routing

## Abstract
P4-Routing models a same-day delivery problem with 20 generated orders, three vans, capacity constraints and delivery time windows. A neural network predicts travel durations from NYC TLC January 2023 yellow-taxi records. A constraint satisfaction problem (CSP) assigns orders to vans; A*, uniform-cost search (UCS) and greedy best-first search order each van's stops. This report records the latest local run using five seeds (0–4). The results show that A* and UCS return matching costs on the tested instances, while greedy is substantially faster and nearly matches their costs. AC-3 reduces CSP search nodes on larger instances but increases runtime. The neural network slightly improves MAE over the historical-average baseline, but not RMSE. The integrated replay experiment does not distinguish the three planning matrices under its current lateness metric.

## 1. Problem
Each generated order has a destination zone, a time window, a demand and a three-minute service time. Three vans depart from depot zone 162 at 09:00. Each van has capacity 15 and a maximum of 10 stops. The system assigns every order to a van and searches for a feasible visit sequence. If no plan exists, the integration layer returns a `NO SOLUTION` status and a reason.

## 2. Data
The real source is the NYC TLC Yellow Taxi January 2023 trip record. `make_data.py` filters trip durations to 1–120 minutes, removes invalid/non-positive or over-100-mile distances, retains weekdays, and restricts pickup and drop-off zones to the 20 most frequent pickup zones. It derives duration, pickup hour, weekday, day-of-month and pickup/drop-off-pair mean distance. Pair means are fitted from training data only; unseen test pairs use the training global mean.

The split is chronological: the first 15 available weekdays are training and the remaining days are held out. The script samples up to 300,000 training and 30,000 test rows using fixed seeds. Order zones, windows and demands are generated separately with a seeded generator. Seed 7 deliberately creates a reproducible time-window conflict for orders 7 and 12. See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for field definitions.

The reported data-generation run used:
- Training rows: 300,000
- Held-out test rows: 30,000
- Top-20 pickup zones: 237, 132, 161, 236, 162, 186, 142, 230, 138, 170, 163, 239, 234, 141, 48, 140, 68, 164, 238, 107
- Training data SHA-256: `1ded704ce9ba546acc563f4c9ef0eb52`

Taxi travel is only a proxy for van travel. The current pipeline applies a factor of 1.2; the project does not contain observed van GPS data.

## 3. Neural Network and Baselines
`src/nn.py` uses embeddings for pickup zone (16), drop-off zone (16), hour (4) and weekday (3), concatenated with standardized pair mean distance. An MLP with hidden widths 128, 64 and 32 predicts duration. Training uses L1 loss, Adam, a batch size of 2048 and a maximum of 30 epochs.

**Evaluation protocol:** the training partition is split into fit and validation subsets using a fixed seed. Normalization statistics are fitted on the fit subset. Early stopping and checkpoint selection use validation MAE only. The held-out test partition is evaluated after checkpoint selection; it is not used for epoch-by-epoch model selection.

### Held-out test metrics

| Model | MAE (minutes) | RMSE (minutes) |
|---|---:|---:|
| Neural network | 3.0792 | 4.7839 |
| Historical average | 3.1263 | 4.7322 |
| Distance baseline | 6.3212 | 8.1400 |

The neural network reduces MAE by approximately 1.51% relative to the historical-average baseline, but its RMSE is approximately 1.09% higher. It substantially outperforms the distance baseline on both metrics. Therefore, the results support a modest improvement in average absolute error, not an unqualified claim that the neural network is the best predictor under every metric.

The historical baseline is the mean for (pickup zone, drop-off zone, hour), falling back to the fit-set global mean. The distance baseline uses observed test-trip distance, which is not available to a pre-trip dispatcher; it is an oracle-style reference and must be interpreted accordingly.

## 4. Search: Experiment 1
Experiment 1 compares A*, UCS and greedy best-first search across route sizes, using five seeds (0–4). The table reports the mean route cost, mean runtime in seconds and fraction of runs solved. The reported mean cost is the evaluator's aggregate for each algorithm/size.

| Orders | Algorithm | Mean cost | Mean runtime (s) | Solved rate |
|---:|---|---:|---:|---:|
| 6 | A* | 204.979 | 0.000472 | 0.60 |
| 6 | UCS | 204.979 | 0.000342 | 0.60 |
| 6 | Greedy | 204.979 | 0.000040 | 0.60 |
| 7 | A* | 210.457 | 0.001043 | 0.60 |
| 7 | UCS | 210.457 | 0.000673 | 0.60 |
| 7 | Greedy | 210.457 | 0.000061 | 0.60 |
| 8 | A* | 199.218 | 0.002052 | 0.40 |
| 8 | UCS | 199.218 | 0.001943 | 0.40 |
| 8 | Greedy | 199.218 | 0.000091 | 0.40 |
| 9 | A* | 213.000 | 0.006143 | 0.40 |
| 9 | UCS | 213.000 | 0.004639 | 0.40 |
| 9 | Greedy | 213.000 | 0.000105 | 0.40 |
| 10 | A* | 210.000 | 0.010319 | 0.40 |
| 10 | UCS | 210.000 | 0.007819 | 0.40 |
| 10 | Greedy | 210.031 | 0.000091 | 0.40 |

A* and UCS report matching mean costs for every tested route size. This is consistent with optimal search behavior on these tested cases, but it does not prove the A* heuristic is admissible. Greedy is much faster in this small benchmark and matches the reported mean cost through nine orders; at ten orders its mean cost is slightly higher. The solved rate falls from 60% at six and seven orders to 40% at eight through ten orders. This feasibility rate is a more important practical limitation than the small runtime differences. The raw results should be consulted to determine whether each unsuccessful run was infeasible, timed out, or failed for another reason.

The A* heuristic is not formally proven admissible because travel times vary by departure hour. Agreement with UCS on selected instances is not a proof of optimality.

## 5. CSP: Experiment 2
Experiment 2 compares backtracking (BT), backtracking with forward checking (BT+FC), and BT+FC with AC-3. Five seeds were evaluated at each problem size. Runtime is in seconds.

| Variables | Mode | Mean runtime | Mean nodes | Mean assignments | Mean backtracks | Solved rate | Timeout rate |
|---:|---|---:|---:|---:|---:|---:|---:|
| 15 | BT | 0.000323 | 16.0 | 21.8 | 6.8 | 1.00 | 0.00 |
| 15 | BT+FC | 0.000301 | 16.0 | 21.8 | 6.8 | 1.00 | 0.00 |
| 15 | BT+FC+AC-3 | 0.000626 | 16.0 | 21.8 | 6.8 | 1.00 | 0.00 |
| 20 | BT | 0.154533 | 8,012.6 | 16,018.8 | 16,006.8 | 0.60 | 0.40 |
| 20 | BT+FC | 0.138564 | 8,012.6 | 16,018.8 | 16,006.8 | 0.60 | 0.40 |
| 20 | BT+FC+AC-3 | 0.400538 | 7,322.6 | 14,638.6 | 14,626.6 | 0.60 | 0.40 |
| 30 | BT | 0.450142 | 16,008.4 | 40,003.2 | 39,997.2 | 0.20 | 0.80 |
| 30 | BT+FC | 0.475223 | 16,008.4 | 40,003.2 | 39,997.2 | 0.20 | 0.80 |
| 30 | BT+FC+AC-3 | 0.801118 | 10,567.6 | 26,832.6 | 26,826.6 | 0.20 | 0.80 |

At 15 variables all modes solve all runs. At 20 variables, AC-3 reduces mean visited nodes by about 8.6% relative to BT; at 30 variables, it reduces mean nodes by about 34.0%. However, AC-3 has higher measured runtime at both sizes and does not improve the solved rate or timeout rate in these runs. BT and BT+FC have identical reported counters at all three sizes, so forward checking did not produce measurable pruning in this evaluation. This should be investigated before claiming a benefit from FC; possible explanations include the generated constraint structure or how propagation is counted.

The evaluation uses a 20,000-node cap and a one-second per-solver time limit. A timed-out run is not a proof that the CSP instance is infeasible. At 30 variables, 80% of runs time out, so conclusions about large-instance solvability are limited by the configured budget.

## 6. Integrated Replay: Experiment 3

Experiment 3 plans with NN, historical-average and distance travel-time matrices, then replays the optimized per-van visit sequences against an independent held-out TLC test-set matrix. The evaluator now preserves the actual route order returned by A* in `solve_all`; it no longer sorts assigned orders by time-window start. The replay records late-delivery count, total late minutes, maximum lateness, on-time rate, route order and solve status for each seed/model attempt.

The solved rate is calculated over **all attempted runs**, including failed plans. Lateness and on-time metrics are averaged over successful plans only; the attempt count and solved-run count are reported alongside them so that performance and feasibility are not conflated.

**Important:** the previously recorded Experiment 3 table (0.3333 mean late deliveries and 100% solved rate for all three models) was produced by the earlier evaluator, which replayed orders in sorted time-window order and calculated solved rate only after filtering failures. Those values are therefore superseded and must not be presented as valid results. Regenerate Experiment 3 with the corrected evaluator before reporting numerical comparisons:

```bash
python -m src.eval --exp 3 --seeds 0,1,2,3,4
```

The corrected raw output is written to `results/exp3_raw.csv`, including the optimized route sequence for each successful run. The aggregate output is written to `results/exp3.csv`. Experiment 3 still uses aggregated held-out zone-pair-hour travel durations, not individual van telemetry or live traffic.

## 7. Reproducibility
The reported local run used:

```bash
python make_data.py --month 2023-01 --seed 0
python -m src.nn
python -m src.eval --exp 1,2,3 --seeds 0,1,2,3,4
pytest -q
```

The project previously passed 12 tests. Two additional regression tests now cover ordered-route replay and solved-rate aggregation; run `pytest -q` after pulling the latest commits to verify all 14 tests.

The evaluator writes per-seed observations and aggregate means/standard deviations to the `results/` directory. Preserve the raw CSVs alongside aggregate tables so that solved rates, timeouts and seed-level variation can be audited. The numerical results in this report are from the latest run reported by the project author; rerun the commands above after any code, data, or configuration change before treating them as current.

## 8. Limitations and Follow-up Work
- Taxi travel times approximate van travel; a fixed factor cannot model vehicle class, road access, loading, parking or delivery dwell time.
- Rare zone/hour pairs have limited observations and use global fallback values.
- Orders and delivery constraints are generated, not observed customer deliveries.
- The A* heuristic's admissibility is not established.
- The distance baseline uses information unavailable to a dispatcher before a trip and is an oracle-style comparator.
- NN MAE is slightly better than historical average, but NN RMSE is slightly worse.
- Experiment 1 solved rates are only 40–60% in the tested sizes; failure reasons should be separated into timeout, infeasibility and search failure.
- Experiment 2 has high timeout rates for 20 and 30 variables. AC-3 reduces node counts but costs more runtime; FC's identical counters to BT warrant investigation.
- The earlier Experiment 3 aggregates are invalid because the evaluator replayed sorted time windows and excluded failed attempts from the solved-rate denominator. Re-run Experiment 3 with the corrected ordered-route replay before making model comparisons.
- Experiment 3 evaluates aggregated held-out zone-pair-hour averages, not individual van telemetry or live traffic.

## 9. Conclusions
The latest run verifies that the full test suite passes and that the pipeline produces results across all three experiments. A* and UCS have matching reported route costs on the tested cases, while greedy is substantially faster with only a small cost difference at the largest tested size. AC-3 reduces CSP search nodes on larger instances but increases runtime and does not improve solve rates under the current limits. The NN offers a small MAE improvement over the historical-average baseline, but its RMSE is slightly worse. Corrected Experiment 3 results are pending and no conclusion about comparative delivery timeliness should be drawn from the superseded table. The project is therefore functional, but the current experiments do not justify a broad claim that the NN or propagation-enhanced CSP is superior. The next priority is to investigate search failures/timeouts, explain why FC has no measured effect, and make the replay metric sensitive enough to distinguish route quality.

## References
- NYC Taxi & Limousine Commission, Yellow Taxi Trip Records, January 2023.