# P4-Routing: Neural Travel Times, CSP Van Assignment and Heuristic Routing

## Abstract

P4-Routing models a same-day delivery problem with 20 generated orders, three vans, capacity constraints and delivery time windows. A neural network predicts travel durations from NYC TLC January 2023 yellow-taxi records. A constraint satisfaction problem (CSP) assigns orders to vans; A*, uniform-cost search (UCS) and greedy best-first search order each van's stops.

This report reflects the aggregate result files currently committed in `results/exp1.csv`, `results/exp2.csv` and `results/exp3.csv`, each based on five seeds (0–4). A* and UCS have matching mean costs in the tested route sizes. Greedy is substantially faster and has matching mean cost through nine orders, with a small cost increase at ten orders. AC-3 reduces CSP search effort for the larger tested instances but increases runtime and does not improve solved rates. The neural network has slightly lower MAE than the historical-average baseline but slightly higher RMSE. In the corrected integrated replay, NN and historical average both achieve zero mean late deliveries among successful plans; the distance model has more lateness. All three models solve 60% of attempted plans.

## 1. Problem

Each generated order has a destination zone, a time window, a demand and a three-minute service time. Three vans depart from depot zone 162 at 09:00. Each van has capacity 15 and a maximum of 10 stops. The system assigns every order to a van and searches for a feasible visit sequence. If no plan exists, the integration layer returns a `NO SOLUTION` status and a reason.

## 2. Data

The real source is the NYC TLC Yellow Taxi January 2023 trip record. `make_data.py` filters trip durations to 1–120 minutes, removes invalid/non-positive or over-100-mile distances, retains weekdays, and restricts pickup and drop-off zones to the 20 most frequent pickup zones. It derives duration, pickup hour, weekday, day-of-month and pickup/drop-off-pair mean distance. Pair means are fitted from training data only; unseen test pairs use the training global mean.

The split is chronological: the first 15 available weekdays are training and the remaining days are held out. The script samples up to 300,000 training and 30,000 test rows using fixed seeds. Order zones, windows and demands are generated separately with a seeded generator. Seed 7 deliberately creates a reproducible time-window conflict for orders 7 and 12. See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for field definitions.

The reported data-generation run used:
- Training rows: 300,000
- Held-out test rows: 30,000
- Top-20 pickup zones: 237, 132, 161, 236, 162, 186, 142, 230, 138, 170, 163, 239, 234, 141, 48, 140, 68, 164, 238, 107

The earlier report listed `1ded704ce9ba546acc563f4c9ef0eb52` as a training-data SHA-256 value. That value is 32 hexadecimal characters and therefore is not a valid SHA-256 digest. Do not use it as a verified SHA-256 checksum. The data-generation script's checksum implementation should be corrected and the report updated with a newly computed digest before presenting a dataset-integrity claim.

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

The neural network reduces MAE by approximately 1.51% relative to the historical-average baseline, but its RMSE is approximately 1.09% higher. It substantially outperforms the distance baseline on both metrics. These results support a modest improvement in average absolute error, not an unqualified claim that the neural network is the best predictor under every metric.

The historical baseline is the mean for (pickup zone, drop-off zone, hour), falling back to the fit-set global mean. The distance baseline uses observed test-trip distance, which is not available to a pre-trip dispatcher; it is an oracle-style reference and must be interpreted accordingly.

## 4. Search: Experiment 1

Experiment 1 compares A*, UCS and greedy best-first search across route sizes, using five seeds (0–4). Values below are means from the currently committed `results/exp1.csv`. The aggregate file also contains standard deviations for cost, runtime, expanded states and generated states.

| Orders | Algorithm | Mean cost | Mean runtime (s) | Solved rate |
|---:|---|---:|---:|---:|
| 6 | A* | 204.979 | 0.000591 | 0.60 |
| 6 | UCS | 204.979 | 0.000392 | 0.60 |
| 6 | Greedy | 204.979 | 0.000048 | 0.60 |
| 7 | A* | 210.457 | 0.001074 | 0.60 |
| 7 | UCS | 210.457 | 0.000833 | 0.60 |
| 7 | Greedy | 210.457 | 0.000056 | 0.60 |
| 8 | A* | 199.218 | 0.002067 | 0.40 |
| 8 | UCS | 199.218 | 0.001621 | 0.40 |
| 8 | Greedy | 199.218 | 0.000067 | 0.40 |
| 9 | A* | 213.000 | 0.004278 | 0.40 |
| 9 | UCS | 213.000 | 0.003713 | 0.40 |
| 9 | Greedy | 213.000 | 0.000084 | 0.40 |
| 10 | A* | 210.000 | 0.009627 | 0.40 |
| 10 | UCS | 210.000 | 0.007934 | 0.40 |
| 10 | Greedy | 210.031 | 0.000088 | 0.40 |

A* and UCS report matching mean costs for every tested route size. This is consistent with optimal search behavior on these tested cases, but it does not prove the A* heuristic is admissible. Greedy is much faster in this small benchmark and matches the reported mean cost through nine orders; at ten orders its mean cost is slightly higher. The solved rate falls from 60% at six and seven orders to 40% at eight through ten orders. This feasibility rate is a more important practical limitation than the small runtime differences.

The A* heuristic is not formally proven admissible because travel times vary by departure hour. Agreement with UCS on selected instances is not a proof of optimality. The aggregate CSV contains means and standard deviations, but the raw per-seed observations are generated locally and are not currently committed.

## 5. CSP: Experiment 2

Experiment 2 compares backtracking (BT), backtracking with forward checking (BT+FC), and BT+FC with AC-3. Five seeds were evaluated at each problem size. Runtime is in seconds. The table reports mean runtime and mean node count; the aggregate CSV contains standard deviations and additional counters.

| Variables | Mode | Mean runtime (s) | Mean nodes | Solved rate | Timeout rate |
|---:|---|---:|---:|---:|---:|
| 15 | BT | 0.000285 | 16.0 | 1.00 | 0.00 |
| 15 | BT+FC | 0.000240 | 16.0 | 1.00 | 0.00 |
| 15 | BT+FC+AC-3 | 0.000493 | 16.0 | 1.00 | 0.00 |
| 20 | BT | 0.173835 | 8,012.6 | 0.60 | 0.40 |
| 20 | BT+FC | 0.156959 | 8,012.6 | 0.60 | 0.40 |
| 20 | BT+FC+AC-3 | 0.400418 | 6,135.2 | 0.60 | 0.40 |
| 30 | BT | 0.518295 | 16,008.4 | 0.20 | 0.80 |
| 30 | BT+FC | 0.552306 | 16,008.4 | 0.20 | 0.80 |
| 30 | BT+FC+AC-3 | 0.800546 | 9,514.4 | 0.20 | 0.80 |

At 15 variables all modes solve all runs. At 20 variables, AC-3 reduces mean visited nodes by about 23.4% relative to BT; at 30 variables, it reduces mean nodes by about 40.6%. However, AC-3 has higher measured runtime at both sizes and does not improve the solved rate or timeout rate in these runs. BT and BT+FC have identical reported node, assignment and backtrack counters at all three sizes, so forward checking did not produce measurable pruning in this evaluation. This should be investigated before claiming a benefit from FC; possible explanations include the generated constraint structure or how propagation is counted.

The evaluation uses a 20,000-node cap and a one-second per-solver time limit. A timed-out run is not a proof that the CSP instance is infeasible. At 30 variables, 80% of runs time out, so conclusions about large-instance solvability are limited by the configured budget.

## 6. Integrated Replay: Experiment 3

Experiment 3 plans with NN, historical-average and distance travel-time matrices, then replays the optimized per-van visit sequences against an independent held-out TLC test-set matrix. The evaluator preserves the actual route order returned by A* in `solve_all`; it does not sort assigned orders by time-window start. The replay records late-delivery count, total late minutes, maximum lateness, on-time rate, route order and solve status for each seed/model attempt.

The solved rate is calculated over **all attempted runs**, including failed plans. Lateness and on-time metrics are averaged over successful plans only. The attempt count and solved-run count are reported alongside them so that planning feasibility and delivery performance are not conflated.

### Corrected replay aggregates

| Model | Mean late deliveries | Std. dev. late deliveries | Mean late minutes | Mean max lateness (min) | Mean on-time rate | Solved rate |
|---|---:|---:|---:|---:|---:|---:|
| NN | 0.00 | 0.00 | 0.00 | 0.00 | 100.00% | 60% |
| Historical average | 0.00 | 0.00 | 0.00 | 0.00 | 100.00% | 60% |
| Distance | 1.67 | 0.58 | 12.03 | 11.02 | 91.67% | 60% |

Each model has five attempts and three successful plans. Lateness and on-time metrics above are calculated over the successful plans; solved rate uses all five attempts. The current aggregate CSV reports a standard deviation only for late-delivery count, not for late minutes, maximum lateness or on-time rate. Those additional dispersion statistics should not be inferred from the available aggregate file.

On these five-seed results, NN and historical average tie on the reported replay metrics. The distance baseline has higher lateness, but it is an oracle-style comparator based on observed trip distance and is not equivalent to an operational pre-trip predictor. These results do not demonstrate that NN improves integrated delivery timeliness over historical average.

Experiment 3 evaluates aggregated held-out zone-pair-hour travel durations, not individual van telemetry or live traffic. Raw per-seed replay observations, including route sequences, are generated as `results/exp3_raw.csv` but are not currently committed.

## 7. Reproducibility

The reported pipeline commands are:

```bash
python make_data.py --month 2023-01 --seed 0
python -m src.nn
python -m src.eval --exp 1,2,3 --seeds 0,1,2,3,4
pytest -q
```

The evaluator writes per-seed observations and aggregate means/standard deviations to the `results/` directory. The current repository commits aggregate files `exp1.csv`, `exp2.csv` and `exp3.csv`; the corresponding `*_raw.csv` files are generated locally and are not automatically committed. Commit the raw CSVs when seed-level auditability is required.

The report's numeric experiment tables were synchronized with the aggregate CSVs currently committed in the repository. Rerun the commands above after any code, data or configuration change before treating the results as current. Run `pytest -q` on the submission checkout to verify the current test suite.

## 8. Limitations and Follow-up Work

- Taxi travel times approximate van travel; a fixed factor cannot model vehicle class, road access, loading, parking or delivery dwell time.
- Rare zone/hour pairs have limited observations and use global fallback values.
- Orders and delivery constraints are generated, not observed customer deliveries.
- The A* heuristic's admissibility is not established.
- The distance baseline uses information unavailable to a dispatcher before a trip and is an oracle-style comparator.
- NN MAE is slightly better than historical average, but NN RMSE is slightly worse.
- Experiment 1 solved rates are only 40–60% in the tested sizes; failure reasons should be separated into timeout, infeasibility and search failure.
- Experiment 2 has high timeout rates for 20 and 30 variables. AC-3 reduces node counts but costs more runtime; FC's identical counters to BT warrant investigation.
- Experiment 3 has a 60% solved rate for all three models. Among successful plans, NN and historical average tie on the reported lateness metrics; the results do not establish an NN advantage in delivery timeliness.
- Experiment 3 evaluates aggregated held-out zone-pair-hour averages, not individual van telemetry or live traffic.
- The dataset checksum claim previously included in this report was invalid: the recorded value was 32 hexadecimal characters, not a SHA-256 digest. Correct the generator and recompute the checksum before making a verified integrity claim.

## 9. Conclusions

The committed aggregate results show that the project runs all three evaluation tracks, but they also reveal important limits. A* and UCS report matching mean route costs on the tested instances; greedy is much faster and nearly matches those costs. CSP AC-3 reduces search nodes for larger instances but increases runtime without improving solve rates under the current limits. Forward checking has no measurable effect in the reported counters. The NN modestly improves held-out MAE over historical average, while historical average has slightly better RMSE. In integrated replay, NN and historical average tie on lateness, and all models solve only 60% of attempts. The results therefore support a working integrated prototype, not a general claim that the NN or every propagation method is superior. Priorities for further work are to investigate search failures and timeouts, understand the lack of measurable FC pruning, commit raw per-seed experiment outputs, and correct the dataset checksum implementation.

## References

- NYC Taxi & Limousine Commission, Yellow Taxi Trip Records, January 2023.
