# P4-Routing: Neural Travel Times, CSP Van Assignment and Heuristic Routing

## Abstract
P4-Routing studies a same-day delivery problem with 20 generated orders, three vans, capacity constraints and delivery time windows. A neural network predicts travel durations from NYC TLC January 2023 yellow-taxi records. A constraint satisfaction problem assigns orders to vans; A*, uniform-cost search (UCS) and greedy search order each van's stops. The evaluation pipeline supports three experiments over configurable random seeds. Results must be regenerated after the validation, CSP and replay corrections described below; previously committed aggregate CSVs and report tables were produced by earlier versions and must not be treated as final evidence.

## 1. Problem
Each generated order has a destination zone, a time window, demand and a three-minute service time. Three vans depart from depot zone 162 at 09:00. Each van has capacity 15 and a maximum of 10 stops. The system assigns every order to a van and searches for a feasible visit sequence. If no plan exists, the integration layer returns a clear `NO SOLUTION` status and a reason.

## 2. Data
The real source is the NYC TLC Yellow Taxi January 2023 trip record. `make_data.py` filters trip durations to 1–120 minutes, removes invalid/non-positive or over-100-mile distances, retains weekdays, and restricts pickup and drop-off zones to the 20 most frequent pickup zones. It derives duration, pickup hour, weekday, day-of-month and pickup/drop-off-pair mean distance. Pair means are fitted from training data only; unseen test pairs use the training global mean.

The split is chronological: the first 15 available weekdays are training and the remaining days are held out. The script samples up to 300,000 training and 30,000 test rows using fixed seeds. Order zones, windows and demands are generated separately with a seeded generator. Seed 7 deliberately creates a reproducible time-window conflict for orders 7 and 12. See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for every field and its construction.

Taxi travel is only a proxy for van travel. The current pipeline applies a factor of 1.2; the project does not contain observed van GPS data.

## 3. Neural Network and Baselines
`src/nn.py` uses embeddings for pickup zone (16), drop-off zone (16), hour (4) and weekday (3), concatenated with standardized pair mean distance. An MLP with hidden widths 128, 64 and 32 predicts duration. Training uses L1 loss, Adam, a batch size of 2048 and a maximum of 30 epochs.

**Evaluation protocol:** the training partition is split into fit and validation subsets using a fixed seed. Normalization statistics are fitted on the fit subset. Early stopping and checkpoint selection use validation MAE only. The held-out test partition is evaluated after checkpoint selection. The test partition is not used for epoch-by-epoch model selection.

Baselines are a historical mean for (pickup zone, drop-off zone, hour), falling back to the fit-set global mean, and a distance-based estimate. The distance baseline uses observed test-trip distance, which is not available to a pre-trip dispatcher; it is an oracle-style reference and must be interpreted accordingly. Run `python -m src.nn` to regenerate `results/nn_vs_baseline.csv` and the NN travel-time matrix.

## 4. Search
The search state tracks current location, time and visited orders. A* ranks states by accumulated cost plus a heuristic; UCS uses a zero heuristic; greedy best-first uses earliest feasible arrival. Search returns expanded/generated counters. The test suite compares A* with brute force on a small instance with a known optimum.

The current A* heuristic is not proven admissible because travel times vary by departure hour. Agreement with UCS on selected cases is not a proof of optimality. A formal lower-bound proof or a time-independent lower-bound heuristic remains future work.

## 5. CSP
Each order is a variable whose domain is the set of vans. Constraints include pairwise incompatibility, van capacity and maximum stops. `src/csp.py` exposes three switchable modes: BT, BT+FC and BT+FC+AC-3. Variable selection uses minimum remaining values with degree and index tie-breaks. Forward checking removes the selected van from conflicting unassigned neighbours. AC-3 revises arcs for the binary inequality constraints represented by the conflict graph. Counters include visited nodes, assignments and backtracks.

The previous seed-specific hard-coded conflict in the CSP builder has been removed. The demonstration conflict is now defined only by the generated seed-7 time windows and computed travel feasibility, not by a hidden CSP exception.

## 6. Integration
The integration loop checks depot reachability, builds the pairwise conflict matrix, solves the CSP, searches each van's route, and can add different-van constraints when a route is late. It is bounded by a maximum number of repair rounds. Window widening is an explicit demo option, not evidence that the original instance was feasible.

## 7. Experiments and Reproducibility
Run five seeds and all three experiments from the repository root:

```bash
python make_data.py --month 2023-01 --seed 0
python -m src.nn
python -m src.eval --exp 1,2,3 --seeds 0,1,2,3,4
```

- **Experiment 1:** compares A*, UCS and greedy across order counts. The evaluator saves per-seed observations to `results/exp1_raw.csv`, aggregated means/standard deviations to `results/exp1.csv`, and a plot to `results/exp1.png`.
- **Experiment 2:** compares BT, BT+FC and BT+FC+AC-3. It saves per-seed nodes, assignments, backtracks, runtime, solved status and timeout status to `results/exp2_raw.csv`, plus aggregate statistics and a plot.
- **Experiment 3:** plans with NN, historical-average and distance matrices, then replays each plan against a separate held-out TLC test-set matrix. The replay matrix is built from held-out test trip durations grouped by pickup zone, drop-off zone and hour; it is not generated by perturbing the NN matrix. Missing held-out groups fall back to the held-out global mean. The evaluator refuses to run if the real train/test artifacts or trained NN matrix are absent. Raw per-seed outcomes and aggregate means/standard deviations are written to `results/exp3_raw.csv`, `results/exp3.csv` and `results/exp3.png`.

**Results status:** the existing committed `results/exp1.csv`, `results/exp2.csv`, `results/exp3.csv` and numerical tables from earlier report versions are stale relative to the current evaluator and solver. They must be regenerated with the commands above before numerical claims are made. Do not cite the old Experiment 3 advantage: it was produced by synthetic replay biased toward the NN. The corrected evaluator now uses held-out TLC observations, but its corrected numerical output has not yet been generated in this repository revision.

## 8. Limitations
- Taxi travel times approximate van travel; a fixed factor cannot model vehicle class, road access or load.
- Rare zone/hour pairs have limited observations and use global fallback values.
- Orders and delivery constraints are generated, not observed customer deliveries.
- The A* heuristic's admissibility is not established.
- Experiment 3 evaluates aggregated held-out zone-pair-hour averages, not individual van telemetry or live traffic.
- Experiment 2 has a 20,000-node cap and one-second per-solver time limit in the current evaluation script; timed-out runs must not be described as proven infeasible.

## 9. Conclusions
The repository contains a complete pipeline architecture for data preparation, neural travel-time prediction, search, CSP assignment, integration and evaluation. The corrected validation protocol avoids using test data for checkpoint selection; the CSP no longer inserts a seed-specific hard-coded conflict; and Experiment 3 now separates planning estimates from held-out replay observations. Final comparative conclusions should be written only after rerunning all experiments and reviewing the raw seed-level outputs.

## References
- NYC Taxi & Limousine Commission, Yellow Taxi Trip Records, January 2023.
