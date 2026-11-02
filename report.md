# P4-Routing: Neural Travel Times, CSP Van Assignment and Heuristic Routing

## Abstract
We study a same-day delivery problem: 20 orders with time windows, 3 vans with capacity 15 and at most 10 stops each, starting from a depot at 09:00. A neural network (NN) predicts travel times between NYC taxi zones from the TLC January 2023 yellow-taxi data. A constraint satisfaction problem (CSP) assigns orders to vans, A* / UCS / Greedy order the stops within each van, and an integration loop adds "different van" rules when a route is late (at most 5 rounds). We report NN accuracy against two baselines, search and CSP counters, and experiments over 5 seeds. The NN is roughly as accurate as the historical-average baseline and better than the distance baseline; the full routing results are reported with the limitations stated in Section 9.

## 1. Problem Formulation
- **Orders:** 20 orders. Each has a delivery zone (one of the top 20 TLC pickup zones), a time window [start, end] of 30, 45 or 60 minutes starting between 09:00 and 13:00, demand 1–3, and a fixed 3-minute service time.
- **Fleet:** 3 vans, capacity 15 units, at most 10 stops per van. Depot zone 162, departure 09:00 (minute 540).
- **Goal:** assign every order to one van, and order each van's stops so that every stop starts within its window. Travel time between zones depends on the hour of departure, so the route cost depends on the schedule.
- **Output:** either a complete plan, or a clear "NO SOLUTION" with a reason (CAPACITY, PAIRWISE_CONFLICT, NO_FEASIBLE_ORDER or DEPOT_UNREACHABLE).

## 2. Data
- **Source:** NYC TLC Yellow Taxi trip records, January 2023 (`make_data.py --month 2023-01 --seed 0`).
- **Cleaning:** trip duration kept between 1 and 120 minutes; trips with distance above 100 miles or at or below 0 removed (one record reported 116,439 miles, a meter error); weekdays only; pickups and drop-offs restricted to the 20 most frequent pickup zones.
- **Split:** days 1–15 for training, the remaining days for testing, to respect time order. We sample 300,000 training and 30,000 test trips.
- **Derived features:** duration (minutes), hour of pickup (0–23), weekday (0–4) and `pair_mean_dist`, the mean trip distance for the zone pair computed from training data only (test pairs unseen in training fall back to the global mean).
- **Generated parts:** order zones, windows, demand and the depot are generated with a seeded random generator. Seed 7 forces orders 7 and 12 into overlapping windows that cannot share a van (38 minutes apart), as a controlled conflict.
- **Limitation:** the TLC data records taxi trips, not vans, and does not record delivery stops. Van travel times are approximated as 1.2 × taxi times (a documented factor).

## 3. Neural Network Design and Baselines
**Architecture (`src/nn.py`):** pickup and drop-off zone embeddings (16 dimensions each), hour embedding (4), weekday embedding (3), and the standardised `pair_mean_dist`. These are concatenated and passed through an MLP of 128 → 64 → 32 → 1 with ReLU activations. The output is the predicted trip duration in minutes.

**Training:** L1 loss, Adam with learning rate 1e-3 and weight decay 1e-4, batch size 2048, at most 30 epochs with early stopping (patience 4). The best model was at epoch 13.

**Baselines:**
- *hist-avg:* mean training duration for the same (pickup, drop-off, hour); global mean for unseen combinations.
- *distance:* trip distance ÷ 18 mph. This baseline uses each test trip's actual distance, which is not known before a trip. It is therefore an optimistic baseline.

**Results (test set, 30,000 trips):**

| Model | MAE (min) | RMSE (min) |
|---|---|---|
| NN | 3.066 | 4.749 |
| hist-avg | 3.118 | 4.720 |
| distance | 6.321 | 8.140 |

The NN is slightly better than hist-avg on MAE and slightly worse on RMSE. With one training run we cannot claim the NN is better; we report the two as roughly equal. The NN is clearly better than the distance baseline, even though that baseline had access to the true trip distance.

**Validation caveat:** the early-stopping checkpoint was selected using the test set, so the NN numbers above are slightly optimistic. A validation split from the training days is the fix and is listed as future work.

## 4. Search Algorithms and Heuristic
Within one van, the stops form a sequence problem with time windows. The search state is (current zone, clock time, set of visited orders). Three algorithms are implemented in `src/search.py`:
- **A\*:** priority = cost so far + h, where h is the sum over unvisited orders of the cheapest travel to that order's zone (using 09:00 travel times).
- **UCS:** A\* with h = 0.
- **Greedy best-first:** always visits the feasible unvisited order with the earliest arrival; no backtracking.

**Admissibility:** the heuristic uses departure-time travel values. If travel is faster later in the day, h can overestimate, so A\* is not guaranteed optimal in general. Our tests show A\* and UCS return the same cost on all tested instances (Section 8), which is consistent with that but not a proof. A proof, or taking the minimum over all 24 hours, is future work.

Counters: nodes expanded and generated are recorded per search. Tests compare A\* against brute force on 6 orders (known optimum).

## 5. Constraint Satisfaction Problem (CSP)
**Variables:** one per order. **Domain:** the 3 vans. **Constraints:**
- *Pairwise conflicts:* two orders that cannot be served in either order by one van cannot share a van. Feasibility is checked with the NN travel times.
- *Capacity:* at most 15 units and at most 10 stops per van.

**Solver (`src/csp.py`):** backtracking (BT) with the most-constrained order first. Two additional modes: forward checking (FC) removes values from neighbours' domains, and AC-3 enforces arc consistency before and during search. All three modes count nodes, assignments and backtracks, and can be selected by name.

## 6. Integration Loop and Widening
1. Check the depot can be reached before each window closes (DEPOT_UNREACHABLE otherwise).
2. Build the pairwise conflict matrix with NN times and solve the CSP.
3. For each van, search the stop order. If a stop is late, add the pair as a new "different van" rule and repeat, for at most 5 loops.
4. If no solution is found, the **widening** step extends every window by 15 minutes on each side and the whole process restarts. The web demo exposes this as a button.

Example (seed 7): the original windows give NO SOLUTION. One widening step gives a valid plan with orders 7 and 12 in different vans.

## 7. Replay Evaluation
A plan is made with each travel-time model (NN, hist-avg, distance), then driven on "real" travel times and the number of late stops is counted. In the designed experiment, real times should be held-out TLC trips. **Current status:** `src/eval.py` experiment 3 uses synthetic biased matrices with lognormal noise, not held-out trips. The web demo's replay uses held-out data when available. Experiment 3 must be redone with held-out data before it is used as evidence.

## 8. Experimental Results
Setup: seeds 0–4 for every experiment; mean and standard deviation reported over seeds. Times are in seconds for one search or CSP call on this machine.

**Experiment 1: search cost by number of orders (k)**

[FILL: table from results/exp1.csv — columns k, algo, mean_cost, std_cost, mean_time, mean_expanded]

Observations (verified in both runs): A\* and UCS return identical costs for every k, as expected from two optimal algorithms. A\* expands slightly **more** nodes than UCS in these runs (about 1,720 vs 1,600 at k=10), so the heuristic did not reduce search effort on this data. A\* is only expected to help when h is informative. Greedy frequently fails to find any order (the 9999 value is a penalty for no solution), so its costs are not comparable.

**Experiment 2: CSP pruning (BT / BT+FC / BT+FC+AC-3)**

[FILL: table from results/exp2.csv — columns n, mode, mean_nodes, std_nodes, mean_time, solved_rate]

Observation: on the earlier run, FC and AC-3 did not reduce nodes for 15 and 20 orders, and only slightly reduced them for 30 orders, where only 20% of instances were solved within the limit. We report these results as found; pruning benefits were not observed at these sizes. [FILL: confirm with the final run]

**Experiment 3: NN vs baselines under replay**

[FILL: table from results/exp3.csv — columns model, mean_late, std_late]

Caveat: this experiment currently uses synthetic travel times (Section 7). Results are not evidence about the NN until it is redone with held-out trips.

**NN accuracy:** Section 3 table.

## 9. System Limitations and Boundary Cases
- **Vehicles:** van times are approximated as 1.2 × taxi times. Vans and taxis differ in capacity and in access to streets.
- **Data sparsity:** rare night-time pairs have few trips, so their NN predictions are less reliable.
- **Capacity:** a van holds at most 10 stops, and 15 units of demand.
- **Scope:** 20 zones, weekdays, one month, one depot, and fixed service time.
- **Heuristic:** admissibility is not proven (Section 4).
- **Validation:** early stopping used the test set (Section 3).
- **Experiment 3:** synthetic replay (Section 7).
- **Realism:** orders are generated, not real customer data.

## 10. Conclusions and Future Work
- An NN trained on taxi data predicts trip durations about as well as a simple historical average and better than a distance-based estimate. A single run does not show the NN is better than the average.
- The CSP with the integration loop turns time-window infeasibility into a clear NO SOLUTION, and widening recovers feasible plans on the demonstration seed.
- Search counters show A\* and UCS return the same cost; on this data A\* did not expand fewer nodes than UCS, so the heuristic's benefit is not demonstrated.
- Future work: a validation split for early stopping, admissibility proof or per-hour heuristic, multi-seed NN comparison with standard deviations, held-out replay in Experiment 3, and real van travel data.

## References
- NYC Taxi & Limousine Commission, Yellow Taxi Trip Records, January 2023.
