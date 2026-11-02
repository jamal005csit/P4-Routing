# P4-Routing Presentation and Live Demo Plan

**Total time: 15 minutes** — 10-minute presentation followed by a 5-minute live demonstration.

## 10-minute presentation

| Time | Slide | Content |
|---|---|---|
| 0:00–0:45 | 1. Problem | Same-day delivery, 20 orders, time windows, three vans, capacity and stop limits. |
| 0:45–1:30 | 2. Architecture | TLC data → NN travel-time estimates → CSP van assignment → route search → integration/repair. |
| 1:30–2:30 | 3. Dataset | TLC January 2023 source, cleaning, chronological split, derived fields, generated orders and taxi-to-van limitation. |
| 2:30–3:30 | 4. Neural model | Zone/time embeddings, MLP, fit/validation/test protocol, MAE/RMSE baselines. Show only regenerated test metrics. |
| 3:30–4:30 | 5. Search | A*, UCS and Greedy; state representation; expanded/generated counters; heuristic admissibility limitation. |
| 4:30–5:45 | 6. CSP | Variables/domains/constraints; BT vs BT+FC vs BT+FC+AC-3; nodes, assignments and backtracks. |
| 5:45–6:45 | 7. Integration | Conflict matrix, per-van route checks, repair loop, widening and NO SOLUTION behavior. |
| 6:45–8:45 | 8. Experiments | Experiment 1 search comparison, Experiment 2 pruning comparison, Experiment 3 held-out replay. Use fresh CSV/plots from five seeds. |
| 8:45–9:30 | 9. Limitations | Taxi proxy, generated orders, sparse pairs, heuristic proof, aggregated held-out replay. |
| 9:30–10:00 | 10. Conclusion | What works, what the experiments support, and next steps. |

## 5-minute live demonstration

1. **0:00–0:45 — Show the input.** Run `python run_demo.py --seed 7 --widen 0`; point out the generated orders, windows, depot and zones. Explain that the original instance is expected to be infeasible.
2. **0:45–2:00 — Compare CSP modes.** Show the printed BT, BT+FC and BT+FC+AC-3 node, assignment and backtrack counters. Explain that the conflict graph is computed from travel feasibility and no seed-specific conflict is hard-coded in the CSP.
3. **2:00–3:00 — Show recovery.** Run `python run_demo.py --seed 7 --widen 1`; show the resulting assignment and route status after widening.
4. **3:00–4:00 — Show search trace.** Point to A*, UCS and Greedy per-van expanded/generated counters and the route visualization in `results/demo.png`.
5. **4:00–5:00 — NN on versus off.** Explain the NN travel-time matrix path and the synthetic fallback warning. If time permits, show the NN matrix file and the regenerated NN-vs-baseline table. Do not present stale aggregate CSVs as current results.

## Rehearsal checklist

- Run the data-generation and NN-training commands before the presentation if showing real NN output.
- Regenerate all three experiments with at least five seeds and inspect raw per-seed CSVs before presenting numerical conclusions.
- Confirm `python run_demo.py --seed 7 --widen 0` and `python run_demo.py --seed 7 --widen 1` behave as expected on the presentation machine.
- Keep the generated `results/demo.png` and experiment plots ready.
- If real data or the NN checkpoint is unavailable, state that limitation and do not imply the synthetic fallback is a trained model.
