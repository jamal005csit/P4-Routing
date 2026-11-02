# Experiment Results

Run `python -m src.eval --exp 1,2,3 --seeds 0,1,2,3,4` from the repository root to regenerate raw CSVs, aggregate CSVs and plots.

- `exp1_raw.csv` and `exp1.csv`: route-search observations and aggregates.
- `exp2_raw.csv` and `exp2.csv`: CSP observations and aggregates.
- `exp3_raw.csv` and `exp3.csv`: held-out replay observations and aggregates.

Experiment 3's evaluator was corrected to replay the exact A* visit order returned by the integration layer and to calculate solved rate across all attempted runs. Previously committed Experiment 3 aggregates were produced by the superseded evaluator and must not be used as evidence. Regenerate Experiment 3 before citing any lateness or solved-rate values.

Raw CSVs are generated locally and are not automatically committed. Commit the regenerated raw and aggregate files if the submission requires reproducible result artifacts.
