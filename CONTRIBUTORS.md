# Individual Contribution Statement

**Name:** Jamal Hany El-Shenawy

I completed this project alone. My contributions:

- **Data pipeline:** `make_data.py` builds the TLC January 2023 dataset (cleaning, zone filtering, weekday split, zone-pair distance features) with a fixed seed, and removes bad distance readings.
- **Neural network:** `src/nn.py` trains the travel-time model, compares it with the historical-average and distance baselines, and exports the 20×20×24 time matrix.
- **Search and CSP:** `src/search.py` (A\*, UCS and Greedy with node counters) and `src/csp.py` (backtracking, forward checking, AC-3 and dynamic variable ordering).
- **Integration and evaluation:** the integration loop and widening strategy, and `src/eval.py` for the three experiments over five seeds.
- **Demo, tests and UI:** `run_demo.py`, `server.py`, `ui/index.html`, and `tests/test_routing.py` (14 tests: 5 normal, 4 edge, 3 impossible, and 2 Experiment 3 regression tests).
- **Report and documentation:** `report.md`, `README.md` and `DATA_DICTIONARY.md`.
