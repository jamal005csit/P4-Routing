# Project Contributors

This document details individual contributions to the P4-Routing repository.

---

## Member A: Data Pipeline & Neural Network Design

**Primary Focus:** Data Processing, Feature Engineering, Neural Network Architecture

* **Data Engineering & TLC Preprocessing:** Designed and implemented `make_data.py` to process dynamic NYC Taxi and Limousine Commission (TLC) trip datasets. Built robust temporal and spatial feature extraction routines, handling missing records, spatial aggregation into zone nodes, and noise filtering.
* **Neural Network Architecture:** Developed the core spatial-temporal model in `src/nn.py`. Built high-dimensional embeddings for spatial nodes, time-of-day, and day-of-week context, paired with a residual Multi-Layer Perceptron (MLP) backbone to accurately predict live edge traversal weights and dynamic transit times.
* **Model Training & Pipeline Optimization:** Implemented loss metrics, training loops, evaluation hooks, and hyperparameter tuning setups to guarantee fast model convergence and continuous baseline comparison.

---

## Member B: Graph Search & CSP Solver Implementation

**Primary Focus:** Heuristic Pathfinding, Constraint Satisfaction Problem (CSP) Formulations

* **Graph Search Algorithms:** Implemented and benchmarked core pathfinding strategies ($A^*$ Search, Uniform Cost Search, and Greedy Best-First Search). Verified heuristic admissibility ($h(n) \le h^*(n)$) using physical speed limits and spatial bounding to guarantee optimal route generation.
* **CSP Modeling & Algorithms:** Formulated vehicle-to-request assignments as a strict CSP adhering to pickup-before-dropoff, capacity ceilings ($\le 10\text{ stops/van}$), and delivery time window constraints. Developed three solver variants: standard Backtracking (BT), Forward Checking (FC), and Arc Consistency (AC-3).
* **Performance Benchmarking:** Optimized constraint checking and arc pruning mechanisms to minimize node expansions and prevent computational bottlenecks under heavy workload scenarios.

---

## Member C: System Integration, Evaluation & Interactive UI

**Primary Focus:** Loop Integration, Trace-Driven Replay, Interactive Demo & Testing

* **Integration Loop & Widening Strategy:** Designed the master routing loop coupling neural dynamic cost updates, $A^*$ route planning, and CSP schedule verification within a strict ceiling ($\le 5\text{ iterations}$). Built the adaptive relaxation/widening mechanism triggered during initial assignment failures.
* **Replay Evaluation Engine:** Authored `src/eval.py` to run multi-seed validation ($0\text{--}4$) across experiments 1–3. Built trace-driven simulation metrics tracking total vehicle miles traveled (VMT), request latency, and throughput (ms/request).
* **Interactive Demo & Test Suite:** Developed `run_demo.py` for dynamic visualization and live scenario testing. Configured the automated test suite (`pytest -q`) to ensure pipeline reliability and regression prevention.
