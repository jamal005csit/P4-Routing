# Technical Report: P4-Routing Architecture & Performance Analysis

---

## 1. Problem Formulation
Defines the core routing problem statement, constraints, and optimization objectives for the system.

## 2. Data & TLC Accuracy Limits
Details data preprocessing pipelines, dataset sources, and empirical upper bounds on accuracy stemming from TLC (TLC Taxi & Limousine Commission) sampling limitations.

## 3. Neural Network Design & Baselines
- **Architecture**: Core network parameters, input/output embeddings, and layer configurations.
- **Baseline Models**: Standard benchmarks used to measure performance gains.

## 4. Search Algorithms & Heuristic Admissibility
- **A* Search**: Proof of heuristic admissibility and optimality guarantees.
- **Uniform Cost Search (UCS)**: Baseline unguided search strategy.
- **Greedy Best-First Search**: Performance vs. accuracy tradeoffs under computational constraints.

## 5. Constraint Satisfaction Problem (CSP) Formulation
Implementation and benchmarking of constraint propagation strategies:
- **Backtracking (BT)**
- **Forward Checking (FC)**
- **Arc Consistency (AC-3)**

## 6. Integration Loop & Widening Strategy
- Search and CSP integration loop constrained to $\le 5$ iterations.
- Adaptive widening parameters to prevent search space explosion.

## 7. Replay Evaluation Methodology
Framework for offline simulation replay, metrics collection, and variance controls.

## 8. Experimental Results (Exp 1–3)
- **Experiment 1**: Baseline performance comparison across algorithms.
- **Experiment 2**: Robustness across random seeds ($0 \text{--} 4$).
- **Experiment 3**: Stress-testing scaling constraints and runtime limits.

## 9. System Limitations & Boundary Cases
- **Vehicle Heterogeneity**: Capacity scaling factors (Taxi $\neq$ Van, ratio $1.2\times$).
- **Data Sparsity**: Long-tail variance in rare nighttime origin-destination pairs.
- **Capacity Bounds**: Maximum limit of 10 stops per van.

## 10. Conclusions & Future Directions
Summary of key findings, trade-off insights, and recommended next steps for real-world deployment.
