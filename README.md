# P4-Routing

An end-to-end pipeline for dataset generation, neural network training, evaluation, and testing for P4 routing research.

---

## Quick Start

### 1. Installation
Install all required dependencies:
```bash
pip install -r requirements.txt
```
Run all commands from the repository root.

---

## Execution Workflow

Follow the commands below in order to prepare data, train the model, evaluate, and test the system.

### Data Generation
Generate dataset for a specific month and seed:
```bash
python make_data.py --month 2023-01 --seed 0
```

### Model Training
Train the neural network module:
```bash
python -m src.nn
```

### Evaluation
Evaluate the model across multiple experiments and seeds:
```bash
python -m src.eval --exp 1,2,3 --seeds 0,1,2,3,4
```

### Interactive Demo
Run the demonstration script:
```bash
python run_demo.py --seed 7
```

---

### Demo output
`python run_demo.py --seed 7` prints the input, the CSP pruning counters (BT / BT+FC / BT+FC+AC-3), the solution and per-van search counters, and saves `results/demo.png`. It widens the time windows once by default; use `--widen 0` for the original windows (no solution).

## Testing

GitHub Actions runs the test suite on pushes to `main` and pull requests targeting `main`.

Run the test suite locally in quiet mode:
```bash
pytest -q
```

The suite covers 5 normal cases, 3 edge cases, and 3 impossible cases, including CSP mode selection and AC-3 domain propagation.

## Presentation

See [PRESENTATION.md](PRESENTATION.md) for the 10-minute presentation outline and 5-minute live-demo script. Regenerate all experiment outputs before presenting numerical results; the committed aggregate CSVs are historical and may not match the corrected evaluation pipeline.

## Server Execution

Start the web server:
```bash
python server.py
```

Access the UI in your browser:
```bash
http://localhost:8000
```
