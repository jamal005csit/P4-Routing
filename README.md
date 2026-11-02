# P4-Routing

An end-to-end pipeline for dataset generation, neural network training, evaluation, and testing for P4 routing research.

---

## Quick Start

### 1. Installation
Install all required dependencies:
```bash
pip install -r requirements.txt
```

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

## Testing

Run the test suite in quiet mode:
```bash
pytest -q
```
