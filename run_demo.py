import os, numpy as np, matplotlib.pyplot as plt
from src.data import generate
from src.integration import solve_all
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()

inst = generate(seed=a.seed, n=20)
zi = {z:i for i,z in enumerate(inst["top20"])}
if os.path.exists("time_matrix_20x20x24.npy"):
    mat = np.load("time_matrix_20x20x24.npy")
    if mat.shape != (20,20,24):
        print(f"warn matrix shape {mat.shape}, using fallback")
        mat = np.full((20,20,24), 12.0)
else:
    print("warn no npy, using synthetic fallback")
    rng = np.random.RandomState(0)
    mat = np.full((20,20,24), 12.0) + rng.rand(20,20,24)*4.0

print("INPUT orders 7,12:", [o for o in inst["orders"] if o["id"] in [7,12]])
print("top20:", inst["top20"][:5], "... depot:", inst["depot"])
res = solve_all(inst, mat, zi, algo="A*")
print(res["status"], res.get("assign"))