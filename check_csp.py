# Run from the repo root:  python check_csp.py
# Checks whether BT, BT+FC and BT+FC+AC-3 really prune, on the real time matrix.
import numpy as np
from src.data import generate
from src.csp import build_diff_matrix, CSP

mat = np.load("time_matrix_20x20x24.npy")
for seed in range(5):
    inst = generate(seed, 20)
    zi = {z: i for i, z in enumerate(inst["top20"])}
    diff = build_diff_matrix(inst, mat, zi)
    conflicts = int(sum(sum(1 for x in row if x) for row in diff)) // 2
    out = []
    for mode in ["BT", "BT+FC", "BT+FC+AC-3"]:
        c = CSP(inst, diff, max_nodes=20000, time_limit=1.0)
        r = c.solve(mode)
        out.append(f"{mode}: nodes={c.nodes} solved={r is not None} timed_out={c.timed_out}")
    print(f"seed {seed} conflict_pairs={conflicts}")
    for line in out:
        print("   ", line)
