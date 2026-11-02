"""Interactive demo: input -> CSP pruning (BT / BT+FC / BT+FC+AC-3) -> integrated solve -> search counters -> plots.

    python run_demo.py --seed 7              # widens windows once (default), as the web UI does
    python run_demo.py --seed 7 --widen 0    # original windows (expected: NO SOLUTION)
Writes results/demo.png
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data import generate
from src.integration import solve_all, widen
from src.search import route
from src.csp import build_diff_matrix, CSP

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--widen", type=int, default=1, help="widen every window by 15 min each side, this many times")
ap.add_argument("--out", default="results/demo.png")
ap.add_argument(
    "--travel-model", choices=("auto", "nn", "synthetic"), default="auto",
    help="nn uses the trained NN matrix; synthetic explicitly disables the NN",
)
a = ap.parse_args()

# ---- 1. input
inst = generate(seed=a.seed, n=20)
for _ in range(a.widen):
    widen(inst, 15)
zi = {z: i for i, z in enumerate(inst["top20"])}
matrix_path = "time_matrix_20x20x24.npy"
has_nn_matrix = os.path.exists(matrix_path)
if a.travel_model == "nn" and not has_nn_matrix:
    raise FileNotFoundError(
        "NN travel matrix is missing. Run make_data.py and python -m src.nn first."
    )
use_nn = a.travel_model == "nn" or (a.travel_model == "auto" and has_nn_matrix)
if use_nn:
    mat = np.load(matrix_path)
    if mat.shape != (20, 20, 24):
        raise ValueError(f"Expected NN matrix shape (20,20,24), got {mat.shape}")
    print("travel times: trained NN matrix (NN ON)")
else:
    print("travel times: deterministic synthetic matrix (NN OFF; comparison/demo only)")
    rng = np.random.RandomState(0)
    mat = np.full((20, 20, 24), 12.0) + rng.rand(20, 20, 24) * 4.0
print(f"windows widened {a.widen}x15 min")
print("INPUT orders 7,12:", [o for o in inst["orders"] if o["id"] in [7, 12]])
print("top20:", inst["top20"][:5], "... depot:", inst["depot"])

# ---- 2. CSP: the three pruning modes, same instance
diff = build_diff_matrix(inst, mat, zi)
csp_rows = []
for mode in ["BT", "BT+FC", "BT+FC+AC-3"]:
    c = CSP(inst, diff, max_nodes=50000, time_limit=2.0)
    r = c.solve(mode)
    csp_rows.append((mode, c.nodes, c.assignments, c.backtracks, r is not None))
    print(f"CSP {mode:11s} nodes={c.nodes:6d} assignments={c.assignments:6d} backtracks={c.backtracks:6d} solved={r is not None}")

# ---- 3. integrated solve (CSP + A* per van + integration loop)
res = solve_all(inst, mat, zi, algo="A*")
print(res["status"])
print("assign (order -> van):", res.get("assign"))

# ---- 4. search counters per van, for each algorithm
search_rows = []
routes = res.get("routes") or {}
for algo in ["A*", "UCS", "Greedy"]:
    exp = []
    for v, oids in routes.items():
        if not oids:
            continue
        path, cost, st = route(oids, inst, mat, zi, algo, max_expanded=30000, time_limit=2.0)
        exp.append(st["expanded"])
        print(f"search {algo:6s} van {v}: path={path} expanded={st['expanded']} generated={st['generated']}")
    search_rows.append((algo, exp))

# ---- 5. plots: how the parts cooperate
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
ax = axes[0]
ax.bar([r[0] for r in csp_rows], [r[1] for r in csp_rows], color=["#999", "#666", "#222"])
for i, r in enumerate(csp_rows):
    ax.text(i, r[1], f"{r[1]}\nbt={r[3]}", ha="center", va="bottom", fontsize=8)
ax.set_title("CSP pruning: nodes visited per mode"); ax.set_ylabel("nodes")

ax = axes[1]
if search_rows and any(e for _, e in search_rows):
    w = 0.27
    for k, (algo, exp) in enumerate(search_rows):
        xs = np.arange(len(exp)) + (k - 1) * w
        ax.bar(xs, exp, width=w, label=algo)
    ax.set_xticks(range(len(routes))); ax.set_xticklabels([f"van {v}" for v in routes])
    ax.legend(); ax.set_title("Search per van: nodes expanded"); ax.set_ylabel("expanded")
else:
    ax.text(0.5, 0.5, "no routes: CSP found no assignment\n(try --widen 1)", ha="center", va="center")
    ax.set_axis_off()

ax = axes[2]
if res.get("assign"):
    assign = res["assign"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for i, o in enumerate(inst["orders"]):
        ax.barh(i, o["tw"][1] - o["tw"][0], left=o["tw"][0], color=colors[assign[o["id"]]], alpha=0.8)
    ax.set_yticks(range(len(inst["orders"])))
    ax.set_yticklabels([f"order {o['id']}" for o in inst["orders"]], fontsize=7)
    ax.invert_yaxis()
    ax.set_xticks(range(540, 901, 60)); ax.set_xticklabels([f"{m//60:02d}:00" for m in range(540, 901, 60)])
    ax.set_title("Time windows, coloured by van (blue=0 orange=1 green=2)")
else:
    ax.text(0.5, 0.5, res["status"], ha="center", va="center", wrap=True)
    ax.set_axis_off()

fig.suptitle(f"P4-Routing demo, seed {a.seed}, windows widened {a.widen}x15 min, {res['status'][:40]}")
plt.tight_layout()
plt.savefig(a.out, dpi=120)
print("saved", a.out)
