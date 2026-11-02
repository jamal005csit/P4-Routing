import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import generate
from src.search import route
from src.csp import build_diff_matrix, CSP
from src.integration import solve_all


def get_matrix():
    if os.path.exists("time_matrix_20x20x24.npy"):
        return np.load("time_matrix_20x20x24.npy")
    # synthetic fallback: 20x20x24, rush hour peak
    rng = np.random.RandomState(0)
    mat = np.full((20, 20, 24), 12.0) + rng.rand(20, 20, 24) * 4.0
    mat[:, :, 8:11] += 8.0
    mat[:, :, 17:20] += 10.0
    return mat * 1.2  # van factor


def exp1(seeds):
    os.makedirs("results", exist_ok=True)
    mat = get_matrix()
    rows = []
    for s in seeds:
        for k in [6, 7, 8, 9, 10]:
            inst = generate(s, k)
            zi = {z: i for i, z in enumerate(inst["top20"])}
            for algo in ["A*", "UCS", "Greedy"]:
                t0 = time.time()
                p, c, st = route(list(range(k)), inst, mat, zi, algo, max_expanded=20000, time_limit=1.5)
                if c is None or (isinstance(c, float) and np.isinf(c)):
                    c = 9999.0
                    solved = 0
                else:
                    solved = 1 if p is not None else 0
                dt = time.time() - t0
                rows.append({"seed": s, "k": k, "algo": algo,
                             "cost": float(c), "time": dt,
                             "expanded": st.get("expanded", 0),
                             "generated": st.get("generated", 0),
                             "solved": solved})
    df = pd.DataFrame(rows)
    df["cost_solved"] = df.cost.where(df.solved == 1)
    agg = df.groupby(["k", "algo"]).agg(
        mean_cost=("cost", "mean"), std_cost=("cost", "std"),
        mean_time=("time", "mean"), std_time=("time", "std"),
        mean_expanded=("expanded", "mean"),
        solved_rate=("solved", "mean"),
        mean_cost_solved=("cost_solved", "mean")).reset_index()
    agg.to_csv("results/exp1.csv", index=False)
    print(agg.to_string())
    plt.figure()
    agg.pivot(index="k", columns="algo", values="mean_time").plot(kind="bar")
    plt.ylabel("mean time (s)")
    plt.tight_layout()
    plt.savefig("results/exp1.png")
    plt.close()
    print("saved results/exp1.csv, results/exp1.png")


def exp2(seeds):
    os.makedirs("results", exist_ok=True)
    mat = get_matrix()
    rows = []
    for s in seeds:
        for n in [15, 20, 30]:
            inst = generate(s, n)
            zi = {z: i for i, z in enumerate(inst["top20"])}
            diff = build_diff_matrix(inst, mat, zi)
            for mode in ["BT", "BT+FC", "BT+FC+AC-3"]:
                csp = CSP(inst, diff, max_nodes=20000, time_limit=1.0)
                t0 = time.time()
                r = csp.solve(mode)
                dt = time.time() - t0
                rows.append({"seed": s, "n": n, "mode": mode,
                             "time": dt, "nodes": csp.nodes,
                             "assignments": csp.assignments,
                             "backtracks": csp.backtracks,
                             "solved": r is not None})
    df = pd.DataFrame(rows)
    agg = df.groupby(["n", "mode"]).agg(
        mean_time=("time", "mean"), std_time=("time", "std"),
        mean_nodes=("nodes", "mean"), std_nodes=("nodes", "std"),
        solved_rate=("solved", "mean")).reset_index()
    agg.to_csv("results/exp2.csv", index=False)
    print(agg.to_string())
    plt.figure()
    agg.pivot(index="n", columns="mode", values="mean_nodes").plot(kind="bar")
    plt.ylabel("mean nodes")
    plt.tight_layout()
    plt.savefig("results/exp2.png")
    plt.close()
    print("saved results/exp2.csv, results/exp2.png")


def exp3(seeds):
    os.makedirs("results", exist_ok=True)
    mat_nn = get_matrix()
    rng = np.random.RandomState(1)
    # simulate model error: avg and distance are biased vs NN
    mats = {
        "NN": mat_nn,
        "hist-avg": mat_nn * 1.10 + 1.0,
        "distance": mat_nn * 0.85 + 4.0
    }
    # real world for replay = NN + lognormal noise
    real_mat = mat_nn * rng.lognormal(0, 0.15, size=mat_nn.shape)

    rows = []
    for s in seeds:
        for model in ["NN", "hist-avg", "distance"]:
            inst = generate(s, 20)
            zi = {z: i for i, z in enumerate(inst["top20"])}
            plan_mat = mats[model]
            res = solve_all(inst, plan_mat, zi, algo="A*")
            if res["status"] != "OK":
                rows.append({"seed": s, "model": model, "late": 20})
                continue
            # replay routes on real_mat
            mp = {o["id"]: o for o in inst["orders"]}
            late = 0
            for v, oids in res["routes"].items():
                # need order: sort by planned path if available, else tw
                path = sorted(oids, key=lambda oid: mp[oid]["tw"][0])
                clock = inst["depot_start"]
                cur = inst["depot"]
                for oid in path:
                    t = float(real_mat[zi[cur], zi[mp[oid]["zone"]], int(clock // 60) % 24])
                    if clock + t > mp[oid]["tw"][1]:
                        late += 1
                    clock = max(clock + t, mp[oid]["tw"][0]) + mp[oid]["service"]
                    cur = mp[oid]["zone"]
            rows.append({"seed": s, "model": model, "late": late})
    df = pd.DataFrame(rows)
    agg = df.groupby("model").agg(mean_late=("late", "mean"), std_late=("late", "std")).reset_index()
    agg.to_csv("results/exp3.csv", index=False)
    print(agg.to_string())
    plt.figure()
    plt.bar(agg["model"], agg["mean_late"])
    plt.ylabel("mean late deliveries")
    plt.tight_layout()
    plt.savefig("results/exp3.png")
    plt.close()
    print("saved results/exp3.csv, results/exp3.png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="1,2,3", help="e.g. 1,2,3 or 1")
    p.add_argument("--seeds", default="0,1,2,3,4", help="e.g. 0,1,2,3,4")
    a = p.parse_args()
    exps = [int(x) for x in str(a.exp).replace(" ", ",").split(",") if str(x).strip() != ""]
    seeds = [int(x) for x in str(a.seeds).replace(" ", ",").split(",") if str(x).strip() != ""]
    return exps, seeds


def main():
    exps, seeds = parse_args()
    print(f"running exps={exps} seeds={seeds}")
    if 1 in exps:
        exp1(seeds)
    if 2 in exps:
        exp2(seeds)
    if 3 in exps:
        exp3(seeds)


if __name__ == "__main__":
    main()
