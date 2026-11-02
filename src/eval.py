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
    # Synthetic fallback is acceptable for algorithm smoke tests, not data claims.
    rng = np.random.RandomState(0)
    mat = np.full((20, 20, 24), 12.0) + rng.rand(20, 20, 24) * 4.0
    mat[:, :, 8:11] += 8.0
    mat[:, :, 17:20] += 10.0
    return mat * 1.2


def exp1(seeds):
    os.makedirs("results", exist_ok=True)
    mat = get_matrix()
    rows = []
    for seed in seeds:
        for k in [6, 7, 8, 9, 10]:
            inst = generate(seed, k)
            zi = {zone: i for i, zone in enumerate(inst["top20"])}
            for algorithm in ["A*", "UCS", "Greedy"]:
                started = time.perf_counter()
                path, cost, stats = route(
                    list(range(k)), inst, mat, zi, algorithm,
                    max_expanded=20000, time_limit=1.5,
                )
                solved = int(path is not None and np.isfinite(cost))
                rows.append({
                    "seed": seed, "k": k, "algo": algorithm,
                    "cost": float(cost) if solved else np.nan,
                    "time": time.perf_counter() - started,
                    "expanded": stats.get("expanded", 0),
                    "generated": stats.get("generated", 0),
                    "solved": solved,
                })
    df = pd.DataFrame(rows)
    agg = df.groupby(["k", "algo"]).agg(
        mean_cost=("cost", "mean"), std_cost=("cost", "std"),
        mean_time=("time", "mean"), std_time=("time", "std"),
        mean_expanded=("expanded", "mean"), std_expanded=("expanded", "std"),
        mean_generated=("generated", "mean"), std_generated=("generated", "std"),
        solved_rate=("solved", "mean"),
    ).reset_index()
    df.to_csv("results/exp1_raw.csv", index=False)
    agg.to_csv("results/exp1.csv", index=False)
    ax = agg.pivot(index="k", columns="algo", values="mean_time").plot(kind="bar")
    ax.set_ylabel("Mean runtime (s)")
    plt.tight_layout()
    plt.savefig("results/exp1.png")
    plt.close()
    print(agg.to_string(index=False))


def exp2(seeds):
    os.makedirs("results", exist_ok=True)
    mat = get_matrix()
    rows = []
    for seed in seeds:
        for n in [15, 20, 30]:
            inst = generate(seed, n)
            zi = {zone: i for i, zone in enumerate(inst["top20"])}
            diff = build_diff_matrix(inst, mat, zi)
            for mode in ["BT", "BT+FC", "BT+FC+AC-3"]:
                solver = CSP(inst, diff, max_nodes=20000, time_limit=1.0)
                started = time.perf_counter()
                solution = solver.solve(mode)
                rows.append({
                    "seed": seed, "n": n, "mode": mode,
                    "time": time.perf_counter() - started,
                    "nodes": solver.nodes, "assignments": solver.assignments,
                    "backtracks": solver.backtracks,
                    "solved": solution is not None, "timed_out": solver.timed_out,
                })
    df = pd.DataFrame(rows)
    agg = df.groupby(["n", "mode"]).agg(
        mean_time=("time", "mean"), std_time=("time", "std"),
        mean_nodes=("nodes", "mean"), std_nodes=("nodes", "std"),
        mean_assignments=("assignments", "mean"),
        mean_backtracks=("backtracks", "mean"),
        solved_rate=("solved", "mean"), timeout_rate=("timed_out", "mean"),
    ).reset_index()
    df.to_csv("results/exp2_raw.csv", index=False)
    agg.to_csv("results/exp2.csv", index=False)
    ax = agg.pivot(index="n", columns="mode", values="mean_nodes").plot(kind="bar")
    ax.set_ylabel("Mean search nodes")
    plt.tight_layout()
    plt.savefig("results/exp2.png")
    plt.close()
    print(agg.to_string(index=False))


def _load_replay_matrices():
    """Build three planning matrices and one independent held-out replay matrix."""
    required = [
        "train_20zones.parquet", "test_20zones.parquet",
        "top20.json", "time_matrix_20x20x24.npy",
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Experiment 3 requires real held-out TLC data and a trained NN; "
            f"missing: {', '.join(missing)}. Run make_data.py and src.nn first."
        )

    train = pd.read_parquet("train_20zones.parquet")
    test = pd.read_parquet("test_20zones.parquet")
    top20 = pd.read_json("top20.json", typ="series").tolist()
    nn_matrix = np.load("time_matrix_20x20x24.npy").astype(float)
    if nn_matrix.shape != (20, 20, 24):
        raise ValueError(f"Expected NN matrix shape (20,20,24), got {nn_matrix.shape}")

    shape = (20, 20, 24)
    hist_matrix = np.full(shape, float(train.duration.mean()), dtype=float)
    distance_matrix = np.full(shape, float(train.trip_distance.mean()) / 18.0 * 60.0, dtype=float)
    replay_matrix = np.full(shape, float(test.duration.mean()), dtype=float)

    train_group = train.groupby(["PULocationID", "DOLocationID", "hour"]).duration.mean()
    test_group = test.groupby(["PULocationID", "DOLocationID", "hour"]).duration.mean()
    pair_distance = train.groupby(["PULocationID", "DOLocationID"]).trip_distance.mean()
    global_distance = float(train.trip_distance.mean())
    zone_index = {zone: i for i, zone in enumerate(top20)}

    for i, pickup in enumerate(top20):
        for j, dropoff in enumerate(top20):
            distance = float(pair_distance.get((pickup, dropoff), global_distance))
            distance_matrix[i, j, :] = distance / 18.0 * 60.0
            for hour in range(24):
                hist_matrix[i, j, hour] = float(
                    train_group.get((pickup, dropoff, hour), train.duration.mean())
                )
                replay_matrix[i, j, hour] = float(
                    test_group.get((pickup, dropoff, hour), test.duration.mean())
                )

    # Convert observed taxi duration to the same van-time scale as NN export.
    replay_matrix *= 1.2
    hist_matrix *= 1.2
    distance_matrix *= 1.2
    return top20, {
        "NN": nn_matrix,
        "hist-avg": hist_matrix,
        "distance": distance_matrix,
    }, replay_matrix, zone_index


def exp3(seeds):
    os.makedirs("results", exist_ok=True)
    top20, planning_matrices, replay_matrix, zi = _load_replay_matrices()
    rows = []
    for seed in seeds:
        inst = generate(seed, 20, top20=top20)
        for model_name, planning_matrix in planning_matrices.items():
            plan = solve_all(inst, planning_matrix, zi, algo="A*")
            if plan["status"] != "OK":
                rows.append({
                    "seed": seed, "model": model_name, "late": np.nan,
                    "status": plan["status"], "solved": False,
                })
                continue

            orders = {order["id"]: order for order in inst["orders"]}
            late = 0
            for van, order_ids in plan["routes"].items():
                path = sorted(order_ids, key=lambda oid: orders[oid]["tw"][0])
                clock = float(inst["depot_start"])
                current_zone = inst["depot"]
                for order_id in path:
                    order = orders[order_id]
                    travel_time = float(replay_matrix[
                        zi[current_zone], zi[order["zone"]],
                        int(clock // 60) % 24,
                    ])
                    arrival = clock + travel_time
                    if arrival > order["tw"][1]:
                        late += 1
                    clock = max(arrival, order["tw"][0]) + order["service"]
                    current_zone = order["zone"]
            rows.append({
                "seed": seed, "model": model_name, "late": late,
                "status": "OK", "solved": True,
            })

    raw = pd.DataFrame(rows)
    raw.to_csv("results/exp3_raw.csv", index=False)
    successful = raw[raw.solved & raw.late.notna()]
    if successful.empty:
        raise RuntimeError("Experiment 3 produced no feasible plans; inspect exp3_raw.csv")
    agg = successful.groupby("model").agg(
        mean_late=("late", "mean"), std_late=("late", "std"),
        solved_rate=("solved", "mean"),
    ).reset_index()
    agg.to_csv("results/exp3.csv", index=False)
    plt.figure()
    plt.bar(agg["model"], agg["mean_late"])
    plt.ylabel("Mean late deliveries on held-out TLC replay")
    plt.tight_layout()
    plt.savefig("results/exp3.png")
    plt.close()
    print(agg.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="1,2,3", help="e.g. 1,2,3 or 1")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="at least five seeds recommended")
    args = parser.parse_args()
    experiments = [int(x) for x in str(args.exp).replace(" ", ",").split(",") if x.strip()]
    seeds = [int(x) for x in str(args.seeds).replace(" ", ",").split(",") if x.strip()]
    if not seeds:
        parser.error("provide at least one seed")
    return experiments, seeds


def main():
    experiments, seeds = parse_args()
    print(f"running experiments={experiments}, seeds={seeds}")
    if 1 in experiments:
        exp1(seeds)
    if 2 in experiments:
        exp2(seeds)
    if 3 in experiments:
        exp3(seeds)


if __name__ == "__main__":
    main()
