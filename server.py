"""
Tiny backend for the P4-Routing demo UI (standard library + numpy only).

    python server.py            ->  http://localhost:8000

What it does
  * serves ui/index.html
  * POST /api/solve   : generate instance -> NN time matrix -> CSP (BT+FC+AC-3) -> A*/UCS/Greedy per van
  * POST /api/replay  : plan with NN / hist-avg / distance matrices, then replay each plan
                        on "real" travel times and count late deliveries (Experiment 3, live)
  * GET  /api/meta    : which artifacts are loaded (real NN matrix? real held-out trips?)

Artifacts used if present in the repo root (created by make_data.py + src/nn.py):
    time_matrix_20x20x24.npy   -> NN travel times (x1.2 van factor already applied)
    top20.json, train_20zones.parquet, test_20zones.parquet, results/nn_vs_baseline.csv
Anything missing falls back to a clearly-labelled synthetic matrix, and /api/meta says so.
"""
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)                      # src.data / src.nn read files relative to the repo root
sys.path.insert(0, ROOT)

from src.data import generate                       # noqa: E402
from src.integration import solve_all, widen        # noqa: E402
from src.search import route                        # noqa: E402

VAN_FACTOR = 1.2
MODELS = ["NN", "hist-avg", "distance"]


# --------------------------------------------------------------------------- matrices
def _synthetic_nn():
    rng = np.random.RandomState(0)
    mat = np.full((20, 20, 24), 12.0) + rng.rand(20, 20, 24) * 4.0
    mat[:, :, 8:11] += 8.0
    mat[:, :, 17:20] += 10.0
    return mat * VAN_FACTOR


def load_matrices():
    info = {"nn_real": False, "baselines_real": False, "replay_real": False}
    if os.path.exists("time_matrix_20x20x24.npy"):
        nn = np.load("time_matrix_20x20x24.npy")
        info["nn_real"] = nn.shape == (20, 20, 24)
        if not info["nn_real"]:
            nn = _synthetic_nn()
    else:
        nn = _synthetic_nn()

    mats = {"NN": nn}
    real = None
    try:                                           # real baselines + held-out replay from the parquet files
        import pandas as pd
        top20 = json.load(open("top20.json"))
        tr = pd.read_parquet("train_20zones.parquet")
        te = pd.read_parquet("test_20zones.parquet")
        zi = {z: i for i, z in enumerate(top20)}
        gmean = float(tr.duration.mean())

        def table(df, col, agg):
            g = df.groupby(["PULocationID", "DOLocationID", "hour"])[col].agg(agg)
            return g.to_dict()

        hist = table(tr, "duration", "mean")
        dist = tr.groupby(["PULocationID", "DOLocationID"])["trip_distance"].mean().to_dict()
        dglob = float(tr.trip_distance.mean())
        test_dur = table(te, "duration", "mean")

        h_mat = np.zeros((20, 20, 24)); d_mat = np.zeros((20, 20, 24)); r_mat = np.zeros((20, 20, 24))
        for a, pu in enumerate(top20):
            for b, do in enumerate(top20):
                for h in range(24):
                    h_mat[a, b, h] = hist.get((pu, do, h), gmean) * VAN_FACTOR
                    d_mat[a, b, h] = dist.get((pu, do), dglob) / 18 * 60 * VAN_FACTOR
                    r_mat[a, b, h] = test_dur.get((pu, do, h), nn[a, b, h] / VAN_FACTOR) * VAN_FACTOR
        mats["hist-avg"], mats["distance"], real = h_mat, d_mat, r_mat
        info["baselines_real"] = info["replay_real"] = True
    except Exception as e:                         # no parquet / pandas / pyarrow -> synthetic stand-ins (as in src/eval.py exp3)
        info["fallback_reason"] = f"{type(e).__name__}: {e}"
        mats["hist-avg"] = nn * 1.10 + 1.0
        mats["distance"] = nn * 0.85 + 4.0
        real = nn * np.random.RandomState(1).lognormal(0, 0.15, size=nn.shape)
    return mats, real, info


MATS, REAL, INFO = load_matrices()


def nn_table():
    p = "results/nn_vs_baseline.csv"
    if not os.path.exists(p):
        return None
    rows = [l.strip().split(",") for l in open(p) if l.strip()]
    head, body = rows[0], rows[1:]
    return [dict(zip(head, r)) for r in body]


# --------------------------------------------------------------------------- core
def fmt_min(m):
    return f"{int(m // 60):02d}:{int(m % 60):02d}"


def simulate(path, inst, mat, zi):
    """Walk a route on a given time matrix. Returns stops with arrival/wait/on_time, and total drive minutes."""
    mp = {o["id"]: o for o in inst["orders"]}
    clock, cur, drive, stops = inst["depot_start"], inst["depot"], 0.0, []
    for oid in path:
        o = mp[oid]
        t = float(mat[zi.get(cur, 0), zi.get(o["zone"], 0), (int(clock) // 60) % 24])
        arr = clock + t
        on_time = arr <= o["tw"][1]
        wait = max(0.0, o["tw"][0] - arr)
        stops.append({"id": oid, "zone": o["zone"], "tw": o["tw"], "tw_txt": f'{fmt_min(o["tw"][0])}–{fmt_min(o["tw"][1])}',
                      "ride": round(t, 1), "arrive": fmt_min(arr), "wait": round(wait, 1),
                      "demand": o["demand"], "on_time": bool(on_time)})
        drive += t
        clock = max(arr, o["tw"][0]) + o["service"]
        cur = o["zone"]
    return stops, drive


def plan(seed, n, algo, widen_steps, model):
    inst = generate(seed=seed, n=n)
    for _ in range(widen_steps):
        widen(inst, 15)
    mat = MATS[model]
    zi = {z: i for i, z in enumerate(inst["top20"])}
    res = solve_all(inst, mat, zi, algo=algo)
    out = {"model": model, "status": res["status"], "ok": res["status"] == "OK",
           "reason": res.get("reason"), "inst": inst, "zi": zi, "mat": mat, "res": res}
    if not out["ok"]:
        return out
    vans = {}
    for v, oids in res["routes"].items():
        if not oids:
            vans[v] = {"path": [], "expanded": 0, "generated": 0, "timed_out": False}
            continue
        path, cost, st = route(oids, inst, mat, zi, algo, max_expanded=30000, time_limit=2.0)
        vans[v] = {"path": path or [], "cost": None if path is None else round(cost, 1),
                   "expanded": st["expanded"], "generated": st["generated"], "timed_out": st["timed_out"]}
    out["vans"] = vans
    return out


def solve_endpoint(body):
    seed, n = int(body.get("seed", 7)), int(body.get("n", 20))
    algo = body.get("algo", "A*")
    ws = int(body.get("widen", 0))
    model = body.get("model", "NN")
    t0 = time.time()
    p = plan(seed, n, algo, ws, model)
    if not p["ok"]:
        return {"ok": False, "status": p["status"], "reason": p["reason"], "model": model, "widen": ws,
                "assign": p["res"].get("assign")}
    inst, zi = p["inst"], p["zi"]
    mp = {o["id"]: o for o in inst["orders"]}
    vans, late_total, drive_total, n_stops = {}, 0, 0.0, 0
    for v, d in p["vans"].items():
        stops, drive = simulate(d["path"], inst, p["mat"], zi)
        late = sum(1 for s in stops if not s["on_time"])
        late_total += late; drive_total += drive; n_stops += len(stops)
        vans[str(v)] = {"stops": stops, "drive_min": round(drive, 1), "late": late,
                        "load": sum(mp[s["id"]]["demand"] for s in stops), "capacity": inst["capacity"],
                        "expanded": d["expanded"], "generated": d["generated"], "timed_out": d["timed_out"]}
    return {"ok": True, "status": "OK", "model": model, "algo": algo, "widen": ws, "seed": seed, "n": n,
            "loops": p["res"]["loops"] + 1, "csp": p["res"]["csp_stats"],
            "vans": vans, "on_time_pct": round(100 * (n_stops - late_total) / max(1, n_stops)),
            "drive_total": round(drive_total, 1), "unserviceable": 0,
            "forced_pair": {"a": 7, "b": 12,
                            "van_a": p["res"]["assign"].get(7), "van_b": p["res"]["assign"].get(12)} if n > 12 else None,
            "runtime_s": round(time.time() - t0, 3)}


def replay_endpoint(body):
    """Plan with each travel-time model, replay the plan on 'real' times, count late stops."""
    seed, n = int(body.get("seed", 7)), int(body.get("n", 20))
    algo, ws = body.get("algo", "A*"), int(body.get("widen", 0))
    rows = []
    for model in MODELS:
        p = plan(seed, n, algo, ws, model)
        if not p["ok"]:
            rows.append({"model": model, "status": p["status"], "late": None, "stops": n})
            continue
        late = stops_n = 0
        for d in p["vans"].values():
            stops, _ = simulate(d["path"], p["inst"], REAL, p["zi"])
            late += sum(1 for s in stops if not s["on_time"]); stops_n += len(stops)
        rows.append({"model": model, "status": "OK", "late": late, "stops": stops_n,
                     "on_time_pct": round(100 * (stops_n - late) / max(1, stops_n))})
    return {"ok": True, "rows": rows, "replay_real": INFO["replay_real"],
            "note": "replayed on held-out TLC trips" if INFO["replay_real"]
            else "SYNTHETIC replay (lognormal noise) - held-out parquet not found"}


def meta_endpoint():
    return {**INFO, "nn_vs_baseline": nn_table(), "models": MODELS}


# --------------------------------------------------------------------------- http
class H(BaseHTTPRequestHandler):
    def _send(self, code, payload, ctype="application/json"):
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, open("ui/index.html", "rb").read(), "text/html; charset=utf-8")
        if self.path == "/api/meta":
            return self._send(200, meta_endpoint())
        self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/api/solve":
                return self._send(200, solve_endpoint(body))
            if self.path == "/api/replay":
                return self._send(200, replay_endpoint(body))
            self._send(404, {"error": "not found"})
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"P4-Routing demo on http://localhost:{port}   artifacts: {INFO}")
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()