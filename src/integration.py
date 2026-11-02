from src.csp import build_diff_matrix, CSP
from src.search import route

def solve_all(inst, mat, zi, algo="A*", mode="BT+FC+AC-3"):
    depot = inst["depot"] if inst["depot"] in zi else inst["top20"][0]
    # DEPOT_UNREACHABLE
    for o in inst["orders"]:
        ia = zi.get(depot, 0); ib = zi.get(o["zone"], 0)
        try:
            td = float(mat[ia, ib, (inst["depot_start"]//60)%24])
        except Exception:
            td = 12.0
        if inst["depot_start"] + td > o["tw"][1]:
            return {"status": f"NO SOLUTION: DEPOT_UNREACHABLE order {o['id']}",
                    "reason": "DEPOT_UNREACHABLE", "order": o["id"], "assign": None}

    diff = build_diff_matrix(inst, mat, zi)

    for it in range(5):
        c = CSP(inst, diff)
        assign = c.solve(mode)
        if assign is None:
            total = sum(o["demand"] for o in inst["orders"])
            if total > inst["vans"] * inst["capacity"]:
                reason = "CAPACITY"
            elif getattr(c, "timed_out", False):
                reason = "PAIRWISE_CONFLICT+CAPACITY (timeout)"
            else:
                reason = "PAIRWISE_CONFLICT"
            return {"status": f"NO SOLUTION: {reason} unsat", "reason": reason,
                    "assign": None,
                    "stats": {"nodes": c.nodes, "assignments": c.assignments, "backtracks": c.backtracks}}

        routes = {v: [o for o, a in assign.items() if a == v] for v in range(inst["vans"])}
        # Preserve the optimized visit order separately from CSP assignment membership.
        route_paths = {}
        late = None
        for v, oids in routes.items():
            if not oids:
                continue
            path, cost, st = route(oids, inst, mat, zi, algo, max_expanded=30000, time_limit=2.0)
            if path is None:
                if len(oids) >= 2:
                    late = (oids[0], oids[1])
                    break
                return {"status": f"NO SOLUTION: NO_FEASIBLE_ORDER van {v} single {oids[0]}",
                        "reason": "NO_FEASIBLE_ORDER", "assign": assign}
            route_paths[v] = path
            clock, cur = inst["depot_start"], depot
            mp = {o["id"]: o for o in inst["orders"]}
            for idx, oid in enumerate(path):
                t = float(mat[zi.get(cur,0), zi.get(mp[oid]["zone"],0), (int(clock)//60)%24])
                if clock + t > mp[oid]["tw"][1]:
                    prev = path[max(0, idx-1)]
                    late = (prev, oid)
                    break
                clock = max(clock+t, mp[oid]["tw"][0]) + mp[oid]["service"]
                cur = mp[oid]["zone"]
            if late:
                break

        if late is None:
            return {"status": "OK", "assign": assign, "routes": routes,
                    "route_paths": route_paths, "loops": it,
                    "csp_stats": {"nodes": c.nodes, "assignments": c.assignments, "backtracks": c.backtracks}}
        if late[0] == late[1]:
            return {"status": f"NO SOLUTION: NO_FEASIBLE_ORDER late pair {late}",
                    "reason": "NO_FEASIBLE_ORDER", "late_pair": late, "assign": assign}
        diff[late[0]][late[1]] = diff[late[1]][late[0]] = True

    return {"status": "NO SOLUTION: NO_FEASIBLE_ORDER after 5 loops",
            "reason": "NO_FEASIBLE_ORDER", "late_pair": late, "assign": assign}

def widen(inst, step=15):
    for o in inst["orders"]:
        o["tw"] = [o["tw"][0]-step, o["tw"][1]+step]
    return inst