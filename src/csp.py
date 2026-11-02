"""Graph-colouring CSP for assigning orders to vans."""
import time


def _idx(zi, zone):
    return zi.get(zone, 0) if isinstance(zi, dict) else 0


def build_diff_matrix(inst, mat, zi):
    """Return symmetric conflicts: True means two orders cannot share a van."""
    n = len(inst["orders"])
    diff = [[False] * n for _ in range(n)]
    depot = inst["depot"] if inst["depot"] in zi else inst["top20"][0]
    for i in range(n):
        for j in range(i + 1, n):
            feasible_order = False
            for first, second in (
                (inst["orders"][i], inst["orders"][j]),
                (inst["orders"][j], inst["orders"][i]),
            ):
                depot_time = float(mat[
                    _idx(zi, depot), _idx(zi, first["zone"]),
                    (inst["depot_start"] // 60) % 24
                ])
                if inst["depot_start"] + depot_time > first["tw"][1]:
                    continue
                travel_time = float(mat[
                    _idx(zi, first["zone"]), _idx(zi, second["zone"]),
                    (first["tw"][0] // 60) % 24
                ])
                arrival = max(
                    first["tw"][0] + first["service"] + travel_time,
                    second["tw"][0],
                )
                if arrival <= second["tw"][1]:
                    feasible_order = True
                    break
            if not feasible_order:
                diff[i][j] = diff[j][i] = True
    return diff


class CSP:
    def __init__(self, inst, diff, max_nodes=50000, time_limit=2.0):
        self.inst = inst
        self.diff = diff
        self.n = len(inst["orders"])
        self.V = inst["vans"]
        self.max_stops = inst.get("max_stops", 10)
        self.max_nodes = max_nodes
        self.time_limit = time_limit
        self.nodes = 0
        self.assignments = 0
        self.backtracks = 0
        self.timed_out = False
        self._t0 = None

    def solve(self, mode="BT+FC+AC-3", time_limit=None):
        if mode not in {"BT", "BT+FC", "BT+FC+AC-3"}:
            raise ValueError("mode must be BT, BT+FC, or BT+FC+AC-3")
        if time_limit is not None:
            self.time_limit = time_limit
        self.nodes = self.assignments = self.backtracks = 0
        self.timed_out = False
        self._t0 = time.monotonic()
        domains = [set(range(self.V)) for _ in range(self.n)]
        if mode == "BT+FC+AC-3" and not self.ac3(domains):
            return None
        return self.bt({}, domains, mode)

    def bt(self, assignment, domains, mode):
        if self.nodes >= self.max_nodes or time.monotonic() - self._t0 > self.time_limit:
            self.timed_out = True
            return None
        self.nodes += 1
        if len(assignment) == self.n:
            return dict(assignment) if self.capacity_ok(assignment) else None

        # MRV, with degree as a deterministic tie-breaker.
        variable = min(
            (v for v in range(self.n) if v not in assignment),
            key=lambda v: (len(domains[v]), -sum(self.diff[v]), v),
        )
        for value in sorted(domains[variable]):
            if any(
                assignment.get(other) == value and self.diff[variable][other]
                for other in assignment
            ):
                continue

            self.assignments += 1
            assignment[variable] = value
            if not self.partial_capacity_ok(assignment):
                del assignment[variable]
                self.backtracks += 1
                continue

            next_domains = [set(domain) for domain in domains]
            next_domains[variable] = {value}
            consistent = True

            if mode in {"BT+FC", "BT+FC+AC-3"}:
                for neighbor in range(self.n):
                    if neighbor not in assignment and self.diff[variable][neighbor]:
                        next_domains[neighbor].discard(value)
                        if not next_domains[neighbor]:
                            consistent = False
                            break

            if consistent and mode == "BT+FC+AC-3":
                consistent = self.ac3(next_domains)

            if consistent:
                result = self.bt(assignment, next_domains, mode)
                if result is not None:
                    return result

            del assignment[variable]
            self.backtracks += 1
            if self.timed_out:
                return None
        return None

    def ac3(self, domains):
        """Enforce arc consistency for every binary inequality constraint."""
        queue = [
            (i, j)
            for i in range(self.n)
            for j in range(self.n)
            if self.diff[i][j]
        ]
        while queue:
            i, j = queue.pop()
            revised = False
            for value in tuple(domains[i]):
                # For X_i != X_j, value has support iff X_j has another value.
                if not any(value != other for other in domains[j]):
                    domains[i].remove(value)
                    revised = True
            if not domains[i]:
                return False
            if revised:
                for neighbor in range(self.n):
                    if neighbor != j and self.diff[neighbor][i]:
                        queue.append((neighbor, i))
        return True

    def partial_capacity_ok(self, assignment):
        loads = [0] * self.V
        stops = [0] * self.V
        for order_id, van in assignment.items():
            loads[van] += self.inst["orders"][order_id]["demand"]
            stops[van] += 1
            if loads[van] > self.inst["capacity"] or stops[van] > self.max_stops:
                return False
        return True

    def capacity_ok(self, assignment):
        return self.partial_capacity_ok(assignment)
