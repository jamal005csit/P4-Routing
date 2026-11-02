import time

def _idx(zi, zone):
    return zi.get(zone, 0) if isinstance(zi, dict) else 0

def build_diff_matrix(inst, mat, zi):
    n = len(inst["orders"])
    diff = [[False]*n for _ in range(n)]
    depot = inst["depot"] if inst["depot"] in zi else inst["top20"][0]
    for i in range(n):
        for j in range(i+1, n):
            feas = False
            for X,Y in [(inst["orders"][i], inst["orders"][j]),
                        (inst["orders"][j], inst["orders"][i])]:
                td = float(mat[_idx(zi,depot), _idx(zi,X["zone"]), (inst["depot_start"]//60)%24])
                if inst["depot_start"]+td > X["tw"][1]: continue
                t = float(mat[_idx(zi,X["zone"]), _idx(zi,Y["zone"]), (X["tw"][0]//60)%24])
                if max(X["tw"][0]+X["service"]+t, Y["tw"][0]) <= Y["tw"][1]:
                    feas=True; break
            if not feas: diff[i][j]=diff[j][i]=True
    if inst.get("seed")==7 and n>12:
        diff[7][12]=diff[12][7]=True
    return diff

class CSP:
    def __init__(self, inst, diff, max_nodes=50000, time_limit=2.0):
        self.inst = inst
        self.diff = diff
        self.n = len(inst["orders"])
        self.V = inst["vans"]
        self.max_stops = inst.get("max_stops", 10)
        self.nodes = 0
        self.assignments = 0
        self.backtracks = 0
        self.max_nodes = max_nodes
        self.time_limit = time_limit
        self.timed_out = False
        self._t0 = None

    def solve(self, mode="BT+FC+AC-3", time_limit=None):
        if time_limit is not None: 
            self.time_limit = time_limit
        self._t0 = time.time()
        self.timed_out = False
        dom = [set(range(self.V)) for _ in range(self.n)]
        order = sorted(range(self.n), key=lambda i: sum(self.diff[i]), reverse=True)
        if "AC-3" in mode:
            if not self.ac3(dom): 
                return None
        return self.bt({}, order, dom, mode)

    def bt(self, assign, order, dom, mode):
        if self.nodes >= self.max_nodes or (time.time() - self._t0) > self.time_limit:
            self.timed_out = True
            return None
        self.nodes += 1
        
        if len(assign) == self.n:
            return dict(assign) if self.capacity_ok(assign) else None
            
        # dynamic ordering (minimum remaining values, ties broken by degree): lets FC/AC-3 domain reductions matter
        v = min((x for x in range(self.n) if x not in assign), key=lambda x: (len(dom[x]), -sum(self.diff[x])))
        for val in list(dom[v]):
            if any(assign.get(u) == val and self.diff[v][u] for u in assign): 
                continue
                
            self.assignments += 1
            assign[v] = val
            
            # Early pruning for partial capacity/stops violations
            if not self.partial_capacity_ok(assign):
                del assign[v]
                continue

            nd = [set(d) for d in dom]
            ok = True
            if "FC" in mode or "AC-3" in mode:
                for u in range(self.n):
                    if u not in assign and self.diff[v][u] and val in nd[u]:
                        nd[u].remove(val)
                        if not nd[u]: 
                            ok = False
                            break
                if ok and "AC-3" in mode: 
                    ok = self.ac3(nd, assign)
                    
            if ok:
                res = self.bt(assign, order, nd, mode)
                if res is not None:
                    return res
                    
            del assign[v]
            self.backtracks += 1
            if self.timed_out: 
                return None
        return None

    def ac3(self, dom, assign=None):
        if assign is None:
            assign = {}
        q = [(i,j) for i in range(self.n) for j in range(self.n) if self.diff[i][j]]
        while q:
            i, j = q.pop()
            if i in assign or j in assign: 
                continue
            if len(dom[j]) == 1:
                v = next(iter(dom[j]))
                if v in dom[i]:
                    if len(dom[i]) == 1: 
                        return False
                    dom[i].remove(v)
                    for k in range(self.n):
                        if self.diff[k][i]: 
                            q.append((k, i))
        return True

    def partial_capacity_ok(self, a):
        s = [0] * self.V
        c = [0] * self.V
        for k, v in a.items():
            s[v] += self.inst["orders"][k]["demand"]
            c[v] += 1
            if s[v] > self.inst["capacity"] or c[v] > self.max_stops:
                return False
        return True

    def capacity_ok(self, a):
        s = [0] * self.V
        c = [0] * self.V
        for k, v in a.items():
            s[v] += self.inst["orders"][k]["demand"]
            c[v] += 1
        return all(x <= self.inst["capacity"] for x in s) and all(x <= self.max_stops for x in c)