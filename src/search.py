import heapq, itertools, time

def travel(mat, zi, a, b, clock):
    try:
        ia = zi.get(a, 0) if isinstance(zi, dict) else 0
        ib = zi.get(b, 0) if isinstance(zi, dict) else 0
        return float(mat[ia, ib, (int(clock)//60)%24])
    except Exception:
        return 12.0

def route(oids, inst, mat, zi, algo="A*", max_expanded=30000, time_limit=2.0):
    orders = {o["id"]: o for o in inst["orders"]}
    depot = inst["depot"] if inst["depot"] in zi else inst["top20"][0]
    start = inst["depot_start"]
    stats = {"expanded":0,"generated":0,"timed_out":False}
    if len(oids)==0:
        return [], 0.0, stats
    t0 = time.time()
    if algo=="Greedy":
        cur, clock, un, path, cost = depot, start, set(oids), [], 0.0
        while un:
            best, bestArr = None, float('inf')
            for oid in un:
                stats["generated"]+=1
                t = travel(mat, zi, cur, orders[oid]["zone"], clock)
                arr = max(clock+t, orders[oid]["tw"][0])
                if arr <= orders[oid]["tw"][1] and arr < bestArr:
                    bestArr, best = arr, oid
            if best is None:
                return None, float('inf'), stats
            t = travel(mat, zi, cur, orders[best]["zone"], clock)
            clock_before = clock
            stats["expanded"]+=1
            clock = max(clock+t, orders[best]["tw"][0])+orders[best]["service"]
            wait=max(0,orders[best]["tw"][0]-(clock_before+t))
            cost+=t+wait; cur=orders[best]["zone"]; un.remove(best); path.append(best)
        return path, cost, stats

    min_in = {}
    for oid in oids:
        zt = orders[oid]["zone"]
        best = travel(mat, zi, depot, zt, start)
        for o in oids:
            if o==oid: continue
            t = travel(mat, zi, orders[o]["zone"], zt, start)
            if t < best: best = t
        min_in[oid]=best
    def h(vis):
        return sum(min_in[o] for o in oids if o not in vis)

    pq, ctr, seen = [], itertools.count(), {}
    h0 = h(set()) if algo=="A*" else 0
    heapq.heappush(pq,(h0,next(ctr),0.0,start,depot,frozenset(),[]))
    best, bestP = float('inf'), None
    while pq:
        if stats["expanded"]>=max_expanded or (time.time()-t0)>time_limit:
            stats["timed_out"]=True
            break
        f,_,g,clock,cur,vis,path = heapq.heappop(pq)
        stats["expanded"]+=1
        if len(vis)==len(oids):
            if g<best: best,bestP=g,path
            continue
        if (cur,vis) in seen and seen[(cur,vis)]<=g: continue
        seen[(cur,vis)]=g
        for oid in oids:
            if oid in vis: continue
            o=orders[oid]
            t=travel(mat,zi,cur,o["zone"],clock)
            if clock+t>o["tw"][1]: continue
            wait=max(0,o["tw"][0]-(clock+t))
            ng=g+t+wait
            nc=max(clock+t,o["tw"][0])+o["service"]
            nvis=frozenset(set(vis)|{oid})
            nf=ng+(h(nvis) if algo=="A*" else 0)
            stats["generated"]+=1
            heapq.heappush(pq,(nf,next(ctr),ng,nc,o["zone"],nvis,path+[oid]))
    if bestP is None:
        return None, float('inf'), stats
    return bestP, best, stats

def brute_force(oids, inst, mat, zi):
    import itertools
    orders={o["id"]:o for o in inst["orders"]}
    best,bestP=float('inf'),None
    for perm in itertools.permutations(oids):
        clock,cur,cost,ok=inst["depot_start"],inst["depot"],0.0,True
        if cur not in zi: cur=inst["top20"][0]
        for oid in perm:
            o=orders[oid]; t=travel(mat,zi,cur,o["zone"],clock); arr=clock+t
            if arr>o["tw"][1]: ok=False; break
            cost+=t+max(0,o["tw"][0]-arr); clock=max(arr,o["tw"][0])+o["service"]; cur=o["zone"]
        if ok and cost<best: best,bestP=cost,list(perm)
    return bestP,best
