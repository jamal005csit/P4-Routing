import heapq, itertools
def travel(mat,zi,a,b,clock): return float(mat[zi[a],zi[b],(int(clock)//60)%24])
def route(oids,inst,mat,zi,algo="A*"):
    orders={o["id"]:o for o in inst["orders"]}; depot=inst["depot"]; start=inst["depot_start"]
    stats={"expanded":0,"generated":0}
    if algo=="Greedy":
        cur,clock,un,path,cost=depot,start,set(oids),[],0
        while un:
            best=None; bestArr=float('inf')
            for oid in un:
                t=travel(mat,zi,cur,orders[oid]["zone"],clock); arr=max(clock+t,orders[oid]["tw"][0])
                if arr<=orders[oid]["tw"][1] and arr<bestArr: bestArr=arr; best=oid; stats["generated"]+=1
            if best is None: return None,float('inf'),stats
            t=travel(mat,zi,cur,orders[best]["zone"],clock); stats["expanded"]+=1
            clock=max(clock+t,orders[best]["tw"][0])+orders[best]["service"]; cost+=t; cur=orders[best]["zone"]; un.remove(best); path.append(best)
        return path,cost,stats
    min_in={oid:min(travel(mat,zi,orders[o]["zone"] if o!=oid else depot,orders[oid]["zone"],start) for o in oids+[-1] if True) for oid in oids}
    # fix -1: use depot
    for oid in oids: min_in[oid]=min([travel(mat,zi,depot,orders[oid]["zone"],start)]+[travel(mat,zi,orders[o]["zone"],orders[oid]["zone"],start) for o in oids if o!=oid])
    def h(vis): return sum(min_in[o] for o in oids if o not in vis)
    pq=[]; ctr=itertools.count()
    h0=h(set()) if algo=="A*" else 0
    heapq.heappush(pq,(h0,next(ctr),0,start,depot,frozenset(),[])); seen={}; best=float('inf'); bestP=None
    while pq:
        f,_,g,clock,cur,vis,path=heapq.heappop(pq); stats["expanded"]+=1
        if len(vis)==len(oids):
            if g<best: best=g; bestP=path
            continue
        if (cur,vis) in seen and seen[(cur,vis)]<=g: continue
        seen[(cur,vis)]=g
        for oid in oids:
            if oid in vis: continue
            o=orders[oid]; t=travel(mat,zi,cur,o["zone"],clock); arr=clock+t
            if arr>o["tw"][1]: continue
            wait=max(0,o["tw"][0]-arr); ng=g+t+wait; nc=max(arr,o["tw"][0])+o["service"]
            nvis=frozenset(set(vis)|{oid}); nf=ng+(h(nvis) if algo=="A*" else 0)
            stats["generated"]+=1; heapq.heappush(pq,(nf,next(ctr),ng,nc,o["zone"],nvis,path+[oid]))
    return bestP,best,stats
def brute_force(oids,inst,mat,zi):
    import itertools; orders={o["id"]:o for o in inst["orders"]}; best=float('inf'); bestP=None
    for perm in itertools.permutations(oids):
        clock=inst["depot_start"]; cur=inst["depot"]; cost=0; ok=True
        for oid in perm:
            o=orders[oid]; t=travel(mat,zi,cur,o["zone"],clock); arr=clock+t
            if arr>o["tw"][1]: ok=False; break
            cost+=t+max(0,o["tw"][0]-arr); clock=max(arr,o["tw"][0])+o["service"]; cur=o["zone"]
        if ok and cost<best: best=cost; bestP=list(perm)
    return bestP,best
