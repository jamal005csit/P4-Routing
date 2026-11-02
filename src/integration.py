from src.csp import build_diff_matrix, CSP
from src.search import route
def solve_all(inst,mat,zi,algo="A*",mode="BT+FC+AC-3"):
    diff=build_diff_matrix(inst,mat,zi)
    # depot check
    for o in inst["orders"]:
        td=mat[zi[inst["depot"]],zi[o["zone"]],(inst["depot_start"]//60)%24]
        if inst["depot_start"]+td>o["tw"][1]:
            return {"status":"UNSAT","reason":"DEPOT_UNREACHABLE","order":o["id"],"assign":None}
    for it in range(5):
        c=CSP(inst,diff); assign=c.solve(mode)
        if assign is None: return {"status":"NO SOLUTION: PAIRWISE_CONFLICT+CAPACITY unsat","reason":"PAIRWISE_CONFLICT","assign":None,"stats":vars(c)}
        routes={v:[o for o,a in assign.items() if a==v] for v in range(inst["vans"])}
        late=None
        for v,oids in routes.items():
            if not oids: continue
            path,cost,st=route(oids,inst,mat,zi,algo)
            if path is None: late=(oids[0],oids[1] if len(oids)>1 else oids[0]); break
            clock=inst["depot_start"]; cur=inst["depot"]; mp={o["id"]:o for o in inst["orders"]}
            for idx,oid in enumerate(path):
                t=mat[zi[cur],zi[mp[oid]["zone"]],(int(clock)//60)%24]
                if clock+t>mp[oid]["tw"][1]: late=(path[max(0,idx-1)],oid); break
                clock=max(clock+t,mp[oid]["tw"][0])+mp[oid]["service"]; cur=mp[oid]["zone"]
            if late: break
        if not late: return {"status":"OK","assign":assign,"routes":routes,"loops":it,"csp_stats":{"nodes":c.nodes,"assignments":c.assignments,"backtracks":c.backtracks}}
        diff[late[0]][late[1]]=diff[late[1]][late[0]]=True
    return {"status":"NO SOLUTION: NO_FEASIBLE_ORDER after 5 loops","reason":"NO_FEASIBLE_ORDER","late_pair":late,"assign":assign}
def widen(inst,step=15):
    for o in inst["orders"]: o["tw"]=[o["tw"][0]-step,o["tw"][1]+step]
    return inst
