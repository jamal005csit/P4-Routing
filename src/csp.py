def build_diff_matrix(inst,mat,zi):
    n=len(inst["orders"]); diff=[[False]*n for _ in range(n)]; depot=inst["depot"]
    for i in range(n):
        for j in range(i+1,n):
            feas=False
            for X,Y in [(inst["orders"][i],inst["orders"][j]),(inst["orders"][j],inst["orders"][i])]:
                td=mat[zi[depot],zi[X["zone"]],(inst["depot_start"]//60)%24]
                if inst["depot_start"]+td>X["tw"][1]: continue
                t=mat[zi[X["zone"]],zi[Y["zone"]],(X["tw"][0]//60)%24]
                arr=max(X["tw"][0]+X["service"]+t,Y["tw"][0])
                if arr<=Y["tw"][1]: feas=True; break
            if not feas: diff[i][j]=diff[j][i]=True
    return diff
class CSP:
    def __init__(self,inst,diff):
        self.inst=inst; self.diff=diff; self.n=len(inst["orders"]); self.V=inst["vans"]
        self.nodes=0; self.assignments=0; self.backtracks=0
    def solve(self,mode="BT+FC+AC-3"):
        dom=[set(range(self.V)) for _ in range(self.n)]
        order=sorted(range(self.n),key=lambda i: sum(self.diff[i]),reverse=True)
        if "AC-3" in mode:
            if not self.ac3(dom): return None
        return self.bt({},order,dom,mode)
    def bt(self,assign,order,dom,mode):
        self.nodes+=1
        if len(assign)==self.n: return dict(assign) if self.capacity_ok(assign) else None
        v=next(x for x in order if x not in assign)
        for val in list(dom[v]):
            if any(assign.get(u)==val and self.diff[v][u] for u in assign): continue
            self.assignments+=1; assign[v]=val
            nd=[set(d) for d in dom]; ok=True
            if "FC" in mode or "AC-3" in mode:
                for u in range(self.n):
                    if u not in assign and self.diff[v][u] and val in nd[u]:
                        nd[u].remove(val)
                        if not nd[u]: ok=False; break
                if ok and "AC-3" in mode: ok=self.ac3(nd,assign)
            if ok:
                s=[0]*self.V
                for k,vv in assign.items(): s[vv]+=self.inst["orders"][k]["demand"]
                if all(x<=self.inst["capacity"] for x in s):
                    r=self.bt(assign,order,nd,mode)
                    if r: return r
            del assign[v]; self.backtracks+=1
        return None
    def ac3(self,dom,assign={}):
        q=[(i,j) for i in range(self.n) for j in range(self.n) if self.diff[i][j]]
        while q:
            i,j=q.pop()
            if i in assign or j in assign: continue
            if len(dom[j])==1:
                v=next(iter(dom[j]))
                if v in dom[i]:
                    if len(dom[i])==1: return False
                    dom[i].remove(v)
                    for k in range(self.n):
                        if self.diff[k][i]: q.append((k,i))
        return True
    def capacity_ok(self,a):
        s=[0]*self.V
        for k,v in a.items(): s[v]+=self.inst["orders"][k]["demand"]
        return all(x<=self.inst["capacity"] for x in s)
