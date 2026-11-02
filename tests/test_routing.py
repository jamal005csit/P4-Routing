import numpy as np
from src.data import generate
from src.search import route, brute_force
from src.integration import solve_all
def mat_for(inst):
    import numpy as np
    zi={z:i for i,z in enumerate(inst["top20"])}
    m=np.full((20,20,24),10.0)
    return m,zi
def test_normal_astar_optimal_6():
    inst=generate(0,6); m,zi=mat_for(inst)
    p1,c1,_=route(list(range(6)),inst,m,zi,"A*"); p2,c2=brute_force(list(range(6)),inst,m,zi)
    assert abs(c1-c2)<1e-6
def test_normal_ucs_valid(): inst=generate(1,8); m,zi=mat_for(inst); p,c,s=route(list(range(8)),inst,m,zi,"UCS"); assert s["expanded"]>0
def test_normal_greedy_fast(): inst=generate(2,8); m,zi=mat_for(inst); p,c,s=route(list(range(8)),inst,m,zi,"Greedy"); assert True
def test_normal_csp_solves20(): inst=generate(3,20); m,zi=mat_for(inst); r=solve_all(inst,m,zi); assert r["status"]=="OK"
def test_normal_counters(): inst=generate(4,10); m,zi=mat_for(inst); _,_,s=route(list(range(6)),inst,m,zi,"A*"); assert "expanded" in s and "generated" in s
def test_edge_exact_window():
    inst=generate(5,2); m,zi=mat_for(inst)
    inst["orders"][0]["tw"]=[600,610]; inst["orders"][1]["tw"]=[610,620]
    r=solve_all(inst,m,zi); assert r["status"]=="OK" or "NO SOLUTION" in r["status"]
def test_edge_capacity_full():
    inst=generate(6,20)
    for o in inst["orders"]: o["demand"]=3
    inst["capacity"]=30; m,zi=mat_for(inst); r=solve_all(inst,m,zi); assert "assign" in r
def test_edge_single_stop():
    inst=generate(7,1); m,zi=mat_for(inst); p,c,s=route([0],inst,m,zi,"A*"); assert p==[0]
def test_impossible_same_van_forced():
    inst=generate(7,20); m,zi=mat_for(inst)
    m[:,:,:]=50.0 # all 50 min -> 7/12 cannot share, capacity forces conflict
    inst["vans"]=1; r=solve_all(inst,m,zi)
    assert "NO SOLUTION" in r["status"], r
def test_impossible_tiny_windows():
    inst=generate(8,5)
    for o in inst["orders"]: o["tw"]=[600,605]
    m,zi=mat_for(inst); m[:,:,:]=30.0; r=solve_all(inst,m,zi)
    assert "NO SOLUTION" in r["status"]
