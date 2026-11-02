import numpy as np
from src.data import generate
from src.search import route, brute_force
from src.integration import solve_all

def easy_instance(seed=0, n=6, travel=5.0, capacity=15, vans=3):
    inst = generate(seed, n)
    for o in inst["orders"]:
        o["tw"] = [540, 900]  # wide -> always feasible
        o["demand"] = 1
        o["service"] = 3
    inst["capacity"] = capacity
    inst["vans"] = vans
    if inst["depot"] not in inst["top20"]:
        inst["depot"] = inst["top20"][0]
    inst["seed"] = 99  # disable forced 7/12 rule
    zi = {z:i for i,z in enumerate(inst["top20"])}
    mat = np.full((20,20,24), travel)
    return inst, mat, zi

# 5 normal
def test_normal_astar_optimal_6():
    inst, m, zi = easy_instance(0, 6)
    p1, c1, _ = route(list(range(6)), inst, m, zi, "A*")
    p2, c2 = brute_force(list(range(6)), inst, m, zi)
    assert abs(c1-c2) < 1e-6

def test_normal_ucs_valid():
    inst, m, zi = easy_instance(1, 6)
    p, c, s = route(list(range(6)), inst, m, zi, "UCS")
    assert p is not None and c < float('inf') and s["expanded"] > 0

def test_normal_greedy_valid():
    inst, m, zi = easy_instance(2, 8)
    p, c, s = route(list(range(8)), inst, m, zi, "Greedy")
    assert p is not None and len(p) == 8

def test_normal_csp_solves20():
    inst, m, zi = easy_instance(3, 20, travel=5.0, capacity=15)
    r = solve_all(inst, m, zi)
    assert r["status"] == "OK", r

def test_normal_counters():
    inst, m, zi = easy_instance(4, 6)
    _, _, s = route(list(range(6)), inst, m, zi, "A*")
    assert "expanded" in s and "generated" in s

# 3 edge
def test_edge_exact_window():
    inst, m, zi = easy_instance(5, 2, travel=5.0)
    inst["orders"][0]["tw"] = [600, 610]
    inst["orders"][1]["tw"] = [615, 625]
    r = solve_all(inst, m, zi)
    assert r["status"] == "OK", r

def test_edge_capacity_full():
    inst, m, zi = easy_instance(6, 3, travel=5.0, capacity=5, vans=3)
    for o in inst["orders"]:
        o["demand"] = 5  # total 15 == 3*5 full
    r = solve_all(inst, m, zi)
    assert r["status"] == "OK", r

def test_edge_single_stop():
    inst, m, zi = easy_instance(7, 1)
    p, c, s = route([0], inst, m, zi, "A*")
    assert p == [0]

# 2 impossible -> must say NO SOLUTION
def test_impossible_same_van_forced():
    inst, m, zi = easy_instance(8, 2, travel=50.0)
    for o in inst["orders"]:
        o["tw"] = [600, 610]  # 50 min apart -> cannot share
    inst["vans"] = 1
    inst["seed"] = 99
    r = solve_all(inst, m, zi)
    assert "NO SOLUTION" in r["status"], r

def test_impossible_tiny_windows():
    inst, m, zi = easy_instance(9, 2, travel=20.0)
    for o in inst["orders"]:
        o["tw"] = [540, 545]  # depot 540+20=560 > 545
    r = solve_all(inst, m, zi)
    assert "NO SOLUTION" in r["status"], r
    assert r["reason"] == "DEPOT_UNREACHABLE"