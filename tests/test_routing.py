import numpy as np
import pytest

from src.data import generate
from src.search import route, brute_force
from src.integration import solve_all
from src.csp import CSP


def easy_instance(seed=0, n=6, travel=5.0, capacity=15, vans=3):
    inst = generate(seed, n)
    for order in inst["orders"]:
        order["tw"] = [540, 900]
        order["demand"] = 1
        order["service"] = 3
    inst["capacity"] = capacity
    inst["vans"] = vans
    if inst["depot"] not in inst["top20"]:
        inst["depot"] = inst["top20"][0]
    zi = {zone: i for i, zone in enumerate(inst["top20"])}
    mat = np.full((20, 20, 24), travel)
    return inst, mat, zi


# Normal cases
def test_normal_astar_matches_brute_force():
    inst, matrix, zi = easy_instance(0, 6)
    _, astar_cost, _ = route(list(range(6)), inst, matrix, zi, "A*")
    _, exact_cost = brute_force(list(range(6)), inst, matrix, zi)
    assert astar_cost == pytest.approx(exact_cost)


def test_normal_ucs_returns_valid_route_and_counters():
    inst, matrix, zi = easy_instance(1, 6)
    path, cost, stats = route(list(range(6)), inst, matrix, zi, "UCS")
    assert path is not None and np.isfinite(cost)
    assert stats["expanded"] > 0 and stats["generated"] > 0


def test_normal_greedy_returns_all_stops():
    inst, matrix, zi = easy_instance(2, 8)
    path, _, _ = route(list(range(8)), inst, matrix, zi, "Greedy")
    assert path is not None and len(path) == 8


def test_normal_integrated_solver():
    inst, matrix, zi = easy_instance(3, 20)
    result = solve_all(inst, matrix, zi)
    assert result["status"] == "OK", result


def test_normal_csp_all_modes_solve_conflict_free_graph():
    inst, matrix, zi = easy_instance(4, 6)
    diff = [[False] * 6 for _ in range(6)]
    for mode in ("BT", "BT+FC", "BT+FC+AC-3"):
        solver = CSP(inst, diff)
        assignment = solver.solve(mode)
        assert assignment is not None
        assert len(assignment) == 6
        assert solver.nodes > 0 and solver.assignments > 0


# Edge cases
def test_edge_exact_window_boundary():
    inst, matrix, zi = easy_instance(5, 2)
    inst["orders"][0]["tw"] = [600, 610]
    inst["orders"][1]["tw"] = [615, 625]
    assert solve_all(inst, matrix, zi)["status"] == "OK"


def test_edge_capacity_exactly_full():
    inst, matrix, zi = easy_instance(6, 3, capacity=5, vans=3)
    for order in inst["orders"]:
        order["demand"] = 5
    assert solve_all(inst, matrix, zi)["status"] == "OK"


def test_edge_single_stop():
    inst, matrix, zi = easy_instance(7, 1)
    path, _, _ = route([0], inst, matrix, zi, "A*")
    assert path == [0]


def test_edge_ac3_propagates_inequality_chain():
    inst, _, _ = easy_instance(8, 3)
    diff = [[False] * 3 for _ in range(3)]
    diff[0][1] = diff[1][0] = True
    diff[1][2] = diff[2][1] = True
    solver = CSP(inst, diff)
    domains = [{0}, {0, 1}, {1, 2}]
    assert solver.ac3(domains)
    assert domains == [{0}, {1}, {2}]


# Impossible cases: solver must report no solution
def test_impossible_one_van_for_incompatible_orders():
    inst, matrix, zi = easy_instance(9, 2, travel=50.0, vans=1)
    for order in inst["orders"]:
        order["tw"] = [600, 610]
    result = solve_all(inst, matrix, zi)
    assert "NO SOLUTION" in result["status"]


def test_impossible_depot_unreachable():
    inst, matrix, zi = easy_instance(10, 2, travel=20.0)
    for order in inst["orders"]:
        order["tw"] = [540, 545]
    result = solve_all(inst, matrix, zi)
    assert "NO SOLUTION" in result["status"]
    assert result["reason"] == "DEPOT_UNREACHABLE"


def test_impossible_ac3_empty_domain():
    inst, _, _ = easy_instance(11, 2)
    diff = [[False, True], [True, False]]
    solver = CSP(inst, diff)
    domains = [{0}, {0}]
    assert solver.ac3(domains) is False



def test_exp3_replays_route_order_instead_of_sorting_time_windows():
    from src.eval import _replay_route

    orders = {
        0: {"id": 0, "zone": 2, "tw": [540, 545], "service": 0},
        1: {"id": 1, "zone": 3, "tw": [550, 560], "service": 0},
    }
    zi = {1: 0, 2: 1, 3: 2}
    matrix = np.ones((3, 3, 24), dtype=float)
    matrix[0, 1, :] = 50.0
    matrix[0, 2, :] = 1.0
    matrix[2, 1, :] = 1.0

    replay = _replay_route([1, 0], orders, matrix, zi, 540, 1)

    assert replay["late"] == 1
    assert replay["late_minutes"] == pytest.approx(6.0)
    assert replay["on_time_rate"] == pytest.approx(0.5)


def test_exp3_solved_rate_includes_failed_attempts():
    import pandas as pd
    from src.eval import _aggregate_exp3

    raw = pd.DataFrame([
        {"seed": 0, "model": "NN", "late": 2, "late_minutes": 8.0,
         "max_late_minutes": 5.0, "on_time_rate": 0.5, "solved": True},
        {"seed": 1, "model": "NN", "late": np.nan, "late_minutes": np.nan,
         "max_late_minutes": np.nan, "on_time_rate": np.nan, "solved": False},
    ])
    aggregate = _aggregate_exp3(raw).iloc[0]

    assert aggregate["solved_rate"] == pytest.approx(0.5)
    assert aggregate["attempts"] == 2
    assert aggregate["solved_runs"] == 1
    assert aggregate["mean_late"] == pytest.approx(2.0)
