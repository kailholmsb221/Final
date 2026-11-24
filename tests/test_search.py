import random

import pytest

from campus_courier.grid import Grid
from campus_courier.scenarios import CAMPUS_MAP
from campus_courier.search import astar, bfs, manhattan


def random_grid(rng: random.Random, n: int, wall_ratio: float) -> Grid:
    lines = []
    for r in range(n):
        row = []
        for c in range(n):
            if (r, c) == (0, 0):
                row.append("S")
            elif (r, c) == (n - 1, n - 1):
                row.append("G")
            else:
                row.append("#" if rng.random() < wall_ratio else ".")
        lines.append("".join(row))
    return Grid.from_strings(lines)


def is_valid_path(grid: Grid, path) -> bool:
    if path[0] != grid.start or path[-1] != grid.goal:
        return False
    return all(grid.passable(p) for p in path) and all(
        manhattan(a, b) == 1 for a, b in zip(path, path[1:]))


def test_campus_route_is_shortest_and_valid():
    grid = Grid.from_strings(CAMPUS_MAP)
    res = astar(grid)
    assert res.found and res.cost == 7
    assert is_valid_path(grid, res.path)
    assert grid.corridor in res.path


def test_astar_expands_fewer_nodes_than_bfs_on_campus():
    grid = Grid.from_strings(CAMPUS_MAP)
    assert astar(grid).nodes_expanded < bfs(grid).nodes_expanded


def test_detour_avoids_corridor():
    grid = Grid.from_strings(CAMPUS_MAP)
    res = astar(grid.with_blocked(grid.corridor))
    assert res.found and res.cost == 11
    assert grid.corridor not in res.path


def test_unreachable_goal_returns_none():
    grid = Grid.from_strings(["S#.", "##.", "..G"])
    assert astar(grid).path is None
    assert bfs(grid).path is None


def test_start_equals_neighbour_of_goal():
    grid = Grid.from_strings(["SG"])
    assert astar(grid).path == [(0, 0), (0, 1)]


@pytest.mark.parametrize("seed", range(200))
def test_astar_matches_bfs_optimal_cost_on_random_maps(seed):
    """A* with an admissible heuristic must return the same optimal cost as BFS."""
    rng = random.Random(seed)
    grid = random_grid(rng, rng.randint(4, 10), rng.uniform(0.1, 0.35))
    a, b = astar(grid), bfs(grid)
    assert a.found == b.found
    if a.found:
        assert a.cost == b.cost
        assert is_valid_path(grid, a.path)
        assert a.nodes_expanded <= b.nodes_expanded


def test_heuristic_is_admissible_on_campus():
    grid = Grid.from_strings(CAMPUS_MAP)
    for r in range(grid.rows):
        for c in range(grid.cols):
            cell = (r, c)
            if grid.passable(cell):
                true_cost = astar(grid, start=cell).cost
                if true_cost is not None:
                    assert manhattan(cell, grid.goal) <= true_cost


def test_bad_map_is_rejected():
    with pytest.raises(ValueError):
        Grid.from_strings(["S..", ".."])
    with pytest.raises(ValueError):
        Grid.from_strings(["...", "..G"])
