"""Topic 1 — Informed search: A* (with BFS as an uninformed baseline).

A* expands the node with the smallest f(n) = g(n) + h(n):
    g(n) = cost of the best known path from the start to n,
    h(n) = Manhattan distance from n to the goal.
On a 4-connected grid with unit step cost the Manhattan distance never
overestimates the true remaining cost (it is admissible and consistent),
so the first time A* pops the goal its path is optimal.
"""
from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field

from .grid import Cell, Grid


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass
class SearchResult:
    algorithm: str
    path: list[Cell] | None            # start ... goal, or None if unreachable
    expanded: list[Cell] = field(default_factory=list)  # order in which nodes were expanded

    @property
    def found(self) -> bool:
        return self.path is not None

    @property
    def cost(self) -> int | None:
        return None if self.path is None else len(self.path) - 1

    @property
    def nodes_expanded(self) -> int:
        return len(self.expanded)


def _rebuild(came_from: dict[Cell, Cell], end: Cell) -> list[Cell]:
    path = [end]
    while path[-1] in came_from:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def astar(grid: Grid, start: Cell | None = None, goal: Cell | None = None) -> SearchResult:
    start = grid.start if start is None else start
    goal = grid.goal if goal is None else goal
    result = SearchResult("A*", None)
    if not grid.passable(start) or not grid.passable(goal):
        return result

    g: dict[Cell, int] = {start: 0}
    came_from: dict[Cell, Cell] = {}
    # Heap entries: (f, h, tie-break counter, cell). Lower h first on equal f
    # makes A* head towards the goal; the counter keeps the order deterministic.
    counter = 0
    frontier = [(manhattan(start, goal), manhattan(start, goal), counter, start)]
    closed: set[Cell] = set()

    while frontier:
        _, _, _, cell = heapq.heappop(frontier)
        if cell in closed:
            continue  # stale entry: a cheaper copy was already expanded
        closed.add(cell)
        result.expanded.append(cell)
        if cell == goal:
            result.path = _rebuild(came_from, cell)
            return result
        for nxt in grid.neighbors(cell):
            new_g = g[cell] + 1
            if new_g < g.get(nxt, float("inf")):
                g[nxt] = new_g
                came_from[nxt] = cell
                h = manhattan(nxt, goal)
                counter += 1
                heapq.heappush(frontier, (new_g + h, h, counter, nxt))
    return result


def bfs(grid: Grid, start: Cell | None = None, goal: Cell | None = None) -> SearchResult:
    """Breadth-first search: optimal for unit costs but ignores where the goal is."""
    start = grid.start if start is None else start
    goal = grid.goal if goal is None else goal
    result = SearchResult("BFS", None)
    if not grid.passable(start) or not grid.passable(goal):
        return result

    came_from: dict[Cell, Cell] = {}
    seen = {start}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        result.expanded.append(cell)
        if cell == goal:
            result.path = _rebuild(came_from, cell)
            return result
        for nxt in grid.neighbors(cell):
            if nxt not in seen:
                seen.add(nxt)
                came_from[nxt] = cell
                queue.append(nxt)
    return result
