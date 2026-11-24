"""Campus map as a small 2D grid.

Legend used in map strings:
    S  start (mail room)          G  goal (delivery office)
    #  wall                       C  suspicious corridor cell (monitored by a sensor)
    .  free cell
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

Cell = tuple[int, int]  # (row, col)

# 4-connected moves: up, down, left, right. Each move costs 1.
MOVES: tuple[Cell, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass
class Grid:
    rows: int
    cols: int
    walls: set[Cell]
    start: Cell
    goal: Cell
    corridor: Cell | None = None
    extra_blocked: set[Cell] = field(default_factory=set)

    @classmethod
    def from_strings(cls, lines: list[str]) -> "Grid":
        """Parse a map such as ["S.#", "..G"] (spaces are ignored)."""
        rows = [line.replace(" ", "") for line in lines if line.strip()]
        if not rows:
            raise ValueError("map is empty")
        width = len(rows[0])
        if any(len(r) != width for r in rows):
            raise ValueError("all map rows must have the same length")

        walls: set[Cell] = set()
        start = goal = corridor = None
        for r, line in enumerate(rows):
            for c, ch in enumerate(line):
                if ch == "#":
                    walls.add((r, c))
                elif ch == "S":
                    start = (r, c)
                elif ch == "G":
                    goal = (r, c)
                elif ch == "C":
                    corridor = (r, c)
                elif ch != ".":
                    raise ValueError(f"unknown map symbol {ch!r} at {(r, c)}")
        if start is None or goal is None:
            raise ValueError("map needs exactly one S and one G")
        return cls(len(rows), width, walls, start, goal, corridor)

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def passable(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and cell not in self.walls and cell not in self.extra_blocked

    def neighbors(self, cell: Cell) -> Iterator[Cell]:
        r, c = cell
        for dr, dc in MOVES:
            nxt = (r + dr, c + dc)
            if self.passable(nxt):
                yield nxt

    def with_blocked(self, cell: Cell) -> "Grid":
        """Copy of the map where `cell` is treated as an obstacle (used for the detour)."""
        return Grid(self.rows, self.cols, set(self.walls), self.start, self.goal,
                    self.corridor, set(self.extra_blocked) | {cell})

    def render(self, path: list[Cell] | None = None) -> str:
        """ASCII picture of the map; '*' marks the path."""
        on_path = set(path or [])
        out = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                cell = (r, c)
                if cell == self.start:
                    ch = "S"
                elif cell == self.goal:
                    ch = "G"
                elif cell in self.walls:
                    ch = "#"
                elif cell in self.extra_blocked:
                    ch = "X"
                elif cell in on_path:
                    ch = "*"
                elif cell == self.corridor:
                    ch = "C"
                else:
                    ch = "."
                row.append(ch)
            out.append(" ".join(row))
        return "\n".join(out)
