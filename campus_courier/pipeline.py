"""The three algorithms working together on one goal: deliver the parcel.

    1. A*   plans the shortest route S -> G and the detour that avoids corridor C.
    2. Bayes turns the noisy sensor readings about C into P(C is blocked).
    3. MDP  uses the two A* route lengths as rewards, solves the model with value
            iteration and, weighted by the Bayes belief, picks Proceed or Detour.
    The final route is the A* path that matches the chosen action.
"""
from __future__ import annotations

from dataclasses import dataclass

from .bayes import BayesStep, SensorModel, bayes_sequence
from .grid import Cell, Grid
from .mdp import (DETOUR, PROCEED, CourierRewards, ValueIterationResult, belief_q,
                  build_courier_mdp, decision_threshold, value_iteration)
from .search import SearchResult, astar, bfs


@dataclass
class MissionReport:
    grid: Grid
    main: SearchResult
    main_bfs: SearchResult
    detour: SearchResult | None
    corridor_on_route: bool
    dist_to_corridor: int | None
    prior: float
    bayes_steps: list[BayesStep]
    belief: float
    vi: ValueIterationResult | None
    q_belief: dict[str, float] | None
    threshold: float | None
    action: str
    final_route: list[Cell] | None
    rewards: CourierRewards = CourierRewards()
    sensor: SensorModel = SensorModel()

    @property
    def delivered(self) -> bool:
        return self.final_route is not None


def run_mission(grid: Grid, prior: float = 0.2, readings: list[str] | None = None,
                sensor: SensorModel = SensorModel(),
                rewards: CourierRewards = CourierRewards()) -> MissionReport:
    readings = list(readings or [])

    # 1) Search ---------------------------------------------------------------
    main = astar(grid)
    main_bfs = bfs(grid)

    # 2) Bayes ----------------------------------------------------------------
    steps = bayes_sequence(prior, readings, sensor)
    belief = steps[-1].posterior if steps else prior

    corridor = grid.corridor
    on_route = bool(main.found and corridor is not None and corridor in main.path)
    if not main.found:
        return MissionReport(grid, main, main_bfs, None, False, None, prior, steps, belief,
                             None, None, None, "No route", None, rewards, sensor)
    if not on_route:
        # The sensor does not matter: the optimal route never enters the corridor.
        return MissionReport(grid, main, main_bfs, None, False, None, prior, steps, belief,
                             None, None, None, PROCEED, main.path, rewards, sensor)

    detour = astar(grid.with_blocked(corridor))
    dist = main.path.index(corridor)

    # 3) MDP ------------------------------------------------------------------
    mdp = build_courier_mdp(main.cost, detour.cost, dist, rewards)
    vi = value_iteration(mdp)
    q_b = belief_q(vi.q, belief)
    action = max(q_b, key=q_b.get)
    if action == DETOUR and not detour.found:
        action = "Abort"  # no way around the corridor and the risk is too high
    final = main.path if action == PROCEED else detour.path
    return MissionReport(grid, main, main_bfs, detour, True, dist, prior, steps, belief,
                         vi, q_b, decision_threshold(vi.q), action, final, rewards, sensor)


__all__ = ["MissionReport", "run_mission", "PROCEED", "DETOUR"]
