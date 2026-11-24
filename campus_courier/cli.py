"""Console demo: input -> key steps -> output -> one-sentence explanation for each algorithm."""
from __future__ import annotations

import argparse
import sys

from .bayes import SensorModel
from .grid import Grid
from .mdp import BLOCKED, FREE, CourierRewards
from .pipeline import MissionReport, run_mission
from .scenarios import CAMPUS_MAP, SCENARIOS

LINE = "=" * 64


def _fmt_path(path) -> str:
    return " -> ".join(f"({r},{c})" for r, c in path) if path else "no path"


def print_report(rep: MissionReport, title: str = "") -> None:
    g = rep.grid
    print(LINE)
    print("CAMPUS COURIER AI" + (f" — {title}" if title else ""))
    print(LINE)
    print("Map (S = mail room, G = office, C = monitored corridor, # = wall):")
    print(g.render())

    # 1. Search
    print(f"\n[1] A* SEARCH  (heuristic: Manhattan distance)")
    print(f"  input : start {g.start}, goal {g.goal}, {len(g.walls)} walls")
    print(f"  steps : A* expanded {rep.main.nodes_expanded} cells, "
          f"BFS would expand {rep.main_bfs.nodes_expanded}")
    print(f"  output: shortest route, {rep.main.cost} steps: {_fmt_path(rep.main.path)}")
    if not rep.main.found:
        print("  -> the goal is unreachable; the mission stops here.")
        return
    if not rep.corridor_on_route:
        print("  -> the shortest route does not use the corridor, so no risk decision is needed.")
        return
    if rep.detour.found:
        print(f"  detour: corridor C={g.corridor} removed, {rep.detour.cost} steps: "
              f"{_fmt_path(rep.detour.path)}")
    else:
        print(f"  detour: none — without C={g.corridor} the goal is unreachable")
    print("  why   : A* gives both candidate routes and their lengths, which become MDP rewards.")

    # 2. Bayes
    print(f"\n[2] BAYESIAN INFERENCE  (is the corridor blocked?)")
    print(f"  input : prior P(Blocked) = {rep.prior:.2f}, sensor P(alarm|Blocked) = "
          f"{rep.sensor.sensitivity:.2f}, P(alarm|Free) = {rep.sensor.false_alarm_rate:.2f}")
    print(f"          readings = {[st.reading for st in rep.bayes_steps] or 'none'}")
    for i, st in enumerate(rep.bayes_steps, 1):
        print(f"  step {i}: {st.explain()}")
    print(f"  output: P(Blocked | evidence) = {rep.belief:.4f}")
    print("  why   : the sensor is noisy, so we reason with a probability instead of trusting it blindly.")

    # 3. MDP
    vi = rep.vi
    print(f"\n[3] MDP + VALUE ITERATION  (Proceed through C or Detour?)")
    rw = rep.rewards
    print(f"  input : states {{Free, Blocked}}, actions {{Proceed, Detour}}, gamma = {rw.gamma}")
    print(f"          goal reward {rw.goal_reward:g}, step cost {rw.step_cost:g}, delay penalty "
          f"{rw.delay_penalty:g}, P(unblock) = {rw.p_unblock:g}")
    print(f"  steps : converged after {vi.iterations} iterations; "
          f"V(Free) = {vi.values[FREE]:.2f}, V(Blocked) = {vi.values[BLOCKED]:.2f}")
    for state in (FREE, BLOCKED):
        qs = ", ".join(f"Q({state},{a}) = {v:.2f}" for a, v in vi.q[state].items())
        print(f"          {qs}")
    qb = ", ".join(f"{a}: {v:.2f}" for a, v in rep.q_belief.items())
    print(f"          belief-weighted values with b = {rep.belief:.3f}: {qb}")
    if rep.threshold is not None:
        print(f"          Detour is better whenever P(Blocked) > {rep.threshold:.3f}")
    print(f"  output: ACTION = {rep.action}")
    print("  why   : the MDP trades the shorter route against the cost of hitting a blocked corridor.")

    if rep.final_route is None:
        print("\nMISSION ABORTED: the corridor is almost surely blocked and there is no detour.")
        return
    print(f"\nFINAL ROUTE ({len(rep.final_route) - 1} steps): {_fmt_path(rep.final_route)}")
    print(g.render(rep.final_route))
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m campus_courier",
                                     description="Campus Courier AI — A* + Bayes + MDP demo")
    parser.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all")
    parser.add_argument("--readings", nargs="*", choices=["alarm", "clear"],
                        help="custom sensor readings (overrides --scenario)")
    parser.add_argument("--prior", type=float, default=0.20)
    parser.add_argument("--sensitivity", type=float, default=0.90)
    parser.add_argument("--false-alarm", type=float, default=0.20)
    args = parser.parse_args(argv)

    grid = Grid.from_strings(CAMPUS_MAP)
    if args.readings is not None:
        sensor = SensorModel(args.sensitivity, args.false_alarm)
        rep = run_mission(grid, args.prior, args.readings, sensor, CourierRewards())
        print_report(rep, "custom run")
        return 0

    keys = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for key in keys:
        sc = SCENARIOS[key]
        rep = run_mission(grid, sc.prior, sc.readings, sc.sensor, sc.rewards)
        print_report(rep, sc.title)
    return 0


if __name__ == "__main__":
    sys.exit(main())
