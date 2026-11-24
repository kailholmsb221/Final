import pytest

from campus_courier.cli import main
from campus_courier.grid import Grid
from campus_courier.mdp import DETOUR, PROCEED
from campus_courier.pipeline import run_mission
from campus_courier.scenarios import CAMPUS_MAP, SCENARIOS


@pytest.fixture
def grid():
    return Grid.from_strings(CAMPUS_MAP)


@pytest.mark.parametrize("key, action, steps", [
    ("no-sensor", PROCEED, 7),
    ("clear", PROCEED, 7),
    ("alarm", DETOUR, 11),
    ("two-alarms", DETOUR, 11),
    ("mixed", PROCEED, 7),
])
def test_scenarios(grid, key, action, steps):
    sc = SCENARIOS[key]
    rep = run_mission(grid, sc.prior, sc.readings, sc.sensor, sc.rewards)
    assert rep.action == action
    assert len(rep.final_route) - 1 == steps
    assert rep.delivered


def test_detour_route_never_uses_corridor(grid):
    rep = run_mission(grid, readings=["alarm"])
    assert grid.corridor not in rep.final_route


def test_decision_flips_exactly_at_threshold(grid):
    thr = run_mission(grid).threshold
    assert run_mission(grid, prior=thr - 0.01).action == PROCEED
    assert run_mission(grid, prior=thr + 0.01).action == DETOUR


def test_corridor_off_route_skips_decision():
    g = Grid.from_strings(["S..G", "....", "..C."])
    rep = run_mission(g, readings=["alarm"])
    assert not rep.corridor_on_route and rep.action == PROCEED and rep.final_route == rep.main.path


def test_unreachable_goal_is_reported():
    g = Grid.from_strings(["S#G"])
    rep = run_mission(g)
    assert not rep.delivered and rep.action == "No route"


def test_cli_runs_all_scenarios(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert out.count("FINAL ROUTE") == len(SCENARIOS)
    assert "ACTION = Detour" in out and "ACTION = Proceed" in out


def test_cli_custom_readings(capsys):
    assert main(["--readings", "alarm", "alarm", "--prior", "0.1"]) == 0
    assert "ACTION = Detour" in capsys.readouterr().out


def test_no_detour_and_certain_blockage_aborts():
    # One long corridor, no way around it, C is 6 steps away: retrying is expensive.
    g = Grid.from_strings(["S.....C.G"])
    assert run_mission(g, readings=["alarm"]).action == PROCEED     # belief 0.53: still try
    rep = run_mission(g, prior=0.999, readings=["alarm"])
    assert rep.action == "Abort" and not rep.delivered
