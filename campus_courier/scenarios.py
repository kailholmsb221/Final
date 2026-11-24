"""Default campus map and the demo scenarios (all numbers are assumptions, not real data)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .bayes import SensorModel
from .mdp import CourierRewards

# 6 x 6 toy campus. S = mail room, G = dean's office, C = short corridor with a sensor.
CAMPUS_MAP: list[str] = [
    "S . . # . .",
    ". # . # . .",
    ". # . C . G",
    ". # . # # .",
    ". . . . . .",
    "# # . # . .",
]

DEFAULT_PRIOR = 0.20  # P(corridor blocked) before any reading (assumed from past days)


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    readings: list[str] = field(default_factory=list)
    prior: float = DEFAULT_PRIOR
    sensor: SensorModel = SensorModel()
    rewards: CourierRewards = CourierRewards()


SCENARIOS: dict[str, Scenario] = {
    s.key: s
    for s in [
        Scenario("no-sensor", "No sensor reading — decide from the prior only"),
        Scenario("clear", "Sensor says the corridor is clear", ["clear"]),
        Scenario("alarm", "Sensor raises one 'blocked' alarm", ["alarm"]),
        Scenario("two-alarms", "Two independent alarms in a row", ["alarm", "alarm"]),
        Scenario("mixed", "An alarm, then a clear reading", ["alarm", "clear"]),
    ]
}
