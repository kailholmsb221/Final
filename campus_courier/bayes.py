"""Topic 2 — Probabilistic reasoning: Bayes' rule for a noisy sensor.

Hidden variable  B in {Blocked, Free}  — is the monitored corridor blocked?
Evidence         E in {alarm, clear}   — what the corridor sensor reports.

Sensor model (given, not learned):
    P(alarm | Blocked) = sensitivity       (true-positive rate)
    P(alarm | Free)    = false_alarm_rate  (false-positive rate)

Bayes' rule:
    P(Blocked | e) = P(e | Blocked) P(Blocked) / P(e)
    P(e)           = P(e | Blocked) P(Blocked) + P(e | Free) P(Free)
Several readings are treated as conditionally independent given B, so we
simply apply the rule again with the previous posterior as the new prior.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SensorModel:
    sensitivity: float = 0.90       # P(alarm | Blocked)
    false_alarm_rate: float = 0.20  # P(alarm | Free)

    def __post_init__(self) -> None:
        for name in ("sensitivity", "false_alarm_rate"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a probability, got {value}")

    def likelihoods(self, reading: str) -> tuple[float, float]:
        """Return (P(reading | Blocked), P(reading | Free))."""
        if reading == "alarm":
            return self.sensitivity, self.false_alarm_rate
        if reading == "clear":
            return 1.0 - self.sensitivity, 1.0 - self.false_alarm_rate
        raise ValueError(f"reading must be 'alarm' or 'clear', got {reading!r}")


@dataclass(frozen=True)
class BayesStep:
    reading: str
    prior: float
    likelihood_blocked: float
    likelihood_free: float
    evidence: float   # P(e)
    posterior: float  # P(Blocked | e)

    def explain(self) -> str:
        return (
            f"P(Blocked | {self.reading}) = {self.likelihood_blocked:.2f} x {self.prior:.3f} / "
            f"({self.likelihood_blocked:.2f} x {self.prior:.3f} + {self.likelihood_free:.2f} x "
            f"{1 - self.prior:.3f}) = {self.likelihood_blocked * self.prior:.4f} / "
            f"{self.evidence:.4f} = {self.posterior:.4f}"
        )


def bayes_update(prior: float, reading: str, sensor: SensorModel) -> BayesStep:
    if not 0.0 <= prior <= 1.0:
        raise ValueError(f"prior must be a probability, got {prior}")
    l_blocked, l_free = sensor.likelihoods(reading)
    evidence = l_blocked * prior + l_free * (1.0 - prior)
    if evidence == 0.0:
        raise ValueError("this reading is impossible under the sensor model (P(e) = 0)")
    posterior = l_blocked * prior / evidence
    return BayesStep(reading, prior, l_blocked, l_free, evidence, posterior)


def bayes_sequence(prior: float, readings: list[str], sensor: SensorModel) -> list[BayesStep]:
    """Apply Bayes' rule once per reading; returns every intermediate step."""
    steps: list[BayesStep] = []
    belief = prior
    for reading in readings:
        step = bayes_update(belief, reading, sensor)
        steps.append(step)
        belief = step.posterior
    return steps
