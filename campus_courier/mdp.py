"""Topic 3 — Sequential decision making: Markov Decision Process + value iteration.

An MDP is (S, A, T, R, gamma). Value iteration repeats the Bellman optimality update
    V_{k+1}(s) = max_a  sum_{s'} T(s, a, s') [ R(s, a, s') + gamma * V_k(s') ]
until the largest change is below `theta`. Then
    Q(s, a)  = sum_{s'} T(s, a, s') [ R(s, a, s') + gamma * V(s') ]
    pi(s)    = argmax_a Q(s, a).

The courier cannot observe the corridor directly, it only has a belief b = P(Blocked)
from the Bayes module. We choose the action with the highest belief-weighted value
    Q_b(a) = b * Q(Blocked, a) + (1 - b) * Q(Free, a)
(the QMDP approximation: solve the fully observable MDP, then average over the belief).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# transitions[state][action] = [(probability, next_state, reward), ...]
Transitions = dict[str, dict[str, list[tuple[float, str, float]]]]

FREE, BLOCKED, DELIVERED = "Free", "Blocked", "Delivered"
PROCEED, DETOUR = "Proceed", "Detour"


@dataclass
class MDP:
    states: list[str]
    transitions: Transitions
    gamma: float = 0.9

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must be in [0, 1)")
        for s, actions in self.transitions.items():
            for a, outcomes in actions.items():
                total = sum(p for p, _, _ in outcomes)
                if abs(total - 1.0) > 1e-9:
                    raise ValueError(f"T({s}, {a}, .) sums to {total}, expected 1")

    def actions(self, state: str) -> list[str]:
        return list(self.transitions.get(state, {}))  # terminal states have no actions

    def q_value(self, state: str, action: str, values: dict[str, float]) -> float:
        return sum(p * (r + self.gamma * values[nxt])
                   for p, nxt, r in self.transitions[state][action])


@dataclass
class ValueIterationResult:
    values: dict[str, float]
    q: dict[str, dict[str, float]]
    policy: dict[str, str]
    iterations: int
    history: list[dict[str, float]] = field(default_factory=list)  # V_0, V_1, ...


def value_iteration(mdp: MDP, theta: float = 1e-6, max_iter: int = 1000) -> ValueIterationResult:
    values = {s: 0.0 for s in mdp.states}
    history = [dict(values)]
    for k in range(1, max_iter + 1):
        new_values = {}
        for s in mdp.states:
            acts = mdp.actions(s)
            new_values[s] = max(mdp.q_value(s, a, values) for a in acts) if acts else 0.0
        delta = max(abs(new_values[s] - values[s]) for s in mdp.states)
        values = new_values
        history.append(dict(values))
        if delta < theta:
            break
    q = {s: {a: mdp.q_value(s, a, values) for a in mdp.actions(s)} for s in mdp.states}
    policy = {s: max(q[s], key=q[s].get) for s in mdp.states if q[s]}
    return ValueIterationResult(values, q, policy, k, history)


@dataclass(frozen=True)
class CourierRewards:
    goal_reward: float = 20.0     # reward for delivering the parcel
    step_cost: float = 1.0        # cost of every grid step
    delay_penalty: float = 4.0    # extra cost when the robot hits a blocked corridor
    p_unblock: float = 0.4        # P(corridor becomes free while the robot comes back)
    gamma: float = 0.9


def build_courier_mdp(main_len: int, detour_len: int | None, dist_to_corridor: int,
                      rw: CourierRewards = CourierRewards()) -> MDP:
    """Two-state / two-action courier MDP whose rewards come from the A* route lengths.

    Free,    Proceed -> Delivered            R = goal - cost * main_len
    *,       Detour  -> Delivered            R = goal - cost * detour_len
    Blocked, Proceed -> back at start: Free (p_unblock) or Blocked (1 - p_unblock)
                                             R = -(cost * 2 * dist_to_corridor + delay_penalty)
    If no detour exists, Detour means failing the delivery: R = -goal.
    """
    deliver_main = rw.goal_reward - rw.step_cost * main_len
    deliver_detour = (rw.goal_reward - rw.step_cost * detour_len
                      if detour_len is not None else -rw.goal_reward)
    wasted_trip = -(rw.step_cost * 2 * dist_to_corridor + rw.delay_penalty)
    transitions: Transitions = {
        FREE: {
            PROCEED: [(1.0, DELIVERED, deliver_main)],
            DETOUR: [(1.0, DELIVERED, deliver_detour)],
        },
        BLOCKED: {
            PROCEED: [(rw.p_unblock, FREE, wasted_trip),
                      (1.0 - rw.p_unblock, BLOCKED, wasted_trip)],
            DETOUR: [(1.0, DELIVERED, deliver_detour)],
        },
    }
    return MDP([FREE, BLOCKED, DELIVERED], transitions, rw.gamma)


def belief_q(q: dict[str, dict[str, float]], p_blocked: float) -> dict[str, float]:
    """Q_b(a) = b * Q(Blocked, a) + (1 - b) * Q(Free, a)."""
    return {a: p_blocked * q[BLOCKED][a] + (1.0 - p_blocked) * q[FREE][a] for a in q[FREE]}


def decision_threshold(q: dict[str, dict[str, float]]) -> float | None:
    """Belief P(Blocked) above which Detour beats Proceed (None if one action always wins)."""
    gain_free = q[FREE][PROCEED] - q[FREE][DETOUR]          # >0: proceeding is better if free
    gain_blocked = q[BLOCKED][PROCEED] - q[BLOCKED][DETOUR]  # <0: proceeding is worse if blocked
    denom = gain_free - gain_blocked
    if denom == 0:
        return None
    b = gain_free / denom
    return b if 0.0 <= b <= 1.0 else None
