# Campus Courier AI — Technical Report

**Course:** AI Fundamentals (QAIRU), Final Project · **Team:** Alikhan Ashimkhan, Ansar Amangeldy
**Repository:** https://github.com/kailholmsb221/Final

---

## Abstract

We build a decision system for a campus delivery robot that must choose between a short
corridor that may be blocked and a longer safe detour. Three algorithms from three course
topics are chained: **A\* search** produces both candidate routes and their lengths,
**Bayesian inference** converts noisy sensor readings into a probability that the corridor
is blocked, and a **Markov Decision Process** solved by **value iteration** turns route
lengths into rewards and, weighted by that probability, selects the action. On the demo
map the robot keeps the 7-step route when the risk is below 22.9% and switches to the
11-step detour above it. In 20 000 simulated deliveries the combined system earns a mean
reward of 11.26, versus 10.94 for trusting the raw sensor, 9.00 for always detouring and
5.97 for always proceeding (an oracle that knows the truth reaches 12.19). All 235
automated tests pass.

## 1. Problem statement

A robot carries documents from the mail room (S) to the dean's office (G). The fastest
way passes through corridor C, which is sometimes blocked (cleaning, crowds, deliveries).
A sensor at C reports `alarm` or `clear`, but it makes mistakes. Walking into a blocked
corridor wastes time; always detouring wastes time too. The goal is a route choice that
maximises the expected delivery reward.

The problem naturally splits into three sub-problems, each matching one course topic:

| Sub-problem | Course topic | Algorithm |
|---|---|---|
| Which paths exist and how long are they? | Search | A\* (BFS as a baseline) |
| How likely is C blocked, given imperfect readings? | Probabilistic reasoning | Bayes' rule |
| Given that uncertainty, which action is best in the long run? | Sequential decisions | MDP, value iteration |

## 2. System design

```
map ──► A* ──► main route (7), detour (11) ─┐
                                            ├──► MDP rewards ──► value iteration ──► Q(s,a)
readings ──► Bayes ──► b = P(Blocked) ──────┴──────────────────────► Q_b(a) ──► action
                                                       action ──► A* route to execute
```

Information flows between the modules. The MDP cannot be built without the A\* route
lengths, and it cannot pick an action without the Bayes belief. The output of the MDP
selects which A\* path is executed. Code: `campus_courier/pipeline.py`, function
`run_mission`.

### 2.1 Environment and assumptions

* 6×6 grid, 4-connected moves, every move costs 1. Ten walls, S = (0,0), G = (2,5),
  C = (2,3) (see `campus_courier/scenarios.py`).
* Prior P(Blocked) = 0.20 (assumed from past observations).
* Sensor model: P(alarm | Blocked) = 0.90, P(alarm | Free) = 0.20.
* Rewards: +20 for delivery, −1 per step, −4 extra delay when the robot reaches a blocked
  corridor; after a failed attempt the corridor clears with probability 0.4; γ = 0.9.
* These numbers are hand-picked assumptions, as allowed by the course guidelines. Every
  one of them is a parameter in the UI and the CLI.

## 3. Algorithm 1 — A\* search

**Idea.** A\* is best-first search ordered by `f(n) = g(n) + h(n)`: `g(n)` is the exact
cost from S to n, `h(n)` estimates the remaining cost. We use the Manhattan distance
`h(n) = |r − r_G| + |c − c_G|`. With unit-cost 4-connected moves no path can be shorter
than that, so `h` is **admissible**. It is also **consistent**:
`h(n) ≤ 1 + h(n')` for neighbours. Therefore the first time A\* removes G from the
priority queue, the path is optimal.

**Implementation** (`search.py`). A binary heap holds `(f, h, counter, cell)`. Ties on `f`
are broken by smaller `h` (prefer cells closer to the goal), then by insertion order, so
runs are deterministic. A closed set skips stale heap entries. `came_from` rebuilds the
path. BFS uses the same interface and serves as an uninformed baseline.

**Why A\*.** We need the *optimal* lengths of two routes because they become rewards. A
non-optimal search would distort the decision. A\* gives optimality and expands fewer
nodes than BFS.

**Worked example.** From S, A\* expands 10 cells (BFS: 18) and returns
`(0,0)→(0,1)→(0,2)→(1,2)→(2,2)→(2,3)→(2,4)→(2,5)`, cost 7. With C turned into a wall
(`Grid.with_blocked`) it returns the detour through row 4, cost 11. C is the 5th step of
the main route (`dist_to_corridor = 5`).

## 4. Algorithm 2 — Bayesian inference

**Idea.** Let B = "C is blocked" and e = sensor reading. Bayes' rule gives

```
P(B | e) = P(e | B) P(B) / [ P(e | B) P(B) + P(e | ¬B) P(¬B) ]
```

For several readings we assume they are **conditionally independent given B**, so we
apply the rule repeatedly and use each posterior as the next prior (`bayes_sequence`).

**Why Bayes.** The sensor has a 20% false-alarm rate. Treating every alarm as the truth
would send the robot on needless detours, while ignoring the sensor wastes information.
Bayes combines the prior frequency of blockages with the sensor's reliability.

**Worked example.**

| Readings | Calculation | P(Blocked) |
|---|---|---|
| none | prior | 0.200 |
| alarm | 0.9·0.2 / (0.9·0.2 + 0.2·0.8) = 0.18 / 0.34 | **0.529** |
| clear | 0.1·0.2 / (0.1·0.2 + 0.8·0.8) = 0.02 / 0.66 | 0.030 |
| alarm, alarm | 0.81·0.2 / (0.81·0.2 + 0.04·0.8) | 0.835 |
| alarm, clear | order does not matter | 0.123 |

An alarm does not make blockage certain: false alarms (0.2·0.8 = 0.16) are almost as
likely as true ones (0.9·0.2 = 0.18).

## 5. Algorithm 3 — MDP and value iteration

**Model** (`mdp.py`, `build_courier_mdp`).

| State | Action | Next state (probability) | Reward |
|---|---|---|---|
| Free | Proceed | Delivered (1.0) | 20 − 7 = **13** |
| Free | Detour | Delivered (1.0) | 20 − 11 = **9** |
| Blocked | Proceed | Free (0.4) / Blocked (0.6) | −(2·5 + 4) = **−14** |
| Blocked | Detour | Delivered (1.0) | **9** |

Delivered is terminal (V = 0). The rewards 13 and 9 come directly from the A\* route
lengths, and 2·5 is the trip to C and back.

**Value iteration** applies the Bellman optimality update until the largest change is
below 10⁻⁶:

```
V_{k+1}(s) = max_a Σ_{s'} T(s,a,s') [ R(s,a,s') + γ V_k(s') ]
```

| k | V(Free) | V(Blocked) |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 13 | 9 |
| 2 | 13 | 9 (converged) |

Q-values at convergence:
`Q(Free,Proceed) = 13`, `Q(Free,Detour) = 9`, `Q(Blocked,Detour) = 9`,
`Q(Blocked,Proceed) = −14 + 0.9·(0.4·13 + 0.6·9) = −14 + 9.54 = −4.46`.
The optimal policy is Proceed when Free and Detour when Blocked.

**Acting under uncertainty (QMDP).** The robot does not observe the true state. It
has only the belief b = P(Blocked) from Bayes, so it scores each action by
`Q_b(a) = b·Q(Blocked,a) + (1−b)·Q(Free,a)`:

* Proceed: `13 − 17.46·b`
* Detour: `9`

They are equal at **b\* = 4 / 17.46 = 0.229**. After one alarm (b = 0.529):
Proceed = 3.76 < Detour = 9, so the robot **detours**. With no reading (b = 0.2):
Proceed = 9.51 > 9, so it **proceeds**.

**Why an MDP.** The decision is not a single comparison of lengths. Proceeding into a
blocked corridor has a future: the robot returns and must decide again, and the corridor
may have cleared. The discounted Bellman equation accounts for this. QMDP is the standard
cheap approximation for using an MDP with a probabilistic belief.

**Edge case.** If no detour exists, "Detour" means aborting the delivery (reward −20). The
threshold then rises close to 1: the robot keeps trying unless it is almost sure the
corridor is blocked and the corridor is far away (tested in
`test_no_detour_and_certain_blockage_aborts`).

## 6. Implementation and UI

* Pure-Python core (standard library only), type hints, dataclasses, 8 small modules.
* `python -m campus_courier` prints, for each algorithm, *input → key steps → output →
  one-sentence explanation*, as the guidelines require.
* `streamlit run app.py` gives an interactive dashboard. Every assumption is a control,
  the map is editable, and each algorithm has its own tab with intermediate results.
* Invalid input (bad map, probabilities outside [0,1], impossible evidence, unknown
  readings) raises a clear error, which the UI shows instead of crashing.

## 7. Testing

235 `pytest` tests, run in CI on every push:

| File | What is checked |
|---|---|
| `test_search.py` | campus route cost 7, detour 11 avoids C; A\* cost equals BFS cost on 200 random maps; A\* never expands more nodes than BFS; heuristic admissible at every cell; unreachable goal; invalid maps |
| `test_bayes.py` | hand-calculated posteriors for alarm, clear and two alarms; reading order does not matter; uninformative sensor keeps the prior; certainty stays certain; invalid probabilities and impossible evidence raise errors |
| `test_mdp.py` | hand-calculated Q-values (13, 9, −4.46, 9); policy; Bellman equation holds at convergence; convergence on an infinite-loop MDP (V = 1/(1−γ) = 10); threshold value; no-detour and long-detour cases; probabilities must sum to 1 |
| `test_pipeline.py` | all five scenarios give the expected action and route length; detour never uses C; decision flips exactly at the threshold; corridor not on the route; unreachable goal; abort case; CLI output |

## 8. Results

**A\* vs BFS** (200 random maps per size, 25% walls):

| n | BFS nodes | A\* nodes | saved | same optimal cost |
|---|---|---|---|---|
| 6 | 26.0 | 12.7 | 51% | 100% |
| 10 | 72.6 | 26.0 | 64% | 100% |
| 20 | 294.1 | 69.5 | 76% | 100% |
| 30 | 664.9 | 140.1 | 79% | 100% |

**Policies** (20 000 simulated deliveries; the true state is sampled from the prior, the
sensor gives one noisy reading, and a failed attempt costs the wasted trip):

| Policy | Mean reward | Mean steps |
|---|---|---|
| always proceed | 5.97 | 12.02 |
| always detour | 9.00 | 11.00 |
| trust the raw sensor | 10.94 | 8.87 |
| **Bayes + MDP (ours)** | **11.26** | **8.66** |
| oracle (knows the truth) | 12.19 | 7.81 |

Across priors 0.05–0.8, Bayes+MDP has the highest mean reward of all non-oracle policies
(table in `results/RESULTS.md`). The gain over "trust the sensor" comes from three
situations. When blockages are rare, one alarm is probably false. When they are frequent,
a "clear" reading is not enough to risk the corridor. And after a failed attempt the
robot detours instead of retrying the same corridor.

## 9. Discussion, limitations and improvements

* **Assumed numbers.** Probabilities and rewards are illustrative. With logged data the
  sensor model and prior could be estimated by counting.
* **QMDP** ignores the value of gathering information: it never chooses "wait and read the
  sensor again". A POMDP solver or an explicit *Sense* action would fix that.
* **Independence of readings** may fail for a sensor stuck in one state. An HMM over time
  would model a corridor state that changes during the day.
* **Scale.** One uncertain corridor gives a 2-state MDP. With k corridors the state space
  becomes 2^k, which is still small for value iteration up to roughly 10–15 corridors.
* **Next steps:** the real QAIRU floor plan, several corridors, and a physical robot or
  simulator.

## 10. Conclusion

The three algorithms each solve a different part of the problem and depend on each
other. A\* supplies optimal route lengths, Bayes supplies a calibrated belief, and the
MDP converts both into the action with the highest expected reward. The combination
measurably beats simpler strategies in simulation and behaves as predicted by hand
calculation in every test case.

## References

1. S. Russell, P. Norvig. *Artificial Intelligence: A Modern Approach*, 4th ed. — Ch. 3
   (A\* search), Ch. 12–13 (probability, Bayes' rule), Ch. 17 (MDPs, value iteration).
2. AI Fundamentals course materials, QAIRU, 2026: practical works on search, Bayesian
   inference and MDPs; Final Project guidelines (*MiniAI_FPTopics*).
3. M. L. Littman, A. R. Cassandra, L. P. Kaelbling. "Learning policies for partially
   observable environments: scaling up", ICML 1995 (QMDP approximation).
