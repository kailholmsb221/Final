# AI Usage Declaration

The course allows AI assistance as long as AI-assisted code is declared and explained
(Final Project guidelines: *"explain any AI-assisted code"*). This is our declaration.

## Tool

**Claude** (Anthropic), used through the Claude app in November 2026.

## What the AI did

| Part | AI contribution |
|---|---|
| Project idea | Proposed the "campus courier" scenario, adapted from example topics 02 and 05 in the course list, and the way the three algorithms are chained |
| Code (`campus_courier/`, `app.py`) | Generated the first version of all modules, the CLI and the Streamlit UI |
| Tests (`tests/`) | Generated the test suite and the hand-calculated expected values |
| Experiments (`experiments.py`, `results/`) | Wrote the benchmark and the Monte-Carlo simulation, produced the charts |
| Documentation | Drafted the README, this report, the demo script and the slide content |

## What the team is responsible for

The AI produced drafts. The team is responsible for understanding, checking and defending
them. Before the defense each member ticks the items below:

- [ ] Ran `python -m pytest -q` and `python -m campus_courier` on our own machine
- [ ] Re-did by hand the Bayes posterior after one alarm (0.18 / 0.34 = 0.529)
- [ ] Re-did by hand Q(Blocked, Proceed) = −14 + 0.9·(0.4·13 + 0.6·9) = −4.46 and the threshold 0.229
- [ ] Traced A\* on the campus map and can explain why the Manhattan heuristic is admissible
- [ ] Read every module and can explain each function in our own words
- [ ] Changed at least one parameter in the UI and predicted the decision before running it

## What was not done by AI

No course-graded material was copied from other students. No AI was used during the
oral defense.
