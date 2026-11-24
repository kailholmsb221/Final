# Defense plan and demo script

Total about 10 minutes, then individual Q&A.

## 1. Problem and objectives — 2 min

* The robot carries documents from the mail room to the dean's office.
* The short corridor C is sometimes blocked, and the sensor at C makes mistakes.
* Objective: choose between the short risky route and the long safe one so that the
  expected delivery reward is maximal.
* Three course topics, one per sub-problem: **search** (which routes?),
  **probability** (how risky?), **MDP** (what to do?).

## 2. Live demo — 3–5 min

Terminal:

```bash
python -m campus_courier --scenario no-sensor   # Proceed, 7 steps
python -m campus_courier --scenario alarm       # Detour, 11 steps
```

UI (`streamlit run app.py`):

1. Preset *"Sensor raises one 'blocked' alarm"*: P(Blocked) = 52.9%, decision **Detour**.
2. Clear the readings field: the belief returns to 20%, decision **Proceed**. Below the
   threshold 22.9%.
3. Type `alarm, clear`: 12.3%, decision **Proceed**. The clear reading outweighs the alarm, because P(clear | Free) = 0.8 is much larger than P(clear | Blocked) = 0.1.
4. Move the false-alarm rate to 0.45: one alarm gives only 33%. Still Detour, but close
   to the threshold.
5. Tab **A\***: 10 expanded cells vs 18 for BFS, detour drawn with dots.
6. Tab **MDP**: the Q-table and the two lines crossing at 0.229.
7. Edit the map: change row 4 to `. # . # # #` (wall right below G). The detour disappears,
   so even with 4 alarms (P = 99%) the robot still proceeds: giving up costs more than
   retrying. Only above 99.6% does it abort.

## 3. The three AI topics — 2 min

* **A\***: f = g + h, Manhattan h is admissible, so the route is optimal. It produces the
  two lengths 7 and 11.
* **Bayes**: 0.9·0.2 / 0.34 = 0.529. A noisy alarm is evidence, not certainty.
* **MDP**: rewards 13 / 9 / −14 from the A\* lengths. Value iteration gives Q(Blocked,
  Proceed) = −4.46. The belief-weighted values give the threshold 0.229.

## 4. Q&A — up to 3 min

See the question list in the team's notes.

---

## Demo video (2–3 min) — narration

| Time | Screen | Narration |
|---|---|---|
| 0:00–0:20 | title, map | "Campus Courier AI. A robot must deliver a parcel across campus. The short corridor may be blocked and the sensor is noisy." |
| 0:20–0:50 | UI, A\* tab | "A\* with the Manhattan heuristic finds the 7-step route and, with the corridor removed, the 11-step detour. It expands 10 cells, BFS 18." |
| 0:50–1:20 | Bayes tab | "One alarm: 0.9 times 0.2 over 0.34 gives 53%. A clear reading would drop it to 3%." |
| 1:20–1:55 | MDP tab | "Route lengths become rewards. Value iteration gives the Q-values; weighted by the belief, Detour wins above 22.9%." |
| 1:55–2:30 | change inputs live | "No reading: proceed. Alarm then clear: 12%, proceed. Higher false-alarm rate: closer to the threshold." |
| 2:30–2:50 | results chart | "In 20 000 simulated deliveries our system beats always-proceed, always-detour and trusting the sensor." |
