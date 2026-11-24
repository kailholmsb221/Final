"""Streamlit UI for Campus Courier AI.   Run:  streamlit run app.py"""
from __future__ import annotations

import html

import streamlit as st

from campus_courier.bayes import SensorModel
from campus_courier.grid import Grid
from campus_courier.mdp import BLOCKED, DETOUR, FREE, PROCEED, CourierRewards, belief_q
from campus_courier.pipeline import run_mission
from campus_courier.scenarios import CAMPUS_MAP, SCENARIOS

st.set_page_config(page_title="Campus Courier AI", page_icon="📦", layout="wide")

COLORS = {
    "wall": "#3d3d3d", "start": "#2e7d32", "goal": "#c62828", "corridor": "#f9a825",
    "route": "#1e88e5", "detour": "#8e24aa", "explored": "#dbe9f7", "free": "#ffffff",
}


def draw_grid(grid: Grid, route=None, explored=None, other=None) -> str:
    route, explored, other = set(route or []), set(explored or []), set(other or [])
    cells = []
    for r in range(grid.rows):
        row = []
        for c in range(grid.cols):
            cell, label, fg = (r, c), "", "#000"
            if cell in grid.walls:
                bg = COLORS["wall"]
            elif cell == grid.start:
                bg, label, fg = COLORS["start"], "S", "#fff"
            elif cell == grid.goal:
                bg, label, fg = COLORS["goal"], "G", "#fff"
            elif cell == grid.corridor:
                bg, label = COLORS["corridor"], "C"
                if cell in route:
                    label = "C●"
            elif cell in route:
                bg, label, fg = COLORS["route"], "●", "#fff"
            elif cell in other:
                bg, label = "#efe0f5", "·"
            elif cell in explored:
                bg = COLORS["explored"]
            else:
                bg = COLORS["free"]
            row.append(f'<td style="background:{bg};color:{fg}">{html.escape(label)}</td>')
        cells.append("<tr>" + "".join(row) + "</tr>")
    return ("<table style='border-collapse:collapse;margin:4px 0'>"
            "<style>td{width:42px;height:42px;text-align:center;font-weight:700;"
            "border:1px solid #bbb;font-size:15px}</style>" + "".join(cells) + "</table>")


# ---------------------------------------------------------------- sidebar inputs
st.sidebar.header("Inputs")
preset = st.sidebar.selectbox("Scenario preset", list(SCENARIOS),
                              format_func=lambda k: SCENARIOS[k].title, index=2)
sc = SCENARIOS[preset]

st.sidebar.subheader("Sensor (Bayes)")
prior = st.sidebar.slider("Prior P(corridor blocked)", 0.0, 1.0, sc.prior, 0.01)
sens = st.sidebar.slider("Sensitivity P(alarm | Blocked)", 0.5, 1.0, sc.sensor.sensitivity, 0.01)
far = st.sidebar.slider("False-alarm rate P(alarm | Free)", 0.0, 0.5, sc.sensor.false_alarm_rate, 0.01)
readings_txt = st.sidebar.text_input("Readings (comma separated: alarm / clear)",
                                     ", ".join(sc.readings))
readings = [r.strip() for r in readings_txt.split(",") if r.strip()]

st.sidebar.subheader("Rewards (MDP)")
goal_reward = st.sidebar.number_input("Delivery reward", 1.0, 100.0, 20.0, 1.0)
step_cost = st.sidebar.number_input("Cost per step", 0.0, 10.0, 1.0, 0.5)
delay = st.sidebar.number_input("Delay penalty if blocked", 0.0, 50.0, 4.0, 1.0)
p_unblock = st.sidebar.slider("P(corridor clears while robot returns)", 0.0, 1.0, 0.4, 0.05)
gamma = st.sidebar.slider("Discount γ", 0.0, 0.99, 0.9, 0.01)

with st.sidebar.expander("Edit campus map"):
    st.caption("S start · G goal · C monitored corridor · # wall · . free")
    map_txt = st.text_area("Map", "\n".join(CAMPUS_MAP), height=170)

# ---------------------------------------------------------------- run
st.title("📦 Campus Courier AI")
st.caption("A delivery robot crosses the campus: **A\\*** plans the route, **Bayes** reads a noisy "
           "sensor, an **MDP** decides whether to risk the short corridor.")

try:
    grid = Grid.from_strings(map_txt.splitlines())
    sensor = SensorModel(sens, far)
    if any(r not in ("alarm", "clear") for r in readings):
        raise ValueError("readings must be 'alarm' or 'clear'")
    rewards = CourierRewards(goal_reward, step_cost, delay, p_unblock, gamma)
    rep = run_mission(grid, prior, readings, sensor, rewards)
except ValueError as err:
    st.error(f"Input error: {err}")
    st.stop()

# ---------------------------------------------------------------- headline
c1, c2, c3, c4 = st.columns(4)
c1.metric("Shortest route (A*)", f"{rep.main.cost} steps" if rep.main.found else "none")
c2.metric("P(corridor blocked)", f"{rep.belief:.1%}", f"{rep.belief - prior:+.1%} vs prior")
c3.metric("Decision (MDP)", rep.action)
c4.metric("Final route", f"{len(rep.final_route) - 1} steps" if rep.final_route else "—")

if rep.action == DETOUR:
    st.warning(f"The corridor is probably blocked ({rep.belief:.0%} > threshold "
               f"{rep.threshold:.0%}). The robot takes the detour.")
elif rep.action == PROCEED:
    st.success("The robot takes the short route" +
               (f" ({rep.belief:.0%} ≤ threshold {rep.threshold:.0%})." if rep.threshold else "."))
else:
    st.error("Mission cannot be completed: " + rep.action)

left, right = st.columns([1, 1.25])
with left:
    st.subheader("Final route")
    st.markdown(draw_grid(grid, rep.final_route), unsafe_allow_html=True)
    st.caption("🟩 start · 🟥 goal · 🟨 corridor C · 🟦 route · ⬛ wall")

with right:
    t1, t2, t3 = st.tabs(["1 · A* search", "2 · Bayes", "3 · MDP"])

    with t1:
        st.markdown(f"A\\* expanded **{rep.main.nodes_expanded}** cells; BFS would expand "
                    f"**{rep.main_bfs.nodes_expanded}** for the same {rep.main.cost}-step route.")
        st.markdown(draw_grid(grid, rep.main.path, rep.main.expanded,
                              rep.detour.path if rep.detour and rep.detour.found else None),
                    unsafe_allow_html=True)
        st.caption("Light blue = expanded by A\\*, dots = detour route without C. "
                   "f(n) = g(n) + h(n), h = Manhattan distance (admissible).")
        if rep.detour is not None:
            st.markdown(f"Detour without C: **{rep.detour.cost if rep.detour.found else '—'}** steps.")

    with t2:
        st.latex(r"P(B\mid e)=\frac{P(e\mid B)\,P(B)}{P(e\mid B)P(B)+P(e\mid \neg B)P(\neg B)}")
        if not rep.bayes_steps:
            st.info("No readings: the belief stays equal to the prior.")
        for i, s in enumerate(rep.bayes_steps, 1):
            st.markdown(f"**Reading {i}: `{s.reading}`** — prior {s.prior:.3f} → posterior "
                        f"**{s.posterior:.3f}**")
            st.code(s.explain(), language=None)
        st.bar_chart({"P(Blocked)": [prior] + [s.posterior for s in rep.bayes_steps]})
        st.caption("Bar 0 = prior, bar k = belief after k readings.")

    with t3:
        if rep.vi is None:
            st.info("The shortest route does not use the corridor, so no decision is needed.")
        else:
            q = rep.vi.q
            st.markdown(f"Value iteration converged in **{rep.vi.iterations}** iterations "
                        f"(γ = {gamma}).")
            st.table({"state": [FREE, BLOCKED],
                      "Q(s, Proceed)": [round(q[FREE][PROCEED], 2), round(q[BLOCKED][PROCEED], 2)],
                      "Q(s, Detour)": [round(q[FREE][DETOUR], 2), round(q[BLOCKED][DETOUR], 2)],
                      "best": [rep.vi.policy[FREE], rep.vi.policy[BLOCKED]]})
            st.latex(r"Q_b(a)=b\,Q(\text{Blocked},a)+(1-b)\,Q(\text{Free},a)")
            st.markdown(f"With b = {rep.belief:.3f}: Proceed = **{rep.q_belief[PROCEED]:.2f}**, "
                        f"Detour = **{rep.q_belief[DETOUR]:.2f}** → **{rep.action}**")
            xs = [i / 50 for i in range(51)]
            st.line_chart({"Proceed": [belief_q(q, b)[PROCEED] for b in xs],
                           "Detour": [belief_q(q, b)[DETOUR] for b in xs]})
            st.caption("x axis: belief P(Blocked) from 0 to 1 (50 steps). "
                       + (f"Lines cross at {rep.threshold:.3f}." if rep.threshold else ""))
            with st.expander("Value-iteration history"):
                st.table([{"k": k, **{s: round(v, 3) for s, v in h.items()}}
                          for k, h in enumerate(rep.vi.history)])
