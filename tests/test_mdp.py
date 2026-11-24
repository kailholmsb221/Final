import pytest

from campus_courier.mdp import (BLOCKED, DETOUR, FREE, MDP, PROCEED, CourierRewards, belief_q,
                                build_courier_mdp, decision_threshold, value_iteration)


def campus_q():
    return value_iteration(build_courier_mdp(main_len=7, detour_len=11, dist_to_corridor=5)).q


def test_q_values_match_hand_calculation():
    q = campus_q()
    assert q[FREE][PROCEED] == pytest.approx(13.0)   # 20 - 7
    assert q[FREE][DETOUR] == pytest.approx(9.0)     # 20 - 11
    assert q[BLOCKED][DETOUR] == pytest.approx(9.0)
    # -(2*5 + 4) + 0.9 * (0.4 * V(Free) + 0.6 * V(Blocked)) = -14 + 0.9 * (5.2 + 5.4)
    assert q[BLOCKED][PROCEED] == pytest.approx(-4.46)


def test_policy_per_state():
    res = value_iteration(build_courier_mdp(7, 11, 5))
    assert res.policy == {FREE: PROCEED, BLOCKED: DETOUR}
    assert res.values[FREE] == pytest.approx(13.0)
    assert res.values[BLOCKED] == pytest.approx(9.0)


def test_threshold_and_belief_decision():
    q = campus_q()
    thr = decision_threshold(q)
    assert thr == pytest.approx(4.0 / (4.0 + 13.46))
    below, above = belief_q(q, thr - 0.01), belief_q(q, thr + 0.01)
    assert below[PROCEED] > below[DETOUR]
    assert above[DETOUR] > above[PROCEED]


def test_bellman_equation_holds_at_convergence():
    mdp = build_courier_mdp(7, 11, 5)
    res = value_iteration(mdp)
    for s in (FREE, BLOCKED):
        assert res.values[s] == pytest.approx(max(mdp.q_value(s, a, res.values)
                                                  for a in mdp.actions(s)), abs=1e-5)


def test_value_iteration_converges_on_a_loop_mdp():
    # Stay forever earning 1 per step: V = 1 / (1 - gamma) = 10.
    mdp = MDP(["A"], {"A": {"stay": [(1.0, "A", 1.0)]}}, gamma=0.9)
    res = value_iteration(mdp, theta=1e-9)
    assert res.values["A"] == pytest.approx(10.0, rel=1e-6)
    assert res.iterations > 50


def test_no_detour_means_giving_up_only_when_almost_sure():
    # Without a detour, "Detour" = abort the delivery (reward -20). Retrying a blocked
    # corridor is slightly worse than aborting, so the threshold moves close to 1.
    q = value_iteration(build_courier_mdp(7, None, 5)).q
    assert q[FREE][PROCEED] > q[FREE][DETOUR]
    assert decision_threshold(q) > 0.99


def test_very_long_detour_makes_proceed_dominant():
    q = value_iteration(build_courier_mdp(7, 40, 5)).q
    assert belief_q(q, 0.95)[PROCEED] > belief_q(q, 0.95)[DETOUR]


def test_transition_probabilities_must_sum_to_one():
    with pytest.raises(ValueError):
        MDP(["A"], {"A": {"go": [(0.5, "A", 0.0)]}})
    with pytest.raises(ValueError):
        build_courier_mdp(7, 11, 5, CourierRewards(gamma=1.0))
