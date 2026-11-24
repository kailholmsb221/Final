import pytest

from campus_courier.bayes import SensorModel, bayes_sequence, bayes_update

SENSOR = SensorModel(sensitivity=0.9, false_alarm_rate=0.2)


def test_alarm_posterior_matches_hand_calculation():
    # 0.9*0.2 / (0.9*0.2 + 0.2*0.8) = 0.18 / 0.34
    step = bayes_update(0.2, "alarm", SENSOR)
    assert step.evidence == pytest.approx(0.34)
    assert step.posterior == pytest.approx(0.18 / 0.34)


def test_clear_posterior_matches_hand_calculation():
    # 0.1*0.2 / (0.1*0.2 + 0.8*0.8) = 0.02 / 0.66
    assert bayes_update(0.2, "clear", SENSOR).posterior == pytest.approx(0.02 / 0.66)


def test_alarm_raises_and_clear_lowers_belief():
    assert bayes_update(0.2, "alarm", SENSOR).posterior > 0.2
    assert bayes_update(0.2, "clear", SENSOR).posterior < 0.2


def test_two_alarms_stronger_than_one():
    one = bayes_sequence(0.2, ["alarm"], SENSOR)[-1].posterior
    two = bayes_sequence(0.2, ["alarm", "alarm"], SENSOR)[-1].posterior
    assert two > one
    assert two == pytest.approx(0.81 * 0.2 / (0.81 * 0.2 + 0.04 * 0.8))


def test_order_of_readings_does_not_matter():
    a = bayes_sequence(0.3, ["alarm", "clear"], SENSOR)[-1].posterior
    b = bayes_sequence(0.3, ["clear", "alarm"], SENSOR)[-1].posterior
    assert a == pytest.approx(b)


def test_useless_sensor_does_not_change_belief():
    coin = SensorModel(0.5, 0.5)
    assert bayes_update(0.37, "alarm", coin).posterior == pytest.approx(0.37)


def test_certain_prior_stays_certain():
    assert bayes_update(1.0, "clear", SENSOR).posterior == pytest.approx(1.0)
    assert bayes_update(0.0, "alarm", SENSOR).posterior == pytest.approx(0.0)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        SensorModel(1.2, 0.1)
    with pytest.raises(ValueError):
        bayes_update(-0.1, "alarm", SENSOR)
    with pytest.raises(ValueError):
        bayes_update(0.5, "maybe", SENSOR)
    with pytest.raises(ValueError):  # impossible evidence: P(e) = 0
        bayes_update(1.0, "clear", SensorModel(1.0, 0.0))
