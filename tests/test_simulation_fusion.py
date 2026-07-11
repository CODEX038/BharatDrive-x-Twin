"""Counterfactual simulator, journey safety, alerts, storage."""
import time

from alerts.manager import AlertManager, level_for
from digital_twin.world_state import list_scenarios, load_scenario
from fusion.journey_safety import compute_journey_safety
from simulation.engine import CounterfactualEngine
from simulation.physics import braking_distance_m, stopping_distance_m, ttc_s
from storage.db import Store


def test_physics_sanity():
    assert braking_distance_m(14, 0.7) < braking_distance_m(14, 0.4)  # wet is longer
    assert stopping_distance_m(14, 0.7, 2.0) > stopping_distance_m(14, 0.7, 1.0)
    assert ttc_s(30, 10) == 3.0
    assert ttc_s(30, 0.0) == float("inf")


def test_all_scenarios_load_and_simulate():
    engine = CounterfactualEngine()
    names = list_scenarios()
    assert len(names) >= 5  # minimum demonstration requirement
    for name in names:
        world = load_scenario(name)
        res = engine.simulate(world)
        assert len(res.outcomes) >= 3  # ≥3 actions per scenario
        assert res.recommended.risk <= max(o.risk for o in res.outcomes)
        assert all(0 <= o.risk <= 1 for o in res.outcomes)
        assert res.recommended.assumptions
        assert "SIMULATION" in res.label
        assert res.explanation


def test_fatigued_driver_prefers_conservative_action():
    world = load_scenario("drowsy_near_junction")
    res = CounterfactualEngine().simulate(world)
    assert res.recommended.action != "continue"
    continue_risk = next(o.risk for o in res.outcomes if o.action == "continue")
    assert res.recommended.risk < continue_risk


def test_lane_change_into_occupied_lane_ranks_worse():
    world = load_scenario("motorcycle_cut_in")  # motorcycle in LEFT lane
    res = CounterfactualEngine().simulate(world)
    risks = {o.action: o.risk for o in res.outcomes}
    assert risks["lane_change_left"] > risks["gradual_decel"]


def test_simulation_reproducible():
    world = load_scenario("cattle_on_road")
    a = CounterfactualEngine(seed=7).simulate(world)
    b = CounterfactualEngine(seed=7).simulate(world)
    assert [o.risk for o in a.outcomes] == [o.risk for o in b.outcomes]


def test_journey_safety_fusion():
    good = compute_journey_safety(
        readiness=85, fatigue_risk=0.1, risk_trend="Stable", reliability=0.9,
        complexity=20, complexity_reasons=[], max_hazard_level=0.1,
        weather_multiplier=1.0, speed_kmh=40, best_action_risk=0.1,
        recommendation=None, sources=["cam"])
    bad = compute_journey_safety(
        readiness=35, fatigue_risk=0.7, risk_trend="Increasing", reliability=0.7,
        complexity=85, complexity_reasons=["Dense two-wheeler movement"],
        max_hazard_level=0.9, weather_multiplier=1.6, speed_kmh=60,
        best_action_risk=0.5, recommendation="Reduce speed", sources=["cam"])
    assert good.score > bad.score
    assert bad.danger_level in ("High", "Critical")
    assert bad.reasons
    unreliable = compute_journey_safety(
        readiness=None, fatigue_risk=None, risk_trend="Stable", reliability=0.2,
        complexity=30, complexity_reasons=[], max_hazard_level=0.2,
        weather_multiplier=1.0, speed_kmh=None, best_action_risk=None,
        recommendation=None, sources=[])
    assert unreliable.confidence == "Low"


def test_alert_levels_and_cooldown():
    assert level_for("CRITICAL", 0.8, "GOOD")[0] == 4
    assert level_for("ALERT", 0.8, "GOOD")[0] == 3
    assert level_for("PRE_DROWSY", 0.1, "GOOD")[0] == 2
    assert level_for("ALERT", 0.1, "POOR")[0] == 2  # unreliable → ask to fix camera
    assert level_for("ALERT", 0.1, "GOOD")[0] == 0
    am = AlertManager(cooldown_s=5.0)
    e1 = am.raise_alert(0.0, 2, "caution_fatigue", risk_before=0.5)
    e2 = am.raise_alert(1.0, 2, "caution_fatigue", risk_before=0.5)  # suppressed
    e3 = am.raise_alert(2.0, 4, "critical", risk_before=0.6)          # escalation passes
    assert e1 is not None and e2 is None and e3 is not None
    am.acknowledge(3.0)
    assert am.history[-1].response_time_s is not None
    assert am.effectiveness() is None  # no measured risk_after yet — no claim


def test_alert_multilingual():
    hi = AlertManager(language="hi")
    ev = hi.raise_alert(0.0, 2, "caution_fatigue")
    assert "थकान" in ev.text
    mr = AlertManager(language="mr")
    ev2 = mr.raise_alert(0.0, 4, "critical")
    assert "थकवा" in ev2.text


def test_storage_roundtrip_and_report(tmp_path):
    st = Store(tmp_path / "t.sqlite3")
    st.start_session("test")
    st.driver_event(1.0, "ALERT", 0.2, 80, 0.9, {})
    st.driver_event(2.0, "PRE_DROWSY", 0.55, 60, 0.85, {})
    st.hazard(2.0, "motorcycle", 0.7, "front_camera", {})
    st.journey(2.0, 55, "Moderate", "Medium", 60, {})
    st.end_session()
    rep = st.session_report()
    assert rep["driver_samples"] == 2
    assert rep["max_fatigue_risk"] == 0.55
    assert rep["hazards"] == {"motorcycle": 1}
    assert rep["limitations"]
    path = st.export_report(tmp_path)
    assert path.exists()
    st.close()
