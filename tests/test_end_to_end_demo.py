"""End-to-end: full demo pipeline headless (fast mode, no dashboard, no camera)."""
from app.config import Config
from app.main import run_demo


def test_demo_drowsy_scenario_end_to_end(tmp_path):
    cfg = Config.load()
    cfg.db_path = str(tmp_path / "e2e.sqlite3")
    snap = run_demo(cfg, "drowsy_near_junction", duration_s=75.0,
                    use_dashboard=False, realtime=False)
    d = snap["driver"]
    # after 60+ s of scripted microsleeps the system must not think the driver is fine
    assert d["state"] in ("PRE_DROWSY", "DROWSY", "CRITICAL")
    assert d["fatigue_risk"] is not None and d["fatigue_risk"] >= 0.5
    assert snap["simulation"] is not None
    assert snap["simulation"]["recommended"]
    assert snap["journey"]["score"] is not None
    assert snap["journey"]["score"] < 65  # unsafe situation reflected
    assert snap["road"]["complexity"] > 0
    assert any(c["available"] for c in snap["context"])


def test_demo_alert_scenario_stays_calm(tmp_path):
    cfg = Config.load()
    cfg.db_path = str(tmp_path / "e2e2.sqlite3")
    snap = run_demo(cfg, "unmarked_speed_breaker", duration_s=45.0,
                    use_dashboard=False, realtime=False)
    d = snap["driver"]
    assert d["state"] in ("ALERT", "CALIBRATING", "SLIGHT_FATIGUE")
    assert d["fatigue_risk"] is None or d["fatigue_risk"] < 0.5
