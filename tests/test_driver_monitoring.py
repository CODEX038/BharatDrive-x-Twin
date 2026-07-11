"""Unit tests for driver-monitoring logic (synthetic data — no camera)."""
import math

from driver_monitoring.blink_analysis import BlinkTracker
from driver_monitoring.eye_analysis import EyeAnalyzer, eye_aspect_ratio
from driver_monitoring.fatigue_signature import FatigueSignature
from driver_monitoring.landmarks import SyntheticBackend
from driver_monitoring.perclos import Perclos
from driver_monitoring.readiness_engine import ReadinessEngine
from driver_monitoring.reliability import ReliabilityEngine
from driver_monitoring.state_machine import DriverStateMachine
from driver_monitoring.yawn_analysis import YawnDetector, mouth_aspect_ratio


def _eye(openness: float):
    h = 4.0 * openness + 0.4
    return [(-15, 0), (-7, -h), (7, -h), (15, 0), (7, h), (-7, h)]


def test_ear_open_vs_closed():
    assert eye_aspect_ratio(_eye(1.0)) > 0.25
    assert eye_aspect_ratio(_eye(0.05)) < 0.1
    assert eye_aspect_ratio(_eye(1.0)[:4]) is None  # invalid geometry


def test_invisible_eye_is_not_closed():
    be = SyntheticBackend(lambda t: {"left_eye_occluded": True, "right_eye_occluded": True})
    lm = be.extract(None, 0.0)
    m = EyeAnalyzer().update(lm)
    assert m.ear is None and not m.eyes_valid  # unknown, NOT closed


def test_blink_detection_and_long_blink():
    bt = BlinkTracker(long_blink_s=0.4)
    t = 0.0
    # 3 normal blinks (150 ms) + 1 long blink (600 ms)
    for start, dur in [(1.0, 0.15), (4.0, 0.15), (7.0, 0.15), (10.0, 0.6)]:
        while t < start + dur + 0.5:
            ear = 0.05 if start <= t < start + dur else 0.32
            stats = bt.update(t, ear, 0.2, valid=True)
            t += 0.05
    assert stats.long_blink_count == 1
    assert 3 <= stats.blink_rate_per_min <= 5  # 4 blinks in 60 s window
    assert 0.1 < stats.mean_duration_s < 0.35


def test_blink_gap_does_not_count_hidden_eyes():
    bt = BlinkTracker()
    for i in range(20):
        bt.update(i * 0.1, None, 0.2, valid=False)
    stats = bt.update(2.1, 0.3, 0.2, valid=True)
    assert stats.blink_rate_per_min == 0
    assert stats.current_closure_s == 0


def test_perclos_counts_only_valid_time():
    p = Perclos(windows_s=(30.0,))
    t = 0.0
    while t <= 40.0:
        # eyes closed 20% of the time, valid throughout
        ear = 0.05 if (t % 1.0) < 0.2 else 0.3
        out = p.update(t, ear, 0.2, valid=True)
        t += 0.05
    assert out[30.0] is not None and 0.1 < out[30.0] < 0.3
    p2 = Perclos(windows_s=(30.0,))
    out2 = p2.update(0.0, None, 0.2, valid=False)
    assert out2[30.0] is None  # insufficient observed time → None, not 0 or 1


def test_yawn_requires_sustained_opening():
    y = YawnDetector(mar_threshold=0.6, min_duration_s=1.5)
    mouth_open = [(300.0, 300.0), (310.0, 270.0), (320.0, 267.0), (330.0, 270.0),
                  (340.0, 300.0), (330.0, 330.0), (320.0, 333.0), (310.0, 330.0)]
    mouth_closed = [(300.0, 300.0), (310.0, 298.0), (320.0, 297.7), (330.0, 298.0),
                    (340.0, 300.0), (330.0, 302.0), (320.0, 302.3), (310.0, 302.0)]
    # brief opening (speech) — not a yawn
    for t in [0.0, 0.3, 0.6]:
        s = y.update(t, mouth_open, valid=True)
    s = y.update(0.9, mouth_closed, valid=True)
    assert s.yawn_count_5min == 0
    # sustained opening — yawn
    for i in range(40):
        s = y.update(2.0 + i * 0.05, mouth_open, valid=True)
    assert s.yawn_count_5min == 1


def test_mar_math():
    wide = [(0, 0), (2, -5), (5, -6), (8, -5), (10, 0), (8, 5), (5, 6), (2, 5)]
    assert mouth_aspect_ratio(wide) > 0.6
    assert mouth_aspect_ratio(wide[:5]) is None


def test_reliability_gates_on_occlusion_and_darkness():
    be = SyntheticBackend(lambda t: {"left_eye_occluded": True, "right_eye_occluded": True,
                                     "face_confidence": 0.9})
    lm = be.extract(None, 0.0)
    r = ReliabilityEngine().assess(lm, {"brightness": 30, "blur_var": 100})
    assert r.score < 0.35
    assert any("eyes" in x.lower() for x in r.reasons)
    be2 = SyntheticBackend(lambda t: {"face_missing": True})
    r2 = ReliabilityEngine().assess(be2.extract(None, 0.0))
    assert r2.state == "UNAVAILABLE" and r2.score == 0.0


def test_signature_learns_and_flags_deviation():
    sig = FatigueSignature(min_samples=40)
    for i in range(60):
        sig.observe(i * 0.5, {"ear": 0.31 + 0.005 * math.sin(i), "blink_duration": 0.17,
                              "blink_rate": 14.0, "perclos": 0.05, "head_pitch": 3.0},
                    reliable=True, suspected_fatigue=False)
    assert sig.calibrated
    rep_ok = sig.compare({"ear": 0.31, "blink_duration": 0.17, "blink_rate": 14.0,
                          "perclos": 0.05, "head_pitch": 3.0})
    rep_bad = sig.compare({"ear": 0.22, "blink_duration": 0.42, "blink_rate": 22.0,
                           "perclos": 0.23, "head_pitch": 14.0})
    assert rep_bad.fatigue_deviation > rep_ok.fatigue_deviation
    assert rep_bad.level in ("Moderate", "High")
    thr = sig.personalized_ear_threshold()
    assert 0.12 <= thr <= 0.35 and abs(thr - 0.72 * 0.31) < 0.03


def test_signature_does_not_adapt_during_fatigue():
    sig = FatigueSignature(min_samples=10)
    for i in range(20):
        sig.observe(i, {"ear": 0.31}, reliable=True, suspected_fatigue=False)
    before = sig.compare({"ear": 0.31}).baseline["ear"].median
    for i in range(50):
        sig.observe(20 + i, {"ear": 0.15}, reliable=True, suspected_fatigue=True)
    after = sig.compare({"ear": 0.31}).baseline["ear"].median
    assert abs(before - after) < 1e-9


def test_readiness_smooth_and_gated():
    r = ReadinessEngine()
    out = r.update(0.0, fatigue_risk=None, reliability=0.2, distracted=False,
                   looking_away=False, journey_hours=0.1)
    assert out.score is None and out.band == "Unknown"
    v1 = r.update(1.0, 0.1, 0.9, False, False, 0.1).score
    v2 = r.update(1.1, 0.95, 0.9, False, False, 0.1).score  # sudden spike
    assert v1 is not None and v2 is not None
    assert v1 - v2 <= 2  # rate-limited: no violent jump within 0.1 s


def test_state_machine_reliability_and_face_rules():
    sm = DriverStateMachine(dwell_s=1.0, face_missing_s=2.0)
    for t in (0.0, 0.5, 1.2):  # ALERT requires dwell-time confirmation
        s = sm.update(t, calibrated=True, face_found=True, reliability_state="GOOD",
                      state_hint="alert")
    assert s == "ALERT"
    # poor reliability must never produce DROWSY
    s = sm.update(1.0, calibrated=True, face_found=True, reliability_state="POOR",
                  state_hint="drowsy")
    assert s == "OBSERVATION_UNRELIABLE"
    # missing face → FACE_NOT_VISIBLE after threshold, not drowsy/recovering
    for t in (2.0, 3.0, 4.5):
        s = sm.update(t, calibrated=True, face_found=False, reliability_state="GOOD",
                      state_hint="alert")
    assert s == "FACE_NOT_VISIBLE"
