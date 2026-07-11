"""Driver-monitoring pipeline: one `update(frame, ts)` call per frame ties together
landmarks -> eye/blink/yawn/PERCLOS/head-pose -> reliability -> signature -> forecast
-> readiness -> state machine. Frame may be None when using the synthetic backend.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.config import Config
from .blink_analysis import BlinkTracker
from .eye_analysis import EyeAnalyzer
from .fatigue_signature import FatigueSignature
from .forecasting import FatigueForecaster, Snapshot
from .head_pose import HeadPoseEstimator
from .landmarks import LandmarkBackend, get_backend
from .perclos import Perclos
from .readiness_engine import ReadinessEngine
from .reliability import ReliabilityEngine, image_quality_metrics
from .state_machine import DriverStateMachine
from .yawn_analysis import YawnDetector

log = logging.getLogger(__name__)


@dataclass
class DriverFrameResult:
    ts: float
    state: str
    reliability: float
    reliability_state: str
    reliability_reasons: list
    fatigue_risk: Optional[float]
    risk_trend: str
    horizon_s: Optional[tuple]
    contributors: list
    readiness: Optional[int]
    readiness_band: str
    ear: Optional[float]
    personalized_ear_threshold: float
    perclos: Dict[float, Optional[float]]
    blink: Any = None
    yawn: Any = None
    head: Any = None
    signature: Any = None
    fps: Optional[float] = None


class DriverMonitor:
    def __init__(self, cfg: Config, backend: Optional[LandmarkBackend] = None) -> None:
        self.cfg = cfg
        self.backend = backend or get_backend()
        self.eyes = EyeAnalyzer()
        self.blinks = BlinkTracker(cfg.long_blink_s, cfg.microsleep_s)
        self.yawns = YawnDetector(cfg.mar_threshold)
        self.perclos = Perclos(cfg.perclos_windows_s)
        self.head = HeadPoseEstimator()
        self.reliability = ReliabilityEngine(cfg.reliability_poor, cfg.reliability_good)
        self.signature = FatigueSignature(cfg.calibration_s)
        self.forecaster = FatigueForecaster()
        self.readiness = ReadinessEngine()
        self.fsm = DriverStateMachine(cfg.state_dwell_s, cfg.face_missing_s)
        self._session_start: Optional[float] = None
        self._fps_ts: list = []

    def update(self, frame_bgr, ts: Optional[float] = None) -> DriverFrameResult:
        ts = time.monotonic() if ts is None else ts
        if self._session_start is None:
            self._session_start = ts
        self._fps_ts = [t for t in self._fps_ts if t > ts - 2.0] + [ts]
        fps = len(self._fps_ts) / 2.0 if len(self._fps_ts) > 3 else None

        lm = self.backend.extract(frame_bgr, ts)
        quality = image_quality_metrics(frame_bgr) if frame_bgr is not None else {}
        head = self.head.update(lm)
        rel = self.reliability.assess(lm, quality, head.yaw_deg, fps)
        reliable = rel.score >= self.cfg.reliability_poor and lm.face_found

        eye = self.eyes.update(lm)
        thr = self.signature.personalized_ear_threshold(self.cfg.ear_threshold)
        valid_eyes = reliable and eye.eyes_valid
        # raw EAR for event detection (smoothing would stretch blink durations);
        # smoothed EAR only for display and slow features
        blink = self.blinks.update(ts, eye.ear, thr, valid_eyes)
        yawn = self.yawns.update(ts, lm.mouth_inner, reliable and lm.face_found)
        pcl = self.perclos.update(ts, eye.ear, thr, valid_eyes)

        current = {
            "ear": eye.ear_smoothed,
            "blink_duration": blink.mean_duration_s or None,
            "blink_rate": blink.blink_rate_per_min or None,
            "perclos": pcl.get(60.0),
            "head_pitch": head.pitch_deg,
        }
        sig = self.signature.compare(current)
        snap = Snapshot(
            ts=ts, ear=eye.ear_smoothed, perclos_60=pcl.get(60.0),
            blink_rate=blink.blink_rate_per_min, blink_duration=blink.mean_duration_s,
            long_blinks=blink.long_blink_count, yawns_5min=yawn.yawn_count_5min,
            nods_60s=head.nod_count_60s, head_down=head.head_down,
            looking_away=head.looking_away, reliability=rel.score,
            baseline_deviation=sig.fatigue_deviation, microsleep=blink.microsleep,
        )
        fc = self.forecaster.predict(snap)
        # baseline learns only from reliable, non-fatigued observations
        self.signature.observe(ts, current, reliable and rel.score >= self.cfg.reliability_good,
                               suspected_fatigue=(fc.reliable and fc.risk >= 0.4))
        journey_h = (ts - self._session_start) / 3600.0
        rd = self.readiness.update(ts, fc.risk if fc.reliable else None, rel.score,
                                   distracted=False, looking_away=head.looking_away,
                                   journey_hours=journey_h)
        state = self.fsm.update(
            ts, calibrated=self.signature.calibrated, face_found=lm.face_found,
            reliability_state=rel.state, state_hint=fc.state_hint,
            distracted=head.looking_away and not head.head_down)
        return DriverFrameResult(
            ts=ts, state=state, reliability=rel.score, reliability_state=rel.state,
            reliability_reasons=rel.reasons,
            fatigue_risk=fc.risk if fc.reliable else None, risk_trend=fc.trend,
            horizon_s=fc.horizon_s, contributors=fc.contributors,
            readiness=rd.score, readiness_band=rd.band, ear=eye.ear_smoothed,
            personalized_ear_threshold=round(thr, 3), perclos=pcl,
            blink=blink, yawn=yawn, head=head, signature=sig, fps=fps,
        )
