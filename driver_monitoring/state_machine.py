"""Driver state machine with dwell-time hysteresis and reliability gating."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

STATES = ("CALIBRATING", "ALERT", "SLIGHT_FATIGUE", "PRE_DROWSY", "DROWSY",
          "CRITICAL", "DISTRACTED", "UNRESPONSIVE", "FACE_NOT_VISIBLE",
          "OBSERVATION_UNRELIABLE", "CAMERA_ERROR", "PAUSED", "ENDED")

_FATIGUE_ORDER = {"alert": "ALERT", "slight_fatigue": "SLIGHT_FATIGUE",
                  "pre_drowsy": "PRE_DROWSY", "drowsy": "DROWSY", "critical": "CRITICAL"}
_SEVERITY = {s: i for i, s in enumerate(
    ("ALERT", "SLIGHT_FATIGUE", "PRE_DROWSY", "DROWSY", "CRITICAL"))}


@dataclass
class StateChange:
    state: str
    previous: str
    ts: float


class DriverStateMachine:
    def __init__(self, dwell_s: float = 1.5, face_missing_s: float = 2.0,
                 unresponsive_s: float = 6.0) -> None:
        self.dwell_s = dwell_s
        self.face_missing_s = face_missing_s
        self.unresponsive_s = unresponsive_s
        self.state = "CALIBRATING"
        self._candidate: Optional[str] = None
        self._candidate_since: float = 0.0
        self._face_missing_since: Optional[float] = None
        self.last_change: Optional[StateChange] = None

    def update(self, ts: float, *, calibrated: bool, face_found: bool,
               reliability_state: str, state_hint: str, distracted: bool = False,
               camera_ok: bool = True) -> str:
        target = self._target(ts, calibrated, face_found, reliability_state,
                              state_hint, distracted, camera_ok)
        # immediate (non-dwell) transitions for hard conditions
        immediate = {"CAMERA_ERROR", "FACE_NOT_VISIBLE", "UNRESPONSIVE",
                     "OBSERVATION_UNRELIABLE", "CALIBRATING", "CRITICAL"}
        if target == self.state:
            self._candidate = None
            return self.state
        if target in immediate:
            self._commit(target, ts)
            return self.state
        if self._candidate != target:
            self._candidate, self._candidate_since = target, ts
            return self.state
        if ts - self._candidate_since >= self.dwell_s:
            self._commit(target, ts)
        return self.state

    def _target(self, ts, calibrated, face_found, rel_state, hint, distracted, camera_ok) -> str:
        if not camera_ok:
            return "CAMERA_ERROR"
        if not face_found:
            if self._face_missing_since is None:
                self._face_missing_since = ts
            missing = ts - self._face_missing_since
            was_fatigued = _SEVERITY.get(self.state, 0) >= _SEVERITY["DROWSY"]
            if was_fatigued and missing >= self.unresponsive_s:
                return "UNRESPONSIVE"
            if missing >= self.face_missing_s:
                return "FACE_NOT_VISIBLE"
            return self.state
        self._face_missing_since = None
        if rel_state in ("POOR", "UNAVAILABLE"):
            return "OBSERVATION_UNRELIABLE"   # never classify fatigue blind
        if not calibrated:
            # universal-threshold evidence of serious fatigue is still acted on;
            # calibration only delays the *personalized* refinement
            if hint in ("pre_drowsy", "drowsy", "critical"):
                return _FATIGUE_ORDER[hint]
            return "CALIBRATING"
        if distracted and hint in ("alert", "slight_fatigue"):
            return "DISTRACTED"
        return _FATIGUE_ORDER.get(hint, "ALERT")

    def _commit(self, target: str, ts: float) -> None:
        self.last_change = StateChange(target, self.state, ts)
        self.state = target
        self._candidate = None
