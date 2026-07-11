"""Driver Readiness Score (0–100) with smoothing, hysteresis and reliability gating.

Never jumps violently between frames: EWMA + per-second rate limiting.
Unknown (None) when observation is unreliable — never a fabricated number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

BANDS = [(81, "Ready"), (61, "Slightly reduced"), (41, "Reduced"),
         (21, "Unsafe"), (0, "Critical")]


@dataclass
class Readiness:
    score: Optional[int]      # None = Unknown
    band: str
    reasons: List[str] = field(default_factory=list)


class ReadinessEngine:
    def __init__(self, max_change_per_s: float = 8.0, alpha: float = 0.25) -> None:
        self.max_rate = max_change_per_s
        self.alpha = alpha
        self._value: Optional[float] = None
        self._last_ts: Optional[float] = None

    def update(self, ts: float, fatigue_risk: Optional[float], reliability: float,
               distracted: bool, looking_away: bool, journey_hours: float,
               alert_ignored_recently: bool = False) -> Readiness:
        if fatigue_risk is None or reliability < 0.35:
            return Readiness(None, "Unknown", ["Observation unreliable — readiness not estimated"])
        raw = 100.0 * (1.0 - fatigue_risk)
        reasons: List[str] = []
        if distracted:
            raw -= 20
            reasons.append("Driver distraction detected")
        if looking_away:
            raw -= 10
            reasons.append("Gaze away from road")
        if journey_hours > 2.0:
            raw -= min(15.0, 5.0 * (journey_hours - 2.0))
            reasons.append(f"Long journey duration ({journey_hours:.1f} h)")
        if alert_ignored_recently:
            raw -= 8
            reasons.append("Recent alert not acknowledged")
        raw *= (0.85 + 0.15 * reliability)   # mild reliability weighting
        raw = max(0.0, min(100.0, raw))
        if self._value is None:
            self._value = raw
        else:
            target = self.alpha * raw + (1 - self.alpha) * self._value
            dt = max(1e-3, ts - (self._last_ts or ts))
            max_step = self.max_rate * dt
            self._value += max(-max_step, min(max_step, target - self._value))
        self._last_ts = ts
        score = int(round(self._value))
        band = next(b for lo, b in BANDS if score >= lo)
        if fatigue_risk >= 0.5:
            reasons.insert(0, f"Fatigue risk at {fatigue_risk:.0%}")
        return Readiness(score, band, reasons)

    def reset(self) -> None:
        self._value = None
        self._last_ts = None
