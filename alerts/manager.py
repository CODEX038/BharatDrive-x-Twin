"""Adaptive alert manager: levels 0–4, cooldown, duplicate suppression, escalation,
acknowledgement, non-blocking audio (pygame optional), response tracking for
alert-effectiveness learning (§28).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .languages import msg

log = logging.getLogger(__name__)

LEVELS = {0: "NORMAL", 1: "INFO", 2: "CAUTION", 3: "HIGH", 4: "CRITICAL"}
# levels: 0 NORMAL · 1 INFO · 2 CAUTION · 3 HIGH · 4 CRITICAL


@dataclass
class AlertEvent:
    ts: float
    level: int
    key: str
    text: str
    risk_before: Optional[float] = None
    acknowledged_at: Optional[float] = None
    risk_after: Optional[float] = None
    response_time_s: Optional[float] = None


class AudioSink:
    """Non-blocking audio via pygame when available; logs otherwise."""

    def __init__(self, alarm_path: Optional[str] = None) -> None:
        self.alarm_path = alarm_path
        self._ok = False
        try:
            import pygame
            pygame.mixer.init()
            self._pg = pygame
            self._ok = alarm_path is not None
        except Exception:
            self._pg = None

    def play(self, loop: bool = False) -> None:
        if self._ok:
            try:
                if not self._pg.mixer.music.get_busy():
                    self._pg.mixer.music.load(self.alarm_path)
                    self._pg.mixer.music.play(-1 if loop else 0)
                return
            except Exception as exc:  # pragma: no cover
                log.warning("audio failed: %s", exc)
        log.warning("AUDIO ALERT (fallback log)")

    def stop(self) -> None:
        if self._ok and self._pg.mixer.music.get_busy():
            self._pg.mixer.music.stop()


class AlertManager:
    def __init__(self, language: str = "en", cooldown_s: float = 8.0,
                 audio: Optional[AudioSink] = None,
                 on_alert: Optional[Callable[[AlertEvent], None]] = None) -> None:
        self.language = language
        self.cooldown_s = cooldown_s
        self.audio = audio or AudioSink()
        self.on_alert = on_alert
        self.history: List[AlertEvent] = []
        self._last_fired: Dict[str, float] = {}
        self._active_level = 0
        self._clear_since: Optional[float] = None
        self.clear_dwell_s = 4.0  # level must stay calm this long before reset

    def raise_alert(self, ts: float, level: int, key: str,
                    risk_before: Optional[float] = None, **fmt) -> Optional[AlertEvent]:
        if level <= 0:
            self.clear(ts)
            return None
        self._clear_since = None  # situation is active again
        last = self._last_fired.get(key, -1e9)
        escalating = level > self._active_level
        if ts - last < self.cooldown_s and not escalating:
            return None  # duplicate suppression within cooldown
        text = msg(self.language, key, **fmt)
        ev = AlertEvent(ts=ts, level=level, key=key, text=text, risk_before=risk_before)
        self.history.append(ev)
        self._last_fired[key] = ts
        self._active_level = max(self._active_level, level)
        if level >= 4:
            self.audio.play(loop=True)
        elif level == 3:
            self.audio.play(loop=False)
        log.info("ALERT L%d [%s] %s", level, LEVELS[level], text)
        if self.on_alert:
            self.on_alert(ev)
        return ev

    def acknowledge(self, ts: float) -> None:
        for ev in reversed(self.history):
            if ev.acknowledged_at is None:
                ev.acknowledged_at = ts
                ev.response_time_s = round(ts - ev.ts, 2)
                break
        self.audio.stop()
        self._active_level = 0

    def record_risk_after(self, ts: float, risk: Optional[float],
                          window_s: float = 20.0) -> None:
        for ev in reversed(self.history):
            if ev.risk_after is None and ts - ev.ts >= window_s:
                ev.risk_after = risk
            if ts - ev.ts > 2 * window_s:
                break

    def clear(self, ts: float) -> None:
        """Reset only after the situation stays calm for `clear_dwell_s` —
        prevents flickering detections from re-triggering escalation."""
        if not self._active_level:
            return
        if self._clear_since is None:
            self._clear_since = ts
            return
        if ts - self._clear_since >= self.clear_dwell_s:
            self.audio.stop()
            self._active_level = 0
            self._clear_since = None

    def effectiveness(self) -> Optional[float]:
        """Mean risk reduction after alerts (None until measurable). Never claimed
        without measurement (§28)."""
        pairs = [(e.risk_before, e.risk_after) for e in self.history
                 if e.risk_before is not None and e.risk_after is not None]
        if not pairs:
            return None
        return round(sum(b - a for b, a in pairs) / len(pairs), 3)


def level_for(state: str, max_hazard: float, reliability_state: str) -> tuple:
    """Map fused situation → (level, message key)."""
    if reliability_state in ("POOR", "UNAVAILABLE"):
        return 2, "unreliable"
    # Fatigue-driven critical and hazard-driven critical are distinct situations.
    # Only claim a road hazard when one is actually present (max_hazard >= 0.7);
    # otherwise report severe fatigue alone. In driver-only/live mode max_hazard
    # is always 0.0, so this prevents the alert from inventing a road hazard.
    if state == "CRITICAL" and max_hazard >= 0.7:
        return 4, "critical"                       # severe fatigue AND a real hazard
    if state in ("DROWSY", "UNRESPONSIVE") and max_hazard >= 0.7:
        return 4, "critical"                       # fatigued driver into a hazard
    if state == "CRITICAL":
        return 4, "critical_fatigue"               # severe fatigue only, no hazard
    if max_hazard >= 0.7:
        return 3, "high_hazard"
    if state in ("PRE_DROWSY", "DROWSY"):
        return 2, "caution_fatigue"
    if max_hazard >= 0.45:
        return 1, "info_congestion"
    return 0, ""
