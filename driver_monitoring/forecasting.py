"""Temporal fatigue-risk forecasting over rolling windows.

MVP model: interpretable weighted evidence model (logistic squashing of named
signals) so every risk % is explainable. The `FatigueForecaster.predict` interface
is designed for a drop-in scikit-learn model once labelled, subject-independent
data exists (Phase 10). Timestamp-based throughout.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple


@dataclass
class Snapshot:
    ts: float
    ear: Optional[float]
    perclos_60: Optional[float]
    blink_rate: float
    blink_duration: float
    long_blinks: int
    yawns_5min: int
    nods_60s: int
    head_down: bool
    looking_away: bool
    reliability: float
    baseline_deviation: float
    microsleep: bool


@dataclass
class Forecast:
    risk: float                       # 0–1 current fatigue risk
    trend: str                        # Increasing | Stable | Decreasing
    state_hint: str                   # alert|slight_fatigue|pre_drowsy|drowsy|critical
    horizon_s: Optional[Tuple[int, int]]  # est. seconds until warning level, if rising
    contributors: List[str] = field(default_factory=list)
    reliable: bool = True


class TemporalBuffer:
    def __init__(self, horizon_s: float = 180.0) -> None:
        self.horizon = horizon_s
        self._snaps: Deque[Snapshot] = deque()

    def add(self, s: Snapshot) -> None:
        self._snaps.append(s)
        while self._snaps and self._snaps[0].ts < s.ts - self.horizon:
            self._snaps.popleft()

    def window(self, now: float, seconds: float) -> List[Snapshot]:
        return [s for s in self._snaps if s.ts >= now - seconds]

    def ear_slope(self, now: float, seconds: float = 60.0) -> Optional[float]:
        pts = [(s.ts, s.ear) for s in self.window(now, seconds) if s.ear is not None]
        if len(pts) < 8:
            return None
        n = len(pts)
        t0 = pts[0][0]
        xs = [t - t0 for t, _ in pts]
        ys = [v for _, v in pts]
        mx, my = sum(xs) / n, sum(ys) / n
        denom = sum((x - mx) ** 2 for x in xs)
        if denom < 1e-9:
            return None
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom  # EAR/sec


class FatigueForecaster:
    """Weighted-evidence fatigue model with trend + horizon estimation."""

    W = {
        "perclos": 3.2,        # PERCLOS-60 above 15%
        "long_blinks": 1.4,
        "blink_duration": 1.6,  # vs 250 ms
        "yawns": 1.1,
        "nods": 1.8,
        "baseline_dev": 2.4,
        "ear_slope": 1.2,      # falling EAR
        "microsleep": 3.5,
    }
    BIAS = -3.0

    def __init__(self) -> None:
        self.buffer = TemporalBuffer()
        self._risk_hist: Deque[Tuple[float, float]] = deque()

    def predict(self, s: Snapshot) -> Forecast:
        self.buffer.add(s)
        if s.reliability < 0.35:
            return Forecast(risk=0.0, trend="Stable", state_hint="unknown",
                            horizon_s=None, contributors=["Observation unreliable"],
                            reliable=False)
        x = self.BIAS
        contributors: List[str] = []

        def add(cond: float, weight_key: str, label: str) -> None:
            nonlocal x
            if cond > 0:
                x += self.W[weight_key] * min(cond, 1.5)
                contributors.append(label)

        if s.perclos_60 is not None:
            add(max(0.0, (s.perclos_60 - 0.15) / 0.25), "perclos",
                f"PERCLOS(60s) at {s.perclos_60:.0%}")
        add(min(s.long_blinks / 4.0, 1.0) if s.long_blinks else 0.0, "long_blinks",
            f"{s.long_blinks} long blinks in the last minute")
        if s.blink_duration > 0.25:
            add((s.blink_duration - 0.25) / 0.3, "blink_duration",
                f"Mean blink duration {s.blink_duration*1000:.0f} ms")
        add(min(s.yawns_5min / 3.0, 1.0) if s.yawns_5min else 0.0, "yawns",
            f"{s.yawns_5min} yawns in 5 min")
        add(min(s.nods_60s / 2.0, 1.0) if s.nods_60s else 0.0, "nods",
            f"{s.nods_60s} head-nod events")
        add(s.baseline_deviation, "baseline_dev",
            f"Personal-baseline deviation {s.baseline_deviation:.0%}")
        slope = self.buffer.ear_slope(s.ts)
        if slope is not None and slope < -0.0005:
            add(min(-slope / 0.002, 1.0), "ear_slope", "Eye openness trending down")
        if s.microsleep:
            add(1.0, "microsleep", "Ongoing prolonged eye closure")

        risk = 1.0 / (1.0 + math.exp(-x))
        self._risk_hist.append((s.ts, risk))
        while self._risk_hist and self._risk_hist[0][0] < s.ts - 120.0:
            self._risk_hist.popleft()
        trend, rate = self._trend(s.ts)
        horizon = None
        if trend == "Increasing" and rate > 1e-5 and risk < 0.7:
            secs = (0.7 - risk) / rate
            if secs < 600:
                horizon = (max(5, int(secs * 0.6)), int(secs * 1.5))
        state = ("critical" if risk >= 0.85 or s.microsleep else
                 "drowsy" if risk >= 0.7 else
                 "pre_drowsy" if risk >= 0.5 else
                 "slight_fatigue" if risk >= 0.3 else "alert")
        return Forecast(risk=round(risk, 3), trend=trend, state_hint=state,
                        horizon_s=horizon, contributors=contributors, reliable=True)

    def _trend(self, now: float) -> Tuple[str, float]:
        pts = [(t, r) for t, r in self._risk_hist if t >= now - 60.0]
        if len(pts) < 6:
            return "Stable", 0.0
        n = len(pts)
        t0 = pts[0][0]
        xs = [t - t0 for t, _ in pts]
        ys = [r for _, r in pts]
        mx, my = sum(xs) / n, sum(ys) / n
        denom = sum((x - mx) ** 2 for x in xs) or 1e-9
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
        if slope > 0.0008:
            return "Increasing", slope
        if slope < -0.0008:
            return "Decreasing", slope
        return "Stable", slope
