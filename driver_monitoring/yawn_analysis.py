"""Yawn detection on the *inner mouth* (replaces bottom_lip MAR — audit W-12).

A yawn = sustained large mouth opening (≥ min_duration). Brief openings (speech,
laughter) are rejected by the duration gate; downstream confidence is reduced when
head pose or reliability is poor.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Sequence, Tuple

Point = Tuple[float, float]


def mouth_aspect_ratio(inner: Sequence[Point]) -> Optional[float]:
    """MAR over an 8-point inner-mouth ring: [L, TL, T, TR, R, BR, B, BL]."""
    if len(inner) < 8:
        return None
    width = math.hypot(inner[0][0] - inner[4][0], inner[0][1] - inner[4][1])
    if width < 1e-6:
        return None
    v1 = math.hypot(inner[1][0] - inner[7][0], inner[1][1] - inner[7][1])
    v2 = math.hypot(inner[2][0] - inner[6][0], inner[2][1] - inner[6][1])
    v3 = math.hypot(inner[3][0] - inner[5][0], inner[3][1] - inner[5][1])
    return (v1 + v2 + v3) / (3.0 * width)


@dataclass
class YawnStats:
    mar: Optional[float]
    yawning: bool
    yawn_count_5min: int
    current_open_s: float
    confidence: float


class YawnDetector:
    def __init__(self, mar_threshold: float = 0.6, min_duration_s: float = 1.5,
                 window_s: float = 300.0) -> None:
        self.threshold = mar_threshold
        self.min_duration = min_duration_s
        self.window_s = window_s
        self._open_since: Optional[float] = None
        self._counted_current = False
        self._yawns: Deque[float] = deque()

    def update(self, ts: float, inner_mouth: Sequence[Point], valid: bool) -> YawnStats:
        mar = mouth_aspect_ratio(inner_mouth) if valid else None
        yawning = False
        open_s = 0.0
        if mar is None:
            self._open_since = None
            self._counted_current = False
        elif mar > self.threshold:
            if self._open_since is None:
                self._open_since = ts
                self._counted_current = False
            open_s = ts - self._open_since
            if open_s >= self.min_duration:
                yawning = True
                if not self._counted_current:
                    self._yawns.append(ts)
                    self._counted_current = True
        else:
            self._open_since = None
            self._counted_current = False
        while self._yawns and self._yawns[0] < ts - self.window_s:
            self._yawns.popleft()
        conf = 0.0 if mar is None else (0.85 if yawning else 0.7)
        return YawnStats(mar=mar, yawning=yawning, yawn_count_5min=len(self._yawns),
                         current_open_s=open_s, confidence=conf)
