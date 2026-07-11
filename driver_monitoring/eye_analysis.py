"""Per-eye EAR with visibility. An invisible eye is never treated as closed (audit W-7/W-8)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, float]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(eye: Sequence[Point]) -> Optional[float]:
    """dlib 6-point EAR. Returns None when geometry is invalid."""
    if len(eye) != 6:
        return None
    c = _dist(eye[0], eye[3])
    if c < 1e-6:
        return None
    return (_dist(eye[1], eye[5]) + _dist(eye[2], eye[4])) / (2.0 * c)


@dataclass
class EyeMetrics:
    timestamp: float
    left_ear: Optional[float]
    right_ear: Optional[float]
    ear: Optional[float]            # combined over visible eyes only
    eyes_valid: bool                # at least one visible eye with valid EAR
    both_visible: bool
    ear_smoothed: Optional[float] = None


class EyeAnalyzer:
    """Computes per-frame EyeMetrics with EWMA smoothing over valid observations."""

    def __init__(self, smoothing_alpha: float = 0.35) -> None:
        self.alpha = smoothing_alpha
        self._smoothed: Optional[float] = None

    def update(self, lm) -> EyeMetrics:
        left = eye_aspect_ratio(lm.left_eye) if lm.left_eye_visible else None
        right = eye_aspect_ratio(lm.right_eye) if lm.right_eye_visible else None
        vals = [v for v in (left, right) if v is not None]
        ear = sum(vals) / len(vals) if vals else None
        if ear is not None:
            self._smoothed = ear if self._smoothed is None else \
                self.alpha * ear + (1 - self.alpha) * self._smoothed
        return EyeMetrics(
            timestamp=lm.timestamp, left_ear=left, right_ear=right, ear=ear,
            eyes_valid=bool(vals), both_visible=(left is not None and right is not None),
            ear_smoothed=self._smoothed if ear is not None else None,
        )

    def reset(self) -> None:
        self._smoothed = None
