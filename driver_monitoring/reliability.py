"""Observation Reliability Engine.

Separate score answering "how much can the driver-camera observation be trusted
right now?". Poor reliability gates fatigue classification to Unknown — the system
never claims drowsiness it cannot see (audit W-6/W-8).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Reliability:
    score: float                 # 0–1
    state: str                   # EXCELLENT | GOOD | LIMITED | POOR | UNAVAILABLE
    reasons: List[str] = field(default_factory=list)


class ReliabilityEngine:
    def __init__(self, poor: float = 0.35, good: float = 0.65) -> None:
        self.poor = poor
        self.good = good
        self._ema: Optional[float] = None

    def assess(self, lm, image_quality: Optional[dict] = None,
               head_yaw_deg: Optional[float] = None, fps: Optional[float] = None) -> Reliability:
        reasons: List[str] = []
        if not lm.face_found:
            return Reliability(0.0, "UNAVAILABLE", ["No face detected"])
        score = lm.face_confidence
        if not lm.left_eye_visible and not lm.right_eye_visible:
            score *= 0.25
            reasons.append("Both eyes not visible (occlusion/eyewear)")
        elif not (lm.left_eye_visible and lm.right_eye_visible):
            score *= 0.75
            reasons.append("One eye not visible")
        q = image_quality or {}
        blur = q.get("blur_var")          # Laplacian variance
        if blur is not None and blur < 40:
            score *= 0.6
            reasons.append("Image is blurry")
        bright = q.get("brightness")      # 0–255 mean
        if bright is not None:
            if bright < 50:
                score *= 0.6
                reasons.append("Scene is underexposed")
            elif bright > 210:
                score *= 0.7
                reasons.append("Scene is overexposed / glare")
        if head_yaw_deg is not None and abs(head_yaw_deg) > 35:
            score *= 0.6
            reasons.append("Strong head rotation")
        if lm.face_box and lm.frame_size[0]:
            if lm.face_box[2] / lm.frame_size[0] < 0.10:
                score *= 0.6
                reasons.append("Face too small / too far from camera")
        if fps is not None and fps < 8:
            score *= 0.8
            reasons.append(f"Low frame rate ({fps:.0f} FPS)")
        score = max(0.0, min(1.0, score))
        self._ema = score if self._ema is None else 0.3 * score + 0.7 * self._ema
        s = self._ema
        state = ("EXCELLENT" if s >= 0.85 else "GOOD" if s >= self.good
                 else "LIMITED" if s >= self.poor else "POOR")
        return Reliability(round(s, 3), state, reasons)


def image_quality_metrics(frame_bgr) -> dict:
    """Blur/brightness/contrast via cv2 when available; {} otherwise."""
    try:
        import cv2
        import numpy as np
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return {
            "blur_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "brightness": float(np.mean(gray)),
            "contrast": float(np.std(gray)),
        }
    except Exception:
        return {}
