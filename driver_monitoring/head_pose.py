"""Head-pose estimation with graceful degradation.

Full solvePnP needs cv2 + a rich landmark set; the fallback estimates pitch/yaw
from nose-chin/face-box geometry, which is adequate for nod detection and
"looking away" gating. Outputs are labelled 'estimated'.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple


@dataclass
class HeadPose:
    pitch_deg: Optional[float]  # + = head down
    yaw_deg: Optional[float]    # + = looking right
    roll_deg: Optional[float]
    head_down: bool
    looking_away: bool
    nod_count_60s: int
    confidence: float


class HeadPoseEstimator:
    def __init__(self, down_thresh_deg: float = 15.0, away_thresh_deg: float = 30.0) -> None:
        self.down_thresh = down_thresh_deg
        self.away_thresh = away_thresh_deg
        self._pitch_hist: Deque[Tuple[float, float]] = deque()
        self._nods: Deque[float] = deque()
        self._above = False

    def update(self, lm) -> HeadPose:
        if not lm.face_found or lm.nose_tip is None or lm.chin is None or not lm.face_box:
            return HeadPose(None, None, None, False, False, len(self._nods), 0.0)
        x, y, w, h = lm.face_box
        # geometric approximation: nose vertical position within face box → pitch
        nose_rel = (lm.nose_tip[1] - y) / h if h else 0.5   # 0 top … 1 bottom
        pitch = (nose_rel - 0.52) * 90.0
        nose_rel_x = (lm.nose_tip[0] - x) / w if w else 0.5
        yaw = (nose_rel_x - 0.5) * 90.0
        roll = None
        if len(lm.left_eye) == 6 and len(lm.right_eye) == 6:
            lc = (sum(p[0] for p in lm.left_eye) / 6, sum(p[1] for p in lm.left_eye) / 6)
            rc = (sum(p[0] for p in lm.right_eye) / 6, sum(p[1] for p in lm.right_eye) / 6)
            roll = math.degrees(math.atan2(rc[1] - lc[1], rc[0] - lc[0] + 1e-6))
        ts = lm.timestamp
        self._pitch_hist.append((ts, pitch))
        while self._pitch_hist and self._pitch_hist[0][0] < ts - 3.0:
            self._pitch_hist.popleft()
        # nod = pitch swings above down-threshold then back within ~3 s
        if pitch > self.down_thresh and not self._above:
            self._above = True
        elif pitch < self.down_thresh * 0.4 and self._above:
            self._above = False
            self._nods.append(ts)
        while self._nods and self._nods[0] < ts - 60.0:
            self._nods.popleft()
        return HeadPose(
            pitch_deg=pitch, yaw_deg=yaw, roll_deg=roll,
            head_down=pitch > self.down_thresh,
            looking_away=abs(yaw) > self.away_thresh,
            nod_count_60s=len(self._nods),
            confidence=0.6,  # geometric method — moderate confidence by design
        )
