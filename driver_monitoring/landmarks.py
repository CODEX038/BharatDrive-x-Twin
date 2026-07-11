"""Landmark backend abstraction.

Backends: face_recognition (dlib, user's existing venv), mediapipe, synthetic (tests/demo).
Only geometric landmarks are extracted — never identity. Missing/occluded features are
reported explicitly so downstream code never confuses "not visible" with "closed" (audit W-7/W-8).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

Point = Tuple[float, float]


@dataclass
class FaceLandmarks:
    """One face's geometric observation for a single frame."""
    timestamp: float
    face_found: bool = False
    face_box: Optional[Tuple[float, float, float, float]] = None  # x, y, w, h
    face_confidence: float = 0.0
    left_eye: List[Point] = field(default_factory=list)    # 6 pts, dlib order
    right_eye: List[Point] = field(default_factory=list)
    mouth_outer: List[Point] = field(default_factory=list)  # >=8 pts
    mouth_inner: List[Point] = field(default_factory=list)
    nose_tip: Optional[Point] = None
    chin: Optional[Point] = None
    left_eye_visible: bool = False
    right_eye_visible: bool = False
    frame_size: Tuple[int, int] = (0, 0)  # w, h
    backend: str = "none"


class LandmarkBackend:
    name = "base"

    def extract(self, frame_bgr, timestamp: float) -> FaceLandmarks:  # pragma: no cover
        raise NotImplementedError


class FaceRecognitionBackend(LandmarkBackend):
    """dlib 68-point via face_recognition. Selects the largest face only (fix W-11)."""
    name = "face_recognition"

    def __init__(self) -> None:
        import face_recognition  # noqa: F401 - fail fast if missing
        self._fr = face_recognition

    def extract(self, frame_bgr, timestamp: float) -> FaceLandmarks:
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        out = FaceLandmarks(timestamp=timestamp, frame_size=(w, h), backend=self.name)
        locations = self._fr.face_locations(rgb)
        if not locations:
            return out
        # largest face = driver (fix W-11: never mix passenger landmarks in)
        top, right, bottom, left = max(locations, key=lambda x: (x[1] - x[3]) * (x[2] - x[0]))
        box = (float(left), float(top), float(right - left), float(bottom - top))
        marks_list = self._fr.face_landmarks(rgb, [(top, right, bottom, left)])
        if not marks_list:
            out.face_found = True
            out.face_box = box
            out.face_confidence = 0.3
            return out
        m = marks_list[0]
        out.face_found = True
        out.face_box = box
        out.face_confidence = 0.9  # HOG gives no score; heuristic
        out.left_eye = [tuple(map(float, p)) for p in m.get("left_eye", [])]
        out.right_eye = [tuple(map(float, p)) for p in m.get("right_eye", [])]
        out.mouth_outer = [tuple(map(float, p)) for p in m.get("top_lip", [])] + \
                          [tuple(map(float, p)) for p in m.get("bottom_lip", [])]
        # dlib inner mouth: points 60-67 sit at the tail of top/bottom lip lists
        out.mouth_inner = ([tuple(map(float, p)) for p in m.get("top_lip", [])[7:]] +
                           [tuple(map(float, p)) for p in m.get("bottom_lip", [])[7:]])
        if m.get("nose_tip"):
            out.nose_tip = tuple(map(float, m["nose_tip"][len(m["nose_tip"]) // 2]))
        if m.get("chin"):
            out.chin = tuple(map(float, m["chin"][len(m["chin"]) // 2]))
        out.left_eye_visible = len(out.left_eye) == 6
        out.right_eye_visible = len(out.right_eye) == 6
        return out


class MediaPipeBackend(LandmarkBackend):
    """MediaPipe FaceMesh (468 pts) mapped down to the dlib-style subset."""
    name = "mediapipe"
    _LEFT = [362, 385, 387, 263, 373, 380]
    _RIGHT = [33, 160, 158, 133, 153, 144]
    _MOUTH_IN = [78, 81, 13, 311, 308, 402, 14, 178]

    def __init__(self) -> None:
        import mediapipe as mp
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def extract(self, frame_bgr, timestamp: float) -> FaceLandmarks:
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        out = FaceLandmarks(timestamp=timestamp, frame_size=(w, h), backend=self.name)
        res = self._mesh.process(rgb)
        if not res.multi_face_landmarks:
            return out
        lm = res.multi_face_landmarks[0].landmark
        pt = lambda i: (lm[i].x * w, lm[i].y * h)  # noqa: E731
        out.face_found = True
        xs = [p.x * w for p in lm]
        ys = [p.y * h for p in lm]
        out.face_box = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        out.face_confidence = 0.9
        out.left_eye = [pt(i) for i in self._LEFT]
        out.right_eye = [pt(i) for i in self._RIGHT]
        out.mouth_inner = [pt(i) for i in self._MOUTH_IN]
        out.mouth_outer = out.mouth_inner
        out.nose_tip = pt(1)
        out.chin = pt(152)
        out.left_eye_visible = out.right_eye_visible = True
        return out


class SyntheticBackend(LandmarkBackend):
    """Deterministic synthetic driver for demo/tests. `script` maps time->behaviour."""
    name = "synthetic"

    def __init__(self, script=None) -> None:
        self.script = script or (lambda t: {})

    def extract(self, frame_bgr, timestamp: float) -> FaceLandmarks:
        s = self.script(timestamp)
        out = FaceLandmarks(timestamp=timestamp, frame_size=(640, 480), backend=self.name)
        if s.get("face_missing"):
            return out
        out.face_found = True
        out.face_box = (220.0, 140.0, 200.0, 220.0)
        out.face_confidence = s.get("face_confidence", 0.95)
        openness = max(0.0, min(1.0, s.get("eye_openness", 1.0)))  # 1=open, 0=closed
        half_h = 6.0 * openness + 0.3  # open EAR ~ 0.42, closed ~ 0.02

        def eye(cx: float) -> List[Point]:
            return [(cx - 15, 200.0), (cx - 7, 200.0 - half_h), (cx + 7, 200.0 - half_h),
                    (cx + 15, 200.0), (cx + 7, 200.0 + half_h), (cx - 7, 200.0 + half_h)]

        out.left_eye, out.right_eye = eye(280.0), eye(360.0)
        out.left_eye_visible = not s.get("left_eye_occluded", False)
        out.right_eye_visible = not s.get("right_eye_occluded", False)
        mouth_open = max(0.0, min(1.0, s.get("mouth_openness", 0.05)))
        mh = 30.0 * mouth_open + 1.0
        out.mouth_inner = [(300.0, 300.0), (310.0, 300.0 - mh), (320.0, 300.0 - mh * 1.1),
                           (330.0, 300.0 - mh), (340.0, 300.0), (330.0, 300.0 + mh),
                           (320.0, 300.0 + mh * 1.1), (310.0, 300.0 + mh)]
        out.mouth_outer = out.mouth_inner
        pitch_off = s.get("head_pitch_deg", 0.0) * 1.5
        out.nose_tip = (320.0, 240.0 + pitch_off)
        out.chin = (320.0, 350.0 + pitch_off * 1.8)
        return out


def get_backend(preferred: Optional[str] = None, script=None) -> LandmarkBackend:
    order = [preferred] if preferred else ["face_recognition", "mediapipe", "synthetic"]
    for name in order:
        try:
            if name == "face_recognition":
                return FaceRecognitionBackend()
            if name == "mediapipe":
                return MediaPipeBackend()
            if name == "synthetic":
                return SyntheticBackend(script)
        except Exception as exc:
            log.info("landmark backend %s unavailable: %s", name, exc)
    return SyntheticBackend(script)
