"""Road object detection abstraction.

Backends: MockDetector (scenario JSON playback — default, zero dependencies) and
YoloDetector (ultralytics, optional). Indian classes per project label set.
Monocular distances are estimates and are labelled as such.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

INDIAN_CLASSES = (
    "car", "truck", "bus", "motorcycle", "scooter", "bicycle", "auto_rickshaw",
    "pedestrian", "cattle", "dog", "pothole", "speed_breaker", "waterlogging",
    "open_manhole", "road_construction", "wrong_side_vehicle", "stopped_vehicle",
    "traffic_light", "debris", "parked_truck",
)

# COCO → project class mapping for the YOLO backend
_COCO_MAP = {"person": "pedestrian", "bicycle": "bicycle", "car": "car",
             "motorcycle": "motorcycle", "bus": "bus", "truck": "truck",
             "traffic light": "traffic_light", "cow": "cattle", "dog": "dog"}

HAZARD_BASE = {  # 0–1 innate hazard weight per class
    "pedestrian": 0.8, "cattle": 0.85, "dog": 0.6, "motorcycle": 0.7, "scooter": 0.7,
    "auto_rickshaw": 0.6, "wrong_side_vehicle": 0.95, "pothole": 0.7,
    "speed_breaker": 0.5, "waterlogging": 0.7, "open_manhole": 0.9,
    "road_construction": 0.6, "stopped_vehicle": 0.6, "debris": 0.65,
    "parked_truck": 0.7, "car": 0.4, "truck": 0.5, "bus": 0.5, "bicycle": 0.6,
    "traffic_light": 0.2,
}


@dataclass
class Detection:
    cls: str
    confidence: float
    box: Tuple[float, float, float, float]      # x, y, w, h (normalized 0–1)
    track_id: Optional[int] = None
    distance_m: Optional[float] = None           # ESTIMATED (monocular)
    rel_speed_ms: Optional[float] = None         # + = approaching (estimated)
    ttc_s: Optional[float] = None                # only when rel speed valid
    direction: str = "unknown"                   # left|right|ahead
    hazard_level: float = 0.0
    source: str = "front_camera"
    ts: float = 0.0

    def to_dict(self) -> dict:
        return {"class": self.cls, "confidence": round(self.confidence, 2),
                "box": [round(v, 3) for v in self.box], "track_id": self.track_id,
                "distance_m_est": self.distance_m, "ttc_s_est": self.ttc_s,
                "direction": self.direction, "hazard_level": round(self.hazard_level, 2),
                "source": self.source}


class ObjectDetector:
    name = "base"

    def detect(self, frame_bgr, ts: float) -> List[Detection]:  # pragma: no cover
        raise NotImplementedError


class MockDetector(ObjectDetector):
    """Plays back timed detections from a scenario JSON (digital_twin/scenarios/*.json).

    Scenario `road_objects`: list of {class, appears_at, disappears_at, box, distance_m,
    rel_speed_ms, direction}. Distance decreases linearly with rel_speed over time.
    """
    name = "mock"

    def __init__(self, scenario_path: Path) -> None:
        data = json.loads(Path(scenario_path).read_text(encoding="utf-8"))
        self.objects = data.get("road_objects", [])
        self.scenario_name = data.get("name", scenario_path.stem)

    def detect(self, frame_bgr, ts: float) -> List[Detection]:
        out: List[Detection] = []
        for i, o in enumerate(self.objects):
            t0, t1 = o.get("appears_at", 0), o.get("disappears_at", 1e9)
            if not (t0 <= ts <= t1):
                continue
            dist = o.get("distance_m")
            rel = o.get("rel_speed_ms", 0.0)
            if dist is not None:
                dist = max(0.5, dist - rel * (ts - t0))
            ttc = (dist / rel) if (dist and rel and rel > 0.3) else None
            out.append(Detection(
                cls=o["class"], confidence=o.get("confidence", 0.9),
                box=tuple(o.get("box", (0.4, 0.4, 0.2, 0.2))), track_id=i,
                distance_m=round(dist, 1) if dist else None,
                rel_speed_ms=rel, ttc_s=round(ttc, 1) if ttc else None,
                direction=o.get("direction", "ahead"),
                hazard_level=HAZARD_BASE.get(o["class"], 0.4),
                source="scenario_playback", ts=ts))
        return out


# Custom fine-tuned model classes → project classes.
# Current model (2026-07-12, IDD Detection + RDD2022, mAP50 0.442) emits project
# class names directly; identity fallback below handles them. This map only
# covers legacy/alternate names.
_CUSTOM_MAP = {"living_thing": "pedestrian", "vehicle": "car"}
_CUSTOM_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "indian_hazards.pt"


class YoloDetector(ObjectDetector):
    """Ultralytics YOLO (optional). Auto-prefers fine-tuned Indian weights at
    models/indian_hazards.pt when present; falls back to pretrained yolov8n."""
    name = "yolo"

    def __init__(self, weights: Optional[str] = None, conf: float = 0.35) -> None:
        from ultralytics import YOLO
        if weights is None:
            weights = str(_CUSTOM_WEIGHTS) if _CUSTOM_WEIGHTS.exists() else "yolov8n.pt"
        log.info("YoloDetector loading weights: %s", weights)
        self.model = YOLO(weights)
        self.conf = conf

    def detect(self, frame_bgr, ts: float) -> List[Detection]:
        h, w = frame_bgr.shape[:2]
        res = self.model.predict(frame_bgr, conf=self.conf, verbose=False)[0]
        out: List[Detection] = []
        for b in res.boxes:
            raw = self.model.names[int(b.cls)]
            cls = _COCO_MAP.get(raw) or _CUSTOM_MAP.get(raw) or \
                (raw if raw in HAZARD_BASE else None)
            if cls is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            box = (x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h)
            cx = box[0] + box[2] / 2
            direction = "left" if cx < 0.35 else "right" if cx > 0.65 else "ahead"
            out.append(Detection(cls=cls, confidence=float(b.conf), box=box,
                                 direction=direction,
                                 hazard_level=HAZARD_BASE.get(cls, 0.4),
                                 source="front_camera", ts=ts))
        return out


def get_detector(road_source: str, scenario_path: Optional[Path] = None) -> ObjectDetector:
    if road_source == "scenario" and scenario_path:
        return MockDetector(scenario_path)
    try:
        return YoloDetector()
    except Exception as exc:
        log.info("YOLO unavailable (%s); using mock detector", exc)
        if scenario_path:
            return MockDetector(scenario_path)
        raise RuntimeError("No detector available and no scenario provided")
