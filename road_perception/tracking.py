"""Minimal IoU tracker: stable IDs, size-growth as approach proxy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .detection import Detection


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 1e-9 else 0.0


@dataclass
class Track:
    track_id: int
    cls: str
    box: Tuple[float, float, float, float]
    last_ts: float
    history: List[Tuple[float, float]] = field(default_factory=list)  # ts, area

    @property
    def approaching(self) -> bool:
        if len(self.history) < 4:
            return False
        return self.history[-1][1] > self.history[0][1] * 1.15


class IouTracker:
    def __init__(self, iou_thresh: float = 0.25, max_age_s: float = 1.5) -> None:
        self.iou_thresh = iou_thresh
        self.max_age = max_age_s
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: List[Detection], ts: float) -> List[Detection]:
        for det in detections:
            best_id, best_iou = None, self.iou_thresh
            for tid, tr in self._tracks.items():
                if tr.cls != det.cls:
                    continue
                i = _iou(det.box, tr.box)
                if i > best_iou:
                    best_id, best_iou = tid, i
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                self._tracks[best_id] = Track(best_id, det.cls, det.box, ts)
            tr = self._tracks[best_id]
            tr.box, tr.last_ts = det.box, ts
            tr.history.append((ts, det.box[2] * det.box[3]))
            tr.history = [(t, a) for t, a in tr.history if t > ts - 3.0]
            det.track_id = best_id
            if det.rel_speed_ms is None and tr.approaching:
                det.hazard_level = min(1.0, det.hazard_level * 1.3)
        self._tracks = {tid: tr for tid, tr in self._tracks.items()
                        if ts - tr.last_ts <= self.max_age}
        return detections
