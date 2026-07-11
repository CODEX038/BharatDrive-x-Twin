"""Indian Road Complexity Index (0–100), explainable and configurable."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .detection import Detection

DEFAULT_WEIGHTS = {
    "vehicle_density": 14, "two_wheeler_density": 14, "pedestrian_density": 14,
    "animals": 10, "wrong_side": 12, "road_damage": 8, "construction": 6,
    "narrow_or_unmarked": 8, "zone": 6, "visibility_weather": 8, "speed_factor": 8,
}

_TWO_WHEELERS = {"motorcycle", "scooter", "bicycle"}
_VEHICLES = {"car", "truck", "bus", "auto_rickshaw", "stopped_vehicle", "parked_truck"}
_DAMAGE = {"pothole", "speed_breaker", "waterlogging", "open_manhole", "debris"}


@dataclass
class ComplexityResult:
    score: int
    level: str
    reasons: List[str] = field(default_factory=list)
    components: Dict[str, float] = field(default_factory=dict)


class RoadComplexityIndex:
    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.w = {**DEFAULT_WEIGHTS, **(weights or {})}

    @classmethod
    def from_config(cls, path: Path) -> "RoadComplexityIndex":
        if path.exists():
            return cls(json.loads(path.read_text(encoding="utf-8")))
        return cls()

    def compute(self, detections: List[Detection], *, road_width_m: Optional[float] = None,
                lane_marked: Optional[bool] = None, zone: Optional[str] = None,
                visibility: str = "good", weather: str = "clear",
                speed_kmh: Optional[float] = None) -> ComplexityResult:
        comp: Dict[str, float] = {}
        reasons: List[str] = []
        n_veh = sum(1 for d in detections if d.cls in _VEHICLES)
        n_two = sum(1 for d in detections if d.cls in _TWO_WHEELERS)
        n_ped = sum(1 for d in detections if d.cls == "pedestrian")
        n_animal = sum(1 for d in detections if d.cls in ("cattle", "dog"))
        comp["vehicle_density"] = min(1.0, n_veh / 6.0)
        comp["two_wheeler_density"] = min(1.0, n_two / 5.0)
        comp["pedestrian_density"] = min(1.0, n_ped / 4.0)
        comp["animals"] = min(1.0, n_animal / 1.0)
        comp["wrong_side"] = 1.0 if any(d.cls == "wrong_side_vehicle" for d in detections) else 0.0
        comp["road_damage"] = min(1.0, sum(1 for d in detections if d.cls in _DAMAGE) / 2.0)
        comp["construction"] = 1.0 if any(d.cls == "road_construction" for d in detections) else 0.0
        narrow = 1.0 if (road_width_m is not None and road_width_m < 6.0) else 0.0
        unmarked = 1.0 if lane_marked is False else 0.0
        comp["narrow_or_unmarked"] = max(narrow, unmarked * 0.7)
        comp["zone"] = 1.0 if zone in ("school", "market", "hospital") else 0.0
        vis = {"good": 0.0, "reduced": 0.6, "poor": 1.0}.get(visibility, 0.0)
        wx = {"clear": 0.0, "rain": 0.6, "heavy_rain": 1.0, "fog": 1.0, "night": 0.5}.get(weather, 0.0)
        comp["visibility_weather"] = max(vis, wx)
        comp["speed_factor"] = 0.0
        if speed_kmh is not None:
            busy = comp["vehicle_density"] + comp["pedestrian_density"] + comp["two_wheeler_density"]
            if speed_kmh > 40 and busy > 1.0:
                comp["speed_factor"] = min(1.0, (speed_kmh - 40) / 30.0)

        score = int(round(sum(self.w[k] * v for k, v in comp.items())))
        score = max(0, min(100, score))
        if comp["two_wheeler_density"] >= 0.5:
            reasons.append(f"Dense two-wheeler movement ({n_two} tracked)")
        if comp["pedestrian_density"] >= 0.4:
            reasons.append(f"Pedestrians near the lane ({n_ped})")
        if comp["wrong_side"]:
            reasons.append("Wrong-side vehicle detected")
        if comp["animals"]:
            reasons.append("Animal on or near road")
        if comp["road_damage"] > 0:
            reasons.append("Road damage ahead (pothole/breaker/water)")
        if comp["construction"]:
            reasons.append("Construction zone")
        if comp["narrow_or_unmarked"] > 0:
            reasons.append("Narrow or unmarked road")
        if comp["zone"]:
            reasons.append(f"{zone.capitalize()} zone")
        if comp["visibility_weather"] > 0:
            reasons.append("Reduced visibility / adverse weather")
        if comp["speed_factor"] > 0:
            reasons.append("Speed high for current density")
        level = ("Very High" if score >= 75 else "High" if score >= 55
                 else "Moderate" if score >= 35 else "Low")
        return ComplexityResult(score, level, reasons, {k: round(v, 2) for k, v in comp.items()})
