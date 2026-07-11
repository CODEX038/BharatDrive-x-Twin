"""World state — the digital-twin snapshot the simulator reasons over.

Built either from a scenario file (demo/eval) or from live fusion output.
SUMO (sumo_runner.py) enriches traffic flow when installed; CARLA is deferred.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"


@dataclass
class Hazard:
    cls: str
    distance_m: Optional[float]
    rel_speed_ms: Optional[float]
    direction: str = "ahead"
    lane: str = "ego"          # ego | left | right | roadside
    confidence: float = 0.8
    source: str = "front_camera"

    @property
    def ttc_s(self) -> Optional[float]:
        if self.distance_m and self.rel_speed_ms and self.rel_speed_ms > 0.3:
            return self.distance_m / self.rel_speed_ms
        return None


@dataclass
class WorldState:
    name: str
    ego_speed_kmh: float
    hazards: List[Hazard] = field(default_factory=list)
    weather: str = "clear"
    friction: float = 0.7
    road_width_m: Optional[float] = None
    lane_marked: Optional[bool] = None
    lane_count: int = 2
    zone: Optional[str] = None
    traffic_density: float = 0.3          # 0–1
    driver_readiness: Optional[int] = 75  # None = unknown
    fatigue_risk: Optional[float] = 0.2
    reliability: float = 0.8
    alt_route_available: bool = False
    alt_route_delay_min: float = 0.0
    sources: List[str] = field(default_factory=lambda: ["front_camera"])

    @property
    def ego_speed_ms(self) -> float:
        return self.ego_speed_kmh / 3.6

    @property
    def reaction_time_s(self) -> float:
        """Driver reaction time scaled by readiness (1.0 s alert → 2.5 s critical)."""
        r = self.driver_readiness if self.driver_readiness is not None else 50
        return 1.0 + 1.5 * (1.0 - r / 100.0)


def load_scenario(name: str) -> WorldState:
    path = SCENARIO_DIR / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    tw = data.get("twin", {})
    hazards = [Hazard(cls=h["class"], distance_m=h.get("distance_m"),
                      rel_speed_ms=h.get("rel_speed_ms"), direction=h.get("direction", "ahead"),
                      lane=h.get("lane", "ego"), confidence=h.get("confidence", 0.8),
                      source=h.get("source", "scenario"))
               for h in tw.get("hazards", [])]
    return WorldState(
        name=data.get("name", name), ego_speed_kmh=tw.get("ego_speed_kmh", 40.0),
        hazards=hazards, weather=tw.get("weather", "clear"),
        friction=tw.get("friction", 0.7), road_width_m=tw.get("road_width_m"),
        lane_marked=tw.get("lane_marked"), lane_count=tw.get("lane_count", 2),
        zone=tw.get("zone"), traffic_density=tw.get("traffic_density", 0.3),
        driver_readiness=tw.get("driver_readiness", 75),
        fatigue_risk=tw.get("fatigue_risk", 0.2),
        alt_route_available=tw.get("alt_route_available", False),
        alt_route_delay_min=tw.get("alt_route_delay_min", 0.0),
        sources=tw.get("sources", ["scenario"]))


def list_scenarios() -> List[str]:
    return sorted(p.stem for p in SCENARIO_DIR.glob("*.json"))
