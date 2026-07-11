"""Offline-first context providers: maps (OSM/optional Google), weather, CCTV,
phone sensors. Every reading carries source, timestamp, freshness, confidence,
availability — fusion degrades gracefully when a provider is absent.

Kept in one module for the MVP; split into maps/ weather/ cctv/ sensors/ packages
as they grow (interfaces already match docs/DESIGN.md §10).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent


@dataclass
class ContextReading:
    source: str
    ts: float
    data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    available: bool = False

    def freshness_s(self, now: Optional[float] = None) -> float:
        return (now or time.time()) - self.ts

    def to_dict(self) -> dict:
        return {"source": self.source, "timestamp": self.ts,
                "freshness_s": round(self.freshness_s(), 1),
                "confidence": self.confidence, "available": self.available,
                "data": self.data}


# ---------------------------------------------------------------- maps ----
class MapProvider:
    """OSM-style offline route context from a local JSON extract (or scenario)."""

    def __init__(self, route_file: Optional[Path] = None, api_key: str = "") -> None:
        self.route_file = route_file
        self.api_key = api_key  # optional Google Maps key — used only if set

    def get(self, scenario_ctx: Optional[dict] = None) -> ContextReading:
        if scenario_ctx:
            return ContextReading("scenario_map", time.time(), scenario_ctx, 0.8, True)
        if self.route_file and self.route_file.exists():
            data = json.loads(self.route_file.read_text(encoding="utf-8"))
            return ContextReading("openstreetmap_offline", time.time(), data, 0.7, True)
        return ContextReading("map", time.time(), {}, 0.0, False)


# -------------------------------------------------------------- weather ----
WEATHER_RISK = {"clear": 1.0, "night": 1.15, "rain": 1.3, "heavy_rain": 1.6,
                "fog": 1.6, "wet_road": 1.25}
FRICTION = {"clear": 0.7, "night": 0.7, "rain": 0.5, "heavy_rain": 0.4,
            "fog": 0.6, "wet_road": 0.45}


class WeatherProvider:
    """Mock by default; live API optional. City-level weather is context, not
    proof of local road state — confidence stays moderate."""

    def __init__(self, provider: str = "mock", api_key: str = "") -> None:
        self.provider = provider
        self.api_key = api_key

    def get(self, scenario_weather: Optional[str] = None) -> ContextReading:
        cond = scenario_weather or "clear"
        return ContextReading(
            source=f"weather_{self.provider}", ts=time.time(),
            data={"condition": cond,
                  "risk_multiplier": WEATHER_RISK.get(cond, 1.0),
                  "friction_estimate": FRICTION.get(cond, 0.7)},
            confidence=0.6 if self.provider == "mock" else 0.75, available=True)


# ----------------------------------------------------------------- cctv ----
class CctvAnalyzer:
    """Area-level context from AUTHORIZED sources only (file/RTSP you control).
    MVP: consumes scenario 'cctv' block or a prerecorded file's precomputed stats."""

    ALLOWED_TYPES = ("file", "rtsp", "hls", "api")

    def __init__(self, source_type: str = "file", source_url: str = "") -> None:
        if source_type not in self.ALLOWED_TYPES:
            raise ValueError(f"CCTV source type must be one of {self.ALLOWED_TYPES}")
        self.source_type = source_type
        self.source_url = source_url

    def get(self, scenario_cctv: Optional[dict] = None) -> ContextReading:
        if scenario_cctv:
            return ContextReading("cctv_prerecorded", time.time(), scenario_cctv, 0.6, True)
        return ContextReading("cctv", time.time(), {}, 0.0, False)


# -------------------------------------------------------------- sensors ----
REQUIRED_SENSOR_FIELDS = ("timestamp", "accelerometer", "gyroscope")


def validate_sensor_packet(packet: dict) -> Optional[str]:
    """Returns an error string, or None when valid."""
    for f in REQUIRED_SENSOR_FIELDS:
        if f not in packet:
            return f"missing field: {f}"
    for axis_set in ("accelerometer", "gyroscope"):
        v = packet[axis_set]
        if not isinstance(v, dict) or not all(k in v for k in "xyz"):
            return f"malformed {axis_set}"
        if not all(isinstance(v[k], (int, float)) and abs(v[k]) < 1000 for k in "xyz"):
            return f"out-of-range {axis_set}"
    spd = packet.get("speed_kmh")
    if spd is not None and not (0 <= spd <= 250):
        return "implausible speed"
    return None


class SensorHub:
    """Accepts validated phone/vehicle packets; conclusions labelled experimental."""

    def __init__(self, stale_s: float = 2.0) -> None:
        self.stale_s = stale_s
        self._last: Optional[dict] = None
        self._last_ts: float = 0.0
        self.rejected = 0

    def ingest(self, packet: dict) -> bool:
        err = validate_sensor_packet(packet)
        if err:
            self.rejected += 1
            log.debug("sensor packet rejected: %s", err)
            return False
        self._last = packet
        self._last_ts = time.time()
        return True

    def get(self) -> ContextReading:
        if self._last is None or time.time() - self._last_ts > self.stale_s:
            return ContextReading("phone_sensors", time.time(), {}, 0.0, False)
        a = self._last["accelerometer"]
        jerk = abs(a["x"]) + abs(a["y"]) + abs(a["z"] - 9.81)
        return ContextReading(
            "phone_sensors", self._last_ts,
            {"speed_kmh": self._last.get("speed_kmh"),
             "sudden_movement": jerk > 4.0,
             "device_mounted": self._last.get("device_mounted", True),
             "label": "experimental"},
            confidence=0.5, available=True)
