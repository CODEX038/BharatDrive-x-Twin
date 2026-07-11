"""Central configuration. Loads defaults, then configs/default.json, then .env / environment."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


@dataclass
class Config:
    # thresholds (legacy-compatible defaults; personalized layer supersedes them)
    ear_threshold: float = 0.25
    mar_threshold: float = 0.60
    # timing (seconds — never frame counts)
    long_blink_s: float = 0.4
    microsleep_s: float = 1.0
    perclos_windows_s: tuple = (30.0, 60.0, 120.0)
    feature_windows_s: tuple = (5.0, 15.0, 30.0, 60.0, 120.0)
    calibration_s: float = 30.0
    face_missing_s: float = 2.0
    state_dwell_s: float = 1.5
    # reliability
    reliability_poor: float = 0.35
    reliability_good: float = 0.65
    # alerts
    alert_cooldown_s: float = 8.0
    language: str = "en"
    alarm_path: str = "assets/alarm.mp3"
    # runtime
    driver_camera: int = 0
    road_source: str = "scenario"  # scenario | camera:<idx> | file:<path>
    dashboard_port: int = 8765
    db_path: str = "storage/bharatdrive.sqlite3"
    record_video: bool = False
    consent_recording: bool = False
    # provider keys (optional)
    google_maps_api_key: str = ""
    weather_api_key: str = ""
    weather_provider: str = "mock"
    cctv_source_type: str = "file"
    cctv_source_url: str = ""

    @classmethod
    def load(cls) -> "Config":
        _load_dotenv(ROOT / ".env")
        cfg = cls()
        json_path = ROOT / "configs" / "default.json"
        if json_path.exists():
            data: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
            for f in fields(cls):
                if f.name in data:
                    setattr(cfg, f.name, data[f.name])
        env_map = {
            "GOOGLE_MAPS_API_KEY": "google_maps_api_key",
            "WEATHER_API_KEY": "weather_api_key",
            "WEATHER_PROVIDER": "weather_provider",
            "CCTV_SOURCE_TYPE": "cctv_source_type",
            "CCTV_SOURCE_URL": "cctv_source_url",
            "ALARM_PATH": "alarm_path",
            "DB_PATH": "db_path",
            "LANGUAGE": "language",
        }
        for env_key, attr in env_map.items():
            if os.environ.get(env_key):
                setattr(cfg, attr, os.environ[env_key])
        if os.environ.get("DASHBOARD_PORT"):
            cfg.dashboard_port = int(os.environ["DASHBOARD_PORT"])
        cfg.record_video = os.environ.get("RECORD_VIDEO", "false").lower() == "true"
        cfg.consent_recording = os.environ.get("CONSENT_RECORDING", "false").lower() == "true"
        if cfg.record_video and not cfg.consent_recording:
            cfg.record_video = False  # privacy: recording requires explicit consent
        return cfg
