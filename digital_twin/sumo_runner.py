"""Guarded SUMO integration. Activates only when SUMO is installed (SUMO_HOME set).

Pipeline when available: OSM extract → netconvert → route flows → TraCI stepping,
enriching WorldState.traffic_density and travel-time estimates. Absent SUMO, the
digital twin runs on scenario files — the rest of the system is unaffected.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def sumo_available() -> bool:
    return bool(os.environ.get("SUMO_HOME")) or shutil.which("sumo") is not None


class SumoRunner:
    def __init__(self, osm_file: Optional[Path] = None) -> None:
        if not sumo_available():
            raise RuntimeError(
                "SUMO not installed. Install Eclipse SUMO and set SUMO_HOME, "
                "or run with scenario-file twin (default).")
        self.osm_file = osm_file

    def import_osm(self, out_dir: Path) -> Path:
        """OSM → SUMO network via netconvert (pilot route)."""
        import subprocess
        out_dir.mkdir(parents=True, exist_ok=True)
        net = out_dir / "pilot.net.xml"
        subprocess.run(["netconvert", "--osm-files", str(self.osm_file),
                        "-o", str(net)], check=True)
        return net

    def estimate_traffic_density(self) -> Optional[float]:  # pragma: no cover
        """Placeholder for TraCI-based density estimation on the pilot route."""
        log.info("SUMO density estimation not yet wired to a pilot route")
        return None
