"""Zero-dependency live dashboard: stdlib HTTP server + single-page UI polling
/api/state. Every value is labelled Measured/Estimated/Simulated/Unavailable.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

_STATIC = Path(__file__).resolve().parent / "static" / "index.html"


class DashboardState:
    """Thread-safe latest-snapshot holder."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {"status": "starting"}

    def update(self, snapshot: Dict[str, Any]) -> None:
        with self._lock:
            self._state = snapshot

    def get(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._state)


def start_dashboard(state: DashboardState, port: int = 8765) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/api/state"):
                body = json.dumps(state.get()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            else:
                body = _STATIC.read_bytes() if _STATIC.exists() else b"dashboard missing"
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a) -> None:  # silence request spam
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server
