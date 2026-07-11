"""SQLite persistence (WAL). Structured events only — raw video is never stored
unless RECORD_VIDEO + CONSENT_RECORDING are both enabled (enforced in Config).
Schema designed to migrate to PostgreSQL without change of shape.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS session (
  id TEXT PRIMARY KEY, started_at REAL, ended_at REAL, mode TEXT, notes TEXT);
CREATE TABLE IF NOT EXISTS driver_state_event (
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, ts REAL, state TEXT,
  fatigue_risk REAL, readiness INTEGER, reliability REAL, detail TEXT);
CREATE TABLE IF NOT EXISTS hazard_event (
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, ts REAL, cls TEXT,
  hazard_level REAL, source TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS journey_risk (
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, ts REAL, score INTEGER,
  danger TEXT, confidence TEXT, complexity INTEGER, detail TEXT);
CREATE TABLE IF NOT EXISTS simulation_run (
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, ts REAL, scenario TEXT,
  recommended TEXT, risk REAL, confidence TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS alert (
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, ts REAL, level INTEGER,
  key TEXT, text TEXT, risk_before REAL, risk_after REAL, response_time_s REAL);
CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, event TEXT, detail TEXT);
"""


class Store:
    def __init__(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(p))
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.executescript(_SCHEMA)
        self.session_id: Optional[str] = None

    def start_session(self, mode: str) -> str:
        self.session_id = uuid.uuid4().hex[:12]
        self.con.execute("INSERT INTO session(id, started_at, mode) VALUES (?,?,?)",
                         (self.session_id, time.time(), mode))
        self.audit("session_start", {"mode": mode})
        self.con.commit()
        return self.session_id

    def end_session(self) -> None:
        if self.session_id:
            self.con.execute("UPDATE session SET ended_at=? WHERE id=?",
                             (time.time(), self.session_id))
            self.audit("session_end", {})
            self.con.commit()

    def driver_event(self, ts: float, state: str, fatigue: Optional[float],
                     readiness: Optional[int], reliability: float, detail: dict) -> None:
        self.con.execute(
            "INSERT INTO driver_state_event(session_id, ts, state, fatigue_risk, readiness, reliability, detail)"
            " VALUES (?,?,?,?,?,?,?)",
            (self.session_id, ts, state, fatigue, readiness, reliability, json.dumps(detail)))

    def hazard(self, ts: float, cls: str, level: float, source: str, detail: dict) -> None:
        self.con.execute(
            "INSERT INTO hazard_event(session_id, ts, cls, hazard_level, source, detail) VALUES (?,?,?,?,?,?)",
            (self.session_id, ts, cls, level, source, json.dumps(detail)))

    def journey(self, ts: float, score: Optional[int], danger: str, confidence: str,
                complexity: int, detail: dict) -> None:
        self.con.execute(
            "INSERT INTO journey_risk(session_id, ts, score, danger, confidence, complexity, detail)"
            " VALUES (?,?,?,?,?,?,?)",
            (self.session_id, ts, score, danger, confidence, complexity, json.dumps(detail)))

    def simulation(self, ts: float, scenario: str, recommended: str, risk: float,
                   confidence: str, detail: dict) -> None:
        self.con.execute(
            "INSERT INTO simulation_run(session_id, ts, scenario, recommended, risk, confidence, detail)"
            " VALUES (?,?,?,?,?,?,?)",
            (self.session_id, ts, scenario, recommended, risk, confidence, json.dumps(detail)))

    def alert(self, ev) -> None:
        self.con.execute(
            "INSERT INTO alert(session_id, ts, level, key, text, risk_before, risk_after, response_time_s)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (self.session_id, ev.ts, ev.level, ev.key, ev.text, ev.risk_before,
             ev.risk_after, ev.response_time_s))

    def audit(self, event: str, detail: dict) -> None:
        self.con.execute("INSERT INTO audit_log(ts, event, detail) VALUES (?,?,?)",
                         (time.time(), event, json.dumps(detail)))

    def commit(self) -> None:
        self.con.commit()

    # ---------------- session report ----------------
    def session_report(self) -> Dict[str, Any]:
        sid = self.session_id
        cur = self.con.cursor()
        row = cur.execute("SELECT started_at, ended_at, mode FROM session WHERE id=?", (sid,)).fetchone()
        events = cur.execute(
            "SELECT COUNT(*), AVG(fatigue_risk), MAX(fatigue_risk), AVG(reliability), AVG(readiness)"
            " FROM driver_state_event WHERE session_id=?", (sid,)).fetchone()
        hazards = cur.execute(
            "SELECT cls, COUNT(*) FROM hazard_event WHERE session_id=? GROUP BY cls", (sid,)).fetchall()
        alerts = cur.execute(
            "SELECT level, COUNT(*) FROM alert WHERE session_id=? GROUP BY level", (sid,)).fetchall()
        jss = cur.execute(
            "SELECT AVG(score), MIN(score) FROM journey_risk WHERE session_id=?", (sid,)).fetchone()
        return {
            "session_id": sid, "mode": row[2] if row else None,
            "duration_s": round((row[1] or time.time()) - row[0], 1) if row else None,
            "driver_samples": events[0],
            "avg_fatigue_risk": round(events[1], 3) if events[1] is not None else None,
            "max_fatigue_risk": round(events[2], 3) if events[2] is not None else None,
            "avg_reliability": round(events[3], 3) if events[3] is not None else None,
            "avg_readiness": round(events[4], 1) if events[4] is not None else None,
            "hazards": {c: n for c, n in hazards},
            "alerts_by_level": {int(l): n for l, n in alerts},
            "avg_journey_safety": round(jss[0], 1) if jss[0] is not None else None,
            "min_journey_safety": jss[1],
            "limitations": [
                "Simulation risks are estimates, not real-world probabilities",
                "Monocular distances are approximate",
                "Small-sample personalization — not universal performance",
            ],
        }

    def export_report(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        report = self.session_report()
        path = out_dir / f"session_{self.session_id}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return path

    def close(self) -> None:
        self.con.commit()
        self.con.close()
