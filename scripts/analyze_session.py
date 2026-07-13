"""Offline session analyzer — shows what drove the fatigue risk in a live run.

Usage:
    py -3.12 scripts/analyze_session.py            # latest session
    py -3.12 scripts/analyze_session.py <session_id>
"""
from __future__ import annotations

import collections
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "storage" / "bharatdrive.sqlite3"


def main() -> None:
    con = sqlite3.connect(str(DB))
    cur = con.cursor()

    sid = sys.argv[1] if len(sys.argv) > 1 else None
    if sid is None:
        cur.execute("SELECT id FROM session ORDER BY rowid DESC LIMIT 1")
        row = cur.fetchone()
        if not row:
            print("no sessions found")
            return
        sid = row[0]
    print(f"session: {sid}\n")

    cur.execute("SELECT count(*) FROM driver_state_event WHERE session_id=?", (sid,))
    total = cur.fetchone()[0]
    print(f"frames: {total}")

    cur.execute(
        "SELECT state, count(*) FROM driver_state_event WHERE session_id=? "
        "GROUP BY state ORDER BY 2 DESC", (sid,))
    print("\nstate distribution:")
    for state, n in cur.fetchall():
        print(f"  {n:6d}  ({n/total:5.1%})  {state}")

    cur.execute(
        "SELECT round(min(fatigue_risk),3), round(avg(fatigue_risk),3), "
        "round(max(fatigue_risk),3) FROM driver_state_event "
        "WHERE session_id=? AND fatigue_risk IS NOT NULL", (sid,))
    lo, avg, hi = cur.fetchone()
    print(f"\nfatigue risk  min={lo}  avg={avg}  max={hi}")

    # what signals fired when the model considered the driver fatigued
    cur.execute(
        "SELECT detail FROM driver_state_event WHERE session_id=? "
        "AND fatigue_risk >= 0.35", (sid,))
    cnt: collections.Counter = collections.Counter()
    n = 0
    for (detail,) in cur.fetchall():
        try:
            d = json.loads(detail) if detail else {}
        except json.JSONDecodeError:
            continue
        n += 1
        for contrib in (d.get("contributors") or []):
            cnt[contrib.split(" at ")[0][:46]] += 1
    print(f"\ncontributors among {n} frames with risk >= 0.35:")
    if not cnt:
        print("  (none logged — re-run needs the diagnostic build of app/main.py)")
    for label, c in cnt.most_common(15):
        print(f"  {c:6d}  {label}")

    con.close()


if __name__ == "__main__":
    main()
