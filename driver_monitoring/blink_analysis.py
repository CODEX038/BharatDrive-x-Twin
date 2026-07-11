"""Timestamp-based blink detection (no frame-count assumptions — fixes audit W-3)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple


@dataclass
class Blink:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class BlinkStats:
    blink_rate_per_min: float = 0.0
    mean_duration_s: float = 0.0
    long_blink_count: int = 0
    inter_blink_interval_s: float = 0.0
    current_closure_s: float = 0.0        # ongoing eye-closure duration
    microsleep: bool = False


class BlinkTracker:
    """Feeds on (timestamp, ear, threshold, valid). Closure below threshold = candidate blink;
    closure longer than `long_blink_s` counts as long blink; longer than `microsleep_s`
    is flagged as ongoing microsleep. Invalid observations pause the tracker (never count
    hidden eyes as closed)."""

    def __init__(self, long_blink_s: float = 0.4, microsleep_s: float = 1.0,
                 window_s: float = 60.0) -> None:
        self.long_blink_s = long_blink_s
        self.microsleep_s = microsleep_s
        self.window_s = window_s
        self._blinks: Deque[Blink] = deque()
        self._closed_since: Optional[float] = None
        self._last_valid_ts: Optional[float] = None

    def update(self, ts: float, ear: Optional[float], threshold: float,
               valid: bool) -> BlinkStats:
        if not valid or ear is None:
            # observation gap: close any open closure without counting it
            self._closed_since = None
            return self._stats(ts)
        self._last_valid_ts = ts
        if ear < threshold:
            if self._closed_since is None:
                self._closed_since = ts
        else:
            if self._closed_since is not None:
                blink = Blink(self._closed_since, ts)
                if 0.03 <= blink.duration <= 5.0:  # reject sensor glitches
                    self._blinks.append(blink)
                self._closed_since = None
        self._trim(ts)
        return self._stats(ts)

    def _trim(self, now: float) -> None:
        while self._blinks and self._blinks[0].end < now - self.window_s:
            self._blinks.popleft()

    def _stats(self, now: float) -> BlinkStats:
        blinks = list(self._blinks)
        closure = (now - self._closed_since) if self._closed_since is not None else 0.0
        n = len(blinks)
        rate = n * (60.0 / self.window_s)
        mean_d = sum(b.duration for b in blinks) / n if n else 0.0
        longs = sum(1 for b in blinks if b.duration >= self.long_blink_s)
        ibi = 0.0
        if n >= 2:
            gaps = [blinks[i + 1].start - blinks[i].end for i in range(n - 1)]
            ibi = sum(gaps) / len(gaps)
        return BlinkStats(
            blink_rate_per_min=rate, mean_duration_s=mean_d, long_blink_count=longs,
            inter_blink_interval_s=ibi, current_closure_s=closure,
            microsleep=closure >= self.microsleep_s,
        )

    def reset(self) -> None:
        self._blinks.clear()
        self._closed_since = None
