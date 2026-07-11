"""PERCLOS over rolling time windows, counting only *valid observation time*.

Missing frames / hidden eyes / low-confidence samples are excluded from both
numerator and denominator (audit W-7/W-8).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Sequence, Tuple


@dataclass
class _Sample:
    ts: float
    dt: float
    closed: bool


class Perclos:
    def __init__(self, windows_s: Sequence[float] = (30.0, 60.0, 120.0)) -> None:
        self.windows = tuple(windows_s)
        self._samples: Deque[_Sample] = deque()
        self._last_ts: Optional[float] = None

    def update(self, ts: float, ear: Optional[float], threshold: float,
               valid: bool) -> Dict[float, Optional[float]]:
        dt = 0.0
        if self._last_ts is not None:
            dt = max(0.0, min(ts - self._last_ts, 1.0))  # cap gaps at 1 s
        self._last_ts = ts
        if valid and ear is not None and dt > 0:
            self._samples.append(_Sample(ts, dt, ear < threshold))
        horizon = max(self.windows)
        while self._samples and self._samples[0].ts < ts - horizon:
            self._samples.popleft()
        out: Dict[float, Optional[float]] = {}
        for w in self.windows:
            valid_t = closed_t = 0.0
            for s in self._samples:
                if s.ts >= ts - w:
                    valid_t += s.dt
                    if s.closed:
                        closed_t += s.dt
            # require at least 25% of the window observed before reporting
            out[w] = (closed_t / valid_t) if valid_t >= 0.25 * w else None
        return out

    def reset(self) -> None:
        self._samples.clear()
        self._last_ts = None
