"""Personal Fatigue Signature — per-session anonymous baseline.

Learns the driver's normal EAR, blink behaviour, PERCLOS and head pitch from
HIGH-reliability ALERT-state observations only. Adapts slowly, rejects outliers,
never adapts during suspected fatigue, supports reset. Stores numbers only —
no identity, no embeddings (privacy requirement).
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_FEATURES = ("ear", "blink_duration", "blink_rate", "perclos", "head_pitch")


@dataclass
class BaselineFeature:
    median: float
    mad: float  # median absolute deviation (robust spread)
    n: int


@dataclass
class SignatureReport:
    calibrated: bool
    baseline: Dict[str, BaselineFeature]
    deviations: Dict[str, float]     # robust z-scores of current vs baseline
    fatigue_deviation: float         # 0..1 aggregate
    level: str                       # None|Low|Moderate|High


class FatigueSignature:
    def __init__(self, calibration_s: float = 30.0, min_samples: int = 40,
                 max_samples: int = 600) -> None:
        self.calibration_s = calibration_s
        self.min_samples = min_samples
        self.max_samples = max_samples
        self._buf: Dict[str, List[float]] = {f: [] for f in _FEATURES}
        self._start_ts: Optional[float] = None

    # -- learning ---------------------------------------------------------
    def observe(self, ts: float, features: Dict[str, Optional[float]],
                reliable: bool, suspected_fatigue: bool) -> None:
        """Add a sample to the baseline. Skipped when unreliable or fatigued."""
        if self._start_ts is None:
            self._start_ts = ts
        if not reliable or suspected_fatigue:
            return
        for name in _FEATURES:
            v = features.get(name)
            if v is None:
                continue
            buf = self._buf[name]
            if len(buf) >= self.min_samples:  # outlier rejection at 3×MAD once formed
                med = statistics.median(buf)
                mad = _mad(buf, med)
                if mad > 1e-9 and abs(v - med) > 3 * 1.4826 * mad:
                    continue
            buf.append(v)
            if len(buf) > self.max_samples:   # slow adaptation: drop oldest
                buf.pop(0)

    @property
    def calibrated(self) -> bool:
        return len(self._buf["ear"]) >= self.min_samples

    def reset(self) -> None:
        self._buf = {f: [] for f in _FEATURES}
        self._start_ts = None

    # -- comparison -------------------------------------------------------
    def compare(self, current: Dict[str, Optional[float]]) -> SignatureReport:
        baseline: Dict[str, BaselineFeature] = {}
        deviations: Dict[str, float] = {}
        for name in _FEATURES:
            buf = self._buf[name]
            if len(buf) < self.min_samples:
                continue
            med = statistics.median(buf)
            mad = _mad(buf, med) or 1e-6
            baseline[name] = BaselineFeature(round(med, 4), round(mad, 4), len(buf))
            v = current.get(name)
            if v is None:
                continue
            z = (v - med) / (1.4826 * mad)
            # direction of concern: EAR down, everything else up
            deviations[name] = -z if name == "ear" else z
        if not self.calibrated:
            return SignatureReport(False, baseline, {}, 0.0, "None")
        concerning = [max(0.0, z) for z in deviations.values()]
        agg = min(1.0, (sum(concerning) / (len(concerning) or 1)) / 4.0)
        level = ("High" if agg >= 0.6 else "Moderate" if agg >= 0.3
                 else "Low" if agg >= 0.12 else "None")
        return SignatureReport(True, baseline, {k: round(v, 2) for k, v in deviations.items()},
                               round(agg, 3), level)

    def personalized_ear_threshold(self, universal: float = 0.25) -> float:
        """Personal closed-eye threshold: fraction of the driver's own open-eye median."""
        buf = self._buf["ear"]
        if len(buf) < self.min_samples:
            return universal
        return max(0.12, min(0.35, 0.72 * statistics.median(buf)))


def _mad(values: List[float], med: Optional[float] = None) -> float:
    if not values:
        return 0.0
    m = statistics.median(values) if med is None else med
    return statistics.median([abs(v - m) for v in values])
