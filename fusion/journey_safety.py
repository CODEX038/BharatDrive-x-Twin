"""Unified Journey Safety Score with separate confidence and prediction horizons."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class JourneySafety:
    score: Optional[int]          # 0–100, higher = safer; None = insufficient data
    danger_level: str             # Low | Moderate | High | Critical | Unknown
    confidence: str               # High | Medium | Low
    reasons: List[str] = field(default_factory=list)
    horizons: Dict[str, str] = field(default_factory=dict)
    recommendation: Optional[str] = None
    sources: List[str] = field(default_factory=list)


def compute_journey_safety(*, readiness: Optional[int], fatigue_risk: Optional[float],
                           risk_trend: str, reliability: float, complexity: int,
                           complexity_reasons: List[str], max_hazard_level: float,
                           weather_multiplier: float, speed_kmh: Optional[float],
                           best_action_risk: Optional[float],
                           recommendation: Optional[str],
                           sources: List[str]) -> JourneySafety:
    reasons: List[str] = []
    parts: List[float] = []
    weights: List[float] = []

    if readiness is not None:
        parts.append(readiness)
        weights.append(0.35)
        if readiness < 60:
            reasons.append(f"Driver readiness reduced ({readiness}/100)")
    if fatigue_risk is not None and risk_trend == "Increasing":
        reasons.append("Driver fatigue is increasing")
    parts.append(100 - complexity)
    weights.append(0.30)
    if complexity >= 55:
        reasons.extend(complexity_reasons[:3])
    hazard_safety = 100 * (1 - max_hazard_level)
    parts.append(hazard_safety)
    weights.append(0.20)
    if max_hazard_level >= 0.6:
        reasons.append("High-severity hazard in view")
    if best_action_risk is not None:
        parts.append(100 * (1 - best_action_risk))
        weights.append(0.15)

    if not parts:
        return JourneySafety(None, "Unknown", "Low", ["Insufficient data"], {}, None, sources)

    score = sum(p * w for p, w in zip(parts, weights)) / sum(weights)
    score /= max(1.0, weather_multiplier * 0.6 + 0.4)  # weather degrades safety mildly
    if speed_kmh and speed_kmh > 50 and complexity >= 55:
        score -= 8
        reasons.append("Current speed is high for the conditions")
    score = int(max(0, min(100, round(score))))

    danger = ("Critical" if score < 20 else "High" if score < 40
              else "Moderate" if score < 65 else "Low")
    confidence = ("High" if reliability >= 0.7 else
                  "Medium" if reliability >= 0.4 else "Low")
    if reliability < 0.4:
        reasons.append("Driver observation reliability is low — treat with caution")

    horizons = {
        "immediate_0_3s": "Hazard in path" if max_hazard_level >= 0.7 else "No immediate threat detected",
        "short_3_10s": "Approaching hazards being tracked" if max_hazard_level >= 0.4 else "Clear",
        "near_10_30s": ("Complex road segment continues" if complexity >= 55 else "Manageable complexity"),
        "route_level": ("Fatigue rising over route" if risk_trend == "Increasing" else "Stable"),
    }
    return JourneySafety(score, danger, confidence, reasons, horizons, recommendation, sources)
