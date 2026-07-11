"""Counterfactual multi-action simulator.

For a WorldState, generates candidate actions, estimates per-action risk with
physics-aware rules + Monte Carlo perturbation, ranks them, and explains the
lowest-risk choice. RL never touches a real vehicle; this engine only compares
hypotheticals. All numbers are SIMULATION ESTIMATES.
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from digital_twin.world_state import Hazard, WorldState
from .physics import risk_from_margin

VULNERABLE = {"pedestrian", "motorcycle", "scooter", "bicycle", "cattle", "dog"}


@dataclass
class ActionOutcome:
    action: str
    label: str
    risk: float                      # 0–1 simulation estimate
    risk_spread: float               # Monte Carlo std — uncertainty
    confidence: str                  # High | Medium | Low
    delay_cost: float                # 0–1 relative travel delay
    workload: float                  # 0–1 driver workload
    components: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)


@dataclass
class SimulationResult:
    scenario: str
    outcomes: List[ActionOutcome]
    recommended: ActionOutcome
    explanation: List[str]
    sources: List[str]
    label: str = "SIMULATION ESTIMATE — not a guaranteed outcome"


# action → (label, speed multiplier, decel fraction, lane shift, delay, workload)
ACTIONS = {
    "continue":        ("Continue at current speed",        1.00, 1.0, 0, 0.00, 0.10),
    "gradual_decel":   ("Gradually reduce speed",           0.70, 0.5, 0, 0.15, 0.20),
    "strong_decel":    ("Strongly reduce speed",            0.40, 0.9, 0, 0.30, 0.45),
    "lane_change_left":("Change lane left",                 0.95, 1.0, -1, 0.05, 0.55),
    "lane_change_right":("Change lane right",               0.95, 1.0, +1, 0.05, 0.55),
    "increase_gap":    ("Increase following distance",      0.85, 0.4, 0, 0.10, 0.15),
    "safe_stop":       ("Stop at a safe location",          0.00, 0.8, 0, 0.80, 0.50),
    "reroute":         ("Reduce speed and reroute",         0.70, 0.5, 0, 0.35, 0.30),
    "recommend_break": ("Recommend a break at next stop",   0.60, 0.5, 0, 0.60, 0.25),
}


class CounterfactualEngine:
    def __init__(self, mc_samples: int = 60, seed: Optional[int] = 7) -> None:
        self.mc_samples = mc_samples
        self.seed = seed

    def simulate(self, world: WorldState) -> SimulationResult:
        actions = self._candidate_actions(world)
        rng = random.Random(self.seed)
        outcomes = [self._evaluate(world, a, rng) for a in actions]
        outcomes.sort(key=lambda o: o.risk + 0.15 * o.delay_cost + 0.1 * o.workload)
        best = outcomes[0]
        explanation = self._explain(world, best, outcomes)
        return SimulationResult(scenario=world.name, outcomes=outcomes,
                                recommended=best, explanation=explanation,
                                sources=list(world.sources))

    # ------------------------------------------------------------------
    def _candidate_actions(self, w: WorldState) -> List[str]:
        acts = ["continue", "gradual_decel", "strong_decel", "increase_gap"]
        if w.lane_count >= 2:
            acts += ["lane_change_left", "lane_change_right"]
        if w.alt_route_available:
            acts.append("reroute")
        if (w.fatigue_risk or 0) >= 0.5 or (w.driver_readiness or 100) <= 40:
            acts += ["safe_stop", "recommend_break"]
        return acts

    def _evaluate(self, w: WorldState, action: str, rng: random.Random) -> ActionOutcome:
        label, v_mult, decel, lane_shift, delay, workload = ACTIONS[action]
        samples: List[float] = []
        comp_acc: Dict[str, List[float]] = {}
        for _ in range(self.mc_samples):
            r, comps = self._single_rollout(w, action, v_mult, decel, lane_shift, rng)
            samples.append(r)
            for k, v in comps.items():
                comp_acc.setdefault(k, []).append(v)
        risk = statistics.mean(samples)
        spread = statistics.pstdev(samples)
        conf = "High" if spread < 0.06 and w.reliability >= 0.65 else \
               "Medium" if spread < 0.15 else "Low"
        comps = {k: round(statistics.mean(v), 2) for k, v in comp_acc.items()}
        reasons = [f"{k.replace('_', ' ')} risk {v:.0%}" for k, v in
                   sorted(comps.items(), key=lambda kv: -kv[1])[:3] if v > 0.15]
        assumptions = [
            f"Reaction time {w.reaction_time_s:.1f}s (readiness-scaled)",
            f"Road friction {w.friction:.2f} ({w.weather})",
            "Monocular distances are estimates",
        ]
        return ActionOutcome(action=action, label=label, risk=round(risk, 2),
                             risk_spread=round(spread, 3), confidence=conf,
                             delay_cost=delay, workload=workload,
                             components=comps, reasons=reasons, assumptions=assumptions)

    def _single_rollout(self, w: WorldState, action: str, v_mult: float,
                        decel: float, lane_shift: int, rng: random.Random):
        # perturb inputs (Monte Carlo)
        speed = max(0.0, w.ego_speed_ms * v_mult * rng.gauss(1.0, 0.06))
        react = max(0.4, w.reaction_time_s * rng.gauss(1.0, 0.15))
        friction = max(0.15, w.friction * rng.gauss(1.0, 0.08))
        fatigue_mult = 1.0 + 0.6 * (w.fatigue_risk or 0.0)

        comps: Dict[str, float] = {"collision": 0.0, "vulnerable_road_user": 0.0,
                                   "rear_collision": 0.0, "road_departure": 0.0,
                                   "sudden_braking": 0.0}
        target_lane = {0: "ego", -1: "left", 1: "right"}[lane_shift] if lane_shift else "ego"

        for h in w.hazards:
            noisy_d = None if h.distance_m is None else max(0.5, h.distance_m * rng.gauss(1.0, 0.12))
            in_path = (h.lane == "ego" and lane_shift == 0) or (h.lane == target_lane)
            if lane_shift and h.lane == target_lane:
                # moving into an occupied lane: proximity risk dominates
                prox = 1.0 if (noisy_d or 5) < 12 else 0.5 if (noisy_d or 20) < 25 else 0.15
                key = "vulnerable_road_user" if h.cls in VULNERABLE else "collision"
                comps[key] = max(comps[key], prox * h.confidence)
                continue
            if not in_path or noisy_d is None:
                continue
            closing = speed if h.rel_speed_ms is None else max(0.0, h.rel_speed_ms * v_mult)
            if action == "safe_stop":
                closing = min(closing, speed)
            # stopping margin vs. hazard at (noisy) distance
            margin = noisy_d - (closing * react +
                                (closing ** 2) / (2 * max(0.1, friction * max(0.1, decel)) * 9.81))
            r = risk_from_margin(margin) * h.confidence
            key = "vulnerable_road_user" if h.cls in VULNERABLE else "collision"
            comps[key] = max(comps[key], r)

        # secondary risks
        if action == "strong_decel":
            comps["rear_collision"] = min(0.9, 0.25 + 0.5 * w.traffic_density)
            comps["sudden_braking"] = 0.4
        if action in ("lane_change_left", "lane_change_right"):
            comps["road_departure"] = 0.15 if (w.road_width_m or 7) < 6 else 0.05
            comps["collision"] = max(comps["collision"], 0.15 + 0.3 * w.traffic_density)
        if action == "continue" and (w.fatigue_risk or 0) >= 0.5:
            comps["collision"] = min(1.0, comps["collision"] + 0.15 * fatigue_mult)
        if action == "safe_stop":
            comps["rear_collision"] = max(comps["rear_collision"], 0.1 + 0.25 * w.traffic_density)

        total = 1.0
        for v in comps.values():
            total *= (1.0 - min(0.98, v))
        risk = (1.0 - total)
        risk = min(0.98, risk * (1.0 + 0.25 * (w.fatigue_risk or 0.0)
                                 * (0.0 if action in ("safe_stop", "recommend_break", "reroute") else 1.0)))
        return risk, comps

    def _explain(self, w: WorldState, best: ActionOutcome,
                 all_: List[ActionOutcome]) -> List[str]:
        lines = [f"Recommended (simulated): {best.label} — estimated risk {best.risk:.0%} "
                 f"(confidence {best.confidence})"]
        worst = max(all_, key=lambda o: o.risk)
        lines.append(f"Highest-risk alternative: {worst.label} at {worst.risk:.0%}")
        for h in w.hazards[:3]:
            d = f"~{h.distance_m:.0f} m (est.)" if h.distance_m else "distance unknown"
            lines.append(f"Hazard: {h.cls.replace('_', ' ')} {h.direction}, {d}, source: {h.source}")
        if (w.fatigue_risk or 0) >= 0.4:
            lines.append(f"Driver fatigue risk {w.fatigue_risk:.0%} lengthens assumed reaction time")
        if w.weather != "clear":
            lines.append(f"Weather '{w.weather}' reduces assumed friction to {w.friction:.2f}")
        lines.append("All values are simulation estimates from the digital twin.")
        return lines
