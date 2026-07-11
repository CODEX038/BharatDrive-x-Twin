"""Physics-aware helper calculations for counterfactual simulation.

Simple, defensible kinematics — all outputs are simulation estimates, never
guaranteed real-world outcomes.
"""
from __future__ import annotations

G = 9.81


def braking_distance_m(speed_ms: float, friction: float, decel_fraction: float = 1.0) -> float:
    """v² / (2 μ g), scaled by how hard the braking action is (0–1]."""
    mu = max(0.1, friction * max(0.1, decel_fraction))
    return (speed_ms ** 2) / (2.0 * mu * G)


def stopping_distance_m(speed_ms: float, friction: float, reaction_s: float,
                        decel_fraction: float = 1.0) -> float:
    return speed_ms * reaction_s + braking_distance_m(speed_ms, friction, decel_fraction)


def ttc_s(distance_m: float, closing_speed_ms: float) -> float:
    if closing_speed_ms <= 0.05:
        return float("inf")
    return distance_m / closing_speed_ms


def collision_margin(distance_m: float, speed_ms: float, friction: float,
                     reaction_s: float, decel_fraction: float = 1.0) -> float:
    """>0: can stop short of the hazard; <0: shortfall in metres."""
    return distance_m - stopping_distance_m(speed_ms, friction, reaction_s, decel_fraction)


def risk_from_margin(margin_m: float, scale_m: float = 15.0) -> float:
    """Map stopping margin to 0–1 risk (0 when ample margin, →1 as shortfall grows)."""
    if margin_m >= scale_m:
        return 0.05
    if margin_m <= -scale_m:
        return 0.98
    return 0.05 + 0.93 * (scale_m - margin_m) / (2 * scale_m)
