"""BharatDrive-X Twin entry point.

Modes:
  --demo [--scenario NAME]   offline replay: synthetic driver + Indian road scenario
  --live                     webcam driver monitoring (landmark backend required)
  --legacy                   original notebook behaviour, preserved
  --dashboard                serve http://localhost:<port> alongside any mode
  --list-scenarios           show bundled Indian scenarios
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Config  # noqa: E402
from alerts.manager import AlertManager, AudioSink, level_for  # noqa: E402
from context_providers import CctvAnalyzer, MapProvider, SensorHub, WeatherProvider  # noqa: E402
from dashboard.server import DashboardState, start_dashboard  # noqa: E402
from digital_twin.world_state import WorldState, list_scenarios, load_scenario  # noqa: E402
from driver_monitoring.landmarks import SyntheticBackend, get_backend  # noqa: E402
from driver_monitoring.pipeline import DriverMonitor  # noqa: E402
from fusion.journey_safety import compute_journey_safety  # noqa: E402
from road_perception.detection import MockDetector, get_detector  # noqa: E402
from road_perception.road_complexity import RoadComplexityIndex  # noqa: E402
from road_perception.tracking import IouTracker  # noqa: E402
from simulation.engine import CounterfactualEngine  # noqa: E402
from storage.db import Store  # noqa: E402

log = logging.getLogger("bharatdrive")


# ---------------------------------------------------------------- driver scripts
def driver_script(kind: str):
    """time → synthetic driver behaviour, for demo/eval without a camera."""
    def alert(t):  # healthy blinking ~15/min
        phase = t % 4.0
        return {"eye_openness": 0.05 if phase < 0.18 else 1.0}

    def pre_drowsy(t):
        if t < 35:
            return alert(t)
        phase = t % 2.6
        return {"eye_openness": 0.04 if phase < 0.45 else 1.0,  # slower, longer blinks
                "mouth_openness": 0.8 if 40 < t % 90 < 43 else 0.05}

    def drowsy_onset(t):
        if t < 30:
            return alert(t)
        if t < 60:
            phase = t % 2.2
            return {"eye_openness": 0.04 if phase < 0.6 else 1.0,
                    "mouth_openness": 0.8 if 33 < t < 36 or 50 < t < 53 else 0.05,
                    "head_pitch_deg": 6.0}
        phase = t % 3.5
        return {"eye_openness": 0.03 if phase < 2.2 else 0.8,  # microsleeps
                "head_pitch_deg": 18.0 if phase < 2.2 else 4.0}

    return {"alert": alert, "pre_drowsy": pre_drowsy, "drowsy_onset": drowsy_onset}.get(kind, alert)


# ---------------------------------------------------------------- demo loop
def run_demo(cfg: Config, scenario_name: str, duration_s: float, use_dashboard: bool,
             realtime: bool = True) -> dict:
    import json
    scen_path = ROOT / "digital_twin" / "scenarios" / f"{scenario_name}.json"
    scen = json.loads(scen_path.read_text(encoding="utf-8"))
    world = load_scenario(scenario_name)
    script = driver_script(scen.get("driver_script", {}).get("type", "alert"))

    monitor = DriverMonitor(cfg, backend=SyntheticBackend(script))
    detector = MockDetector(scen_path)
    tracker = IouTracker()
    rci = RoadComplexityIndex.from_config(ROOT / "configs" / "complexity_weights.json")
    engine = CounterfactualEngine()
    weather = WeatherProvider(cfg.weather_provider, cfg.weather_api_key)
    maps = MapProvider()
    cctv = CctvAnalyzer(cfg.cctv_source_type, cfg.cctv_source_url)
    sensors = SensorHub()
    store = Store(ROOT / cfg.db_path)
    sid = store.start_session(f"demo:{scenario_name}")
    alerts = AlertManager(cfg.language, cfg.alert_cooldown_s,
                          AudioSink(str(ROOT / cfg.alarm_path)), on_alert=store.alert)
    dstate = DashboardState()
    server = start_dashboard(dstate, cfg.dashboard_port) if use_dashboard else None
    if server:
        log.info("dashboard at http://localhost:%d", cfg.dashboard_port)

    t0 = time.monotonic()
    sim_result = None
    last_sim = -10.0
    snapshot = {}
    t = 0.0
    step = 1.0 / 15.0  # 15 Hz virtual frame rate
    try:
        while t < duration_s:
            loop_start = time.perf_counter()
            d = monitor.update(None, ts=t0 + t)

            dets = tracker.update(detector.detect(None, t), t0 + t)
            wx = weather.get(scen["twin"].get("weather"))
            comp = rci.compute(
                dets, road_width_m=world.road_width_m, lane_marked=world.lane_marked,
                zone=world.zone, weather=wx.data["condition"],
                speed_kmh=world.ego_speed_kmh)
            max_hazard = max((x.hazard_level for x in dets), default=0.0)

            # refresh twin with live driver estimate; simulate every 5 s or on high hazard
            world.driver_readiness = d.readiness
            world.fatigue_risk = d.fatigue_risk
            world.reliability = d.reliability
            if (t - last_sim >= 5.0) or (max_hazard >= 0.7 and t - last_sim >= 2.0):
                sim_result = engine.simulate(world)
                last_sim = t
                store.simulation(t0 + t, world.name, sim_result.recommended.label,
                                 sim_result.recommended.risk, sim_result.recommended.confidence,
                                 {"explanation": sim_result.explanation})

            jss = compute_journey_safety(
                readiness=d.readiness, fatigue_risk=d.fatigue_risk, risk_trend=d.risk_trend,
                reliability=d.reliability, complexity=comp.score,
                complexity_reasons=comp.reasons, max_hazard_level=max_hazard,
                weather_multiplier=wx.data["risk_multiplier"],
                speed_kmh=world.ego_speed_kmh,
                best_action_risk=sim_result.recommended.risk if sim_result else None,
                recommendation=sim_result.recommended.label if sim_result else None,
                sources=["driver_camera(synthetic)", "scenario_playback", wx.source])

            lvl, key = level_for(d.state, max_hazard, d.reliability_state)
            if lvl:
                top = max(dets, key=lambda x: x.hazard_level, default=None)
                alerts.raise_alert(t0 + t, lvl, key, risk_before=d.fatigue_risk,
                                   hazard=(top.cls.replace("_", " ") if top else "hazard"),
                                   direction=(f"approaching from the {top.direction}"
                                              if top and top.direction in ("left", "right")
                                              else "ahead"),
                                   advice="Avoid changing lanes." if top and top.direction in ("left", "right")
                                   else "Reduce speed gradually.")
            else:
                alerts.clear(t0 + t)
            alerts.record_risk_after(t0 + t, d.fatigue_risk)

            store.driver_event(t0 + t, d.state, d.fatigue_risk, d.readiness, d.reliability,
                               {"trend": d.risk_trend, "contributors": d.contributors})
            for x in dets:
                if x.hazard_level >= 0.6:
                    store.hazard(t0 + t, x.cls, x.hazard_level, x.source, x.to_dict())
            store.journey(t0 + t, jss.score, jss.danger_level, jss.confidence, comp.score,
                          {"reasons": jss.reasons})

            latency_ms = (time.perf_counter() - loop_start) * 1000
            snapshot = {
                "driver": {"state": d.state, "readiness": d.readiness,
                           "readiness_band": d.readiness_band,
                           "fatigue_risk": d.fatigue_risk, "risk_trend": d.risk_trend,
                           "reliability": d.reliability, "reliability_state": d.reliability_state,
                           "ear": d.ear, "ear_threshold": d.personalized_ear_threshold,
                           "perclos_60": d.perclos.get(60.0), "contributors": d.contributors},
                "road": {"complexity": comp.score, "level": comp.level,
                         "reasons": comp.reasons, "detections": [x.to_dict() for x in dets]},
                "simulation": (None if not sim_result else {
                    "scenario": sim_result.scenario,
                    "recommended": sim_result.recommended.label,
                    "risk": sim_result.recommended.risk,
                    "confidence": sim_result.recommended.confidence,
                    "actions": [{"label": o.label, "risk": o.risk} for o in sim_result.outcomes],
                    "label": sim_result.label}),
                "journey": {"score": jss.score, "danger_level": jss.danger_level,
                            "confidence": jss.confidence, "reasons": jss.reasons,
                            "horizons": jss.horizons},
                "context": [maps.get(scen["twin"]).to_dict(), wx.to_dict(),
                            cctv.get(scen.get("cctv")).to_dict(), sensors.get().to_dict()],
                "alert": ({"level": alerts.history[-1].level, "text": alerts.history[-1].text}
                          if alerts.history and t0 + t - alerts.history[-1].ts < 6 else None),
                "system": {"fps": d.fps, "latency_ms": latency_ms, "session_id": sid},
            }
            dstate.update(snapshot)
            store.commit()
            t += step
            if realtime and server:
                time.sleep(step)
    finally:
        store.end_session()
        report_path = store.export_report(ROOT / "reports")
        store.close()
        if server:
            log.info("demo finished — dashboard stays up 30 s (Ctrl+C to stop)")
            try:
                time.sleep(30)
            except KeyboardInterrupt:
                pass
            server.shutdown()
        log.info("session report: %s", report_path)
    return snapshot


# ------------------------------------------------------------- showcase mode
def run_showcase(cfg: Config, road_video: str, driver: str, script_kind: str,
                 speed_kmh: float, use_dashboard: bool, detect_every: int = 2,
                 cctv_video: str = None) -> None:
    """Full-stack demo: a real road video drives the trained detector, fused with
    driver monitoring (synthetic script or webcam), complexity, counterfactual
    simulation, Journey Safety Score, alerts and the live dashboard."""
    import cv2

    from digital_twin.world_state import Hazard, WorldState
    from road_perception.detection import YoloDetector

    cap = cv2.VideoCapture(road_video)
    if not cap.isOpened():
        log.error("cannot open road video: %s", road_video)
        return
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    detector = YoloDetector(conf=0.4)
    tracker = IouTracker()
    rci = RoadComplexityIndex.from_config(ROOT / "configs" / "complexity_weights.json")
    engine = CounterfactualEngine()
    weather = WeatherProvider(cfg.weather_provider, cfg.weather_api_key)

    driver_cap = None
    if driver == "live":
        driver_cap = cv2.VideoCapture(cfg.driver_camera)
        monitor = DriverMonitor(cfg, backend=get_backend())
    else:
        monitor = DriverMonitor(cfg, backend=SyntheticBackend(driver_script(script_kind)))

    store = Store(ROOT / cfg.db_path)
    sid = store.start_session(f"showcase:{Path(road_video).name}")
    alerts = AlertManager(cfg.language, cfg.alert_cooldown_s,
                          AudioSink(str(ROOT / cfg.alarm_path)), on_alert=store.alert)
    dstate = DashboardState()
    server = start_dashboard(dstate, cfg.dashboard_port) if use_dashboard else None
    if server:
        log.info("dashboard: http://localhost:%d", cfg.dashboard_port)

    cctv_reading = None
    if cctv_video:
        from cctv.video_analysis import analyze_cctv_video
        log.info("analyzing CCTV video (area context)...")
        # separate lower-confidence detector: high-angle CCTV views are far from
        # the dashcam training distribution
        try:
            cctv_det = YoloDetector(conf=0.22)
        except Exception:
            cctv_det = detector
        cctv_reading = analyze_cctv_video(cctv_video, detector=cctv_det)

    t0 = time.monotonic()
    frame_idx = 0
    dets = []
    sim_result = None
    last_sim = -10.0
    wx = weather.get()
    try:
        while True:
            ok, road = cap.read()
            if not ok:
                break
            frame_idx += 1
            t = frame_idx / video_fps
            loop_start = time.perf_counter()
            # real-time pacing: don't play the video faster than its native FPS
            ahead = t - (time.monotonic() - t0)
            if ahead > 0.002:
                time.sleep(min(ahead, 0.25))

            if frame_idx % detect_every == 0 or not dets:
                dets = tracker.update(detector.detect(road, t0 + t), t0 + t)

            if driver_cap is not None:
                okd, dframe = driver_cap.read()
                d = monitor.update(dframe if okd else None, ts=t0 + t)
            else:
                d = monitor.update(None, ts=t0 + t)

            comp = rci.compute(dets, weather=wx.data["condition"], speed_kmh=speed_kmh)
            max_hazard = max((x.hazard_level for x in dets), default=0.0)

            if (t - last_sim >= 5.0) or (max_hazard >= 0.7 and t - last_sim >= 2.0):
                lane_map = {"ahead": "ego", "left": "left", "right": "right"}
                world = WorldState(
                    name=f"live:{Path(road_video).stem}", ego_speed_kmh=speed_kmh,
                    hazards=[Hazard(cls=x.cls, distance_m=x.distance_m,
                                    rel_speed_ms=x.rel_speed_ms,  # tracked estimate; None = unknown
                                    direction=x.direction, lane=lane_map.get(x.direction, "ego"),
                                    confidence=x.confidence, source=x.source)
                             for x in sorted(dets, key=lambda z: -z.hazard_level)[:6]],
                    weather=wx.data["condition"], friction=wx.data["friction_estimate"],
                    traffic_density=min(1.0, len(dets) / 8.0),
                    driver_readiness=d.readiness, fatigue_risk=d.fatigue_risk,
                    reliability=d.reliability, sources=["front_camera", "driver_camera"])
                sim_result = engine.simulate(world)
                last_sim = t
                store.simulation(t0 + t, world.name, sim_result.recommended.label,
                                 sim_result.recommended.risk, sim_result.recommended.confidence,
                                 {"explanation": sim_result.explanation})

            jss = compute_journey_safety(
                readiness=d.readiness, fatigue_risk=d.fatigue_risk, risk_trend=d.risk_trend,
                reliability=d.reliability, complexity=comp.score,
                complexity_reasons=comp.reasons, max_hazard_level=max_hazard,
                weather_multiplier=wx.data["risk_multiplier"], speed_kmh=speed_kmh,
                best_action_risk=sim_result.recommended.risk if sim_result else None,
                recommendation=sim_result.recommended.label if sim_result else None,
                sources=["front_camera(YOLO)", f"driver({driver})", wx.source])

            lvl, key = level_for(d.state, max_hazard, d.reliability_state)
            if lvl:
                top = max(dets, key=lambda x: x.hazard_level, default=None)
                alerts.raise_alert(t0 + t, lvl, key, risk_before=d.fatigue_risk,
                                   hazard=(top.cls.replace("_", " ") if top else "hazard"),
                                   direction=(f"approaching from the {top.direction}"
                                              if top and top.direction in ("left", "right") else "ahead"),
                                   advice="Avoid changing lanes." if top and top.direction in ("left", "right")
                                   else "Reduce speed gradually.")
            else:
                alerts.clear(t0 + t)
            store.driver_event(t0 + t, d.state, d.fatigue_risk, d.readiness, d.reliability, {})
            store.journey(t0 + t, jss.score, jss.danger_level, jss.confidence, comp.score, {})

            latency_ms = (time.perf_counter() - loop_start) * 1000
            dstate.update({
                "driver": {"state": d.state, "readiness": d.readiness,
                           "readiness_band": d.readiness_band, "fatigue_risk": d.fatigue_risk,
                           "risk_trend": d.risk_trend, "reliability": d.reliability,
                           "reliability_state": d.reliability_state, "ear": d.ear,
                           "ear_threshold": d.personalized_ear_threshold,
                           "perclos_60": d.perclos.get(60.0), "contributors": d.contributors},
                "road": {"complexity": comp.score, "level": comp.level, "reasons": comp.reasons,
                         "detections": [x.to_dict() for x in dets]},
                "simulation": (None if not sim_result else {
                    "scenario": sim_result.scenario, "recommended": sim_result.recommended.label,
                    "risk": sim_result.recommended.risk, "confidence": sim_result.recommended.confidence,
                    "actions": [{"label": o.label, "risk": o.risk} for o in sim_result.outcomes],
                    "label": sim_result.label}),
                "journey": {"score": jss.score, "danger_level": jss.danger_level,
                            "confidence": jss.confidence, "reasons": jss.reasons,
                            "horizons": jss.horizons},
                "context": [wx.to_dict()] + ([cctv_reading.to_dict()] if cctv_reading else []),
                "alert": ({"level": alerts.history[-1].level, "text": alerts.history[-1].text}
                          if alerts.history and t0 + t - alerts.history[-1].ts < 6 else None),
                "system": {"fps": video_fps, "latency_ms": latency_ms, "session_id": sid},
            })

            # annotated road view
            H, W = road.shape[:2]
            for x in dets:
                bx, by, bw, bh = x.box
                col = (0, 0, 255) if x.hazard_level >= 0.7 else (0, 200, 255) if x.hazard_level >= 0.5 else (0, 220, 0)
                cv2.rectangle(road, (int(bx * W), int(by * H)),
                              (int((bx + bw) * W), int((by + bh) * H)), col, 2)
                label = x.cls + (f" ~{x.distance_m:.0f}m" if x.distance_m else "")
                cv2.putText(road, label, (int(bx * W), max(14, int(by * H) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            hud = [f"Driver: {d.state}  readiness={d.readiness}",
                   f"Complexity: {comp.score}/100 ({comp.level})",
                   f"Journey Safety: {jss.score}/100 [{jss.danger_level}]"]
            if sim_result:
                hud.append(f"Sim: {sim_result.recommended.label} ({sim_result.recommended.risk:.0%} est.)")
            for i, line in enumerate(hud):
                cv2.putText(road, line, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 0), 4)
                cv2.putText(road, line, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)
            cv2.imshow("BharatDrive-X Twin — showcase (ESC quits, A acknowledges)", road)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            if k in (ord("a"), ord("A")):
                alerts.acknowledge(t0 + t)
            store.commit()
    finally:
        cap.release()
        if driver_cap is not None:
            driver_cap.release()
        cv2.destroyAllWindows()
        store.end_session()
        log.info("session report: %s", store.export_report(ROOT / "reports"))
        store.close()
        if server:
            log.info("dashboard stays up 30 s (Ctrl+C to stop)")
            try:
                time.sleep(30)
            except KeyboardInterrupt:
                pass
            server.shutdown()


# ---------------------------------------------------------------- live loop
def run_live(cfg: Config, use_dashboard: bool) -> None:
    import cv2
    backend = get_backend()
    if backend.name == "synthetic":
        log.error(
            "No real landmark backend available — live monitoring would silently "
            "fake results. Install one:\n"
            "    pip install mediapipe        (needs Python <= 3.12)\n"
            "or run with your dds01env venv (has face_recognition):\n"
            '    & "S:\\PROJECTs\\dlib-20.0.0\\DRIVER DROWSINESS DETECTION SYSTEM\\'
            'dds01env\\Scripts\\python.exe" -m app.main --live --dashboard')
        return
    monitor = DriverMonitor(cfg, backend=backend)
    dstate = DashboardState()
    server = start_dashboard(dstate, cfg.dashboard_port) if use_dashboard else None
    store = Store(ROOT / cfg.db_path)
    sid = store.start_session("live")
    alerts = AlertManager(cfg.language, cfg.alert_cooldown_s,
                          AudioSink(str(ROOT / cfg.alarm_path)), on_alert=store.alert)
    cap = cv2.VideoCapture(cfg.driver_camera)
    if not cap.isOpened():
        log.error("cannot open camera %s", cfg.driver_camera)
        return
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            d = monitor.update(frame)
            lvl, key = level_for(d.state, 0.0, d.reliability_state)
            if lvl:
                alerts.raise_alert(d.ts, lvl, key, risk_before=d.fatigue_risk,
                                   hazard="hazard", direction="ahead",
                                   advice="Reduce speed gradually.")
            else:
                alerts.clear(d.ts)
            store.driver_event(d.ts, d.state, d.fatigue_risk, d.readiness, d.reliability,
                               {"trend": d.risk_trend})
            dstate.update({"driver": {"state": d.state, "readiness": d.readiness,
                                      "readiness_band": d.readiness_band,
                                      "fatigue_risk": d.fatigue_risk,
                                      "risk_trend": d.risk_trend,
                                      "reliability": d.reliability,
                                      "reliability_state": d.reliability_state,
                                      "ear": d.ear, "ear_threshold": d.personalized_ear_threshold,
                                      "perclos_60": d.perclos.get(60.0),
                                      "contributors": d.contributors},
                           "system": {"fps": d.fps, "session_id": sid}})
            ear_txt = f"{d.ear:.3f}" if d.ear is not None else "--"
            pcl = d.perclos.get(60.0)
            cv2.putText(frame, f"{d.state} readiness={d.readiness}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame,
                        f"EAR {ear_txt} / thr {d.personalized_ear_threshold}  "
                        f"PERCLOS60 {f'{pcl:.0%}' if pcl is not None else '--'}  "
                        f"risk {f'{d.fatigue_risk:.0%}' if d.fatigue_risk is not None else '--'}",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("BharatDrive-X Twin (ESC quits, A acknowledges)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            if k in (ord("a"), ord("A")):
                alerts.acknowledge(d.ts)
            store.commit()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        store.end_session()
        log.info("session report: %s", store.export_report(ROOT / "reports"))
        store.close()
        if server:
            server.shutdown()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="BharatDrive-X Twin — AI safety co-pilot (research prototype)")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--live", action="store_true")
    p.add_argument("--legacy", action="store_true")
    p.add_argument("--dashboard", action="store_true")
    p.add_argument("--scenario", default="motorcycle_cut_in")
    p.add_argument("--duration", type=float, default=90.0)
    p.add_argument("--fast", action="store_true", help="demo without realtime sleep")
    p.add_argument("--list-scenarios", action="store_true")
    # showcase mode: real road video + trained detector + full fusion stack
    p.add_argument("--road", default=None, metavar="VIDEO",
                   help="road video file → showcase mode (uses trained YOLO)")
    p.add_argument("--driver", default="synthetic", choices=("synthetic", "live"),
                   help="showcase driver source (synthetic script or webcam)")
    p.add_argument("--driver-script", default="drowsy_onset",
                   choices=("alert", "pre_drowsy", "drowsy_onset"))
    p.add_argument("--speed", type=float, default=40.0, help="assumed ego speed km/h")
    p.add_argument("--detect-every", type=int, default=2,
                   help="run detector every Nth frame (CPU headroom)")
    p.add_argument("--cctv", default=None, metavar="VIDEO",
                   help="authorized traffic-cam video for area-level context")
    args = p.parse_args()
    cfg = Config.load()
    if args.list_scenarios:
        for s in list_scenarios():
            print(s)
        return
    if args.legacy:
        from legacy.legacy_pipeline import run_legacy
        run_legacy(cfg.driver_camera, str(ROOT / cfg.alarm_path))
        return
    if args.road:
        run_showcase(cfg, args.road, args.driver, args.driver_script,
                     args.speed, args.dashboard, args.detect_every, args.cctv)
        return
    if args.live:
        run_live(cfg, args.dashboard)
        return
    run_demo(cfg, args.scenario, args.duration, args.dashboard, realtime=not args.fast)


if __name__ == "__main__":
    main()
