# BharatDrive-X Twin — System Design

Covers the pre-coding deliverables (§39): architecture, data flow, state machine, module designs, DB schema, API, dashboard, testing, experiments, roadmap.

## 1. Claim and constraints

> BharatDrive-X Twin is a context-aware AI safety co-pilot that combines personalized driver-readiness analysis, live road perception, traffic and route context, authorized external observations and a digital twin to simulate multiple possible driving actions and recommend a lower-risk response.

Observe / detect / estimate / predict / simulate / compare / explain / recommend / warn — never actuate. No vehicle control, no auto emergency calls, no unauthorized cameras, no raw-video upload. All risk numbers are labelled simulation estimates.

## 2. Data flow

```
driver cam ─► landmark backend ─► eye/blink/yawn/PERCLOS/head-pose ─► reliability engine
                                                     │                      │ (gates)
                                                     ▼                      ▼
                              personal fatigue signature ─► temporal features ─► fatigue risk
                                                                                │
road cam ─► detector ─► tracker ─► hazards ─► Road Complexity Index            ▼
maps/weather/CCTV/sensors (offline-first providers, each with freshness) ─► FUSION
                                                                                │
                        digital twin (scenario state) ─► counterfactual actions │
                                                                                ▼
                                             Journey Safety Score ─► alert policy ─► dashboard/DB
```

Everything is timestamp-based (`time.monotonic` for durations, UTC wall clock for records). No fixed-FPS assumptions.

## 3. Driver state machine

States: CALIBRATING, ALERT, SLIGHT_FATIGUE, PRE_DROWSY, DROWSY, CRITICAL, DISTRACTED, UNRESPONSIVE, FACE_NOT_VISIBLE, OBSERVATION_UNRELIABLE, CAMERA_ERROR, PAUSED, ENDED.

Rules: transitions require the condition to hold for a dwell time (hysteresis); reliability below threshold forces OBSERVATION_UNRELIABLE (never DROWSY); missing face ≥ 2 s → FACE_NOT_VISIBLE; missing face while previous state ≥ DROWSY → UNRESPONSIVE candidate; alerts fire on state entry, escalate on persistence, cool down on exit.

## 4. Personal Fatigue Signature

Rolling robust baseline per session (median + MAD) over: EAR median, L/R EAR difference, blink duration, blink rate, inter-blink interval, PERCLOS, head pitch. Updated only from HIGH-reliability, ALERT-state windows (no adaptation during suspected fatigue); EWMA with slow alpha; outliers rejected at 3×MAD; reset/recalibrate supported; stored as anonymous numbers keyed by session — no identity, no embeddings, no face recognition.

Deviation score = weighted z-scores of current window vs baseline → feeds fatigue risk.

## 5. Observation Reliability Engine

Inputs: face confidence, landmark validity, per-eye visibility, blur (Laplacian variance), brightness/contrast, glare fraction, head yaw magnitude, face size, FPS stability. Output 0–1 + state {EXCELLENT, GOOD, LIMITED, POOR, UNAVAILABLE} + human-readable reasons. POOR/UNAVAILABLE gates all fatigue classification to "Unknown".

## 6. Temporal prediction

Windows 5/15/30/60/120 s over: EAR mean/median/slope/variance, blink rate, blink-duration trend, long-blink count, PERCLOS + trend, yawn count, head-pitch trend, nod count, gaze-away %, face-visibility %, reliability trend, baseline deviation, journey duration, time since last alert, alert-response score. MVP model: interpretable weighted logistic-style rule model (`forecasting.py`), designed so a scikit-learn LogisticRegression/RandomForest/XGBoost can be dropped in once labelled data exists (Phase 10). Output: risk %, trend, horizon estimate, contributing signals.

## 7. Driver Readiness Score

0–100; bands 81–100 Ready / 61–80 Slightly reduced / 41–60 Reduced / 21–40 Unsafe / 0–20 Critical / Unknown when unreliable. Combines fatigue risk, distraction, gaze, head pose, journey duration, reliability, alert-response history. EWMA smoothing + hysteresis + rate limiting: max change per second bounded.

## 8. Road perception

`ObjectDetector` interface → implementations: `MockDetector` (scenario JSON playback — default, zero deps), `YoloDetector` (ultralytics, optional). IoU-based tracker assigns IDs, direction, relative motion. Hazard levels per class + geometry (lane occupancy, size growth as proxy for approach). Monocular distance labelled "estimated"; TTC only when relative speed is estimable. Indian classes per §34.

## 9. Road Complexity Index

0–100 = weighted, configurable sum (configs/complexity_weights.json) of: vehicle count, two-wheeler density, pedestrian density, animals, congestion, road width/lane quality (map), wrong-side movement, damage, construction, school/market zone, visibility, weather, speed vs conditions. Always returns top contributing reasons.

## 10. External providers

Common `Provider` shape: `get() -> ContextReading{data, source, timestamp, freshness_s, confidence, available}`. Implementations: OSM offline extract/mock, optional Google/Mapbox/HERE (keys via .env), mock+API weather, CCTV file/RTSP(authorized) analyzer, phone-sensor packet validator. Missing provider ⇒ `available=False`, fusion degrades gracefully and reports it.

## 11. Digital twin & counterfactual simulation

MVP twin: `world_state.py` snapshot (ego speed, hazards with distance/velocity, road attrs, weather, driver readiness) built from scenario files (`digital_twin/scenarios/*.json`, ≥5 Indian scenarios) or live fusion. `sumo_runner.py` activates when SUMO_HOME present (OSM→netconvert→flows); CARLA deferred.

Actions: continue, gradual_decel, strong_decel, maintain_lane, lane_change_left/right, delay_lane_change, increase_gap, safe_stop, reroute, warn_driver, recommend_break. Physics: reaction distance (readiness-scaled reaction time) + braking distance v²/2μg (weather-scaled μ), TTC, gap acceptance. Risk per action = base collision/near-miss/rear/pedestrian/two-wheeler/road-departure components × fatigue and weather multipliers, + delay and workload costs. Ranked list with assumptions + confidence; all outputs labelled "simulation estimate". Monte Carlo perturbation (N samples over speed/distance/reaction noise) gives spread → confidence.

## 12. Journey Safety Score & horizons

JSS 0–100 fused from readiness, fatigue trend, reliability, complexity, hazards, weather, speed, route risk, simulation results — with reasons, confidence (separate from risk), and per-horizon summaries: immediate 0–3 s, short 3–10 s, near 10–30 s, route-level.

## 13. Alerts

Levels 0–4 (§27), localized en/hi/mr (`alerts/languages.py`), non-blocking audio (pygame if present else console/dashboard), cooldown + duplicate suppression + escalation + acknowledgement + response tracking. Effectiveness learning per §28 stored in DB.

## 14. Storage

SQLite (`storage/db.py`), WAL mode. Tables: session, driver_baseline, driver_feature_window, driver_state_event, road_detection, hazard_event, context_reading, simulation_run, simulated_action, journey_risk, alert, alert_response, user_report, audit_log. Raw video never stored unless `RECORD_VIDEO=true` + consent flag. Export JSON/CSV session report; PDF optional.

## 15. Dashboard & API

Zero-dependency stdlib HTTP server (`dashboard/server.py`) serving a single-page dashboard polling `/api/state` (JSON snapshot) and `/api/events`. Panels: driver, road, context/map, simulation, system — every value labelled Measured/Estimated/Simulated/Experimental/Unavailable/Unreliable.

## 16. Testing strategy

Pure-logic modules (EAR/MAR math, blink/PERCLOS, reliability, signature, readiness, complexity, physics, ranking, fusion, state machine) tested with pytest + synthetic data — no camera needed. Integration: demo mode replays `test.mp4`/scenario JSONs headless. Subject-independent evaluation protocol (GroupKFold / LOSO) documented for Phase 10; never split one participant's frames across train/test.

## 17. Experiments (Phase 10, need data)

E1 universal vs personalized threshold · E2 single-frame vs temporal · E3 camera vs camera+phone · E4 fixed vs adaptive alert · E5 driver-only vs driver+road fusion · E6 camera vs camera+map · E7 detection vs multi-action simulation · E8 generic vs Indian scenarios · E9 reliability gating on/off · E10 offline vs live-traffic mode. Metrics per §35 (never accuracy alone; false alerts/hour, missed events, early-warning time, mAP, calibration, latency, FPS).

## 18. Roadmap & definitions of done

| Phase | Scope | Done when |
|---|---|---|
| 1 | Audit, stabilize, repo, config, logging, legacy preserved | Audit docs exist; legacy pipeline runs from `main.py --legacy`; tests pass |
| 2 | Personalized monitoring + reliability | Signature/readiness/reliability implemented; unit tests pass; missing eyes ≠ closed |
| 3 | Temporal prediction + adaptive alerts | Windows/trends/forecast + alert levels + response tracking; tests pass |
| 4 | Road perception + complexity | Detector abstraction + tracker + RCI with reasons; mock scenario demo runs |
| 5 | Maps/weather/sensors | Providers with freshness + graceful absence; tests pass |
| 6 | CCTV | File-based analyzer + authorized-stream config | 
| 7 | Digital twin | ≥5 Indian scenario files load into world state; SUMO hook guarded |
| 8 | Counterfactual simulator | ≥3 actions/scenario ranked with physics + explanation + confidence; tests pass |
| 9 | Dashboard + storage + reports | Live dashboard, SQLite persistence, JSON/CSV session report |
| 10 | Research evaluation | Blocked on data collection (see PROJECT_INPUT_REQUIREMENTS §6) |
