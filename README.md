# BharatDrive-X Twin

A personalized driver-readiness, Indian road-hazard prediction, digital-twin and counterfactual safety-simulation platform — an incremental, behaviour-preserving upgrade of an existing EAR/MAR drowsiness-detection notebook.

> BharatDrive-X Twin is a context-aware AI safety co-pilot that combines personalized driver-readiness analysis, live road perception, traffic and route context, authorized external observations and a digital twin to simulate multiple possible driving actions and recommend a lower-risk response.

**It observes, detects, estimates, predicts, simulates, compares, explains, recommends and warns. It never controls a vehicle.** All risk figures are simulation estimates, not guaranteed real-world probabilities. Not medically validated, not automotive-certified.

## Quick start (offline, no API keys, no cameras required)

```bash
pip install -r requirements.txt          # only stdlib + optional extras; see file
python -m app.main --demo                # replay demo: synthetic driver + Indian road scenario
python -m app.main --demo --scenario motorcycle_cut_in
python -m app.main --live                # webcam driver monitoring (needs a landmark backend)
python -m app.main --legacy              # original notebook behaviour, preserved
python -m app.main --demo --dashboard    # + http://localhost:8765
pytest tests/                            # unit tests, no camera needed
```

Landmark backends (auto-selected): `face_recognition` (your existing dds01env venv), `mediapipe` (pip install mediapipe), or synthetic (tests/demo). Road detection backends: `ultralytics` YOLO (optional) or scenario-file mock (default).

## Layout

```
app/                 entry point, config, lifecycle
driver_monitoring/   landmarks, eye/blink/yawn, PERCLOS, head pose, reliability,
                     personal fatigue signature, forecasting, readiness, state machine
road_perception/     detector abstraction, tracking, hazards, Road Complexity Index
maps/ weather/ cctv/ sensors/   offline-first context providers (freshness-tracked)
digital_twin/        world state + Indian scenario library + guarded SUMO hook
simulation/          physics-based counterfactual action engine + ranking
fusion/              Journey Safety Score + prediction horizons
alerts/              graded multilingual adaptive alerts + response learning
storage/             SQLite persistence + session reports
dashboard/           zero-dependency live web dashboard
legacy/              original notebook logic, preserved verbatim-in-spirit
docs/                AUDIT_REPORT, DESIGN, DATA_COLLECTION_GUIDE
tests/               pytest suite (synthetic data)
```

## Privacy & safety

Driver video is processed locally and never stored by default (`RECORD_VIDEO=false`). No face recognition/identification — only geometric landmarks. Only authorized CCTV sources (file/RTSP you own). Secrets via `.env` (see `.env.example`); never committed. The system cannot brake, steer, accelerate or contact anyone.

See `docs/AUDIT_REPORT.md` for the audit of the original project and `docs/DESIGN.md` for the full system design.
