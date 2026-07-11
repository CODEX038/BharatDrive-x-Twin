<div align="center">

# 🚗 BharatDrive-X Twin

### An offline-first AI safety co-pilot for Indian roads

*Personalized driver-readiness analysis · Indian road-hazard perception · Digital-twin counterfactual simulation · Explainable risk recommendations*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-28%20passing-brightgreen)
![Dependencies](https://img.shields.io/badge/core%20deps-stdlib%20only-blueviolet)
![API Keys](https://img.shields.io/badge/API%20keys%20required-0-success)
![License](https://img.shields.io/badge/status-research%20prototype-orange)

**No cloud. No API keys. No GPU. Runs on a laptop.**

</div>

---

## 💡 The idea

Most drowsiness detectors do one thing: *if eyes closed for N frames → beep.* One threshold for every human, no context, no foresight.

BharatDrive-X Twin asks a bigger question: **"Given who this driver is right now, and what this specific Indian road is doing, what is the lowest-risk next move — and why?"**

It evolved from a single-notebook EAR/MAR detector into a 10-module platform that:

1. **Learns the driver** — builds a Personal Fatigue Signature (your normal blink rate, eye openness, PERCLOS, head pose) and detects *deviation from you*, not from a textbook average
2. **Predicts, not just detects** — temporal risk forecasting with a trend and an estimated warning horizon ("risk rising, expect warning level in 40–90 s")
3. **Knows what it doesn't know** — a dedicated Observation Reliability Engine; sunglasses at night produce *"Unknown — eyes obscured"*, never a false "awake"
4. **Sees Indian roads** — auto-rickshaws, cattle, potholes, unmarked speed breakers, wrong-side vehicles, waterlogging → an explainable 0–100 Road Complexity Index
5. **Simulates futures** — a digital twin runs Monte-Carlo counterfactuals over 6–9 candidate actions (brake gently? change lanes? reroute? stop?) with real physics: stopping distance, TTC, friction, fatigue-scaled reaction time
6. **Explains everything** — every recommendation ships with reasons, data sources, assumptions, and a confidence level that is kept *separate* from the risk number

---

## 🎬 60-second demo

```bash
git clone https://github.com/CODEX038/BharatDrive-x-Twin.git
cd BharatDrive-x-Twin
pip install -r requirements.txt
python -m app.main --demo --dashboard        # → open http://localhost:8765
```

You'll watch a scripted driver gradually become drowsy while approaching a congested junction at night — fatigue risk climbs, the readiness score falls, the twin re-simulates actions every 5 seconds, and the alert system escalates from caution to critical, in Hindi/Marathi/English.

```bash
python -m app.main --list-scenarios          # 6 bundled Indian road scenarios
python -m app.main --demo --scenario wrong_side_vehicle --dashboard
python -m app.main --live                    # real webcam monitoring
python -m app.main --legacy                  # the original notebook, preserved
pytest tests/                                # 28 tests, no camera needed
```

---

## 🏗️ Architecture

```
 Driver camera ──► Landmark backends ──► Eye/Blink/Yawn/PERCLOS/Head-pose
 (face_recognition │ mediapipe │ synthetic)        │
                                                   ▼
                    Observation Reliability Engine (gates everything)
                                                   │
                    Personal Fatigue Signature ──► Temporal Forecaster ──► Readiness Score
                                                                               │
 Road camera ──► Detector (YOLO │ scenario mock) ──► IoU Tracker ──► Hazards  │
                                                   │                           │
                                    Road Complexity Index (0–100, explainable) │
                                                   │                           │
 OSM · Weather · CCTV · Phone sensors ─────────────┤   (freshness-tracked)     │
                                                   ▼                           ▼
                                        ┌─────────────────────────────────────────┐
                                        │            DATA FUSION                  │
                                        └───────────────────┬─────────────────────┘
                                                            ▼
                       Digital Twin ──► Counterfactual Simulator (Monte-Carlo physics)
                                                            ▼
                          Journey Safety Score ──► Adaptive Multilingual Alerts
                                                            ▼
                              Live Dashboard · SQLite · Session Reports
```

---

## 🔍 Engineering highlights

| | What | Why it matters |
|---|---|---|
| 🧠 | **Personalized thresholds** | Closed-eye threshold = 72% of *your* median EAR, learned from reliable alert-state windows with 3×MAD outlier rejection — never adapts while you're fatigued |
| ⏱️ | **Timestamp-based everything** | No frame-count logic anywhere; behaviour is identical at 10 FPS and 60 FPS |
| 🕶️ | **Reliability gating** | An invisible eye is *unknown*, never *closed*. A missing face is *investigated*, never *recovering* |
| 🛺 | **Indian hazard taxonomy** | 20 classes including auto_rickshaw, cattle, pothole, speed_breaker, wrong_side_vehicle, waterlogging |
| 🎲 | **Seeded Monte-Carlo simulation** | 60 perturbed rollouts per action (speed, distance, reaction noise) → risk *and* uncertainty; fully reproducible |
| 🗣️ | **Multilingual graded alerts** | 5 levels, English/हिंदी/मराठी, cooldown + duplicate suppression + escalation + response-effectiveness learning |
| 📊 | **Zero-dependency dashboard** | Live web UI on Python stdlib alone (`http.server` + JSON polling) — every value labelled Measured / Estimated / Simulated / Unavailable |
| 🔌 | **Graceful degradation** | Every external provider (maps, weather, CCTV, phone sensors) reports availability + freshness; the system runs fully offline |
| 🧪 | **28 deterministic tests** | Blink math, PERCLOS validity, reliability gating, signature adaptation, physics sanity, action ranking, fusion, storage — all synthetic, no camera required |
| 🔒 | **Privacy by design** | Local processing, geometric landmarks only (no identity), zero raw-video storage by default, consent-gated recording, secrets in `.env` |

---

## 🛣️ Bundled Indian road scenarios

`motorcycle_cut_in` · `pothole_blind_turn` · `cattle_on_road` · `wrong_side_vehicle` · `unmarked_speed_breaker` · `drowsy_near_junction`

Each is a JSON world-state + timed detection playback + scripted driver behaviour — add your own by dropping a file into `digital_twin/scenarios/`.

Example simulator output:

```
Recommended (simulated): Gradually reduce speed — estimated risk 24% (confidence Medium)
Highest-risk alternative: Change lane left at 71%
Hazard: motorcycle left, ~14 m (est.), source: front_camera
Driver fatigue risk 68% lengthens assumed reaction time
Weather 'night' reduces assumed friction to 0.65
All values are simulation estimates from the digital twin.
```

---

## 📁 Repository map

```
app/                 entry point, config, CLI modes
driver_monitoring/   landmarks · eye/blink/yawn · PERCLOS · head pose · reliability
                     fatigue signature · forecasting · readiness · state machine
road_perception/     detector abstraction · IoU tracking · Road Complexity Index
digital_twin/        world state · 6 Indian scenarios · guarded SUMO hook
simulation/          physics (TTC, stopping distance) · Monte-Carlo action engine
fusion/              Journey Safety Score + prediction horizons
alerts/              graded multilingual alerts + effectiveness learning
context_providers.py OSM / weather / CCTV / phone-sensor providers (offline-first)
dashboard/           zero-dependency live web dashboard
storage/             SQLite persistence + session reports
legacy/              the original notebook, preserved and runnable
docs/                full audit report · system design · data-collection guide
tests/               28-test pytest suite
```

---

## ⚖️ Honest limits (read before quoting numbers)

This is a **research and driver-assistance prototype**. It observes, estimates, simulates, explains, and warns — it never controls a vehicle, never contacts anyone, and is not medically validated or automotive-certified. Simulation risks are estimates, not real-world probabilities; monocular distances are approximate; the demo driver model is synthetic and the forecaster needs tuning against real, subject-independent data (protocol in `docs/DESIGN.md §16–17`) before any performance claim.

## 🗺️ Roadmap

Pilot-route OSM→SUMO import → YOLO fine-tune on custom Indian hazard classes → subject-independent evaluation (LOSO) on UTA-RLDD/NTHU → smartphone sensor fusion over WebSocket → CARLA visualization.

---

<div align="center">

**Built by [Shreepad Salvi](https://github.com/CODEX038)** — from a 200-line notebook to a tested, explainable, offline-first safety platform.

`docs/AUDIT_REPORT.md` shows exactly where it started. That's the fun part.

</div>
