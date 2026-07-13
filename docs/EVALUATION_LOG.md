# Evaluation Log

Honest record of every trained model and measured result. Never report accuracy alone;
never claim numbers that weren't measured.

## 2026-07-12 — Road detector pilot: YOLOv8n on IDD Lite (coarse)

| Field | Value |
|---|---|
| Model | YOLOv8n fine-tune → `models/indian_hazards.pt` (6.2 MB) |
| Data | IDD Lite, converted via `scripts/convert_idd_lite.py` (boxes derived from segmentation by connected components) |
| Classes | 2 — living_thing, vehicle (coarse) |
| Train / Val | 1,403 / 204 images, 320×227, 10,479 boxes total |
| Training | Colab T4, epochs≤50 (patience 10), imgsz 320, batch 64 |
| **mAP50** | **0.588** |
| **mAP50-95** | **0.312** |
| Split rule | IDD official train/val (no leakage) |

### Interpretation and limitations

- Purpose was **pipeline validation**, not a production detector — labels are
  connected-component approximations, so touching vehicles merge into one box,
  which caps achievable mAP. 0.588 mAP50 under these conditions confirms the
  converter, training loop and weight hand-off all work.
- Only 2 coarse classes; no auto_rickshaw / cattle / pothole / speed_breaker yet.
- 320px training resolution; small/far objects under-detected.
- Runtime mapping is deliberately conservative: living_thing→pedestrian,
  vehicle→car (`road_perception/detection.py::_CUSTOM_MAP`).

### Next

1. ~~IDD Detection: real instance boxes, fine classes~~ → done, see next entry.
2. RDD2022 / custom photos for pothole and speed_breaker classes → potholes done; speed breakers still need custom data.
3. Report per-class precision/recall and false-hazards-per-km, per docs/DESIGN.md §17.

---

## 2026-07-12 — Road detector v2: YOLOv8n on IDD Detection + RDD2022 India

| Field | Value |
|---|---|
| Model | YOLOv8n fine-tune → `models/indian_hazards.pt` |
| Data | `scripts/convert_indian_roads.py`: IDD Detection subset (front cameras prioritized, VOC→YOLO) + RDD2022 India D40 potholes |
| Classes | 10 — pedestrian, car, truck, bus, motorcycle, bicycle, auto_rickshaw, cattle, traffic_light, pothole |
| Train / Val | 7,382 / 1,648 images (val: 4,674 instances), resized max-side 960 |
| Training | Colab T4, 54/60 epochs (runtime disconnect; best.pt from Drive checkpoints), imgsz 640, batch 32 |
| Validation | Local CPU (i7-14650HX), 24.6 ms/image inference |
| **Overall** | **P 0.672 · R 0.401 · mAP50 0.442 · mAP50-95 0.248** |

Per-class (P / R / mAP50 / mAP50-95):

| Class | Inst. | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| pedestrian | 909 | 0.660 | 0.381 | 0.433 | 0.216 |
| car | 729 | 0.729 | 0.550 | 0.603 | 0.370 |
| truck | 429 | 0.672 | 0.699 | **0.746** | 0.503 |
| bus | 61 | 0.735 | 0.508 | 0.629 | 0.408 |
| motorcycle | 1302 | 0.681 | 0.659 | 0.666 | 0.326 |
| bicycle | 45 | 0.574 | 0.150 | 0.237 | 0.114 |
| auto_rickshaw | 205 | 0.530 | 0.488 | 0.465 | 0.281 |
| cattle | 711 | 0.584 | 0.174 | 0.208 | 0.107 |
| traffic_light | **3** | 1.0 | 0.0 | 0.0 | 0.0 |
| pothole | 280 | 0.560 | 0.404 | 0.430 | 0.154 |

### Interpretation and limitations

- Vehicle classes (truck/motorcycle/bus/car) are solid; the detector is usable for
  the traffic-hazard portion of the Road Complexity Index.
- **cattle** recall is poor (0.17): dense overlapping herds + small distant animals;
  needs more targeted data or larger model.
- **bicycle** and especially **traffic_light** (3 val instances) have too few
  samples for meaningful numbers — do not quote them.
- Trained 54 of 60 epochs due to Colab disconnect; marginal further gain expected.
- Nano model at 640px trades accuracy for CPU real-time speed (24.6 ms/img) — a
  deliberate choice for laptop deployment.
- speed_breaker, waterlogging, open_manhole, wrong_side_vehicle remain undetected
  classes (no training data yet).

---

## 2026-07-13 — Driver-monitoring alert calibration (live, MediaPipe backend)

Investigated false alerts observed in live runs: an awake driver was tripping
repeated L4 CRITICAL alarms. Root-caused across the alert → forecaster → state
machine → feature-detector chain and fixed the clear defects; measured the
residual behaviour over four live self-test sessions (single subject, laptop
webcam, i7-14650HX, MediaPipe FaceMesh backend, no `face_recognition`).

### Defects found and fixed

| # | Defect | Fix | File |
|---|---|---|---|
| 1 | L4 message claimed "an immediate road hazard" in driver-only mode, where `max_hazard` is always 0 | Split into `critical_fatigue` message; only claim a hazard when one is present | `alerts/languages.py`, `alerts/manager.py::level_for` |
| 2 | `CRITICAL` was an immediate (zero-dwell) transition; a single ~1 s eye closure fired L4 | Added `critical_dwell_s` (0.6 s) confirmation window | `driver_monitoring/state_machine.py`, `app/config.py` |
| 3 | Baseline learning froze whenever `risk ≥ 0.4` — circular, since a bad baseline inflates risk and then locks itself | Gate freezing on objective evidence (microsleep / PERCLOS ≥ 25%) | `driver_monitoring/pipeline.py` |
| 4 | Robust z-score used `MAD or 1e-6`; a hyper-stable 30 s baseline collapsed MAD → "baseline deviation" pinned at 100% every frame | Floor MAD at 8% of the feature median | `driver_monitoring/fatigue_signature.py::compare` |
| 5 | Geometric head-pitch jitter fabricated "head-nod" events while the driver sat still | Smooth pitch (EWMA) + require the down phase to persist `min_down_s` (0.35 s) | `driver_monitoring/head_pose.py` |

(Note: a prior session also confirmed the calibration-timing guard — `calibrated`
now requires ≥ 30 s elapsed **and** ≥ 40 samples, not samples alone, so the
baseline is no longer captured during ~3 s of camera warmup.)

### Measured effect (live self-test sessions)

| Session | Fixes active | Avg risk | Max risk | DROWSY+CRITICAL frames | L4 alerts | Top inflators |
|---|---|---|---|---|---|---|
| `3d5246e9` | alert-logic only | 0.461 | 0.927 | ~51% | 1 | (not logged) |
| `d60b44b0` | + diagnostics | 0.189 | 0.846 | 0% | 0 | baseline-dev 100%, blink 1203 ms |
| `2e68be29` | + MAD floor | 0.424 | 0.878 | ~24% | 1 | long blinks, head-nods, baseline-dev (now graded) |
| `75e9bac6` | + nod fix | 0.340 | 0.647 | 0% | 0 | baseline drift, 1 yawn/5 min, EAR trending down |

Alert-effectiveness signals moved in the right direction: L4 storms (dozens per
minute) collapsed to zero under good conditions, max risk fell 0.927 → 0.647, and
head-nod / long-blink artifacts were largely eliminated.

### Known limitation — residual L2 "fatigue increasing" on an awake driver

Under good lighting the system no longer produces false CRITICAL/DROWSY alerts,
but it still emits intermittent **L2 CAUTION** ("fatigue indicators increasing")
on a fully awake driver. This is **not** a detector bug — it is inherent to the
current fatigue model:

- The risk model is an interpretable **hand-weighted heuristic** (`forecasting.py`)
  whose weights were set for the **dlib** EAR scale; it runs here on **MediaPipe**,
  whose EAR distribution differs.
- The personal baseline is captured over a **30 s window** and then held fixed.
  Normal relaxation over the following minute drifts EAR downward, which the model
  scores against the frozen baseline as rising fatigue ("baseline deviation",
  "eye openness trending down").
- Behaviour is **sensitive to input quality** — sessions with more movement or
  face-tracking dropouts (flagged as "camera unreliable") score noticeably higher
  average risk (0.42) than still, well-lit sessions (0.19).

Practical guidance: calibrate under good lighting, facing the camera with eyes
open, and wait past the "camera unreliable" warnings before trusting the output.

### Next (not yet done)

1. Faster / rolling baseline adaptation so natural EAR drift is tracked, not read as fatigue.
2. Re-fit forecaster weights (or a labelled ML model, DESIGN §Phase 10) on the MediaPipe EAR scale.
3. Yawn window / MAR-threshold review (one yawn currently contributes for a full 5 min).
4. Report false-alerts-per-hour on a labelled awake-vs-drowsy set, per DESIGN §28.
