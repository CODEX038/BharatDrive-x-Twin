# Repository Audit Report — Existing Driver Drowsiness Detection Project

Audit date: 2026-07-12
Audited location: `S:\PROJECTs\dlib-20.0.0\DRIVER DROWSINESS DETECTION SYSTEM\`
Auditor: BharatDrive-X Twin upgrade process (Phase 1)

---

## 1. Project structure

| Item | Finding |
|---|---|
| Programming language | Python (single Jupyter notebook) |
| Python version | 3.10 (from venv `pyvenv.cfg` and compiled `.pyd` files) |
| Operating system | Windows (venv contains `.exe` launchers, PyWin32) |
| Main entry point | `dds01env/Scripts/Drowsiness Detection system.ipynb` — 11 cells, run interactively |
| Frameworks / libraries | `face_recognition` (dlib backend), OpenCV (`cv2`), NumPy, SciPy, Pillow, Matplotlib, pygame (audio) |
| Face/eye models | dlib 68-point landmark model bundled inside `face_recognition_models` package; no separate model files in the project |
| Model-weight files | None owned by the project (all inside site-packages) |
| Alarm files | `dds01env/Scripts/alarm.mp3` |
| Test media | `dds01env/test.mp4`, `dds01env/test1.jpg` |
| Dataset folders | None |
| Training / evaluation scripts | None (fully rule-based, no trained model) |
| Frontend / backend / API / database | None |
| Configuration files | None — all thresholds hard-coded in notebook cells |
| `.env` / secrets | None found; no exposed secrets ✔ |
| Logging | None (one `print` on score reset) |
| Tests | None |
| Docker / deployment | None |
| Documentation | None |
| requirements.txt | None — dependencies only recoverable from the venv |
| Version control | None — no `.git` |

**Structural problems**

1. The entire project lives **inside the virtual environment's `Scripts/` folder**. The venv (918 MB) is the project. Recreating the environment on another machine is impossible without archaeology.
2. The parent folder is the **dlib 20.0.0 source tree**, unrelated to the project — dlib was apparently compiled from source here and the project folder was dropped inside it.
3. `pip install "setuptools<81"` is executed as a notebook cell (cell 2) — an environment mutation inside application code.
4. Notebook cell 1 imports `matplotlib`, `PIL`, `os` that the runtime loop never uses.

## 2. Existing detection logic (exact)

Backend: `face_recognition` → HOG face detector (default; the visualization cell uses `model='cnn'`) → dlib 68-point landmarks.

Per processed frame (`process_image`):

```text
EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2·‖p1−p4‖)   per eye, averaged
MAR = computed on the *bottom_lip* landmark set (see W-12)

eye_flag   = EAR < 0.25          (fixed universal threshold)
mouth_flag = MAR > 0.60          (fixed universal threshold)
```

Runtime loop (cell 9):

```text
Webcam 0 → resize to 800×500 → process every 3rd frame
eye_score   += 0.6 if eye_flag   else −0.5 (floor 0)
mouth_score += 0.8 if mouth_flag else −0.5 (floor 0)
score = 1.2·eye_score + mouth_score, capped at 20
if 3 ≤ score < 20 → draw "Drowsy Detected", play looping alarm (5 s cooldown)
if score < 3      → stop alarm
if score = 20     → stop alarm (!)   ← auto-silences at maximum severity
Any keypress → reset scores and stop alarm; ESC → exit
```

**What already works and is worth keeping:** the EAR formula, the MAR concept, score accumulation with decay (a primitive temporal filter — better than single-frame alarms), alarm cooldown, non-repeating alarm check, user reset.

## 3. Current weaknesses (numbered for traceability)

| # | Weakness | Evidence |
|---|---|---|
| W-1 | One universal threshold (EAR 0.25, MAR 0.6) for every driver | cell 7 |
| W-2 | No personal calibration or baseline | — |
| W-3 | Frame-count-based scoring, not timestamp-based: behaviour changes with FPS | cell 9, `count % 3` |
| W-4 | No PERCLOS, no blink duration/frequency, no head pose, no gaze | — |
| W-5 | No early prediction — reacts only after visible signs | — |
| W-6 | No confidence score, no observation-reliability measurement | — |
| W-7 | **Missing face ⇒ flags stay False ⇒ scores decay ⇒ a driver who slumps out of frame is classified as recovering** | cell 7: loop over `face_locations` never runs |
| W-8 | Eye occlusion (sunglasses, glare, rotation) indistinguishable from open eyes; landmark failure indistinguishable from alertness | — |
| W-9 | Score cap bug: at maximum score (20) the alarm is **stopped** — most severe state is silent | cell 9 `stop_alarm()` on cap |
| W-10 | `face_recognition.face_landmarks` indexed `[0]` per location — crashes if landmarks return empty; no try/except anywhere in the loop | cell 7 |
| W-11 | Multiple faces: flags OR-ed across all faces — a drowsy passenger triggers the driver's alarm | cell 7 loop |
| W-12 | MAR computed on `bottom_lip` (12 points of one lip), not the inner/outer mouth pair — indices 2/10, 4/8, 0/6 do not correspond to standard MAR; speaking/laughing easily crosses 0.6 | cell 6–7 |
| W-13 | HOG detector: poor in low light, with head rotation > ~30°, and with partial occlusion | face_recognition default |
| W-14 | `face_locations` + `face_landmarks` recompute detection twice per frame — ~2× cost | cell 7 |
| W-15 | Alarm file path `"alarm.mp3"` relative to CWD — breaks if launched elsewhere; hard-coded `test1.jpg` path | cells 3, 9 |
| W-16 | Camera released only on clean ESC exit; exception ⇒ camera leak | cell 9 |
| W-17 | No logging, no session record, no metrics | — |
| W-18 | No tests, no dataset, no evaluation, no train/val/test methodology | — |
| W-19 | No config; magic numbers embedded in code | — |
| W-20 | Any keypress silently resets the score — an accidental keystroke defeats the safety function | cell 9 |
| W-21 | No front-road camera, maps, sensors, simulation, dashboard, privacy controls | — |
| W-22 | `pygame.mixer.music.play(-1)` loops forever; combined with W-9 the alarm lifecycle is inconsistent | cell 9 |

## 4. Component decision table

| Component | Decision | Reason |
|---|---|---|
| EAR formula (`eye_aspect_ratio`) | **Keep and extend** | Correct; extend with per-eye visibility, smoothing, personal baseline |
| MAR formula (`mouth_aspect_ratio`) | **Replace** | Wrong landmark set (W-12); reimplement on inner-mouth points |
| `process_image` per-frame flags | **Refactor** | Split into landmark extraction, feature computation, reliability check |
| Score accumulate/decay loop | **Refactor** | Keep the idea; make timestamp-based; fix W-9 cap bug; feed a proper state machine |
| `play_alarm`/`stop_alarm` (pygame) | **Keep and extend** | Cooldown logic is sound; wrap in AlertManager with levels, escalation, languages |
| face_recognition/dlib backend | **Keep as optional backend** | Works in the user's venv; abstract behind an interface so MediaPipe/others can drop in |
| `highlight_facial_points` (CNN model) | **Remove from runtime** | Visualization only; CNN model far too slow for real time |
| `pip install` notebook cell | **Remove** | Belongs in requirements.txt |
| Notebook as entry point | **Replace** | Move to a proper package with `main.py`; keep the notebook archived as legacy reference |
| `alarm.mp3`, `test.mp4`, `test1.jpg` | **Keep** | Reused as alert asset and demo/test fixtures |
| venv-as-project layout | **Replace** | New repo + requirements.txt; venv stays untouched on the user's machine |
| Multiple-face handling | **Broken** → fix | Track the largest/most central face only (W-11) |
| Missing-face handling | **Broken** → fix | Route to `face_not_visible` state, never to score decay (W-7) |
| Error handling / camera cleanup | **Requires testing** → fix | try/finally around capture loop (W-10, W-16) |

## 5. Feature-gap analysis vs. BharatDrive-X Twin targets

| Capability | Current | Target | Gap |
|---|---|---|---|
| Fatigue detection | Fixed-threshold EAR/MAR | Personal Fatigue Signature + temporal prediction | Full build (Phase 2–3) |
| Readiness score | — | 0–100 with smoothing/hysteresis | Full build |
| Observation reliability | — | Separate 0–1 score gating all classifications | Full build |
| Road perception | — | Detector abstraction + Indian hazard classes | Full build (Phase 4) |
| Road Complexity Index | — | Explainable 0–100 | Full build |
| Maps/weather/CCTV/sensors | — | Offline-first providers with optional keys | Full build (Phase 5–6) |
| Digital twin + counterfactuals | — | OSM+SUMO, ≥5 Indian scenarios, action ranking | Full build (Phase 7–8) |
| Journey Safety Score, dashboard, storage, reports | — | Fused score, web dashboard, SQLite, session reports | Full build (Phase 9) |
| Evaluation | — | Subject-independent metrics | Full build (Phase 10, needs data) |

## 6. Safety and privacy posture

No vehicle control anywhere (compliant). No network calls (compliant offline). No face recognition *identification* is performed despite the library name — only landmarks are used (compliant; keep it that way). No raw video is stored (compliant). No consent flow exists yet (gap). No secrets exposed (compliant).

## 7. Verdict

The existing system is a working single-file EAR/MAR demo with a reasonable score-decay heuristic and several correctness bugs (W-7, W-9, W-11, W-12 are the serious ones). Nothing prevents incremental upgrade. All working concepts are preserved in `legacy/legacy_pipeline.py` and the new `driver_monitoring` package supersedes them behind config flags.
