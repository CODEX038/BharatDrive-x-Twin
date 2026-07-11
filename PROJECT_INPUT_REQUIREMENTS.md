# PROJECT_INPUT_REQUIREMENTS

Tracks what has been provided and what is still missing. Missing optional items do NOT block development — mocks/fallbacks are used. Status: ✅ provided · 🟡 partial/fallback in use · ❌ missing (blocking noted).

## 1. Existing project assets

| Item | Status | Notes |
|---|---|---|
| Complete existing project folder | ✅ | `dds01env/Scripts/Drowsiness Detection system.ipynb` + assets |
| GitHub repository URL | ❌ optional | No repo exists; new repo initialized here |
| requirements.txt | 🟡 | Reconstructed in this repo from venv inspection |
| Python version | ✅ | 3.10 |
| Operating system | ✅ | Windows |
| Model files | ✅ | dlib 68-point via face_recognition_models (in venv) |
| Alarm audio | ✅ | `alarm.mp3` (copy into `bharatdrive-x-twin/assets/` or set path in config) |
| Demo video / image | ✅ | `test.mp4`, `test1.jpg` |
| Database | ✅ n/a | None existed; SQLite created by this project |
| Known errors / working features | 🟡 | Derived from audit; confirm W-9 (alarm silences at max score) matches your experience |

## 2. Hardware

| Item | Status | Fallback |
|---|---|---|
| Laptop + driver webcam | ✅ assumed | `test.mp4` demo mode |
| Second camera / phone for road | ❌ optional | Prerecorded road video or mock scenario feed |
| Android phone sensors | ❌ optional | Mock sensor packets |
| GPU / Jetson / OBD-II / IR camera | ❌ optional | CPU mode |

## 3. API keys (none required for MVP)

| Key | Status | Fallback |
|---|---|---|
| GOOGLE_MAPS_API_KEY | ❌ optional | OpenStreetMap offline extract / mock |
| WEATHER_API_KEY | ❌ optional | Mock weather provider |
| MAPBOX_TOKEN / HERE_API_KEY | ❌ optional | Not used |
| CCTV credentials | ❌ optional | Prerecorded file mode (`CCTV_SOURCE_TYPE=file`) |

See `.env.example`. Never commit real keys.

## 4. Datasets (needed for Phase 10 evaluation, not for MVP)

| Dataset | Purpose | Status |
|---|---|---|
| UTA-RLDD or NTHU-DDD | Drowsiness evaluation | ❌ — request access; check licence |
| YawDD | Yawn evaluation | ❌ optional |
| India Driving Dataset (IDD) | Indian road perception | ❌ — registration required |
| RDD2022 | Road damage | ❌ optional |
| Custom driver clips (per DATA_COLLECTION_GUIDE.md) | Calibration + testing | ❌ — **blocks Phase 10 driver evaluation only** |
| Custom front-road clips (30–50) | Road perception testing | ❌ — blocks Phase 10 road evaluation only |

## 5. Pilot route (needed for Phase 7 digital twin on real geometry)

| Item | Status |
|---|---|
| 2–5 km route with start/end GPS, junctions, hazards | ❌ — **needed before OSM→SUMO import of a real route.** Until provided, the twin runs on a bundled synthetic route + scenario files. |

## 6. Currently blocking

Nothing blocks Phases 1–9 (all have offline fallbacks). Phase 10 (research evaluation) is blocked by: participant driver clips, road clips, and pilot-route selection.
