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

1. IDD Detection (22.8 GB): real instance boxes, fine classes incl. autorickshaw,
   animal, rider → new converter + retrain at imgsz 640.
2. RDD2022 / custom photos for pothole and speed_breaker classes.
3. Report per-class precision/recall and false-hazards-per-km, per docs/DESIGN.md §17.
