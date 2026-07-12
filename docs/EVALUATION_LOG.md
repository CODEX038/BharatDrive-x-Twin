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
