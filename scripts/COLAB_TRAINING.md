# Training the road detector on Google Colab (free GPU)

The sandbox/laptop is too slow for YOLO training — use Colab's free T4 GPU.
Everything you need is already prepared: `datasets/processed/idd_lite_yolo.zip`.

## Steps (~20 minutes total)

**1.** Go to https://colab.research.google.com → New notebook → Runtime → Change runtime type → **T4 GPU**.

**2.** Upload `idd_lite_yolo.zip` (19 MB) via the Files panel (left sidebar → upload icon), or from Drive.

**3.** Run these cells:

```python
# Cell 1 — setup
!pip -q install ultralytics
!unzip -q idd_lite_yolo.zip
```

```python
# Cell 2 — fix data.yaml path for Colab
import re, pathlib
p = pathlib.Path("idd_lite_yolo/data.yaml")
p.write_text(re.sub(r"path: .*", "path: /content/idd_lite_yolo", p.read_text()))
print(p.read_text())
```

```python
# Cell 3 — train (~15 min on T4)
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="idd_lite_yolo/data.yaml", epochs=50, imgsz=320, batch=64, patience=10)
```

```python
# Cell 4 — evaluate + download weights
metrics = model.val()
print("mAP50:", metrics.box.map50, " mAP50-95:", metrics.box.map)
from google.colab import files
files.download("runs/detect/train/weights/best.pt")
```

**4.** Save the downloaded `best.pt` as `models/indian_hazards.pt` in this repo.
The road-perception module will auto-load it once `ultralytics` is installed locally
(`pip install ultralytics` — CPU inference on yolov8n is real-time-capable).

## What to expect

IDD Lite is coarse (2 classes: living_thing / vehicle, 320×227 images), so treat the
resulting model as a **pipeline validator**, not the final detector. Typical result is
mAP50 in the 0.4–0.6 range — record whatever you get honestly, per class.

## Next dataset: IDD Detection (the real one)

1. Download IDD Detection (22.8 GB) on your machine, extract to `datasets/raw/idd_detection/`
2. Ask Claude for `convert_idd_detection.py` — same output format, but with real
   per-instance boxes and fine classes mapped to project labels
   (autorickshaw→auto_rickshaw, animal→cattle, motorcycle, rider, person→pedestrian, truck, bus, car...)
3. Same Colab flow, `imgsz=640`, `epochs=60` — expect a few hours on T4
4. Potholes/speed breakers still need RDD2022 or your own labelled photos

## Licence

IDD is research-use only. Keep raw data, processed zips and trained weights out of
the public GitHub repo (already gitignored) unless you've confirmed the licence allows it.
