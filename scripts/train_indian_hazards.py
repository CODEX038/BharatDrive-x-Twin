"""Fine-tune a YOLO detector for BharatDrive-X Twin road perception.

Works locally (CPU smoke-run) or on Colab/Kaggle GPU (real training).
Output weights land in models/ and are picked up by road_perception.detection
when named models/indian_hazards.pt.

Local smoke test (verifies the pipeline, not a real model):
    python scripts/train_indian_hazards.py --data datasets/processed/idd_lite_yolo/data.yaml \
        --epochs 3 --imgsz 320 --model yolov8n.pt

Colab (T4 GPU) real run:
    !pip install ultralytics
    # upload/mount the processed dataset, then:
    !python scripts/train_indian_hazards.py --data /content/idd_lite_yolo/data.yaml \
        --epochs 60 --imgsz 640 --batch 32

Note: IDD licence is research-use — check before publishing trained weights.
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

log = logging.getLogger("train")
ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="base weights")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="indian_hazards")
    ap.add_argument("--out", default=str(ROOT / "models" / "indian_hazards.pt"))
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics not installed: pip install ultralytics")
        raise SystemExit(1)

    model = YOLO(args.model)
    results = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                          batch=args.batch, name=args.name, patience=15)
    # copy best weights into models/
    best = Path(results.save_dir) / "weights" / "best.pt"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out)
    log.info("best weights -> %s", out)

    metrics = model.val(data=args.data)
    log.info("mAP50: %.3f  mAP50-95: %.3f",
             metrics.box.map50, metrics.box.map)
    log.info("Report per-class precision/recall in evaluation docs — never accuracy alone.")


if __name__ == "__main__":
    main()
