"""Convert IDD Lite (segmentation) to YOLO detection format.

IDD Lite ships semantic label PNGs (level-1 ids), not boxes. This script derives
bounding boxes via connected-component analysis on the two "thing" classes:

    id 2 = living things (pedestrians, riders, animals)  -> class 0 living_thing
    id 3 = vehicles (cars, trucks, buses, 2-wheelers...) -> class 1 vehicle

Output: datasets/processed/idd_lite_yolo/{images,labels}/{train,val} + data.yaml

This is deliberately coarse — its job is to validate the end-to-end training
pipeline cheaply. Fine-grained Indian classes (auto_rickshaw, cattle, ...) come
from IDD Detection later; this converter's structure is reused for it.

Licence: IDD is research-use. Keep raw and processed data out of version control
(datasets/raw and datasets/processed are gitignored).

Usage:
    python scripts/convert_idd_lite.py [--min-area 45] [--limit N]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("convert_idd_lite")
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets" / "raw" / "idd_lite" / "idd20k_lite"
OUT = ROOT / "datasets" / "processed" / "idd_lite_yolo"

CLASS_MAP = {2: 0, 3: 1}          # IDD level-1 id -> YOLO class id
NAMES = ["living_thing", "vehicle"]


def boxes_from_mask(label_img: np.ndarray, idd_id: int, min_area: int):
    """Connected components of one class mask -> list of (x, y, w, h) pixels."""
    mask = (label_img == idd_id).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):  # 0 = background
        x, y, w, h, area = stats[i]
        if area < min_area or w < 4 or h < 4:
            continue
        # skip absurd blobs (merged traffic wall of the whole frame width)
        if w > 0.95 * label_img.shape[1]:
            continue
        out.append((x, y, w, h))
    return out


def convert_split(split: str, min_area: int, limit: int | None) -> tuple[int, int]:
    img_dir = RAW / "leftImg8bit" / split
    lab_dir = RAW / "gtFine" / split
    out_img = OUT / "images" / split
    out_lab = OUT / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    images = sorted(img_dir.glob("*/*_image.jpg"))
    if limit:
        images = images[:limit]
    n_imgs = n_boxes = 0
    for img_path in images:
        stem = img_path.name.replace("_image.jpg", "")
        lab_path = lab_dir / img_path.parent.name / f"{stem}_label.png"
        if not lab_path.exists():
            continue
        label = cv2.imread(str(lab_path), cv2.IMREAD_GRAYSCALE)
        if label is None:
            continue
        H, W = label.shape
        lines = []
        for idd_id, yolo_id in CLASS_MAP.items():
            for (x, y, w, h) in boxes_from_mask(label, idd_id, min_area):
                cx, cy = (x + w / 2) / W, (y + h / 2) / H
                lines.append(f"{yolo_id} {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")
        # unique output name: seq_stem to avoid collisions across sequence dirs
        out_name = f"{img_path.parent.name}_{stem}"
        # copy image (re-encode not needed)
        (out_img / f"{out_name}.jpg").write_bytes(img_path.read_bytes())
        (out_lab / f"{out_name}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_imgs += 1
        n_boxes += len(lines)
    return n_imgs, n_boxes


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-area", type=int, default=45,
                    help="minimum blob area in pixels (320x227 frames)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap images per split (for smoke tests)")
    args = ap.parse_args()

    if not RAW.exists():
        log.error("IDD Lite not found at %s", RAW)
        sys.exit(1)

    total = {}
    for split in ("train", "val"):
        n_imgs, n_boxes = convert_split(split, args.min_area, args.limit)
        total[split] = (n_imgs, n_boxes)
        log.info("%s: %d images, %d boxes (avg %.1f/img)",
                 split, n_imgs, n_boxes, n_boxes / max(1, n_imgs))

    yaml = OUT / "data.yaml"
    yaml.write_text(
        f"path: {OUT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)),
        encoding="utf-8")
    log.info("wrote %s", yaml)
    log.info("Next: python scripts/train_indian_hazards.py --data %s", yaml)


if __name__ == "__main__":
    main()
