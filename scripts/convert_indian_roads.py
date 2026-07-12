"""Build the unified Indian-roads YOLO dataset from IDD Detection + RDD2022 India.

Sources (already extracted under datasets/raw/):
  idd_detection/IDD_Detection/   VOC XMLs + official train/val lists (traffic classes)
  rdd2022_india/India/train/     VOC XMLs (road damage; D40 = pothole)

Output: datasets/processed/indian_roads_yolo/{images,labels}/{train,val} + data.yaml

Design choices:
- 10 project classes (below); IDD 'rider' skipped (the motorcycle box already
  covers the hazard; keeping both double-counts), traffic signs skipped.
- RDD cracks (D00/D10/D20...) skipped — only D40 potholes are driving hazards
  the perception module acts on. RDD has no official val split, so a deterministic
  10% of its images go to val.
- Images are resized to --max-side (default 960) to keep the Colab zip small
  while comfortably supporting 640px training.
- IDD subset is capped (--idd-train/--idd-val) and prioritizes frontFar/frontNear
  cameras (this project uses a front-facing camera).

Run on your machine (takes a few minutes):
    python scripts/convert_indian_roads.py
Then zip for Colab:
    Compress-Archive datasets/processed/indian_roads_yolo indian_roads_yolo.zip

Licences: IDD and RDD2022 are research datasets — keep raw/processed data and
trained weights out of the public repo.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

log = logging.getLogger("convert_indian_roads")
ROOT = Path(__file__).resolve().parent.parent
IDD = ROOT / "datasets" / "raw" / "idd_detection" / "IDD_Detection"
RDD = ROOT / "datasets" / "raw" / "rdd2022_india" / "India" / "train"
OUT = ROOT / "datasets" / "processed" / "indian_roads_yolo"

NAMES = ["pedestrian", "car", "truck", "bus", "motorcycle", "bicycle",
         "auto_rickshaw", "cattle", "traffic_light", "pothole"]
IDX = {n: i for i, n in enumerate(NAMES)}

IDD_MAP = {"person": "pedestrian", "car": "car", "truck": "truck", "bus": "bus",
           "motorcycle": "motorcycle", "bicycle": "bicycle",
           "autorickshaw": "auto_rickshaw", "animal": "cattle",
           "traffic light": "traffic_light"}
RDD_MAP = {"D40": "pothole"}


def parse_voc(xml_path: Path, class_map: dict) -> tuple[int, int, list]:
    """Returns (width, height, [(class_id, xmin, ymin, xmax, ymax), ...])."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(float(size.findtext("width", "0")))
    h = int(float(size.findtext("height", "0")))
    boxes = []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip()
        cls = class_map.get(name)
        if cls is None:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        x1 = float(bb.findtext("xmin", "0")); y1 = float(bb.findtext("ymin", "0"))
        x2 = float(bb.findtext("xmax", "0")); y2 = float(bb.findtext("ymax", "0"))
        x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        boxes.append((IDX[cls], x1, y1, x2, y2))
    return w, h, boxes


def write_sample(img_path: Path, boxes: list, w: int, h: int, split: str,
                 out_name: str, max_side: int) -> bool:
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    ih, iw = img.shape[:2]
    if not w or not h:
        w, h = iw, ih
    scale = min(1.0, max_side / max(iw, ih))
    if scale < 1.0:
        img = cv2.resize(img, (int(iw * scale), int(ih * scale)),
                         interpolation=cv2.INTER_AREA)
    lines = []
    for cid, x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2 / w; cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w;     bh = (y2 - y1) / h
        if not (0 < bw <= 1 and 0 < bh <= 1):
            continue
        lines.append(f"{cid} {min(max(cx,0),1):.6f} {min(max(cy,0),1):.6f} {bw:.6f} {bh:.6f}")
    cv2.imwrite(str(OUT / "images" / split / f"{out_name}.jpg"), img,
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    (OUT / "labels" / split / f"{out_name}.txt").write_text("\n".join(lines), encoding="utf-8")
    return True


def convert_idd(split: str, cap: int) -> tuple[int, int]:
    list_file = IDD / f"{split}.txt"
    stems = [s.strip() for s in list_file.read_text().splitlines() if s.strip()]
    # front cameras first — this project uses a front-facing camera
    stems.sort(key=lambda s: (not s.startswith(("frontFar", "frontNear")), s))
    n = b = 0
    for stem in stems:
        if n >= cap:
            break
        xml = IDD / "Annotations" / f"{stem}.xml"
        jpg = IDD / "JPEGImages" / f"{stem}.jpg"
        if not xml.exists() or not jpg.exists():
            continue
        try:
            w, h, boxes = parse_voc(xml, IDD_MAP)
        except ET.ParseError:
            continue
        if not boxes:
            continue
        out_name = "idd_" + stem.replace("/", "_")
        if write_sample(jpg, boxes, w, h, split, out_name, ARGS.max_side):
            n += 1
            b += len(boxes)
            if n % 500 == 0:
                log.info("IDD %s: %d/%d", split, n, cap)
    return n, b


def convert_rdd() -> tuple[int, int, int, int]:
    xmls = sorted((RDD / "annotations" / "xmls").glob("*.xml"))
    nt = bt = nv = bv = 0
    for xml in xmls:
        try:
            w, h, boxes = parse_voc(xml, RDD_MAP)
        except ET.ParseError:
            continue
        if not boxes:
            continue  # most RDD images have no potholes — skip empty ones
        jpg = RDD / "images" / f"{xml.stem}.jpg"
        if not jpg.exists():
            continue
        # deterministic 10% val split by filename hash
        split = "val" if int(hashlib.md5(xml.stem.encode()).hexdigest(), 16) % 10 == 0 else "train"
        if write_sample(jpg, boxes, w, h, split, f"rdd_{xml.stem}", ARGS.max_side):
            if split == "train":
                nt += 1; bt += len(boxes)
            else:
                nv += 1; bv += len(boxes)
    return nt, bt, nv, bv


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for src, name in ((IDD, "IDD Detection"), (RDD, "RDD2022 India")):
        if not src.exists():
            log.error("%s not found at %s", name, src)
            sys.exit(1)
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    it, ib = convert_idd("train", ARGS.idd_train)
    iv, ivb = convert_idd("val", ARGS.idd_val)
    log.info("IDD: train %d imgs/%d boxes, val %d imgs/%d boxes", it, ib, iv, ivb)
    rt, rb, rv, rvb = convert_rdd()
    log.info("RDD potholes: train %d imgs/%d boxes, val %d imgs/%d boxes", rt, rb, rv, rvb)

    (OUT / "data.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)), encoding="utf-8")
    log.info("dataset ready: %s (train %d, val %d images)", OUT, it + rt, iv + rv)
    log.info('zip for Colab:  Compress-Archive -Path "%s" -DestinationPath indian_roads_yolo.zip', OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--idd-train", type=int, default=6000,
                    help="max IDD train images (front cameras prioritized)")
    ap.add_argument("--idd-val", type=int, default=1500)
    ap.add_argument("--max-side", type=int, default=960,
                    help="resize longest image side to this (keeps zip small)")
    ARGS = ap.parse_args()
    main()
