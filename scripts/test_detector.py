"""Quick visual test of the trained detector. Run: python scripts/test_detector.py [image_path]"""
import sys, glob, cv2
sys.path.insert(0, ".")
from road_perception.detection import YoloDetector

img_path = sys.argv[1] if len(sys.argv) > 1 else sorted(
    glob.glob("datasets/raw/idd_lite/idd20k_lite/leftImg8bit/val/*/*_image.jpg"))[0]
img = cv2.imread(img_path)
dets = YoloDetector().detect(img, 0.0)
print(f"{img_path}: {len(dets)} detections")
for d in dets:
    print(" ", d.to_dict())
    x, y, w, h = d.box
    H, W = img.shape[:2]
    cv2.rectangle(img, (int(x*W), int(y*H)), (int((x+w)*W), int((y+h)*H)), (0, 255, 0), 2)
    cv2.putText(img, d.cls, (int(x*W), int(y*H)-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
cv2.imshow("detections (any key to close)", img)
cv2.waitKey(0)