"""Faithful port of the original notebook (`Drowsiness Detection system.ipynb`).

Preserved so the original behaviour remains runnable via `python -m app.main --legacy`.
Original logic is kept, with only crash-level fixes:
  * camera released in a finally block (audit W-16)
  * empty-landmark guard (audit W-10)
Known preserved quirks (documented, intentionally NOT silently changed):
  * fixed universal thresholds EAR<0.25, MAR>0.6 (W-1)
  * MAR computed on bottom_lip landmark set (W-12)
  * missing face decays the score (W-7)
  * score cap 20 stops the alarm (W-9)
The new pipeline in `driver_monitoring/` supersedes all of these.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Optional, Sequence, Tuple

log = logging.getLogger(__name__)

EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.6
EYE_INC, MOUTH_INC, DECAY = 0.6, 0.8, 0.5
THRESHOLD, MAX_SCORE = 3, 20


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(eye: Sequence[Sequence[float]]) -> float:
    a = _dist(eye[1], eye[5])
    b = _dist(eye[2], eye[4])
    c = _dist(eye[0], eye[3])
    return (a + b) / (2.0 * c) if c else 0.0


def mouth_aspect_ratio(mouth: Sequence[Sequence[float]]) -> float:
    a = _dist(mouth[2], mouth[10])
    b = _dist(mouth[4], mouth[8])
    c = _dist(mouth[0], mouth[6])
    return (a + b) / (2.0 * c) if c else 0.0


def process_image(frame) -> Tuple[bool, bool]:
    """Original per-frame flags. Requires face_recognition + cv2 + numpy."""
    import cv2  # local imports: legacy mode only
    import face_recognition
    import numpy as np

    if frame is None:
        raise ValueError("Image is not found or unable to open")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    eye_flag = mouth_flag = False
    for loc in face_recognition.face_locations(rgb):
        marks_list = face_recognition.face_landmarks(rgb, [loc])
        if not marks_list:  # fix W-10
            continue
        marks = marks_list[0]
        left = np.array(marks["left_eye"])
        right = np.array(marks["right_eye"])
        mouth = np.array(marks["bottom_lip"])
        ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0
        if ear < EYE_AR_THRESH:
            eye_flag = True
        if mouth_aspect_ratio(mouth) > MOUTH_AR_THRESH:
            mouth_flag = True
    return eye_flag, mouth_flag


def run_legacy(camera_index: int = 0, alarm_path: Optional[str] = None) -> None:
    """Original webcam loop, with resource cleanup added."""
    import cv2

    try:
        import pygame

        pygame.mixer.init()
        audio = True
    except Exception:  # pragma: no cover
        audio = False
        log.warning("pygame unavailable; alarm will be logged only")

    last_alarm, cooldown = 0.0, 5.0

    def play_alarm() -> None:
        nonlocal last_alarm
        now = time.time()
        if now - last_alarm >= cooldown:
            if audio and alarm_path and not pygame.mixer.music.get_busy():
                pygame.mixer.music.load(alarm_path)
                pygame.mixer.music.play(-1)
            else:
                log.warning("ALARM (drowsy)")
            last_alarm = now

    def stop_alarm() -> None:
        if audio and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()

    cap = cv2.VideoCapture(camera_index)
    count, eye_score, mouth_score = 0, 0.0, 0.0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            image = cv2.resize(image, (800, 500))
            count += 1
            if count % 3 == 0:
                eye_flag, mouth_flag = process_image(image)
                eye_score = eye_score + EYE_INC if eye_flag else max(0, eye_score - DECAY)
                mouth_score = mouth_score + MOUTH_INC if mouth_flag else max(0, mouth_score - DECAY)
            score = min(eye_score * 1.2 + mouth_score, MAX_SCORE)
            if score >= MAX_SCORE:
                stop_alarm()  # preserved quirk W-9
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, f"Score: {score:.1f}", (10, image.shape[0] - 10), font, 0.8, (0, 255, 0), 2)
            if THRESHOLD <= score < MAX_SCORE:
                cv2.putText(image, "Drowsy Detected", (image.shape[1] - 250, 60), font, 1, (0, 0, 255), 2)
                play_alarm()
            elif score < THRESHOLD:
                stop_alarm()
            cv2.imshow("drowsiness detection (legacy)", image)
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                eye_score = mouth_score = 0
                stop_alarm()
            if key == 27:
                break
    finally:  # fix W-16
        cap.release()
        cv2.destroyAllWindows()
        if audio:
            stop_alarm()
