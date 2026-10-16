# wrong_lane.py

import cv2
import os
from ultralytics import YOLO
from datetime import datetime
import dbconnection


MODEL_PATH = r"trainedmodels/wrong_lane_yolov8.pt"

WRONG_CLASS_ID = 1 
WRONG_LANE_CONF_THRESHOLD = 0.35

REQUIRED_CONSECUTIVE_FRAMES = 3
COOLDOWN_FRAMES = 25


def detect_wrong_lane(video_path, out_folder="static/uploads/detections"):

    print("\n↪️ Wrong Lane Detection (YOLOv8 – Presence + Temporal Logic)")
    os.makedirs(out_folder, exist_ok=True)

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    base = os.path.basename(video_path)

    if not cap.isOpened():
        print("❌ Cannot open video")
        return

    frame_no = 0
    wrong_lane_counter = 0
    last_saved_frame = 0

    print("📹 Processing:", base)

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1

        # skip some frames
        if frame_no % 3 != 0:
            continue

        results = model(frame, conf=0.25, verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            wrong_lane_counter = 0
            continue

        best_wrong_conf = 0.0
        best_wrong_box = None

        # -------------------------------
        # FIND WRONG LANE ONLY
        # -------------------------------
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == WRONG_CLASS_ID and conf > best_wrong_conf:
                best_wrong_conf = conf
                best_wrong_box = box

        # -------------------------------
        # FRAME-LEVEL CHECK
        # -------------------------------
        if best_wrong_box is not None and best_wrong_conf >= WRONG_LANE_CONF_THRESHOLD:
            wrong_lane_counter += 1
        else:
            wrong_lane_counter = 0

        # -------------------------------
        # TEMPORAL CONFIRMATION
        # -------------------------------
        if wrong_lane_counter < REQUIRED_CONSECUTIVE_FRAMES:
            continue

        if frame_no - last_saved_frame < COOLDOWN_FRAMES:
            continue

        last_saved_frame = frame_no
        wrong_lane_counter = 0

        # -------------------------------
        # SAVE EVIDENCE
        # -------------------------------
        x1, y1, x2, y2 = map(int, best_wrong_box.xyxy[0])
        crop = frame[y1:y2, x1:x2]

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fname = f"wrong_lane_{ts}.jpg"
        cv2.imwrite(os.path.join(out_folder, fname), crop)

        try:
            dbconnection.insert_detection(
                number_plate="UNKNOWN",
                video_file=base,
                module="wrong_lane",
                image_filename=fname,
                extra_info=f"wrong_lane_conf={best_wrong_conf:.2f}"
            )
        except Exception as e:
            print("❌ DB Error:", e)

        print(
            f"🚨 WRONG LANE CONFIRMED | "
            f"conf={best_wrong_conf:.2f} | "
            f"frames={REQUIRED_CONSECUTIVE_FRAMES}"
        )

    cap.release()
    print("✔ Wrong Lane Detection Completed\n")
