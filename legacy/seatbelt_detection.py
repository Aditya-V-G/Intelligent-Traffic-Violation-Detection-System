# seatbelt_detection.py

import cv2
import os
from ultralytics import YOLO
from datetime import datetime
import dbconnection


MODEL_PATH = r"trainedmodels/seatbelt_yolov8.pt"

NO_SEATBELT_CLASS_ID = 0   
SEATBELT_CLASS_ID = 1     

NO_SEATBELT_CONF_THRESHOLD = 0.35
SEATBELT_SAFE_THRESHOLD = 0.40

REQUIRED_CONSECUTIVE_FRAMES = 3 
COOLDOWN_FRAMES = 30


def detect_seatbelt(video_path, out_folder="static/uploads/detections"):

    print("\n🪢 Seatbelt Detection Started (YOLOv8 + Temporal Logic)")
    os.makedirs(out_folder, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print("❌ Seatbelt model not found:", MODEL_PATH)
        return

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video:", video_path)
        return

    base = os.path.basename(video_path)
    frame_no = 0
    last_saved_frame = 0

    no_seatbelt_counter = 0   

    print("📹 Processing:", base)

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1

        # skip alternate frames
        if frame_no % 2 != 0:
            continue

        results = model(frame, conf=0.35, verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            no_seatbelt_counter = 0
            continue

        best_no_seatbelt_conf = 0.0
        best_seatbelt_conf = 0.0
        best_no_seatbelt_box = None

        # ---------------------------------
        # COLLECT BEST CONFIDENCES
        # ---------------------------------
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == NO_SEATBELT_CLASS_ID and conf > best_no_seatbelt_conf:
                best_no_seatbelt_conf = conf
                best_no_seatbelt_box = box

            elif cls == SEATBELT_CLASS_ID:
                best_seatbelt_conf = max(best_seatbelt_conf, conf)

        # ---------------------------------
        # FRAME-LEVEL VIOLATION CHECK
        # ---------------------------------
        frame_violation = (
            best_no_seatbelt_box is not None
            and best_no_seatbelt_conf >= NO_SEATBELT_CONF_THRESHOLD
            and best_seatbelt_conf < SEATBELT_SAFE_THRESHOLD
        )

        if frame_violation:
            no_seatbelt_counter += 1
        else:
            no_seatbelt_counter = 0

        # ---------------------------------
        # TEMPORAL CONFIRMATION
        # ---------------------------------
        if no_seatbelt_counter < REQUIRED_CONSECUTIVE_FRAMES:
            continue

        # cooldown to avoid duplicate saves
        if frame_no - last_saved_frame < COOLDOWN_FRAMES:
            continue

        last_saved_frame = frame_no
        no_seatbelt_counter = 0

        # ---------------------------------
        # SAVE EVIDENCE
        # ---------------------------------
        x1, y1, x2, y2 = map(int, best_no_seatbelt_box.xyxy[0])
        crop = frame[y1:y2, x1:x2]

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fname = f"seatbelt_violation_{ts}.jpg"
        cv2.imwrite(os.path.join(out_folder, fname), crop)

        try:
            dbconnection.insert_detection(
                number_plate="UNKNOWN",
                video_file=base,
                module="seatbelt_violation",
                image_filename=fname,
                extra_info=(
                    f"no_seatbelt_conf={best_no_seatbelt_conf:.2f}, "
                    f"seatbelt_conf={best_seatbelt_conf:.2f}"
                )
            )
        except Exception as e:
            print("❌ DB Insert Error:", e)

        print(
            f"🚨 SEATBELT VIOLATION CONFIRMED | "
            f"Frames={REQUIRED_CONSECUTIVE_FRAMES} | "
            f"no_seatbelt={best_no_seatbelt_conf:.2f}"
        )

    cap.release()
    print("🪢 Seatbelt Detection Completed\n")
