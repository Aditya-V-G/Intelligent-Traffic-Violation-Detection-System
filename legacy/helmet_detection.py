import os
import cv2
from ultralytics import YOLO
from datetime import datetime
import dbconnection
from legacy.plate_reader_yolo import read_plate_from_vehicle


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MODEL_PATH = r"C:\capstone_project\ANPR-and-ATCC-for-Smart-Traffic-Management\trainedmodels\helmetdetection.pt"

HELMET_CLASS_ID = 0
NO_HELMET_CLASS_ID = 1

YOLO_CONF = 0.4
NO_HELMET_CONF_THRESHOLD = 0.65

REQUIRED_FRAMES = 3        
HELMET_LOCK_FRAMES = 50   


# --------------------------------------------------
# UTILS
# --------------------------------------------------

def enlarge_roi(frame, box, pad=180):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    return frame[
        max(0, y1 - pad):min(h, y2 + pad),
        max(0, x1 - pad):min(w, x2 + pad)
    ]


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main_fun(video_path, out_folder="static/uploads/detections"):

    os.makedirs(out_folder, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print("❌ Helmet model not found")
        return

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Cannot open video")
        return

    base = os.path.basename(video_path)
    frame_no = 0

    # temporal state
    violation_counter = 0
    last_violation_frame = 0
    helmet_lock = 0

    print(f"🟢 Helmet detection started → {base}")

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        if frame_no % 2 != 0:
            continue

        results = model(frame, conf=YOLO_CONF, verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            violation_counter = 0
            continue

        helmet_conf = 0.0
        no_helmet_conf = 0.0
        no_helmet_boxes = []

        # ----------------------------------------------
        # SCAN DETECTIONS
        # ----------------------------------------------
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == HELMET_CLASS_ID:
                helmet_conf = max(helmet_conf, conf)

            elif cls == NO_HELMET_CLASS_ID:
                no_helmet_conf = max(no_helmet_conf, conf)
                no_helmet_boxes.append((x1, y1, x2, y2, conf))

        # ----------------------------------------------
        # HELMET PRESENCE LOCK (FINAL FIX)
        # ----------------------------------------------
        if helmet_conf > 0.45:
            helmet_lock = HELMET_LOCK_FRAMES

        if helmet_lock > 0:
            helmet_lock -= 1
            violation_counter = 0
            continue

        # ----------------------------------------------
        # TEMPORAL VIOLATION LOGIC
        # ----------------------------------------------
        if (
            no_helmet_conf >= NO_HELMET_CONF_THRESHOLD
            and no_helmet_conf > helmet_conf
        ):
            violation_counter += 1
        else:
            violation_counter = 0

        # ----------------------------------------------
        # CONFIRM VIOLATION
        # ----------------------------------------------
        if violation_counter >= REQUIRED_FRAMES:

            # avoid duplicate spam
            if frame_no - last_violation_frame < 30:
                continue

            last_violation_frame = frame_no
            violation_counter = 0

            for (x1, y1, x2, y2, conf) in no_helmet_boxes:

                head_crop = frame[y1:y2, x1:x2]
                roi = enlarge_roi(frame, (x1, y1, x2, y2))

                try:
                    plate = read_plate_from_vehicle(roi) or "UNKNOWN"
                except:
                    plate = "UNKNOWN"

                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                fname = f"helmet_violation_{ts}.jpg"
                cv2.imwrite(os.path.join(out_folder, fname), head_crop)

                dbconnection.insert_detection(
                    number_plate=plate,
                    video_file=base,
                    module="helmet_violation",
                    image_filename=fname,
                    extra_info=f"helmet=no, conf={conf:.2f}"
                )

                if plate != "UNKNOWN":
                    dbconnection.update_license_plate(plate)

                print(f"🚨 CONFIRMED No Helmet Violation | conf={conf:.2f}")

    cap.release()
    print("✅ Helmet detection completed")