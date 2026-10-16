# accident_detection.py

import os
import cv2
from ultralytics import YOLO
from datetime import datetime
import dbconnection
from plate_reader_yolo import read_plate_from_vehicle


# accident_v3.pt: retrained on traffic-camera crash footage plus normal-traffic frames
# (mAP50 0.95 vs 0.38 for accident_best.pt on the same test images). The original model is
# the fallback when the retrained weights are missing. Override with ACCIDENT_MODEL.
MODEL_PATH = os.getenv("ACCIDENT_MODEL", r"trainedmodels/accident_v3.pt")
FALLBACK_MODEL_PATH = r"trainedmodels/accident_best.pt"

ACCIDENT_CLASS_ID = 0

# A crash is reported when the model is confident for a sustained moment, once per crash.
# Measured with accident_v3 on accident.mp4 + 10 normal clips not used in training:
# 0.7 for 0.8 s -> 4 crashes, 0 false alerts; 0.6 for 1.2 s -> 5 crashes (adds the night crash
# at 44 s), 0 false alerts; 0.6 for 0.8 s -> 2 false alerts. Normal scenes rarely stay above
# 0.6 for more than a second, real crashes do.
BASE_ACCIDENT_CONF = float(os.getenv("ACCIDENT_CONF", 0.6))   # 0.45 for the old model
DOMINANCE_MARGIN = 0.15
MIN_EVENT_SEC = float(os.getenv("ACCIDENT_MIN_SEC", 1.2))     # sustained detection needed
MAX_GAP_FRAMES = 2                                            # analysed frames that may be missed
EVENT_END_SEC = 3.0                                           # quiet time before a new crash counts


def detect_accident(video_path, out_folder="static/uploads/detections"):

    print("\n💥 Accident Detection Started")
    os.makedirs(out_folder, exist_ok=True)

    path = MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_MODEL_PATH
    model = YOLO(path)
    conf_thr = BASE_ACCIDENT_CONF if path == MODEL_PATH else min(BASE_ACCIDENT_CONF, 0.45)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video")
        return

    base = os.path.basename(video_path)
    frame_no = 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    need_frames = MIN_EVENT_SEC * fps          # in video frames
    end_frames = EVENT_END_SEC * fps
    run_start = None                           # first frame of the current confident stretch
    last_hit = -10 ** 9
    reported = False                           # current crash already saved

    print("📹 Processing:", base)

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        if frame_no % 2 != 0:
            continue

        results = model(frame, conf=0.35, verbose=False)
        boxes = results[0].boxes

        best_accident_conf = 0.0
        best_other_conf = 0.0
        best_box = None

        for box in (boxes if boxes is not None else []):
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == ACCIDENT_CLASS_ID:
                if conf > best_accident_conf:
                    best_accident_conf = conf
                    best_box = box
            else:
                best_other_conf = max(best_other_conf, conf)

        is_strong_accident = (
            best_accident_conf >= conf_thr and
            best_accident_conf >= best_other_conf + DOMINANCE_MARGIN
        )

        if is_strong_accident:
            if frame_no - last_hit > 2 * (MAX_GAP_FRAMES + 1):     # stretch broken: start again
                run_start = frame_no
            if frame_no - last_hit > end_frames:                   # long quiet time: new crash
                reported = False
            last_hit = frame_no

        if reported or run_start is None or not is_strong_accident:
            continue
        if frame_no - run_start < need_frames:
            continue

        reported = True                                            # one record per crash

        # -----------------------------
        # SAVE ACCIDENT
        # -----------------------------
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fname = f"accident_{ts}.jpg"
        evidence = frame.copy()
        if best_box is not None:
            bx1, by1, bx2, by2 = map(int, best_box.xyxy[0])
            cv2.rectangle(evidence, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
            cv2.putText(evidence, f"Accident {best_accident_conf:.2f}", (bx1, max(20, by1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(out_folder, fname), evidence)

        plate_found = "UNKNOWN"

        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            roi = frame[
                max(0, y1 - 60):min(frame.shape[0], y2 + 60),
                max(0, x1 - 60):min(frame.shape[1], x2 + 60)
            ]

            try:
                plate = read_plate_from_vehicle(roi)
                if plate:
                    plate_found = plate
            except:
                pass

        try:
            dbconnection.insert_detection(
                number_plate=plate_found,
                video_file=base,
                module="accident",
                image_filename=fname,
                extra_info=f"conf={best_accident_conf:.2f}, time={frame_no / fps:.1f}s"
            )

            if plate_found != "UNKNOWN":
                dbconnection.update_license_plate(plate_found)

        except Exception as e:
            print("❌ DB Error:", e)

        print(
            f"💥 ACCIDENT CONFIRMED | conf={best_accident_conf:.2f} | t={frame_no / fps:.1f}s"
        )

    cap.release()
    print("✔ Accident Detection Completed\n")
