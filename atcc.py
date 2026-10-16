import cv2
import os
from ultralytics import YOLO
from datetime import datetime
import dbconnection
from plate_reader_yolo import read_plate_from_vehicle

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.35
FRAME_SKIP = 2
COOLDOWN_FRAMES = 30

# COCO CLASS IDS
BICYCLE = 1
CAR = 2
MOTORCYCLE = 3
BUS = 5
TRUCK = 7

# Lane split (3 lanes horizontally)
LANE_REGIONS = [
    (0.0, 0.33),
    (0.33, 0.66),
    (0.66, 1.0)
]

# ==========================================
# IOU FUNCTION (TRACKING)
# ==========================================
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter / float(areaA + areaB - inter)

# =================================================
# MAIN FUNCTION (EXPECTED BY app.py)
# =================================================
def process_atcc_videos(video_paths, model=None, out_folder="static/uploads/detections"):

    os.makedirs(out_folder, exist_ok=True)

    # Load YOLO internally (ignore passed model safely)
    model = YOLO(MODEL_PATH)
    print("✔ ATCC using YOLOv8 (NO speed output)")

    for video_path in video_paths:

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Cannot open:", video_path)
            continue

        base = os.path.basename(video_path)
        print(f"🚦 ATCC Started → {base}")

        prev_boxes = {}
        next_id = 0
        frame_no = 0
        last_saved_frame = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_no += 1
            if frame_no % FRAME_SKIP != 0:
                continue

            h, w = frame.shape[:2]

            results = model(frame, conf=CONF_THRESHOLD, verbose=False)
            boxes = results[0].boxes

            if boxes is None or len(boxes) == 0:
                continue

            vehicle_counts = {
                "car": 0,
                "bike": 0,
                "bus": 0,
                "truck": 0
            }

            cur_boxes = []

            # -------------------------
            # VEHICLE DETECTION
            # -------------------------
            for box in boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls == CAR:
                    vehicle_counts["car"] += 1
                elif cls in (MOTORCYCLE, BICYCLE):
                    vehicle_counts["bike"] += 1
                elif cls == BUS:
                    vehicle_counts["bus"] += 1
                elif cls == TRUCK:
                    vehicle_counts["truck"] += 1
                else:
                    continue

                cur_boxes.append((x1, y1, x2, y2))

            # -------------------------
            # COOLDOWN CHECK
            # -------------------------
            if frame_no - last_saved_frame < COOLDOWN_FRAMES:
                continue

            last_saved_frame = frame_no

            # -------------------------
            # SIMPLE TRACKING (NO SPEED)
            # -------------------------
            tracked = {}
            for cbox in cur_boxes:
                cx = (cbox[0] + cbox[2]) // 2
                cy = (cbox[1] + cbox[3]) // 2

                best_id = None
                best_iou_val = 0.0

                for tid, (pbox, px, py) in prev_boxes.items():
                    val = iou(cbox, pbox)
                    if val > best_iou_val and val > 0.25:
                        best_iou_val = val
                        best_id = tid

                if best_id is None:
                    tracked[next_id] = (cbox, cx, cy)
                    next_id += 1
                else:
                    tracked[best_id] = (cbox, cx, cy)

            prev_boxes = tracked

            # -------------------------
            # LANE ASSIGNMENT
            # -------------------------
            lane_ids = []
            for (x1, y1, x2, y2) in cur_boxes:
                cx = (x1 + x2) // 2
                for i, (s, e) in enumerate(LANE_REGIONS):
                    if w * s <= cx < w * e:
                        lane_ids.append(i + 1)
                        break

            # -------------------------
            # SAVE SNAPSHOT + DB
            # -------------------------
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fname = f"atcc_{ts}.jpg"
            cv2.imwrite(os.path.join(out_folder, fname), frame)

            roi = frame[int(h * 0.55):h, int(w * 0.2):int(w * 0.8)]
            try:
                plate_text = read_plate_from_vehicle(roi) or "UNKNOWN"
            except:
                plate_text = "UNKNOWN"

            extra_info = (
                f"Cars:{vehicle_counts['car']},"
                f"Bikes:{vehicle_counts['bike']},"
                f"Bus:{vehicle_counts['bus']},"
                f"Trucks:{vehicle_counts['truck']},"
                f"Lanes:{lane_ids}"
            )

            try:
                dbconnection.insert_detection(
                    number_plate=plate_text,
                    video_file=base,
                    module="atcc",
                    image_filename=fname,
                    extra_info=extra_info
                )
            except Exception as e:
                print("❌ DB Error:", e)

            print(f"✔ ATCC saved @ frame {frame_no}")

        cap.release()

    print("✅ ATCC Completed")
