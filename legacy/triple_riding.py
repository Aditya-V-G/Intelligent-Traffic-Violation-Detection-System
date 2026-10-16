import cv2
import os
from ultralytics import YOLO
from datetime import datetime
import dbconnection

# =========================
# CONFIGURATION
# =========================
MODEL_PATH = "yolov8n.pt"   # COCO pretrained

PERSON_CLASS_ID = 0
MOTORCYCLE_CLASS_ID = 3

CONF_THRESHOLD = 0.35
FRAME_SKIP = 2

REQUIRED_CONSECUTIVE_FRAMES = 1
TEMPORAL_WINDOW = 3
COOLDOWN_FRAMES = 10


# =========================
# RIDER VALIDATION LOGIC
# =========================
def is_valid_rider(person, bike):
    px1, py1, px2, py2 = person
    bx1, by1, bx2, by2 = bike

    # person center
    pcx = (px1 + px2) // 2
    pcy = (py1 + py2) // 2

    person_h = py2 - py1
    bike_h = by2 - by1

    # 1️⃣ center must lie inside bike bbox
    if not (bx1 < pcx < bx2 and by1 < pcy < by2):
        return False

    # 2️⃣ person must be seated (upper 75% of bike)
    if pcy > by1 + 0.75 * bike_h:
        return False

    # 3️⃣ ignore tiny / duplicate person boxes
    if person_h < 0.25 * bike_h:
        return False

    return True


# =========================
# MAIN FUNCTION
# =========================
def detect_triple_riding(video_path, out_folder="static/uploads/detections"):

    print("\n🚨 Triple Riding Detection Started (YOLOv8 COCO – Rider Logic)")

    os.makedirs(out_folder, exist_ok=True)

    model = YOLO(MODEL_PATH)
    print("✔ Model loaded | Classes:", model.names)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video:", video_path)
        return

    base = os.path.basename(video_path)
    frame_no = 0

    hit_frames = []
    last_saved_frame = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        if frame_no % FRAME_SKIP != 0:
            continue

        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            continue

        persons = []
        motorcycles = []

        # -------------------------
        # SEPARATE DETECTIONS
        # -------------------------
        for box in boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == PERSON_CLASS_ID:
                persons.append((x1, y1, x2, y2))

            elif cls == MOTORCYCLE_CLASS_ID:
                motorcycles.append((x1, y1, x2, y2))

        # -------------------------
        # CHECK EACH MOTORCYCLE
        # -------------------------
        for bike in motorcycles:
            bx1, by1, bx2, by2 = bike

            rider_count = 0
            for person in persons:
                if is_valid_rider(person, bike):
                    rider_count += 1

            if rider_count < 3:
                continue

            # -------------------------
            # TEMPORAL CONFIRMATION
            # -------------------------
            hit_frames.append(frame_no)
            hit_frames = [
                f for f in hit_frames if frame_no - f <= TEMPORAL_WINDOW
            ]

            if len(hit_frames) < REQUIRED_CONSECUTIVE_FRAMES:
                continue

            if frame_no - last_saved_frame < COOLDOWN_FRAMES:
                continue

            last_saved_frame = frame_no
            hit_frames.clear()

            # -------------------------
            # SAVE VIOLATION IMAGE
            # -------------------------
            crop = frame[by1:by2, bx1:bx2]
            if crop.size == 0:
                continue

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fname = f"triple_{ts}.jpg"
            save_path = os.path.join(out_folder, fname)
            cv2.imwrite(save_path, crop)

            # -------------------------
            # DATABASE INSERT
            # -------------------------
            try:
                dbconnection.insert_detection(
                    number_plate="UNKNOWN",
                    video_file=base,
                    module="triple_riding",
                    image_filename=fname,
                    extra_info=f"riders={rider_count}"
                )
            except Exception as e:
                print("❌ DB Error:", e)

            print(
                f"🚨 TRIPLE RIDING CONFIRMED | "
                f"Frame {frame_no} | Riders={rider_count}"
            )

    cap.release()
    print("✅ Triple Riding Detection Completed\n")


# =========================
# SAFE IMPORT WRAPPER
# =========================
def start_triple_riding(video_path, out_folder="static/uploads/detections"):
    detect_triple_riding(video_path, out_folder)
