# phone_detection.py
import cv2, os
from ultralytics import YOLO
from datetime import datetime
import dbconnection

MODEL_PATH = "yolov8s.pt"

PHONE_CLASS = 67
PERSON_CLASS = 0

FRAME_SKIP = 2
COOLDOWN_FRAMES = 30   


def detect_phone_usage(video_path, out_folder="static/uploads/detections"):

    print("\n📱 Phone Usage Detection Started")
    print("Model =", MODEL_PATH)

    os.makedirs(out_folder, exist_ok=True)

    try:
        model = YOLO(MODEL_PATH)
        print("✔ Model loaded successfully")
    except Exception as e:
        print("❌ Model load failed:", e)
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open:", video_path)
        return

    base = os.path.basename(video_path)
    frame_no = 0
    last_saved_frame = 0   

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        if frame_no % FRAME_SKIP != 0:
            continue

        results = model(frame, conf=0.40, verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            continue

        persons = []
        phones = []

        # -------------------------
        # COLLECT PERSONS & PHONES
        # -------------------------
        for box in boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == PERSON_CLASS:
                persons.append((x1, y1, x2, y2))

            elif cls == PHONE_CLASS:
                phones.append((x1, y1, x2, y2))

        # -------------------------
        # MATCH PHONE WITH PERSON
        # -------------------------
        for px1, py1, px2, py2 in persons:

            if frame_no - last_saved_frame < COOLDOWN_FRAMES:
                break

            person_upper = (
                px1,
                py1,
                px2,
                py1 + int((py2 - py1) * 0.55)
            )

            for hx1, hy1, hx2, hy2 in phones:

                phone_center_x = (hx1 + hx2) // 2
                phone_center_y = (hy1 + hy2) // 2

                inside = (
                    person_upper[0] <= phone_center_x <= person_upper[2] and
                    person_upper[1] <= phone_center_y <= person_upper[3]
                )

                if inside:
                    print(f"📱 Phone Usage Violation Detected (Frame {frame_no})")

                    crop = frame[py1:py2, px1:px2]

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    fname = f"phone_usage_{ts}.jpg"
                    cv2.imwrite(os.path.join(out_folder, fname), crop)

                    try:
                        dbconnection.insert_detection(
                            number_plate="UNKNOWN",
                            video_file=base,
                            module="phone_usage",
                            image_filename=fname,
                            extra_info="phone_detected"
                        )
                    except Exception as e:
                        print("❌ DB insert failed:", e)

                    last_saved_frame = frame_no   
                    break

    cap.release()
    print("✔ Phone Usage Detection Completed\n")
