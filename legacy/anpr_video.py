import os
import cv2
import re
from datetime import datetime
import dbconnection
from legacy.plate_reader_yolo import read_plate_from_vehicle


def normalize_plate(text):
    if not text:
        return None

    text = text.upper()

    if len(text) >= 3 and text[2] == "O":
        text = text[:2] + "0" + text[3:]

    if len(text) >= 5 and text[4] == "0":
        text = text[:4] + "O" + text[5:]

    if len(text) >= 7 and text[6] == "O":
        text = text[:6] + "0" + text[7:]

    return text


def is_valid_plate(text):
    if not text:
        return False
    pattern = r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$"
    return re.match(pattern, text) is not None


def start_anpr_ultra(video_paths, out_folder="static/uploads/detections"):

    os.makedirs(out_folder, exist_ok=True)
    detected_set = set()

    for video_path in video_paths:

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Cannot open:", video_path)
            continue

        base = os.path.basename(video_path)
        frame_no = 0

        print(f"🔵 Starting ANPR → {base}")

        while True:

            ret, frame = cap.read()
            if not ret:
                break

            frame_no += 1

            if frame_no % 2 != 0:
                continue

            plate = read_plate_from_vehicle(frame)
            if not plate:
                continue

            plate = normalize_plate(plate)

            if not is_valid_plate(plate):
                print("❌ Invalid plate skipped:", plate)
                continue

            if plate in detected_set:
                print("⏩ Duplicate skipped:", plate)
                continue

            detected_set.add(plate)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fname = f"anpr_{plate}_{ts}.jpg"
            filepath = os.path.join(out_folder, fname)

            cv2.imwrite(filepath, frame)

            try:
                dbconnection.insert_detection(
                    number_plate=plate,
                    video_file=base,
                    module="anpr",
                    image_filename=fname,
                    extra_info=""
                )

                dbconnection.update_license_plate(plate)

            except Exception as e:
                print("❌ DB Error:", e)

            print("✔ ANPR Detected:", plate)

        cap.release()

    print("✅ ANPR Completed")


def start_anpr(video_paths, out_folder="static/uploads/detections"):
    return start_anpr_ultra(video_paths, out_folder)
