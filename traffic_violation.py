import cv2
import os
import math
from ultralytics import YOLO
from datetime import datetime
import dbconnection
from plate_reader_yolo import read_plate_from_vehicle   # NEW PLATE READER


# ============================================================
# MODEL PATHS (replace with your trainedmodels folder)
# ============================================================

VEHICLE_MODEL = r"trainedmodels/vehicledetection.pt"   # {0:'mobil', 1:'motor'}
LANE_MODEL = r"trainedmodels/lane.pt"                  # lane detection model

PIXEL_TO_METER = 0.05   # approx scale


# ============================================================
# IOU TRACKER
# ============================================================

def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0

    areaA = abs((a[2]-a[0])*(a[3]-a[1]))
    areaB = abs((b[2]-b[0])*(b[3]-b[1]))

    return inter / max(1, (areaA + areaB - inter))


# ============================================================
# UTILITY
# ============================================================

def enlarge_crop(frame, box, pad=150):
    x1, y1, x2, y2 = box
    H, W = frame.shape[:2]
    return frame[
        max(0, y1-pad):min(H, y2+pad),
        max(0, x1-pad):min(W, x2+pad)
    ]


# ============================================================
# MAIN WRONG-LANE DETECTOR
# ============================================================

def start_traffic_violation(video_path, out_folder="static/uploads/detections"):

    os.makedirs(out_folder, exist_ok=True)

    print("🚗 Loading models...")
    veh_model = YOLO(VEHICLE_MODEL)
    lane_model = YOLO(LANE_MODEL)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video:", video_path)
        return

    base = os.path.basename(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20

    print(f"🛣 Wrong Lane Detection Started → {base}")

    frame_no = 0
    prev_track = {}
    next_id = 0

    lane_boundary = None   # computed once


    # ============================================================
    # PROCESS VIDEO
    # ============================================================

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        if frame_no % 2 != 0:
            continue   # speed optimization

        H, W = frame.shape[:2]


        # ============================================================
        # 1) DETECT LANES
        # ============================================================

        lane_results = lane_model(frame, conf=0.35, verbose=False)
        lane_boxes = lane_results[0].boxes.data.cpu().numpy()

        lane_x = []
        for x1, y1, x2, y2, conf, cls in lane_boxes:
            cx = int((x1 + x2) // 2)
            lane_x.append(cx)

        if len(lane_x) >= 2:
            lane_x.sort()
            lane_boundary = (lane_x[0] + lane_x[-1]) // 2

        if lane_boundary is None:
            continue  # cannot detect wrong lane yet


        # ============================================================
        # 2) DETECT VEHICLES
        # ============================================================

        veh_results = veh_model(frame, conf=0.45, verbose=False)
        dets = veh_results[0].boxes.data.cpu().numpy()

        tracked = {}


        # ---------------- TRACKING ----------------
        for x1, y1, x2, y2, conf, cls in dets:
            cls = int(cls)
            box = (int(x1), int(y1), int(x2), int(y2))
            cx = (box[0] + box[2]) // 2
            cy = (box[1] + box[3]) // 2

            best_i = 0
            best_id = None

            for tid, (pb, px, py, pcl) in prev_track.items():
                score = iou(box, pb)
                if score > best_i and score > 0.25:
                    best_i = score
                    best_id = tid

            if best_id is None:
                tracked[next_id] = (box, cx, cy, cls)
                next_id += 1
            else:
                tracked[best_id] = (box, cx, cy, cls)


        # ---------------- SPEED ESTIMATION ----------------
        speeds = {}
        for tid, (box, cx, cy, cls) in tracked.items():
            if tid in prev_track:
                _, px, py, _ = prev_track[tid]
                d = math.sqrt((cx - px)**2 + (cy - py)**2)
                meters = d * PIXEL_TO_METER
                kmh = round((meters * fps) * 3.6, 2)
                speeds[tid] = kmh

        prev_track = tracked


        # ============================================================
        # 3) WRONG LANE DETECTION
        # ============================================================

        for tid, (box, cx, cy, cls_id) in tracked.items():

            # Determine if vehicle is left/right of lane center
            side = "LEFT" if cx < lane_boundary else "RIGHT"

            # For example: RIGHT = wrong lane
            WRONG_SIDE = "RIGHT"

            if side != WRONG_SIDE:
                continue  # Vehicle is in correct lane

            x1, y1, x2, y2 = box
            crop = enlarge_crop(frame, box)

            # Read plate
            try:
                plate = read_plate_from_vehicle(crop) or "UNKNOWN"
            except:
                plate = "UNKNOWN"

            # Save
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fname = f"wrong_lane_{ts}.jpg"
            cv2.imwrite(os.path.join(out_folder, fname), crop)

            speed_val = speeds.get(tid, "NA")

            extra = f"side={side},lane_center={lane_boundary},speed={speed_val}"

            # DB insert
            try:
                dbconnection.insert_detection(
                    number_plate=plate,
                    video_file=base,
                    module="wrong_lane",
                    image_filename=fname,
                    extra_info=extra
                )
                if plate != "UNKNOWN":
                    dbconnection.update_license_plate(plate)
            except Exception as e:
                print("❌ DB Error:", e)

            print(f"🚧 WRONG LANE → Plate: {plate} | Side: {side}")


    cap.release()
    print("✅ Wrong Lane Detection Completed.")
