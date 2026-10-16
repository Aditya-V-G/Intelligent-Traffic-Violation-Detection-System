import cv2, os, numpy as np
from ultralytics import YOLO
from datetime import datetime
import dbconnection


# Model auto-load
DEFAULT_MODEL_PATH = "yolov8n.pt"

LANE_REGIONS = [
    (0.0, 0.33),
    (0.33, 0.66),
    (0.66, 1.0)
]


# -------------------------------------------------------------
# Helper: detect lane index
# -------------------------------------------------------------

def get_lane(cx, frame_width):
    for i, (s, e) in enumerate(LANE_REGIONS):
        if frame_width * s <= cx < frame_width * e:
            return i + 1
    return None


# -------------------------------------------------------------
# MAIN HEATMAP ENHANCER
# -------------------------------------------------------------

def process_videos(video_files, model=None, out_folder="static/uploads/heatmap"):

    os.makedirs(out_folder, exist_ok=True)

    if model is None:
        try:
            model = YOLO(DEFAULT_MODEL_PATH)
        except Exception as e:
            print("Heatmap: YOLO load error:", e)
            return

    for video_path in video_files:

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Heatmap: can't open", video_path)
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.basename(video_path)

        heatmap_acc = np.zeros((height, width), dtype=np.float32)
        heatmap_vehicle = np.zeros((height, width), dtype=np.float32)
        heatmap_person = np.zeros((height, width), dtype=np.float32)

        frame_count = 0
        overlay_output = os.path.join(out_folder, f"heatmap_overlay_{base}_{ts}.mp4")
        heatmap_output = os.path.join(out_folder, f"heatmap_raw_{base}_{ts}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer_overlay = cv2.VideoWriter(overlay_output, fourcc, fps, (width, height))
        writer_raw = cv2.VideoWriter(heatmap_output, fourcc, fps, (width, height))

        print("Heatmap Processing Started →", base)

        while True:

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 2 != 0:
                continue  # skip frames for performance

            results = model(frame, conf=0.35, verbose=False)
            dets = results[0].boxes.data.cpu().numpy()

            for x1, y1, x2, y2, conf, cls in dets:

                cls = int(cls)

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if not (0 <= cx < width and 0 <= cy < height):
                    continue

                # Global heatmap
                heatmap_acc[cy, cx] += 1

                # Class-based heatmaps
                if cls in [2, 3, 5, 7, 1]:  
                    heatmap_vehicle[cy, cx] += 1
                elif cls == 0:
                    heatmap_person[cy, cx] += 1

            # Build normalized heatmap
            if heatmap_acc.max() > 0:
                hm_norm = cv2.normalize(heatmap_acc, None, 0, 255, cv2.NORM_MINMAX)
                hm_color = cv2.applyColorMap(hm_norm.astype(np.uint8), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(frame, 0.6, hm_color, 0.4, 0)
            else:
                overlay = frame

            writer_overlay.write(overlay)

            # Raw-only heatmap frame
            raw_frame = hm_color if heatmap_acc.max() > 0 else frame
            writer_raw.write(raw_frame)

        cap.release()
        writer_overlay.release()
        writer_raw.release()

        # ----------------------------------------------
        # Lane Heat Distribution
        # ----------------------------------------------

        lane_heat = {1: 0, 2: 0, 3: 0}

        for (y, x), value in np.ndenumerate(heatmap_acc):
            lane = get_lane(x, width)
            if lane:
                lane_heat[lane] += value

        # ----------------------------------------------
        # Density score
        # ----------------------------------------------

        density_score = float(heatmap_acc.sum()) / max(1, frame_count)

        # ----------------------------------------------
        # Save Summary Image
        # ----------------------------------------------

        if heatmap_acc.max() > 0:
            hm_norm = cv2.normalize(heatmap_acc, None, 0, 255, cv2.NORM_MINMAX)
            hm_color = cv2.applyColorMap(hm_norm.astype(np.uint8), cv2.COLORMAP_JET)
            img_out = os.path.join(out_folder, f"heatmap_image_{base}_{ts}.jpg")
            cv2.imwrite(img_out, hm_color)
        else:
            img_out = None

        # ----------------------------------------------
        # Insert into DB
        # ----------------------------------------------

        try:
            extra = (
                f"frames={frame_count},"
                f"density={density_score:.2f},"
                f"lane_heat={lane_heat}"
            )
            dbconnection.insert_detection(
                "HEATMAP_SUMMARY",
                base,
                "heatmap",
                os.path.basename(overlay_output),
                extra
            )
        except Exception as e:
            print("DB insert error:", e)

        print("Heatmap Completed →", base)

    print("ALL Heatmaps Completed")
