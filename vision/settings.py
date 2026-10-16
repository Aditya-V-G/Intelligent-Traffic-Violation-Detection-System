# vision/settings.py
"""
Central configuration for the violation pipeline.

Every path is relative to the project root and can be overridden with an
environment variable (or a .env file), so nothing is tied to one machine.
Class IDs are NOT hard-coded: models are queried for their class names and
matched against the keyword lists below (see vision/models.py).
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _path(env_key, default):
    value = os.getenv(env_key, default)
    return value if os.path.isabs(value) else os.path.join(PROJECT_ROOT, value)


# ---------------------------------------------------------------
# MODEL WEIGHTS
# ---------------------------------------------------------------
# NOTE: "triple riding.pt" and "yolov8s.pt" in the repo are byte-identical to
# the stock COCO yolov8s model, and "atcc.pt" is the stock COCO yolov8n model.
# The real trained triple-riding weights are triple_riding_yolov8.pt.

VEHICLE_MODEL = _path("VEHICLE_MODEL", "yolov8s.pt")            # COCO: car / motorcycle / bus / truck
PERSON_MODEL = _path("PERSON_MODEL", "yolov8s.pt")              # COCO person + cell phone (fallbacks)
PLATE_MODEL = _path("PLATE_MODEL", "trainedmodels/numberplate_v2.pt")            # retrained weights
PLATE_MODEL_FALLBACK = _path("PLATE_MODEL_FALLBACK", "trainedmodels/numberplate.pt")
HELMET_MODEL = _path("HELMET_MODEL", "trainedmodels/helmet_v3.pt")                  # retrained weights (YOLOv8m)
# trainedmodels/helmetdetection.pt (used before) has classes named "0"/"1" and does not separate
# helmet from no-helmet on the sample videos; helmet_best.pt has named classes.
HELMET_MODEL_FALLBACK = _path("HELMET_MODEL_FALLBACK", "helmet_best.pt")
TRIPLE_MODEL = _path("TRIPLE_MODEL", "trainedmodels/triple_riding_yolov8.pt")
WRONG_LANE_MODEL = _path("WRONG_LANE_MODEL", "trainedmodels/wrong_way_v2.pt")          # retrained weights
WRONG_LANE_MODEL_FALLBACK = _path("WRONG_LANE_MODEL_FALLBACK", "trainedmodels/wrong_lane_yolov8.pt")
# drivers and motorcycle riders on the phone (YOLOv8m); without it the COCO "cell phone" check is used
PHONE_MODEL = _path("PHONE_MODEL", "trainedmodels/phone_v3.pt")           # retrained weights
SEATBELT_MODEL = _path("SEATBELT_MODEL", "trainedmodels/seatbelt_v2.pt")  # retrained weights

# Fallbacks used when the retrained weights are not present yet
SEATBELT_MODEL_FALLBACK = _path("SEATBELT_MODEL_FALLBACK", "trainedmodels/seatbelt_yolov8.pt")

TRACKER_CFG = os.getenv("TRACKER_CFG", "bytetrack.yaml")
DEVICE = os.getenv("YOLO_DEVICE", "")          # "" = auto, "cpu", "0" ...

# ---------------------------------------------------------------
# CLASS-NAME KEYWORDS (lower-case substrings, checked in order)
# ---------------------------------------------------------------
VEHICLE_KEYWORDS = {
    "car": ["car", "mobil", "sedan", "suv", "van", "jeep", "taxi"],
    "motorcycle": ["motorcycle", "motorbike", "motor", "bike", "scooter", "two"],
    "bus": ["bus"],
    "truck": ["truck", "lorry"],
    "auto": ["auto", "rickshaw", "three"],
}
NEGATIVE_KEYWORDS = ["no", "without", "not", "non", "wrong", "violat", "un"]

# ---------------------------------------------------------------
# PIPELINE TUNING
# ---------------------------------------------------------------
FRAME_STRIDE = int(os.getenv("FRAME_STRIDE", 2))      # analyse every Nth frame
VEHICLE_CONF = float(os.getenv("VEHICLE_CONF", 0.30))
PLATE_CONF = float(os.getenv("PLATE_CONF", 0.35))
PLATE_EVERY_N = int(os.getenv("PLATE_EVERY_N", 2))    # OCR each track every N analysed frames
PLATE_MIN_VOTES = int(os.getenv("PLATE_MIN_VOTES", 2))
PLATE_MIN_WIDTH = int(os.getenv("PLATE_MIN_WIDTH", 40))   # px; smaller crops are skipped
PLATE_NOT_READABLE = "Plate not readable"
# Plate text reader: "fast" = plate-specific fast-plate-ocr models (EasyOCR only when they read
# nothing), "easyocr" = the general-purpose EasyOCR, "both" = every engine votes.
PLATE_OCR = os.getenv("PLATE_OCR", "fast")
PLATE_OCR_MODELS = [m for m in os.getenv("PLATE_OCR_MODELS", "cct-s-v2-global-model,cct-xs-v2-global-model").split(",") if m]
PLATE_OCR_DIR = _path("PLATE_OCR_DIR", "trainedmodels/plate_ocr")     # bundled ONNX models (offline)

# A violation is confirmed for a track when it is seen in at least
# CONFIRM_HITS of the last CONFIRM_WINDOW analysed frames of that track.
CONFIRM_HITS = int(os.getenv("CONFIRM_HITS", 3))
CONFIRM_WINDOW = int(os.getenv("CONFIRM_WINDOW", 6))
TRACK_LOST_FRAMES = int(os.getenv("TRACK_LOST_FRAMES", 30))   # finalise a track after this many unseen frames

SAVE_ANNOTATED_VIDEO = os.getenv("SAVE_ANNOTATED_VIDEO", "1") == "1"
