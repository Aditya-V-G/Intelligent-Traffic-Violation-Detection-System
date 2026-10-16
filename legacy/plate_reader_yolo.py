# plate_reader_yolo.py

import cv2
import re
import pytesseract
import easyocr
from ultralytics import YOLO
from difflib import SequenceMatcher

# Load OCR engines
reader = easyocr.Reader(['en'], gpu=False)
import shutil
pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load YOLO number plate model
PLATE_MODEL_PATH = r"trainedmodels/numberplate.pt"
plate_model = YOLO(PLATE_MODEL_PATH)

def clean_text(t):
    t = t.upper()
    return re.sub(r"[^A-Z0-9]", "", t)

def best_match(candidates):
    if not candidates:
        return None
    candidates = sorted(candidates, key=len, reverse=True)
    best = candidates[0]
    for c in candidates:
        if SequenceMatcher(None, best, c).ratio() > 0.5:
            best = c
    return best

def read_plate_from_vehicle(frame):
    """
    Detect number plate using YOLO model, then perform OCR.
    """

    results = plate_model(frame, conf=0.45, verbose=False)

    if len(results[0].boxes) == 0:
        return None

    # Select largest plate 
    plates = []
    for b in results[0].boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        plates.append((x1, y1, x2, y2))

    # Pick the biggest detected box
    plates = sorted(plates, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    x1, y1, x2, y2 = plates[0]

    crop = frame[y1:y2, x1:x2]

    if crop is None or crop.size == 0:
        return None

    # OCR
    candidates = []

    # EasyOCR
    try:
        easy = reader.readtext(crop)
        for _, txt, _ in easy:
            txt = clean_text(txt)
            if len(txt) >= 5:
                candidates.append(txt)
    except:
        pass

    # Tesseract
    try:
        ttxt = pytesseract.image_to_string(
            crop,
            config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
        ttxt = clean_text(ttxt)
        if len(ttxt) >= 5:
            candidates.append(ttxt)
    except:
        pass

    return best_match(candidates)
