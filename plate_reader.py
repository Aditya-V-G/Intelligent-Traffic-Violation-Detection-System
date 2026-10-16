import cv2
import re
import pytesseract
import easyocr
import numpy as np
from difflib import SequenceMatcher

# ----------------------------
# TESSERACT CONFIG
# ----------------------------

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

reader = easyocr.Reader(['en'], gpu=False)


# ----------------------------
# SUPER-RESOLUTION MODEL
# ----------------------------

def load_sr_model():
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    try:
        # You can download EDSR_x2.pb and place it in project folder
        sr.readModel("EDSR_x2.pb")
        sr.setModel("edsr", 2)
        return sr
    except:
        return None    # use fallback


SR = load_sr_model()


# ----------------------------
# PREPROCESSING
# ----------------------------

def enhance_for_ocr(img):
    if img is None or img.size == 0:
        return img

    # Super-resolution
    if SR is not None:
        try:
            img = SR.upsample(img)
        except:
            pass

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sharpening
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    gray = cv2.filter2D(gray, -1, kernel)

    # Adaptive threshold (handles low light)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )

    return thresh


# ----------------------------
# CLEANING + VALIDATION
# ----------------------------

def clean_text(t):
    t = t.upper()
    return re.sub(r"[^A-Z0-9]", "", t)


def is_valid_plate(text):
    """
    Validate using Indian vehicle number format.
    Example: KA09AB1234
    """
    pattern = r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$"
    return re.match(pattern, text) is not None


def best_match(candidates):
    """
    Choose best OCR string based on:
    - length
    - pattern matching
    - similarity
    """
    if not candidates:
        return None

    # Strong priority: valid Indian number plates
    valid = [c for c in candidates if is_valid_plate(c)]
    if valid:
        return valid[0]   # best one

    # else fallback to longest string
    candidates = sorted(candidates, key=len, reverse=True)

    best = candidates[0]
    for c in candidates:
        if SequenceMatcher(None, best, c).ratio() > 0.5:
            best = c

    return best


# ----------------------------
# MAIN FUNCTION to use across system
# ----------------------------

def read_plate_from_vehicle(img):

    if img is None or img.size == 0:
        return None

    img = enhance_for_ocr(img)

    candidates = []

    # ---------- EASY OCR ----------
    try:
        easy_res = reader.readtext(img)
        for _, txt, _ in easy_res:
            txt = clean_text(txt)
            if len(txt) >= 6:
                candidates.append(txt)
    except:
        pass

    # ---------- TESSERACT ----------
    try:
        ttxt = pytesseract.image_to_string(
            img,
            config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
        ttxt = clean_text(ttxt)
        if len(ttxt) >= 6:
            candidates.append(ttxt)
    except:
        pass

    # ---------- BEST RESULT ----------
    final = best_match(candidates)

    return final
