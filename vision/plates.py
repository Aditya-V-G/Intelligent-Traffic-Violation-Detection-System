# vision/plates.py
"""
Number-plate reading for Indian plates.

Pipeline per vehicle track:
  1. plate boxes are detected on the frame (trained numberplate model) and assigned to tracks
  2. the best few crops of each track are kept (size x sharpness x confidence)
  3. each crop is read by plate-specific OCR models (fast-plate-ocr); EasyOCR on enhanced
     crops is the fallback (two-line motorcycle plates are read line by line and joined)
  4. raw text is fitted to the Indian formats with position-aware character fixes
     (O->0 in digit slots, 8->B in letter slots ...); unfixable text is rejected
  5. valid reads from several frames are combined by per-character weighted voting
  6. if there is no clear winner the result is "Plate not readable", never a guess
"""

import re
import shutil
from collections import defaultdict

import cv2
import numpy as np

from vision import settings

# ---------------------------------------------------------------
# INDIAN PLATE FORMATS
# ---------------------------------------------------------------
STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN", "GA", "GJ", "HR", "HP",
    "JK", "JH", "KA", "KL", "LA", "LD", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "OR",
    "PY", "PB", "RJ", "SK", "TN", "TS", "TG", "TR", "UP", "UK", "UA", "WB",
}

STANDARD_RE = re.compile(r"^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{4})$")   # KA09AB1234, DL3CAB1234
BH_RE = re.compile(r"^(\d{2})(BH)(\d{4})([A-Z]{1,2})$")                  # 22BH1234AA

TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "J": "1", "T": "1",
            "Z": "2", "S": "5", "B": "8", "G": "6", "A": "4", "E": "3"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "3": "B", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}

MAX_FIXES = 2   # more substitutions than this = too uncertain, reject


def clean_text(text):
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def is_valid_plate(text):
    m = STANDARD_RE.match(text or "")
    if m:
        return m.group(1) in STATE_CODES
    return BH_RE.match(text or "") is not None


def _layouts(n):
    """All slot layouts ('L' letter / 'D' digit) of length n allowed by the formats."""
    out = []
    for dist in (2, 1):
        for series in (1, 2, 3, 0):
            if 2 + dist + series + 4 == n:
                out.append("LL" + "D" * dist + "L" * series + "DDDD")
    for series in (1, 2):
        if 2 + 2 + 4 + series == n:
            out.append("DDLLDDDD" + "L" * series)   # BH series
    return out


def _fit(text, layout):
    fixed, fixes = [], 0
    for ch, slot in zip(text, layout):
        if slot == "D" and not ch.isdigit():
            ch2 = TO_DIGIT.get(ch)
        elif slot == "L" and ch.isdigit():
            ch2 = TO_LETTER.get(ch)
        else:
            ch2 = ch
        if ch2 is None:
            return None, 99
        fixes += ch2 != ch
        fixed.append(ch2)
    return "".join(fixed), fixes


def correct_plate(raw):
    """
    Fit raw OCR text to an Indian plate format.
    Returns (plate, n_fixes) or (None, None) when it cannot be made valid.
    """
    text = clean_text(raw)
    # strip the "IND" hologram text that OCR often picks up
    for junk in ("IND", "INO", "1ND"):
        if text.startswith(junk) and len(text) - len(junk) >= 7:
            text = text[len(junk):]
            break

    candidates = []
    # also try dropping one stray char at either end (bolts, frame edges)
    for t in {text, text[1:], text[:-1]}:
        for layout in _layouts(len(t)):
            fixed, fixes = _fit(t, layout)
            if fixed and is_valid_plate(fixed):
                penalty = fixes + 1.5 * (len(t) != len(text))   # prefer using every character
                candidates.append((penalty, fixed))
    if not candidates:
        return None, None
    penalty, best = min(candidates)
    if penalty > MAX_FIXES:
        return None, None
    return best, penalty


def format_plate(plate):
    """KA09AB1234 -> 'KA 09 AB 1234' for display."""
    m = STANDARD_RE.match(plate or "")
    if m:
        return " ".join(m.groups())
    m = BH_RE.match(plate or "")
    if m:
        return " ".join(m.groups())
    return plate


# ---------------------------------------------------------------
# IMAGE QUALITY + PREPROCESSING
# ---------------------------------------------------------------
def sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop_quality(crop, det_conf):
    h, w = crop.shape[:2]
    if w < settings.PLATE_MIN_WIDTH or h < 10:
        return 0.0
    # width is capped: false detections on grilles/bumpers are wide and sharp and would
    # otherwise push the real (smaller) plate out of the kept crops
    return min(w, 200) * min(sharpness(crop), 400.0) / 400.0 * det_conf ** 2


def _deskew(gray):
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    pts = cv2.findNonZero(th)
    if pts is None or len(pts) < 50:
        return gray
    angle = cv2.minAreaRect(pts)[-1]
    if angle > 45:
        angle -= 90
    if abs(angle) < 2 or abs(angle) > 20:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess_variants(crop):
    """Return a few enhanced grayscale versions of a plate crop for OCR."""
    h, w = crop.shape[:2]
    scale = max(1.0, 96.0 / max(h, 1))           # plate text needs ~>30px character height
    scale = min(scale, 4.0)
    img = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 8)).apply(gray)
    gray = _deskew(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 2)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [sharp, otsu]


# ---------------------------------------------------------------
# OCR ENGINES
# ---------------------------------------------------------------
_ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_easy = None
_tess = None


def _easyocr():
    global _easy
    if _easy is None:
        import easyocr
        import torch
        _easy = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
    return _easy


def _tesseract():
    """Tesseract is optional; used only if it is installed."""
    global _tess
    if _tess is None:
        try:
            import pytesseract
            exe = shutil.which("tesseract") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            pytesseract.pytesseract.tesseract_cmd = exe
            pytesseract.get_tesseract_version()
            _tess = pytesseract
        except Exception:
            _tess = False
    return _tess or None


def _join_lines(results):
    """
    EasyOCR results -> text, ordering boxes top-to-bottom then left-to-right so that
    two-line motorcycle plates ('KA09' / 'AB1234') are joined correctly.
    """
    if not results:
        return "", 0.0
    items = []
    for box, txt, conf in results:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append((min(ys), max(ys), min(xs), clean_text(txt), float(conf)))
    items.sort(key=lambda r: r[0])
    lines, current = [], [items[0]]
    for it in items[1:]:
        prev_top, prev_bot = current[-1][0], current[-1][1]
        if it[0] < (prev_top + prev_bot) / 2:      # vertically overlaps -> same line
            current.append(it)
        else:
            lines.append(current)
            current = [it]
    lines.append(current)
    text = "".join("".join(i[3] for i in sorted(line, key=lambda r: r[2])) for line in lines)
    confs = [i[4] for i in items if i[3]]
    return text, (sum(confs) / len(confs) if confs else 0.0)


_fast = None


def _fast_readers():
    """fast-plate-ocr recognisers (bundled ONNX files first, then the package's model hub)."""
    global _fast
    if _fast is None:
        _fast = []
        try:
            import glob
            import os
            from fast_plate_ocr import LicensePlateRecognizer
        except ImportError:
            print("ℹ fast-plate-ocr not installed: using EasyOCR for plates (pip install fast-plate-ocr[onnx])")
            return _fast
        for name in settings.PLATE_OCR_MODELS:
            folder = os.path.join(settings.PLATE_OCR_DIR, name)
            onnx = glob.glob(os.path.join(folder, "*.onnx"))
            yml = glob.glob(os.path.join(folder, "*.yaml"))
            try:
                if onnx and yml:
                    _fast.append(LicensePlateRecognizer(onnx_model_path=onnx[0], plate_config_path=yml[0], device="cpu"))
                else:
                    _fast.append(LicensePlateRecognizer(name, device="cpu"))
            except Exception as e:
                print(f"⚠ plate OCR model {name} unavailable: {e}")
    return _fast


def _fast_reads(crop):
    reads = []
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    for rec in _fast_readers():
        try:
            pred = rec.run(rgb, return_confidence=True)[0]
        except Exception as e:
            print("plate OCR error:", e)
            continue
        raw = clean_text(pred.plate)
        probs = np.asarray(pred.char_probs if pred.char_probs is not None else [], dtype=float).ravel()
        conf = float(probs[:max(1, len(raw))].min()) if probs.size else 0.5   # weakest character
        plate, fixes = correct_plate(raw)
        if plate:
            reads.append((plate, conf * (0.85 ** fixes), raw))
    return reads


def ocr_crop(crop):
    """
    OCR one plate crop. Returns list of (plate, confidence) for every variant/engine that
    produced a VALID plate (after correction). Empty list = nothing usable.
    """
    reads = []
    if settings.PLATE_OCR in ("fast", "both"):
        reads = _fast_reads(crop)
        if reads and settings.PLATE_OCR == "fast":
            return reads
        if not _fast_readers() and settings.PLATE_OCR == "fast":
            pass                                   # package missing: fall through to EasyOCR
    for img in preprocess_variants(crop):
        try:
            res = _easyocr().readtext(img, allowlist=_ALLOW, paragraph=False,
                                      decoder="beamsearch", text_threshold=0.5)
            raw, conf = _join_lines(res)
        except Exception as e:
            print("EasyOCR error:", e)
            raw, conf = "", 0.0
        plate, fixes = correct_plate(raw)
        if plate:
            reads.append((plate, conf * (0.85 ** fixes), raw))

    tess = _tesseract()
    if tess is not None:
        img = preprocess_variants(crop)[0]
        for psm in (7, 6):    # single line, then block (two-line plates)
            try:
                raw = tess.image_to_string(img, config=f"--psm {psm} -c tessedit_char_whitelist={_ALLOW}")
            except Exception:
                break
            plate, fixes = correct_plate(raw)
            if plate:
                reads.append((plate, 0.6 * (0.85 ** fixes), raw))
                break
    return reads


# ---------------------------------------------------------------
# MULTI-FRAME VOTING
# ---------------------------------------------------------------
class PlateVoter:
    """Collects valid reads of one vehicle over many frames and decides the final plate."""

    def __init__(self):
        self.reads = []          # (plate, weight)

    def add(self, plate, weight):
        if plate and is_valid_plate(plate):
            self.reads.append((plate, max(weight, 0.05)))

    def __len__(self):
        return len(self.reads)

    def result(self):
        """
        Returns (plate_or_None, info). The plate is returned only when the reads agree:
          - at least PLATE_MIN_VOTES valid reads (or one very confident read), and
          - the per-character vote winner gets >= 70% of the weight in every position.
        """
        if not self.reads:
            return None, "no valid reads"

        by_len = defaultdict(list)
        for p, w in self.reads:
            by_len[len(p)].append((p, w))
        length, group = max(by_len.items(), key=lambda kv: sum(w for _, w in kv[1]))
        total = sum(w for _, w in self.reads)
        if sum(w for _, w in group) < 0.75 * total:
            # frames disagree on the number of characters (a char dropped/added): ambiguous
            return None, f"reads={len(self.reads)}, conflicting lengths {sorted(by_len)}"

        chars, weakest = [], 1.0
        for i in range(length):
            votes = defaultdict(float)
            for p, w in group:
                votes[p[i]] += w
            ch, wt = max(votes.items(), key=lambda kv: kv[1])
            chars.append(ch)
            weakest = min(weakest, wt / sum(votes.values()))
        plate = "".join(chars)

        exact = sum(1 for p, _ in self.reads if p == plate)
        best_single = max(w for p, w in self.reads if p == plate) if exact else 0.0
        info = f"reads={len(self.reads)}, agree={exact}, char_agreement={weakest:.2f}"

        if not is_valid_plate(plate) or weakest < 0.7:
            return None, info
        if exact >= settings.PLATE_MIN_VOTES or (len(self.reads) == 1 and best_single >= 0.8):
            return plate, info
        if len(group) >= settings.PLATE_MIN_VOTES and weakest >= 0.75:
            return plate, info          # reads differ by single chars but agree per position
        return None, info


# ---------------------------------------------------------------
# PLATE DETECTION
# ---------------------------------------------------------------
class PlateDetector:
    def __init__(self):
        from vision.models import load_first
        self.model = load_first(settings.PLATE_MODEL, settings.PLATE_MODEL_FALLBACK)

    def detect(self, frame, imgsz=None):
        """Return [(x1, y1, x2, y2, conf)] plate boxes in frame coordinates."""
        if imgsz is None:
            imgsz = 1280 if frame.shape[1] > 1920 else 960
        res = self.model(frame, conf=settings.PLATE_CONF, imgsz=imgsz,
                         device=settings.DEVICE or None, verbose=False)[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            out.append((x1, y1, x2, y2, float(b.conf[0])))
        return out


def crop_with_margin(frame, box, margin=0.08):
    x1, y1, x2, y2 = box[:4]
    H, W = frame.shape[:2]
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    return frame[max(0, y1 - my):min(H, y2 + my), max(0, x1 - mx):min(W, x2 + mx)].copy()
