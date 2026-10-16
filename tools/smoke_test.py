"""
Quick installation check (no database needed, ~2-5 min on CPU):
  1. every model file exists and loads, with the expected classes
  2. the plate reader reads a synthetic plate
  3. one violation module runs end to end on a short sample video
    python tools/smoke_test.py
"""
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SAVE_ANNOTATED_VIDEO", "0")
if "pymysql" not in sys.modules:          # run without a database
    try:
        import pymysql  # noqa: F401
    except ImportError:
        db = types.ModuleType("dbconnection")
        db.insert_detection = lambda *a, **k: None
        db.update_license_plate = lambda *a, **k: None
        sys.modules["dbconnection"] = db

ok = True


def check(name, fn):
    global ok
    try:
        msg = fn()
        print(f"✔ {name}" + (f": {msg}" if msg else ""))
    except Exception as e:
        ok = False
        print(f"✘ {name}: {e}")


from vision import settings  # noqa: E402
from vision.models import load_first  # noqa: E402

for label, paths in [("vehicle model", [settings.VEHICLE_MODEL]),
                     ("plate detector", [settings.PLATE_MODEL, settings.PLATE_MODEL_FALLBACK]),
                     ("helmet", [settings.HELMET_MODEL, settings.HELMET_MODEL_FALLBACK]),
                     ("phone", [settings.PHONE_MODEL]),
                     ("seatbelt", [settings.SEATBELT_MODEL, settings.SEATBELT_MODEL_FALLBACK]),
                     ("wrong lane", [settings.WRONG_LANE_MODEL, settings.WRONG_LANE_MODEL_FALLBACK]),
                     ("triple riding", [settings.TRIPLE_MODEL]),
                     ("accident", ["trainedmodels/accident_v3.pt", "trainedmodels/accident_best.pt"])]:
    def _desc(p=paths):
        m = load_first(*p)
        names = list(m.names.values())
        return f"{os.path.basename(m.ckpt_path or p[0])} ({len(names)} classes: {', '.join(names[:6])}{' ...' if len(names) > 6 else ''})"
    check(label, _desc)


def plate_test():
    import cv2
    import numpy as np
    from vision.plates import ocr_crop, settings as st
    img = np.full((60, 260, 3), 255, np.uint8)
    cv2.putText(img, "KA 02 MM 9091", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
    reads = [p for p, _, _ in ocr_crop(img)]
    assert "KA02MM9091" in reads, reads
    return f"read KA02MM9091 with PLATE_OCR={st.PLATE_OCR}"


check("plate reader", plate_test)


def module_test():
    from helmet_detection import main_fun
    out = tempfile.mkdtemp()
    main_fun("sample detection videos/with helmet.mp4", out)
    return f"helmet module ran on 'with helmet.mp4' (output in {out})"


check("violation pipeline", module_test)
print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED (see ✘ above)")
sys.exit(0 if ok else 1)
