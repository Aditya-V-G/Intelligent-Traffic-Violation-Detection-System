# vision/models.py
"""Cached model loading and class lookup by name instead of hard-coded IDs."""

import os
import re
from functools import lru_cache

from ultralytics import YOLO

from vision import settings


@lru_cache(maxsize=None)
def load(path):
    if not os.path.exists(path):
        if not os.path.basename(path).startswith(("yolov8", "yolo11")):
            raise FileNotFoundError(f"Model not found: {path}")
        path = os.path.basename(path)   # stock COCO weights: Ultralytics downloads them
    model = YOLO(path)
    print(f"✔ Loaded {os.path.basename(path)} | classes: {model.names}")
    return model


def load_first(*paths):
    """Load the first model path that exists (e.g. retrained weights, then old weights)."""
    for p in paths:
        if p and os.path.exists(p):
            return load(p)
    raise FileNotFoundError(f"None of these models exist: {paths}")


def _is_negative(name):
    """True for names like 'no_helmet', 'Without Seatbelt', 'nohelmet', 'wrong-lane', 'unbelted'."""
    for t in re.split(r"[\s_\-]+", name.lower()):
        if t in settings.NEGATIVE_KEYWORDS:
            return True
        if t.startswith(("no", "without", "wrong", "non", "un")) and not t.startswith(("normal", "north")):
            return True
    return False


def find_classes(model, keywords, negative=None):
    """
    Return class IDs whose name contains any keyword.
    negative=True  -> only names that express a violation ("no_helmet", "without seatbelt")
    negative=False -> only names that do NOT express a violation ("helmet", "seatbelt")
    """
    ids = []
    for cid, name in model.names.items():
        low = name.lower()
        if not any(k in low for k in keywords):
            continue
        if negative is not None and _is_negative(low) != negative:
            continue
        ids.append(cid)
    return ids


def vehicle_type_map(model):
    """Map class id -> canonical vehicle type for any vehicle model (COCO or custom)."""
    mapping = {}
    for cid, name in model.names.items():
        low = name.lower()
        if low in ("motorcycle", "motorbike", "motor", "bike", "scooter"):
            mapping[cid] = "motorcycle"
            continue
        if low == "bicycle":
            continue
        for vtype, words in settings.VEHICLE_KEYWORDS.items():
            if any(w == low or (len(w) > 3 and w in low) for w in words):
                mapping[cid] = vtype
                break
    return mapping
