# triple_riding.py
"""
Triple riding detection, linked to each motorcycle and its number plate.
Uses the trained triple-riding model (trainedmodels/triple_riding_yolov8.pt) and/or
de-duplicated rider counting per tracked motorcycle; see vision/violations.py.
The previous version is kept in legacy/triple_riding.py.
"""
from vision.violations import TripleRidingPipeline


def detect_triple_riding(video_path, out_folder="static/uploads/detections"):
    return TripleRidingPipeline().run(video_path, out_folder)


def start_triple_riding(video_path, out_folder="static/uploads/detections"):
    return detect_triple_riding(video_path, out_folder)
