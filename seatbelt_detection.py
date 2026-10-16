# seatbelt_detection.py
"""
No-seatbelt detection, linked to the vehicle and its number plate.
Uses the retrained model trainedmodels/seatbelt_v2.pt when present.
The previous version is kept in legacy/seatbelt_detection.py.
"""
from vision.violations import SeatbeltPipeline


def detect_seatbelt(video_path, out_folder="static/uploads/detections"):
    return SeatbeltPipeline().run(video_path, out_folder)
