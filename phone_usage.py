# phone_usage.py
"""
Phone usage while driving/riding, linked to the vehicle and its number plate.
Uses the retrained model trainedmodels/phone_v3.pt when present.
The previous version is kept in legacy/phone_usage.py.
"""
from vision.violations import PhoneUsagePipeline


def detect_phone_usage(video_path, out_folder="static/uploads/detections"):
    return PhoneUsagePipeline().run(video_path, out_folder)
