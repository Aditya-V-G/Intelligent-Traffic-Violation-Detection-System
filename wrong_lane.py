# wrong_lane.py
"""
Wrong lane / wrong-way detection from tracked vehicle direction (ByteTrack) instead of
single-frame decisions; see vision/violations.py (WrongLanePipeline) for the options.
The previous version is kept in legacy/wrong_lane.py.
"""
from vision.violations import WrongLanePipeline


def detect_wrong_lane(video_path, out_folder="static/uploads/detections"):
    return WrongLanePipeline().run(video_path, out_folder)
