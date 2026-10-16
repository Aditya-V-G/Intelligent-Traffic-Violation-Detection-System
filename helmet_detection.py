# helmet_detection.py
"""
No-helmet detection per rider, linked to the tracked motorcycle and its number plate.
The previous version is kept in legacy/helmet_detection.py.
"""
from vision.violations import HelmetPipeline


def main_fun(video_path, out_folder="static/uploads/detections"):
    return HelmetPipeline().run(video_path, out_folder)
