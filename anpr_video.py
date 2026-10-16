# anpr_video.py
"""
ANPR with vehicle tracking and multi-frame plate voting: one row per vehicle,
"Plate not readable" instead of a wrong number. Old version: legacy/anpr_video.py.
"""
from vision.plates import correct_plate, is_valid_plate  # noqa: F401  (kept for old imports)
from vision.violations import ANPRPipeline


def start_anpr(video_paths, out_folder="static/uploads/detections"):
    records = []
    for path in video_paths:
        records += ANPRPipeline().run(path, out_folder)
    return records


start_anpr_ultra = start_anpr
