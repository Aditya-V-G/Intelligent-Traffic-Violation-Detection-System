# plate_reader_yolo.py
"""
Single-image plate reading (kept for scripts that call read_plate_from_vehicle).
Video modules use vision/pipeline.py, which adds tracking and multi-frame voting.
"""
from vision.plates import PlateDetector, PlateVoter, crop_with_margin, ocr_crop

_detector = None


def read_plate_from_vehicle(frame):
    """Return the plate text from one image, or None when it cannot be read reliably."""
    global _detector
    if frame is None or frame.size == 0:
        return None
    if _detector is None:
        _detector = PlateDetector()
    voter = PlateVoter()
    for box in sorted(_detector.detect(frame), key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:2]:
        for plate, weight, _ in ocr_crop(crop_with_margin(frame, box)):
            voter.add(plate, weight)
    return voter.result()[0]
