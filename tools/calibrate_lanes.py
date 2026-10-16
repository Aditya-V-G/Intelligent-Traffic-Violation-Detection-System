"""
Learn the normal driving direction of each road area for ONE fixed camera and save it, so the
wrong-lane check works from the first vehicle (no learning period, no minimum traffic).

    python tools/calibrate_lanes.py "recording_of_camera_7.mp4"              # -> lane_zones/<name>.json
    python tools/calibrate_lanes.py cam7.mp4 --out lane_zones/default.json    # used for every video

Use a few minutes of NORMAL traffic from the camera's usual view. The wrong-lane module then loads
lane_zones/<video name>.json (or lane_zones/default.json, or the file in WRONG_LANE_ZONES).
A preview image with the learned arrows is written next to the JSON: check it once.
"""
import argparse
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SAVE_ANNOTATED_VIDEO", "0")

import cv2  # noqa: E402

from vision import pipeline  # noqa: E402
from vision.violations import WrongLanePipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", help="default: lane_zones/<video name>.json")
    ap.add_argument("--grid", type=int, default=8, help="cells per side (default 8)")
    ap.add_argument("--min-vehicles", type=int, default=3, help="vehicles needed per cell")
    ap.add_argument("--consistency", type=float, default=0.8, help="0-1: how much they must agree")
    args = ap.parse_args()

    pipeline.dbconnection = None                       # calibration writes nothing to the database
    os.environ["WRONG_LANE_ZONES_DIR"] = tempfile.mkdtemp()     # ignore existing zones
    WrongLanePipeline.ZONES_DIR = os.environ["WRONG_LANE_ZONES_DIR"]
    WrongLanePipeline.GRID = args.grid
    WrongLanePipeline.mode = "direction"
    p = WrongLanePipeline()
    p.run(args.video, tempfile.mkdtemp())

    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    H, W = frame.shape[:2]
    zones = []
    for (cx, cy), samples in sorted(p.flow.items()):
        tracks = {}                                     # per vehicle: summed unit motion, samples
        for _, _, dx, dy, tid in samples:
            v = tracks.setdefault(tid, [0.0, 0.0, 0])
            v[0] += dx; v[1] += dy; v[2] += 1
        vs = [(v[0] / v[2], v[1] / v[2]) for v in tracks.values() if v[2] >= 2]
        if len(vs) < args.min_vehicles:
            continue
        sx, sy = sum(v[0] for v in vs), sum(v[1] for v in vs)
        n = math.hypot(sx, sy)
        if n / len(vs) < args.consistency:              # traffic in this cell goes both ways: skip
            continue
        x0, y0, x1, y1 = cx / args.grid, cy / args.grid, (cx + 1) / args.grid, (cy + 1) / args.grid
        zones.append({"polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                      "direction": [round(sx / n, 3), round(sy / n, 3)], "vehicles": len(vs)})
        c = (int((x0 + x1) / 2 * W), int((y0 + y1) / 2 * H))
        L = 0.4 * W / args.grid
        cv2.rectangle(frame, (int(x0 * W), int(y0 * H)), (int(x1 * W), int(y1 * H)), (0, 200, 0), 1)
        cv2.arrowedLine(frame, (int(c[0] - sx / n * L / 2), int(c[1] - sy / n * L / 2)),
                        (int(c[0] + sx / n * L / 2), int(c[1] + sy / n * L / 2)), (0, 255, 255), 3, tipLength=0.4)

    out = args.out or os.path.join("lane_zones", os.path.splitext(os.path.basename(args.video))[0] + ".json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(zones, f, indent=1)
    cv2.imwrite(os.path.splitext(out)[0] + "_preview.jpg", frame)
    print(f"✔ {len(zones)} road areas with a clear direction saved to {out}")
    print(f"  check the arrows in {os.path.splitext(out)[0] + '_preview.jpg'}")
    if not zones:
        print("  ⚠ nothing learned: use a longer recording with more traffic, or a fixed (not moving) camera")


if __name__ == "__main__":
    main()
