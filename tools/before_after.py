"""
Before/after comparison on the sample videos.

Runs the ORIGINAL scripts (legacy/) and the NEW pipeline on the same videos without
touching the database, and writes:
    before_after/summary.md          table per module/video
    before_after/summary.csv
    before_after/<module>/<video>/old/   evidence images saved by the old script
    before_after/<module>/<video>/new/   evidence images + annotated video + report csv

Usage (from the project root):
    python tools/before_after.py                       # all modules, default sample videos
    python tools/before_after.py --modules triple wrong_lane
    python tools/before_after.py --modules helmet --videos "path/a.mp4" "path/b.mp4"
"""

import argparse
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

try:
    import dbconnection  # noqa: E402
except ImportError:      # pymysql / dotenv not installed: a stub is enough, writes are captured
    import types
    dbconnection = sys.modules["dbconnection"] = types.ModuleType("dbconnection")

SAMPLES = os.path.join(ROOT, "sample detection videos")
DEFAULT_VIDEOS = {
    "helmet": ["helmet detection.mp4", "with helmet.mp4"],
    "triple": ["triple riding.mp4", "triple riding3.mp4", "triple_riding 2.mp4"],
    "wrong_lane": ["wrong lane.mp4"],
    "phone": ["phoneusage.mp4"],
    "seatbelt": ["seatbelt.mp4", "noseatbelt.mp4"],
    "anpr": ["anpr sample.mp4"],
}

# ---------------------------------------------------------------
# capture DB writes instead of sending them to MySQL
# ---------------------------------------------------------------
CAPTURED = []


def _capture(*args, **kwargs):
    if args:
        keys = ["number_plate", "video_file", "module", "image_filename", "extra_info"]
        kwargs = dict(zip(keys, args), **kwargs)
    CAPTURED.append(kwargs)


dbconnection.insert_detection = _capture
dbconnection.update_license_plate = lambda plate: None


def run_old(module, video, out):
    from legacy import anpr_video, helmet_detection, phone_usage, seatbelt_detection, triple_riding, wrong_lane
    helmet_detection.MODEL_PATH = os.path.join(ROOT, "trainedmodels", "helmetdetection.pt")
    {
        "helmet": lambda: helmet_detection.main_fun(video, out),
        "triple": lambda: triple_riding.detect_triple_riding(video, out),
        "wrong_lane": lambda: wrong_lane.detect_wrong_lane(video, out),
        "phone": lambda: phone_usage.detect_phone_usage(video, out),
        "seatbelt": lambda: seatbelt_detection.detect_seatbelt(video, out),
        "anpr": lambda: anpr_video.start_anpr([video], out),
    }[module]()


def run_new(module, video, out):
    from vision import pipeline, violations as V
    pipeline.dbconnection = dbconnection
    cls = {"helmet": V.HelmetPipeline, "triple": V.TripleRidingPipeline, "wrong_lane": V.WrongLanePipeline,
           "phone": V.PhoneUsagePipeline, "seatbelt": V.SeatbeltPipeline, "anpr": V.ANPRPipeline}[module]
    cls().run(video, out)


def plate_ok(p):
    return bool(p) and p not in ("UNKNOWN", "Plate not readable")


def measure(fn, module, video, out):
    os.makedirs(out, exist_ok=True)
    CAPTURED.clear()
    t0 = time.time()
    error = ""
    try:
        fn(module, video, out)
    except Exception as e:           # report and carry on with the other videos
        error = f"{type(e).__name__}: {e}"
        print("❌", error)
    rows = list(CAPTURED)
    plates = [r.get("number_plate") for r in rows]
    return {
        "records": len(rows),
        "with_plate": sum(plate_ok(p) for p in plates),
        "unreadable": sum(p == "Plate not readable" for p in plates),
        "unknown": sum(p in ("UNKNOWN", "", None) for p in plates),
        "plates": " ".join(sorted({p for p in plates if plate_ok(p)})),
        "seconds": round(time.time() - t0, 1),
        "error": error,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", nargs="*", default=list(DEFAULT_VIDEOS))
    ap.add_argument("--videos", nargs="*", help="override the video list (applied to every module given)")
    ap.add_argument("--out", default=os.path.join(ROOT, "before_after"))
    ap.add_argument("--skip-old", action="store_true")
    args = ap.parse_args()

    results = []
    for module in args.modules:
        videos = args.videos or [os.path.join(SAMPLES, v) for v in DEFAULT_VIDEOS[module]]
        for video in videos:
            if not os.path.exists(video) or os.path.getsize(video) < 1000:
                print(f"⚠ skipping {video} (missing, or a Git LFS pointer: run `git lfs pull`)")
                continue
            name = os.path.splitext(os.path.basename(video))[0]
            base = os.path.join(args.out, module, name)
            print(f"\n=========== {module} | {name} ===========")
            old = None if args.skip_old else measure(run_old, module, video, os.path.join(base, "old"))
            new = measure(run_new, module, video, os.path.join(base, "new"))
            results.append((module, name, old, new))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["module", "video", "version", "records", "with_plate", "plate_not_readable",
                    "plate_unknown", "plates", "seconds", "error"])
        for module, name, old, new in results:
            for tag, r in (("old", old), ("new", new)):
                if r:
                    w.writerow([module, name, tag, r["records"], r["with_plate"], r["unreadable"],
                                r["unknown"], r["plates"], r["seconds"], r["error"]])

    lines = ["# Before / after on sample videos", "",
             "Records = violation rows written to the database. The old scripts wrote one row per "
             "detection burst (duplicates of the same vehicle) with plate `UNKNOWN` for every module "
             "except helmet; the new pipeline writes one row per violating vehicle with its plate "
             "or `Plate not readable`. Check the evidence images in each folder to count true/false positives.", "",
             "| Module | Video | Old records | Old with plate | New records | New with plate | New not readable | New plates |",
             "|---|---|---|---|---|---|---|---|"]
    for module, name, old, new in results:
        o = old or {"records": "-", "with_plate": "-"}
        lines.append(f"| {module} | {name} | {o['records']} | {o['with_plate']} | {new['records']} | "
                     f"{new['with_plate']} | {new['unreadable']} | {new['plates'] or '-'} |")
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
