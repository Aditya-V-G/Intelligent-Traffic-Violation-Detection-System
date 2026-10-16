"""
Train / fine-tune a YOLOv8 model and install it where the pipeline expects it.

    python training/train.py --data dsets/seatbelt/data.yaml --name seatbelt_custom
    python training/train.py --data dsets/phone/data.yaml    --name phone_custom

Writes trainedmodels/<name>.pt and appends test-split metrics to training/metrics.csv.
Point SEATBELT_MODEL / PHONE_MODEL (.env) at the new file to use it.
Use a GPU (e.g. Google Colab or Kaggle); on CPU this takes many hours.
"""

import argparse
import csv
import os
import shutil
from datetime import date

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evaluate(weights, data, split):
    m = YOLO(weights).val(data=data, split=split, verbose=False, plots=False)
    per_class = {m.names[c]: round(float(m.box.maps[c]), 3) for c in range(len(m.names))}
    return {"precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4),
            "mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4),
            "per_class_mAP50_95": per_class}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True, help="output weights name, e.g. seatbelt_custom")
    ap.add_argument("--model", default="yolov8s.pt", help="base weights (yolov8s.pt / yolov8m.pt / yolo11s.pt)")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = auto for the GPU")
    ap.add_argument("--old", help="old weights to compare on the same test split (same class order only)")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        patience=30, cos_lr=True, close_mosaic=15, cache=True,
        # traffic-camera style augmentation: no vertical flips, mild scale/colour changes
        fliplr=0.5, flipud=0.0, degrees=3, scale=0.5, translate=0.1, hsv_v=0.4, mixup=0.1,
        project=os.path.join(ROOT, "runs"), name=args.name, exist_ok=True, plots=True,
    )
    best = os.path.join(ROOT, "runs", args.name, "weights", "best.pt")
    dst = os.path.join(ROOT, "trainedmodels", f"{args.name}.pt")
    shutil.copy(best, dst)
    print("✔ installed", dst)

    with open(args.data) as f:
        split = "test" if "test:" in f.read() else "val"
    rows = [("new", dst, evaluate(dst, args.data, split))]
    if args.old:
        try:
            rows.append(("old", args.old, evaluate(args.old, args.data, split)))
        except Exception as e:
            print("old model could not be evaluated on this data:", e)

    out = os.path.join(ROOT, "training", "metrics.csv")
    new_file = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["date", "name", "version", "weights", "split", "precision", "recall",
                        "mAP50", "mAP50_95", "per_class_mAP50_95"])
        for tag, weights, r in rows:
            w.writerow([date.today(), args.name, tag, os.path.basename(weights), split, r["precision"],
                        r["recall"], r["mAP50"], r["mAP50_95"], r["per_class_mAP50_95"]])
            print(f"{tag:>4} {os.path.basename(weights)} [{split}] {r}")


if __name__ == "__main__":
    main()
