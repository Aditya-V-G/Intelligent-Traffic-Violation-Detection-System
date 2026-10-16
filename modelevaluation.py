from ultralytics import YOLO
import time
import csv
import os
from pathlib import Path

model_dir = r"C:/capstone_project/ANPR-and-ATCC-for-Smart-Traffic-Management/trainedmodels/"
dataset_dir = r"C:/capstone_project/ANPR-and-ATCC-for-Smart-Traffic-Management/dsets/"

models_and_yaml = {
    "accident.pt": dataset_dir + "accident_best/data.yaml",
    "helmetdetection.pt": dataset_dir + "helmet_detection/data.yaml",
    "lane.pt": dataset_dir + "lane/data.yaml",
    "numberplate.pt": dataset_dir + "numberplate/data.yaml",
    "phoneusage.pt": dataset_dir + "phoneusage/data.yaml",
    "seatbelt.pt": dataset_dir + "seatbelt/data.yaml",
    "vehicledetection.pt": dataset_dir + "vehicledetection/data.yaml",
}

# Output CSV
out_csv = Path("evaluation_results.csv")
header = [
    "model",
    "data_yaml",
    "mp_precision",
    "mr_recall",
    "f1 (computed)",
    "mAP@50",
    "mAP@50:0.95",
    "val_images",
    "val_instances",
    "fps",
    "status",
    "notes"
]

# prepare CSV
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

for model_name, yaml_path in models_and_yaml.items():
    print("\n" + "=" * 60)
    print(f" Evaluating Model: {model_name}")
    print("=" * 60)
    print(f" Using dataset: {yaml_path}")

    row = [model_name, yaml_path, "", "", "", "", "", "", "", "", "FAILED", ""]

    try:
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML not found: {yaml_path}")

        model = YOLO(model_path)

        start = time.time()
        results = model.val(data=yaml_path)   # runs full validation (mAP etc.)
        end = time.time()

        # Ultralytics Metric object: use available attributes
        mp = getattr(results.box, "mp", None)    # mean precision
        mr = getattr(results.box, "mr", None)    # mean recall
        map50 = getattr(results.box, "map50", None)
        map_all = getattr(results.box, "map", None)  # mAP@0.5:0.95

        # compute F1 from mp and mr if available
        f1 = None
        if mp is not None and mr is not None and (mp + mr) > 0:
            f1 = 2 * (mp * mr) / (mp + mr)
        elif hasattr(results.box, "f1"):
            # fallback: try mean of per-class f1 list if provided
            f_list = getattr(results.box, "f1", None)
            if f_list:
                try:
                    f1 = sum(f_list) / len(f_list)
                except Exception:
                    f1 = None

        # get basic dataset stats if printed in results (best-effort)
        val_images = ""
        val_instances = ""
        try:
            # results.box.mean_results() returns (mp, mr, map50, map) but not counts.
            # The validator prints counts; sometimes results.files or results[0].orig_shape may exist.
            # We try to extract from results.args or results.files if present.
            # This is best-effort and may remain empty on some versions.
            if hasattr(results, "args") and getattr(results.args, "data", None):
                # no reliable image/instance count available from results consistently
                pass
        except Exception:
            pass

        fps = 1.0 / max((end - start), 1e-9)

        # print to console (friendly)
        print(f" mp (precision) : {mp if mp is not None else 'N/A'}")
        print(f" mr (recall)    : {mr if mr is not None else 'N/A'}")
        print(f" F1 (computed)  : {f1 if f1 is not None else 'N/A'}")
        print(f" mAP@50         : {map50 if map50 is not None else 'N/A'}")
        print(f" mAP@50:0.95    : {map_all if map_all is not None else 'N/A'}")
        print(f" FPS (approx)   : {fps:.2f}")

        # success row
        row[2] = f"{mp:.4f}" if mp is not None else ""
        row[3] = f"{mr:.4f}" if mr is not None else ""
        row[4] = f"{f1:.4f}" if f1 is not None else ""
        row[5] = f"{map50:.4f}" if map50 is not None else ""
        row[6] = f"{map_all:.4f}" if map_all is not None else ""
        row[7] = val_images
        row[8] = val_instances
        row[9] = f"{fps:.2f}"
        row[10] = "OK"
        row[11] = ""

    except Exception as e:
        print(f" ERROR evaluating {model_name}: {e}")
        row[11] = str(e)

    # append to csv
    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

print("\nAll done. Results saved to:", out_csv.resolve())
