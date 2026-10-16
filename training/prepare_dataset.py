"""
Merge one or more YOLO-format datasets (e.g. Roboflow "YOLOv8" exports) into a single
dataset with unified class names, ready for training.

    python training/prepare_dataset.py --config training/seatbelt.yaml
    python training/prepare_dataset.py --config training/phone.yaml

The config lists the target classes, the sources (local folders or Roboflow projects)
and how each source class maps to a target class (null = drop those boxes).
"""

import argparse
import os
import random
import shutil

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def download_roboflow(src, dest):
    from roboflow import Roboflow
    key = os.getenv("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("Set ROBOFLOW_API_KEY (free key: roboflow.com -> Settings -> API Keys)")
    ws, project, version = src["roboflow"].split("/")
    ds = Roboflow(api_key=key).workspace(ws).project(project).version(int(version)).download(
        "yolov8", location=dest, overwrite=False)
    return ds.location


def load_names(folder):
    with open(os.path.join(folder, "data.yaml")) as f:
        names = yaml.safe_load(f)["names"]
    return names if isinstance(names, list) else [names[k] for k in sorted(names)]


def find_split_dirs(folder):
    """Roboflow uses train/valid/test, others use images/train ... return {split: (img_dir, lbl_dir)}."""
    out = {}
    for split, aliases in {"train": ["train"], "val": ["valid", "val"], "test": ["test"]}.items():
        for a in aliases:
            for img, lbl in [(f"{a}/images", f"{a}/labels"), (f"images/{a}", f"labels/{a}")]:
                if os.path.isdir(os.path.join(folder, img)):
                    out[split] = (os.path.join(folder, img), os.path.join(folder, lbl))
                    break
            if split in out:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    targets = cfg["classes"]
    out = cfg["output"]
    raw_dir = cfg.get("download_dir", "dsets/raw")
    random.seed(0)
    if os.path.exists(out):
        shutil.rmtree(out)
    stats = {t: 0 for t in targets}
    n_images = {"train": 0, "val": 0, "test": 0}

    for i, src in enumerate(cfg["sources"]):
        folder = src.get("path") or download_roboflow(src, os.path.join(raw_dir, src["roboflow"].replace("/", "_")))
        names = load_names(folder)
        mapping = src.get("map", {})
        missing = [n for n in names if n not in mapping]
        if missing:
            raise SystemExit(f"{folder}: add these classes to 'map' (target name or null): {missing}")
        remap = {idx: (targets.index(mapping[n]) if mapping[n] is not None else None)
                 for idx, n in enumerate(names)}
        print(f"source {i}: {folder}\n   {dict(zip(names, (mapping[n] for n in names)))}")

        splits = find_split_dirs(folder)
        for split, (img_dir, lbl_dir) in splits.items():
            for fname in os.listdir(img_dir):
                stem, ext = os.path.splitext(fname)
                if ext.lower() not in IMG_EXT:
                    continue
                # sources without val/test: hold out 10% / 10%
                dst_split = split
                if split == "train" and "val" not in splits:
                    r = random.random()
                    dst_split = "val" if r < 0.1 else "test" if r < 0.2 and "test" not in splits else "train"
                lines = []
                lbl = os.path.join(lbl_dir, stem + ".txt")
                if os.path.exists(lbl):
                    for line in open(lbl).read().split("\n"):
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        new = remap.get(int(parts[0]))
                        if new is None:
                            continue
                        lines.append(" ".join([str(new)] + parts[1:]))
                        stats[targets[new]] += 1
                if not lines and not src.get("keep_background", True):
                    continue
                os.makedirs(os.path.join(out, "images", dst_split), exist_ok=True)
                os.makedirs(os.path.join(out, "labels", dst_split), exist_ok=True)
                new_stem = f"s{i}_{stem}"
                shutil.copy(os.path.join(img_dir, fname), os.path.join(out, "images", dst_split, new_stem + ext))
                with open(os.path.join(out, "labels", dst_split, new_stem + ".txt"), "w") as f:
                    f.write("\n".join(lines))
                n_images[dst_split] += 1

    data = {"path": os.path.abspath(out), "train": "images/train", "val": "images/val",
            "names": {i: n for i, n in enumerate(targets)}}
    if n_images["test"]:
        data["test"] = "images/test"
    with open(os.path.join(out, "data.yaml"), "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"\n✔ {out}/data.yaml | images {n_images} | boxes {stats}")


if __name__ == "__main__":
    main()
