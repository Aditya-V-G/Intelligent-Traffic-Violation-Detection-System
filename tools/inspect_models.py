"""Print the class names of every weights file (the pipeline matches classes by name)."""
import glob
import os
import sys

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in sorted(glob.glob(os.path.join(ROOT, "**", "*.pt"), recursive=True)):
    if os.path.getsize(path) < 1000:
        print(f"{os.path.relpath(path, ROOT):45s} Git LFS pointer - run `git lfs pull`")
        continue
    try:
        print(f"{os.path.relpath(path, ROOT):45s} {YOLO(path).names}")
    except Exception as e:
        print(f"{os.path.relpath(path, ROOT):45s} ERROR {e}", file=sys.stderr)
