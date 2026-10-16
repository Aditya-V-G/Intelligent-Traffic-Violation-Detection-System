# Intelligent Traffic Violation Detection System

Detects helmet, triple riding, phone usage, seatbelt, wrong lane and accident violations in
traffic video, reads the number plate of each violating vehicle, and stores the results in MySQL
with an evidence image. ANPR (plate reading) and ATCC (vehicle counting) are also available.
What changed in this version and how each model was tested: **IMPROVEMENTS.md**.

## Quick start

Tested with Python 3.10–3.11, MySQL 8 / MariaDB 10+, Windows and Linux. A GPU is optional (CPU works, slower).

```bash
# 1. code and models (the delivery zip already contains everything below)
git lfs pull                              # only when working from git: sample videos and original .pt files
#    the 6 retrained models are in trainedmodels/ (not stored in git; keep them when updating):
#    helmet_v3.pt phone_v3.pt seatbelt_v2.pt numberplate_v2.pt wrong_way_v2.pt accident_v3.pt
pip install -r requirements.txt

# 2. configuration
cp .env.example .env                      # then edit .env: database login + FLASK_SECRET_KEY

# 3. database (creates the tables, then a web login; you are asked for the password)
python create_tables.py --add-user admin

# 4. check the installation (no database needed, 2-5 min on CPU)
python tools/smoke_test.py                # must end with "ALL CHECKS PASSED"

# 5. run
python app.py                                     # development
python -m waitress --port=5000 app:app            # production web server
```

Open http://127.0.0.1:5000, log in, choose a violation type, upload a video. Results (plate,
evidence image, time in video) appear on the Results page; an annotated video and a CSV report
are written next to the evidence images in `static/uploads/detections/`.

## Useful commands

| Command | Purpose |
|---|---|
| `python tools/before_after.py` | old vs new results on the sample videos → `before_after/summary.md` |
| `python tools/inspect_models.py` | class names of every model file |
| `python tools/calibrate_lanes.py camera.mp4` | learn a fixed camera's lane directions once (wrong-lane check) |
| `python -m pytest tests` | unit tests (plate format, correction, voting, class matching) |

Settings (thresholds, model paths, OCR engine, GPU/CPU) are environment variables, listed in
`.env.example` and explained in IMPROVEMENTS.md.

## Notes for production

* Processing runs inside the web request: a long video keeps the browser waiting until it
  finishes. For many or long videos, run the modules from a job queue or a scheduled script
  (each module is a plain function, e.g. `from helmet_detection import main_fun`).
* Speed on CPU is about 1/3 real time at 720p (the helmet and phone modules use larger models
  and take 2–3× longer); an NVIDIA GPU (`YOLO_DEVICE=0`) is much faster.
* Accuracy was measured on the sample videos and public test sets. Before relying on it,
  check a few hours of your own camera footage; IMPROVEMENTS.md lists the known limits.
