# Improvements: violation ↔ number-plate linking, plate accuracy, retrained models

## What changed (summary)

| Area | Before | After |
|---|---|---|
| **Plate per violation** | Only helmet tried to read a plate (OCR on a 180 px box around the head); every other module saved `UNKNOWN` | Every module tracks each vehicle (ByteTrack), links the violation to that vehicle, reads **that vehicle's** plate over several frames, and saves it in `vehicle_data.number_plate` |
| **Plate accuracy** | One frame, raw crop, "longest OCR string wins", no format check in most modules | Plate-specific OCR models (fast-plate-ocr) with EasyOCR as fallback, crop enhancement (upscale, denoise, CLAHE, deskew, sharpen), two-line motorcycle plates joined in the right order, Indian format check (state codes + standard + BH series), position-aware fixes (`O→0` only in digit slots, `8→B` only in letter slots), per-character voting across frames |
| **Blurry / far plates** | Wrong number saved | `Plate not readable` saved when reads are missing or disagree |
| **Duplicates** | One row per detection burst (same vehicle stored many times) | One row per violating vehicle; the same plate is never stored twice per video |
| **Helmet** | Any helmet anywhere in the frame silenced all violations for 50 frames; hard-coded `C:\capstone_project\...` model path | Retrained model; decided per rider on each motorcycle; relative paths |
| **Triple riding** | `triple_riding.py` used the generic COCO `yolov8n.pt` and counted every person box over a bike in a single frame, so duplicate or overlapping person boxes made 2-rider bikes look like 3 | Counts riders on each tracked bike with de-duplicated person boxes and strict rider geometry, confirmed over several frames of the same bike |
| **Wrong lane** | Single-frame classifier: it cannot know which way a vehicle is moving | Each vehicle's direction is measured from its track over ~1 s and compared with the traffic flow near it (learned from the other vehicles, or set per camera with a calibration file). On a moving / handheld camera, where direction cannot be measured, the retrained model decides |
| **Phone usage** | `phone_usage.py` loaded `yolov8s.pt` (COCO "cell phone"); the trained `phoneusage.pt` was unused | Retrained model for car drivers **and motorcycle riders**, linked to the vehicle + plate |
| **Seatbelt** | Frame-level decision, model class names swapped, plate `UNKNOWN` | Retrained model plus a "no belt visible on a clearly visible driver" rule, decided per vehicle, plate linked |
| **Accident** | Every confident frame stored (13 false alerts on 5 normal clips, the same crash stored many times) | Retrained model; a crash is reported once, only when the model stays confident for 1.2 s |
| **Evidence** | Tight crop of the head / box | Crop of the vehicle and occupants with the boxes drawn, the violation and the plate written on the image; an annotated video and a CSV report per run |

The web app (`app.py`), the per-violation dropdown and the results page work as before.
They call the same function names, and the results page already shows `number_plate`.

## Files

```
vision/settings.py      all paths/thresholds (env-overridable, no machine-specific paths)
vision/models.py        model loading + class lookup BY NAME (no hard-coded class ids)
vision/plates.py        plate detection, enhancement, OCR, Indian format fixing, voting
vision/pipeline.py      tracking → plate collection → per-vehicle confirmation → DB row
vision/violations.py    Helmet / TripleRiding / WrongLane / PhoneUsage / Seatbelt / ANPR
helmet_detection.py, triple_riding.py, wrong_lane.py, phone_usage.py, seatbelt_detection.py,
accident_detection.py, anpr_video.py, plate_reader_yolo.py   entry points (same function names)
legacy/                 the original scripts, unchanged, for the before/after comparison
tools/before_after.py   runs original vs new on the sample videos → before_after/summary.md
tools/smoke_test.py     installation check (models, plate reader, one module end to end)
tools/calibrate_lanes.py  learns a fixed camera's lane directions (wrong lane)
tools/inspect_models.py prints class names of every .pt
training/               dataset merge + training scripts, test metrics of the delivered models
tests/                  unit tests (plate format, correction, voting, class matching)
create_tables.py        schema fixed to match the columns dbconnection writes; hashed logins
```

**Database:** `create_tables.py` created columns (`process_type`, `image_path`, …) that
`dbconnection.insert_detection` never writes, so on a fresh database every insert failed.
It now creates `module`, `image_filename`, `extra_info`, and the `login_details` table with
hashed passwords (`python create_tables.py --add-user admin`). An existing database that
already works needs no change.

## Models: original vs delivered

Both scored on the same held-out test images (mAP50: 1.0 = every object found and boxed correctly).

| Violation | Original model | Delivered model | Original | Delivered |
|---|---|---|---|---|
| Helmet | `helmet_best.pt` / `helmet.pt` / `triple.pt` | `helmet_v3.pt` (YOLOv8m, 12,534 images incl. traffic-camera footage) | 0.19 / 0.30 / 0.29 | **0.84** |
| Phone usage | `phoneusage.pt` | `phone_v3.pt` (YOLOv8m, 17,163 images: drivers, riders, look-alikes such as hands on the wheel) | 0.00 | **0.97** |
| Seatbelt | `seatbelt_yolov8.pt` / `seatbelt.pt` | `seatbelt_v2.pt` | 0.14 / 0.19 | **0.97** |
| Number plate | `numberplate.pt` / `plate.pt` | `numberplate_v2.pt` | 0.76 / 0.75 | **0.97** |
| Wrong lane | `wrong_lane_yolov8.pt` | direction from tracking + `wrong_way_v2.pt` for moving cameras | 0.99 | 0.99 (equal on images; on video the original gave 5 false alarms on 8 clips, the delivered version 0) |
| Accident | `accident_best.pt` | `accident_v3.pt` | 0.38 | **0.95** |
| Triple riding | COCO person count per frame | rider counting per tracked bike (no extra model) | – | – (see video results) |

A high test score is not enough on its own: every module was also run on the sample videos and
the evidence images were checked by eye (next section). Test metrics and training curves of the
delivered models are in `training/results/`.

## Results on the sample videos

Original scripts (`legacy/`) vs the delivered version, same videos. Regenerate with
`python tools/before_after.py` (writes `before_after/summary.md` and the evidence images).

| Module | Original | Delivered |
|---|---|---|
| Helmet | helmet clip: 1 rider found, stored 5 times, no plate; with-helmet clip: 0 | helmet clip: 5 different bare-headed riders; on 4 other rider clips 6 more bare-headed riders. 1 false alarm (a rider in a dark helmet in `with helmet.mp4`) |
| Triple riding | 8 records on 3 clips, including 2-rider bikes | 1 correct (the 3-rider bike), 0 false alarms on 5 clips (misses one partly hidden 3-rider scooter) |
| Wrong lane | 2 records on the wrong-way clip; per-frame classifier, no direction | wrong-way rider caught (1 record); 0 false alarms on 7 normal clips (CCTV, ATCC, ANPR, traffic, helmet and triple-riding clips) |
| Phone usage | 13 records for one driver, no plate | the driver caught (2 records: two camera shots); 0 false alarms on 7 other clips |
| Seatbelt | unbelted clip: 3 records for one driver; class names swapped | belted clip: 0; unbelted clip: 1 correct; 0 on the ANPR clip |
| ANPR | 2 plates: `KA02MM9091` correct, `KA02HN1826` wrong (real: KA02MN1826) | all 3 plates correct: `KA02MN1826`, `KA02MM9091`, `KA02MH7256` |
| Accident | 13 false alerts on 5 normal clips; the same crash stored many times | 5 of the ~8 crashes in `accident.mp4`, one record each, crash boxed; 0 false alerts on 10 normal clips |

Plate reading, 79 real plate crops from `anpr sample.mp4` (3 plates, many frames), one read per crop:

| Reader | exactly right | wrong number | unreadable |
|---|---|---|---|
| EasyOCR (original) | 6 | 13 | 60 |
| fast-plate-ocr (delivered) | 51–60 | 1–7 | 12–27 |

With multi-frame voting (the default) all three plates are read correctly and no wrong number is
produced. Synthetic plates (12 per condition, a third two-line): 160 px sharp 12/12, 110 px motion
blur 12/12 (EasyOCR: 8 right, 1 wrong), 80 px heavy blur: 0 read and 0 wrong.
Set `PLATE_OCR=easyocr` to use EasyOCR.

## Tuning (environment variables or `.env`)

| Variable | Default | Meaning |
|---|---|---|
| `FRAME_STRIDE` | 2 | analyse every Nth frame |
| `CONFIRM_HITS` / `CONFIRM_WINDOW` | 3 / 6 | violation must be seen in 3 of the vehicle's last 6 analysed frames |
| `PLATE_MIN_VOTES` | 2 | agreeing reads needed before a plate is accepted |
| `PLATE_OCR` | fast | `fast`, `easyocr`, `both` |
| `NO_HELMET_CONF`, `TRIPLE_CONF`, `PHONE_CONF`, `NO_SEATBELT_CONF` | 0.45–0.5 | detection thresholds |
| `TRIPLE_MODE` | count | `count` (count riders on each tracked bike), `assist` (3 counted, or the original triple model + 2 counted), `model`, `both`, `auto` |
| `WRONG_LANE_MODE` | either | `either` (fixed camera: direction from tracking; moving camera: model), `direction`, `model`, `both` |
| `WRONG_LANE_ZONES` | – | per-camera calibration file (below); without it the flow is learned from the traffic |
| `WRONG_LANE_NEAR` | 1.0 | a vehicle is compared with vehicles that passed within this many vehicle sizes of it |
| `MOVING_CAM_PX` | 1.5 | background shift per frame above which the camera counts as moving |
| `SEATBELT_ABSENCE_RULE` | 1 | also flag a clearly visible driver on whom no belt is found over several frames |
| `PHONE_CONFIRM_COCO` | 0 | 1 = a phone must also be seen by the generic COCO model (stricter, misses riders) |
| `ACCIDENT_CONF` / `ACCIDENT_MIN_SEC` | 0.6 / 1.2 | a crash is reported when the model stays this confident for this long |
| `VEHICLE_MODEL` | yolov8s.pt | e.g. `trainedmodels/vehicledetection.pt` (car/motorcycle only) |
| `SAVE_ANNOTATED_VIDEO` | 1 | write `annotated_<module>_<video>.mp4` |
| `YOLO_DEVICE` | auto | `cpu`, `0` … |

## Wrong lane: camera calibration

* `python tools/calibrate_lanes.py camera7.mp4` learns the normal direction of every road area of a
  fixed camera from a few minutes of normal traffic and saves `lane_zones/camera7.json` (+ a preview
  with arrows). With it, a wrong-way vehicle is flagged from the first vehicle, also on quiet roads.
  Without it, directions are learned from each video's own traffic (needs ≥ 5 vehicles nearby).
* Camera movement (pan, zoom, a handheld or following camera) is detected; positions recorded
  while the view moves are ignored and the retrained model decides instead.

Zones file format (coordinates are 0–1 fractions of the frame):

```json
[
  {"polygon": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]], "direction": "down"},
  {"polygon": [[0.5, 0], [1, 0], [1, 1], [0.5, 1]], "direction": "up"}
]
```

## Retraining on your own footage

Training needs a GPU (Google Colab, Kaggle or a local NVIDIA card):

```bash
export ROBOFLOW_API_KEY=...
python training/prepare_dataset.py --config training/seatbelt.yaml   # merge sources, unify class names
python training/train.py --data dsets/seatbelt/data.yaml --name seatbelt_custom
# then set SEATBELT_MODEL=trainedmodels/seatbelt_custom.pt in .env
```

`train.py` copies the weights to `trainedmodels/<name>.pt` and records test-split mAP in
`training/metrics.csv`. Adding a few hundred labelled frames from your own cameras as a `path:`
source helps the most on your footage.

## Known limits

* Tested on the sample videos and public test sets only. Footage from your own cameras (angle,
  night, rain) is the real test; a few hundred labelled frames from it is the best next improvement.
* Helmet: a rider in a dark helmet can be taken for a bare head (1 false alarm on the samples).
  If the tracker briefly loses a rider, the same rider can be recorded twice.
* Triple riding: a third rider hidden behind the others is not counted.
* Seatbelt: the decision relies mainly on seeing (or not seeing) the belt on a clearly visible
  driver; tinted glass, night footage and a hidden chest make it uncertain. Only people sitting
  inside a detected vehicle count as drivers (pedestrians next to a car, standing people and
  two-wheeler riders are ignored): on a street dashcam clip this removed 4 false alarms while the
  unbelted driver in the sample is still caught.
* Accident: a crash is reported only when the model stays confident for 1.2 s, so an alert comes
  about a second after the impact, and partly hidden or night crashes can be missed (5 of ~8 caught
  on the sample). This setting was tuned on the sample videos; check it on your own footage.
* Plates narrower than ~80 px or with strong motion blur are not readable in a single frame. The
  system then says so instead of guessing. Higher resolution or a camera closer to the road fixes this.
* Speed: about 4 analysed frames/s on a 4-core CPU at 720p; with `FRAME_STRIDE=2` a 25 fps video runs
  at about 1/3 real time. The helmet and phone models are larger and take 2–3× longer on CPU. An
  NVIDIA GPU (`YOLO_DEVICE=0`) is many times faster.
