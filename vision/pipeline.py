# vision/pipeline.py
"""
Shared video pipeline for every violation module.

  frame -> ByteTrack vehicle tracking -> plate detection assigned to tracks
        -> module-specific violation check (subclass .evaluate)
        -> per-track temporal confirmation (k hits in last n frames)
        -> when the track ends: vote the plate, save evidence, write ONE DB row
           per violating vehicle with its number plate (or "Plate not readable")

Subclasses implement:
    module, label, vehicle_types
    setup(self)                       load extra models
    evaluate(self, frame, tracks)     -> {track_id: Observation}
"""

import csv
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import cv2

from vision import settings
from vision.models import load, vehicle_type_map
from vision.plates import (PlateDetector, PlateVoter, crop_quality, crop_with_margin,
                           format_plate, ocr_crop)

try:
    import dbconnection
except Exception as e:          # DB is optional (offline runs / before-after tests)
    print("DB helper unavailable:", e)
    dbconnection = None

PLATE_NOT_READABLE = settings.PLATE_NOT_READABLE
MAX_PLATE_CROPS = 8          # best crops kept per track
MAX_OCR_PER_TRACK = 16


# ---------------------------------------------------------------
# GEOMETRY
# ---------------------------------------------------------------
def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def intersection(a, b):
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def iou(a, b):
    i = intersection(a, b)
    return i / max(1, area(a) + area(b) - i)


def ioa(inner, outer):
    """Fraction of `inner` that lies inside `outer`."""
    return intersection(inner, outer) / max(1, area(inner))


def center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def rider_zone(box, vtype):
    """Region where the occupants of a vehicle appear (riders sit above a bike's box)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if vtype == "motorcycle":
        return (x1 - 0.25 * w, y1 - 1.3 * h, x2 + 0.25 * w, y2)
    return (x1 - 0.05 * w, y1 - 0.05 * h, x2 + 0.05 * w, y2 + 0.05 * h)


# ---------------------------------------------------------------
# DATA
# ---------------------------------------------------------------
@dataclass
class Observation:
    hit: bool
    conf: float = 0.0
    box: tuple = None            # evidence box (head, riders, driver ...)
    info: str = ""


@dataclass
class Track:
    tid: int
    vtype: str
    box: tuple
    first_frame: int
    last_frame: int
    history: deque = field(default_factory=lambda: deque(maxlen=120))   # (frame, cx, cy, box)
    hits: deque = field(default_factory=lambda: deque(maxlen=settings.CONFIRM_WINDOW))
    confirmed: bool = False
    best: Observation = None
    evidence: object = None      # image region saved at confirmation
    evidence_frame: int = 0
    plate_crops: list = field(default_factory=list)    # [quality, crop, already_ocrd]
    voter: PlateVoter = field(default_factory=PlateVoter)
    ocr_calls: int = 0
    extra: dict = field(default_factory=dict)          # per-module state

    @property
    def candidate(self):
        return self.confirmed or any(self.hits)


# ---------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------
class ViolationPipeline:
    module = "violation"                 # value stored in vehicle_data.module
    label = "Violation"
    vehicle_types = {"car", "motorcycle", "bus", "truck", "auto"}
    read_plates_for_all = False          # ANPR: OCR every vehicle, not just violators

    def __init__(self):
        self.vehicle_model = load(settings.VEHICLE_MODEL)
        self.vehicle_classes = {c: t for c, t in vehicle_type_map(self.vehicle_model).items()
                                if t in self.vehicle_types}
        self.plates = PlateDetector()
        self.setup()

    # ---- to override ------------------------------------------------
    def setup(self):
        pass

    def evaluate(self, frame, tracks):
        """Return {track_id: Observation} for tracks seen in this frame."""
        raise NotImplementedError

    def accept(self, track):
        """Final per-vehicle check before a violation is recorded (override in subclasses)."""
        return True

    def describe(self, track):
        """Extra info stored with the DB row."""
        b = track.best
        return f"conf={b.conf:.2f}" + (f", {b.info}" if b and b.info else "") if b else ""

    # ---- helpers for subclasses ----------------------------------------
    def link_boxes(self, boxes, tracks, min_overlap=0.5):
        """
        Assign evidence boxes [(x1,y1,x2,y2,conf,info)] to vehicle tracks.
        Returns {tid: [box, ...]}. A box goes to the track whose occupant zone contains
        most of it; ties go to the closest vehicle.
        """
        linked = {}
        for b in boxes:
            best, best_key = None, None
            for t in tracks:
                score = ioa(b[:4], rider_zone(t.box, t.vtype))
                if score < min_overlap:
                    continue
                bc, tc = center(b), center(t.box)
                key = (round(score, 1), -abs(bc[0] - tc[0]))
                if best_key is None or key > best_key:
                    best, best_key = t, key
            if best is not None:
                linked.setdefault(best.tid, []).append(b)
        return linked

    def orphan(self, box, vtype, frame_shape):
        """
        Pseudo-track for evidence that no detected vehicle contains (close-up footage where
        the vehicle fills the frame). Matched frame-to-frame by IoU; negative ids.
        Its 'vehicle box' extends down to the frame bottom so the plate below is assigned.
        """
        H = frame_shape[0]
        vbox = (int(box[0]), int(box[1]), int(box[2]), H)
        # hands and phones move a lot between frames: match by overlap OR by position
        W = frame_shape[1]
        best, best_score = None, 0.0
        for t in self.tracks.values():
            if t.tid < 0 and t not in self._orphans:
                ov = iou(t.extra["ebox"], box[:4])
                dist = abs(center(t.extra["ebox"])[0] - center(box)[0]) / W
                score = ov + (1.0 - dist if dist < 0.35 else 0.0)
                if score > best_score:
                    best, best_score = t, score
        if best is None:
            self._next_orphan = getattr(self, "_next_orphan", 0) - 1
            best = self.tracks[self._next_orphan] = Track(self._next_orphan, vtype, vbox,
                                                          self.frame_no, self.frame_no)
        best.box, best.last_frame = vbox, self.frame_no
        best.extra["ebox"] = tuple(box[:4])
        best.history.append((self.frame_no, *center(vbox), vbox))
        self._orphans.append(best)
        return best

    # ---- main loop -----------------------------------------------------
    def run(self, video_path, out_folder="static/uploads/detections"):
        os.makedirs(out_folder, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Cannot open video:", video_path)
            return []

        self.video_file = os.path.basename(video_path)
        self.out_folder = out_folder
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25
        self.tracks, self.records, self._recorded = {}, [], []
        writer = None
        frame_no, analysed = 0, 0
        print(f"\n▶ {self.label}: {self.video_file}")

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            if frame_no % settings.FRAME_STRIDE:
                continue
            analysed += 1

            self.frame_no, self._orphans = frame_no, []
            active = self._track(frame, frame_no, first=(analysed == 1))
            observations = self.evaluate(frame, active)
            active = active + self._orphans
            if active and analysed % settings.PLATE_EVERY_N == 0:
                self._collect_plates(frame, active)

            for t in active:
                obs = observations.get(t.tid, Observation(False))
                self._update_violation(t, obs, frame, frame_no)

            self._finalise_lost(frame_no)

            if settings.SAVE_ANNOTATED_VIDEO:
                if writer is None:
                    h, w = frame.shape[:2]
                    name = f"annotated_{self.module}_{os.path.splitext(self.video_file)[0]}.mp4"
                    self.annotated_path = os.path.join(out_folder, name)
                    writer = cv2.VideoWriter(self.annotated_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                             self.fps / settings.FRAME_STRIDE, (w, h))
                writer.write(self._annotate(frame, active))

        for tid in list(self.tracks):
            self._finalise(tid)
        cap.release()
        if writer is not None:
            writer.release()
        self._write_report()
        print(f"✅ {self.label} done: {len(self.records)} violation(s)")
        return self.records

    # ---- tracking ---------------------------------------------------------
    def _track(self, frame, frame_no, first):
        if not self.vehicle_classes:
            return []
        res = self.vehicle_model.track(frame, persist=not first, tracker=settings.TRACKER_CFG,
                                       classes=list(self.vehicle_classes), conf=settings.VEHICLE_CONF,
                                       device=settings.DEVICE or None, verbose=False)[0]
        active = []
        if res.boxes is None or res.boxes.id is None:
            return active
        for box, cls, tid in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.id.tolist()):
            tid = int(tid)
            box = tuple(int(v) for v in box)
            t = self.tracks.get(tid)
            if t is None:
                t = self.tracks[tid] = Track(tid, self.vehicle_classes[int(cls)], box, frame_no, frame_no)
            t.box, t.last_frame = box, frame_no
            cx, cy = center(box)
            t.history.append((frame_no, cx, cy, box))
            active.append(t)
        return active

    # ---- plates -------------------------------------------------------------
    def _collect_plates(self, frame, active):
        wanted = [t for t in active if self.read_plates_for_all or t.candidate]
        if not wanted:
            # still remember crops for every vehicle: the plate may only be visible before
            # the violation is (e.g. a bike approaching the camera, then passing)
            wanted = active
        for x1, y1, x2, y2, conf in self.plates.detect(frame):
            pc = center((x1, y1, x2, y2))
            owners = [t for t in wanted
                      if t.box[0] <= pc[0] <= t.box[2] and t.box[1] <= pc[1] <= t.box[3] + 0.1 * (t.box[3] - t.box[1])]
            if not owners:
                continue
            t = min(owners, key=lambda o: area(o.box))     # innermost vehicle
            crop = crop_with_margin(frame, (x1, y1, x2, y2))
            q = crop_quality(crop, conf)
            if q <= 0:
                continue
            if len(t.plate_crops) < MAX_PLATE_CROPS or q > t.plate_crops[-1][0]:
                entry = [q, crop, False]
                t.plate_crops.append(entry)
                t.plate_crops.sort(key=lambda c: -c[0])
                del t.plate_crops[MAX_PLATE_CROPS:]
                if (self.read_plates_for_all or t.candidate) and t.ocr_calls < MAX_OCR_PER_TRACK:
                    self._ocr(t, entry)        # live read (budgeted) so the annotated video shows plates

    def _ocr(self, t, entry):
        t.ocr_calls += 1
        entry[2] = True
        for plate, w, _raw in ocr_crop(entry[1]):
            t.voter.add(plate, w)

    def _resolve_plate(self, t):
        # Always read the best stored crops that were not read yet. The live budget is often
        # spent on small early crops while the vehicle is far away; the close-up crops that
        # arrive later are the ones that matter.
        for entry in t.plate_crops:
            if not entry[2]:
                self._ocr(t, entry)
        plate, info = t.voter.result()
        return plate, info

    # ---- violations ----------------------------------------------------------
    def _update_violation(self, t, obs, frame, frame_no):
        t.hits.append(bool(obs.hit))
        if obs.hit and (t.best is None or obs.conf >= t.best.conf):
            t.best = obs                                  # keep the clearest evidence
            t.evidence = self._evidence_image(frame, t, obs)
            t.evidence_frame = frame_no
        if not t.confirmed and sum(t.hits) >= settings.CONFIRM_HITS:
            t.confirmed = True
        if t.confirmed and t.evidence is not None and t.extra.get("printed") is None:
            t.extra["printed"] = True
            print(f"🚨 {self.label} | vehicle #{t.tid} ({t.vtype}) | frame {frame_no}")

    def _evidence_image(self, frame, t, obs):
        img = frame.copy()
        cv2.rectangle(img, t.box[:2], t.box[2:4], (0, 200, 255), 2)
        if obs.box is not None:
            b = tuple(int(v) for v in obs.box[:4])
            cv2.rectangle(img, b[:2], b[2:4], (0, 0, 255), 2)
        zx1, zy1, zx2, zy2 = rider_zone(t.box, t.vtype)
        if obs.box is not None:
            zx1, zy1 = min(zx1, obs.box[0]), min(zy1, obs.box[1])
            zx2, zy2 = max(zx2, obs.box[2]), max(zy2, obs.box[3])
        H, W = img.shape[:2]
        if t.tid < 0:
            # close-up pseudo-track: its box is only as wide as the evidence (e.g. a hand + phone),
            # which gave thin strips; show at least 3x that width / 40 % of the frame around it
            cx, half = (zx1 + zx2) / 2, max(1.5 * (zx2 - zx1), 0.2 * W)
            zx1, zx2 = min(zx1, cx - half), max(zx2, cx + half)
        pad = 40
        return img[max(0, int(zy1) - pad):min(H, int(zy2) + pad), max(0, int(zx1) - pad):min(W, int(zx2) + pad)]

    # ---- finalisation ---------------------------------------------------------
    def _finalise_lost(self, frame_no):
        for tid, t in list(self.tracks.items()):
            # close-up pseudo-tracks: the subject is often hidden for a few seconds
            limit = settings.TRACK_LOST_FRAMES if tid >= 0 else int(5 * self.fps)
            if frame_no - t.last_frame > limit:
                self._finalise(tid)

    def _finalise(self, tid):
        t = self.tracks.pop(tid)
        if not t.confirmed or t.evidence is None or not self.accept(t):
            return
        plate, plate_info = self._resolve_plate(t)
        plate_text = plate or PLATE_NOT_READABLE
        if plate and any(r["number_plate"] == plate for r in self.records):
            return      # same vehicle re-entered / tracker ID switch: already recorded
        t_sec = t.evidence_frame / self.fps
        if t.tid < 0 and not plate and any(abs(r["time_s"] - t_sec) < 10 and r["number_plate"] == PLATE_NOT_READABLE
                                           for r in self.records):
            return      # close-up footage: the same subject re-detected, not a new vehicle
        if not plate and any(self._same_vehicle(t, p) for p in self._recorded):
            return      # tracker lost the vehicle and gave it a new (or reused) ID
        self._recorded.append((t.tid, t.vtype, plate, list(t.history)))

        img = t.evidence.copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(img, f"{self.label} | {format_plate(plate) if plate else PLATE_NOT_READABLE}",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fname = f"{self.module}_{plate or 'unreadable'}_{ts}.jpg"
        cv2.imwrite(os.path.join(self.out_folder, fname), img)

        t_sec = t.evidence_frame / self.fps
        extra = (f"vehicle={t.vtype}, track={t.tid}, time={int(t_sec // 60):02d}:{t_sec % 60:04.1f}, "
                 f"{self.describe(t)}, plate_{plate_info}")
        rec = {"module": self.module, "video_file": self.video_file, "track_id": t.tid,
               "vehicle": t.vtype, "number_plate": plate_text, "image_filename": fname,
               "time_s": round(t_sec, 1), "extra_info": extra}
        self.records.append(rec)
        print(f"   ↳ vehicle #{t.tid}: plate = {plate_text}")

        if dbconnection is not None:
            try:
                dbconnection.insert_detection(number_plate=plate_text, video_file=self.video_file,
                                              module=self.module, image_filename=fname, extra_info=extra)
                if plate:
                    dbconnection.update_license_plate(plate)
            except Exception as e:
                print("❌ DB Error:", e)

    def _same_vehicle(self, t, prev):
        """True when track `t` continues an already recorded unreadable-plate track."""
        tid, vtype, plate, hist = prev
        if plate or not hist or not t.history:
            return False
        if tid == t.tid:
            return True
        if vtype != t.vtype:
            return False
        if t.history[0][0] <= hist[-1][0] and hist[0][0] <= t.history[-1][0]:
            # seen at the same time: a different vehicle, unless one box sits inside the
            # other (the detector drew a second, partial box on the same vehicle)
            boxes = {f: b for f, _, _, b in hist}
            inside = [max(ioa(b, boxes[f]), ioa(boxes[f], b)) > 0.6 for f, _, _, b in t.history if f in boxes]
            return len(inside) >= 3 and sum(inside) >= 0.6 * len(inside)
        a, b = (hist[-1], t.history[0]) if hist[-1][0] < t.history[0][0] else (t.history[-1], hist[0])
        if b[0] - a[0] > 2 * self.fps:
            return False
        w = max(a[3][2] - a[3][0], b[3][2] - b[3][0], 1)
        return ((a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5 < w

    def _write_report(self):
        path = os.path.join(self.out_folder, f"report_{self.module}_{os.path.splitext(self.video_file)[0]}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["module", "video_file", "track_id", "vehicle", "number_plate",
                                              "time_s", "image_filename", "extra_info"])
            w.writeheader()
            w.writerows(self.records)
        self.report_path = path

    # ---- visualisation ----------------------------------------------------------
    def _annotate(self, frame, active):
        img = frame.copy()
        for t in active:
            color = (0, 0, 255) if t.confirmed else (0, 165, 255) if any(t.hits) else (0, 200, 0)
            cv2.rectangle(img, t.box[:2], t.box[2:4], color, 2)
            label = f"#{t.tid} {t.vtype}"
            if t.confirmed:
                plate, _ = t.voter.result()
                label += f" {self.label}" + (f" {plate}" if plate else "")
            cv2.putText(img, label, (t.box[0], max(15, t.box[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            if t.best is not None and t.best.box is not None and any(t.hits):
                b = tuple(int(v) for v in t.best.box[:4])
                cv2.rectangle(img, b[:2], b[2:4], (255, 0, 255), 1)
        return img
