# vision/violations.py
"""Violation modules built on the shared tracking + plate pipeline."""

import json
import math
import os

import numpy as np

from vision import settings
from vision.models import find_classes, load, load_first
from vision.pipeline import Observation, ViolationPipeline, center, ioa, iou, rider_zone


def _boxes(result, class_ids, min_conf=0.0):
    """[(x1,y1,x2,y2,conf,cls)] for the given classes."""
    out = []
    for b in result.boxes:
        cls, conf = int(b.cls[0]), float(b.conf[0])
        if cls in class_ids and conf >= min_conf:
            out.append((*map(int, b.xyxy[0]), conf, cls))
    return out


def _suppressed(box, others, margin=0.0):
    """True if an overlapping box of the opposite class is more confident (e.g. helmet vs no-helmet)."""
    return any(iou(box[:4], o[:4]) > 0.5 and o[4] > box[4] + margin for o in others)


def _dedupe(boxes, thr=0.6, contain=0.9):
    """Greedy NMS + containment removal (duplicate person boxes inflate rider counts)."""
    keep = []
    for b in sorted(boxes, key=lambda b: -b[4]):
        if all(iou(b[:4], k[:4]) < thr and ioa(b[:4], k[:4]) < contain for k in keep):
            keep.append(b)
    return keep


# =================================================================
# HELMET
# =================================================================
class HelmetPipeline(ViolationPipeline):
    module, label = "helmet_violation", "No Helmet"
    vehicle_types = {"motorcycle"}
    min_conf = float(os.getenv("NO_HELMET_CONF", 0.45))

    def setup(self):
        self.model = load_first(settings.HELMET_MODEL, settings.HELMET_MODEL_FALLBACK)
        self.bad = find_classes(self.model, ["helmet"], negative=True)
        self.good = find_classes(self.model, ["helmet"], negative=False)
        if os.getenv("NO_HELMET_CLASS_ID"):          # for models with unnamed classes ("0", "1")
            self.bad = [int(os.environ["NO_HELMET_CLASS_ID"])]
            self.good = [c for c in self.model.names if c not in self.bad]
        if not self.bad:
            raise ValueError(f"No 'no helmet' class in {self.model.names}; set NO_HELMET_CLASS_ID")

    def evaluate(self, frame, tracks):
        if not tracks:
            return {}
        r = self.model(frame, conf=0.25, device=settings.DEVICE or None, verbose=False)[0]
        good = _boxes(r, self.good)
        bad = [b for b in _boxes(r, self.bad, self.min_conf) if not _suppressed(b, good)]
        obs = {}
        for tid, heads in self.link_boxes(bad, tracks, min_overlap=0.6).items():
            top = max(heads, key=lambda b: b[4])
            obs[tid] = Observation(True, top[4], top[:4], f"heads_without_helmet={len(heads)}")
        return obs


# =================================================================
# TRIPLE RIDING
# =================================================================
class TripleRidingPipeline(ViolationPipeline):
    """
    Uses the trained triple-riding model when it has a 'triple' class; otherwise (or with
    TRIPLE_MODE=count) counts de-duplicated riders per tracked motorcycle.
    TRIPLE_MODE: auto | model | count | both | assist (3+ riders counted, or model + 2 counted)
    """
    module, label = "triple_riding", "Triple Riding"
    vehicle_types = {"motorcycle"}
    # default "count": on the sample videos the trained model (single class, very tall boxes)
    # flagged 2-rider bikes; per-bike rider counting did not. "model"/"both"/"auto" remain available.
    mode = os.getenv("TRIPLE_MODE", "count")
    min_conf = float(os.getenv("TRIPLE_CONF", 0.5))
    MIN_BIKE_H = int(os.getenv("TRIPLE_MIN_BIKE_PX", 48))

    def setup(self):
        self.model = self.pos = None
        if self.mode in ("auto", "model", "both", "assist") and os.path.exists(settings.TRIPLE_MODEL):
            self.model = load(settings.TRIPLE_MODEL)
            self.pos = find_classes(self.model, ["tripl", "triple", "3", "three", "more", "overload"])
            self.neg = [c for c in find_classes(self.model, ["single", "double", "two", "normal", "legal", "no"])
                        if c not in self.pos]
            if not self.pos:
                print(f"⚠ Triple model has no triple class {self.model.names}; using rider counting")
                self.model = None
        if self.model is None and self.mode == "model":
            raise ValueError("TRIPLE_MODE=model but no usable triple-riding model")
        self.counting = self.model is None or self.mode in ("count", "both", "assist")
        if self.counting:
            self.person_model = load(settings.PERSON_MODEL)
            self.person_ids = find_classes(self.person_model, ["person"]) or [0]

    def _model_hits(self, frame, tracks):
        r = self.model(frame, conf=0.25, device=settings.DEVICE or None, verbose=False)[0]
        neg = _boxes(r, self.neg)
        pos = [b for b in _boxes(r, self.pos, self.min_conf) if not _suppressed(b, neg)]
        hits = {}
        for b in pos:
            # the triple box should cover one bike + its riders. The model sometimes emits
            # frame-sized boxes: require a real match with the bike's rider zone, not mere containment.
            best, best_iou = None, 0.25
            for t in tracks:
                if t.box[3] - t.box[1] < self.MIN_BIKE_H:      # too small to judge riders
                    continue
                score = iou(rider_zone(t.box, t.vtype), b[:4])
                if score > best_iou and ioa(t.box, b[:4]) > 0.5:
                    best, best_iou = t, score
            if best is not None and (best.tid not in hits or b[4] > hits[best.tid][4]):
                hits[best.tid] = b
        return hits

    def _rider_counts(self, frame, tracks):
        r = self.person_model(frame, conf=0.35, classes=self.person_ids,
                              device=settings.DEVICE or None, verbose=False)[0]
        persons = _dedupe(_boxes(r, set(self.person_ids)))
        counts = {t.tid: [] for t in tracks}
        for p in persons:
            px = center(p)[0]
            ph = p[3] - p[1]
            best, best_d = None, None
            for t in tracks:
                bx1, by1, bx2, by2 = t.box
                bh = by2 - by1
                if not (bx1 <= px <= bx2):                       # rider is over the bike
                    continue
                if ioa(p[:4], rider_zone(t.box, t.vtype)) < 0.7:
                    continue
                if not (by1 + 0.1 * bh <= p[3] <= by2 + 0.15 * bh):   # legs/seat inside the bike box
                    continue
                if ph < 0.6 * bh:                                # too small: background person
                    continue
                d = abs(px - center(t.box)[0])
                if best is None or d < best_d:
                    best, best_d = t, d
            if best is not None:
                counts[best.tid].append(p)
        return counts

    def evaluate(self, frame, tracks):
        if not tracks:
            return {}
        model_hits = self._model_hits(frame, tracks) if self.model is not None else {}
        counts = self._rider_counts(frame, tracks) if self.counting else {}
        obs = {}
        for t in tracks:
            n = len(counts.get(t.tid, []))
            m = model_hits.get(t.tid)
            if self.model is not None and self.mode == "assist":
                # a third rider is often hidden behind the others: let the model vouch for it
                hit = n >= 3 or (m is not None and n >= 2)
            elif self.model is not None and self.counting:        # both must agree
                hit = m is not None and n >= 3
            elif self.model is not None:
                hit = m is not None
            else:
                hit = n >= 3
            if hit:
                conf = m[4] if m else min(1.0, 0.5 + 0.1 * n)
                box = m[:4] if m else (min(p[0] for p in counts[t.tid]), min(p[1] for p in counts[t.tid]),
                                       max(p[2] for p in counts[t.tid]), max(p[3] for p in counts[t.tid]))
                obs[t.tid] = Observation(True, conf, box, f"riders={n}" if self.counting else "")
            t.extra.setdefault("riders", []).append(n)
        return obs

    def describe(self, track):
        base = super().describe(track)
        if self.counting and track.extra.get("riders"):
            seen = sorted(track.extra["riders"])
            base += f", median_riders={seen[len(seen) // 2]}"
        return base


# =================================================================
# WRONG LANE / WRONG WAY  (direction from tracking)
# =================================================================
class WrongLanePipeline(ViolationPipeline):
    """
    A vehicle is going the wrong way when its tracked motion over the last ~1 s opposes
    the traffic flow at its location. The flow is either
      * given in a zones file (WRONG_LANE_ZONES=path.json):
          [{"polygon": [[x,y], ...] (0-1 normalised), "direction": "up"|"down"|"left"|"right"|[dx,dy]}]
      * or learned from all other vehicles in a coarse grid (default; works on two-way roads
        because each side of the road falls into different grid cells).
    WRONG_LANE_MODE: either (default: direction OR trained model) | direction | model | both
    The trained model is needed for moving/handheld cameras, where image motion is not
    real motion; the direction check is needed for fixed CCTV views the model was not trained on.
    """
    module, label = "wrong_lane", "Wrong Lane"
    mode = os.getenv("WRONG_LANE_MODE", "either")
    GRID = int(os.getenv("WRONG_LANE_GRID", 6))
    MIN_SUPPORT = 5            # other vehicles that must have defined the flow near a vehicle
    NEAR = float(os.getenv("WRONG_LANE_NEAR", 1.0))   # "near" = within this many vehicle sizes
    MAX_SAMPLES = 400          # remembered positions per grid cell (oldest dropped)
    MOVING_CAM_PX = float(os.getenv("MOVING_CAM_PX", 1.5))   # background shift per frame => handheld/moving camera
    WINDOW_S = 1.0             # motion is measured over this many seconds
    MIN_MOVE = 0.03            # ...and must exceed this fraction of the frame diagonal
    # the single-frame model is trusted only with context: at least this many vehicles in the same
    # frame judged "right way" and wrong-way vehicles in the minority (alone, it flagged ordinary riders)
    MIN_CONTEXT = int(os.getenv("WRONG_LANE_MIN_CONTEXT", 3))

    DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

    ZONES_DIR = os.getenv("WRONG_LANE_ZONES_DIR", "lane_zones")   # <video name>.json or default.json

    def setup(self):
        self.flow = {}         # cell -> [(x, y, dx, dy, tid)] positions + unit motion of vehicles seen there
        self.zones = []
        self.model = None
        if self.mode in ("model", "both", "either"):
            self.model = load_first(settings.WRONG_LANE_MODEL, settings.WRONG_LANE_MODEL_FALLBACK)
            self.wrong = find_classes(self.model, ["wrong", "lane", "violat", "opposite"], negative=True) or [1]

    def _load_zones(self, video_path):
        """Allowed direction per road area: WRONG_LANE_ZONES file, else lane_zones/<video name>.json,
        else lane_zones/default.json (made with tools/calibrate_lanes.py)."""
        stem = os.path.splitext(os.path.basename(video_path))[0]
        for path in (os.getenv("WRONG_LANE_ZONES"), os.path.join(self.ZONES_DIR, stem + ".json"),
                     os.path.join(self.ZONES_DIR, "default.json")):
            if path and os.path.exists(path):
                with open(path) as f:
                    zones = []
                    for z in json.load(f):
                        d = z["direction"]
                        d = self.DIRS[d] if isinstance(d, str) else d
                        n = math.hypot(*d)
                        zones.append((np.array(z["polygon"], dtype=np.float32), (d[0] / n, d[1] / n)))
                print(f"✔ Wrong-lane zones: {len(zones)} from {path}")
                return zones
        return []

    def run(self, video_path, out_folder="static/uploads/detections"):
        self.zones = self._load_zones(video_path)
        self.flow, self.still_since, self._cut_prev, self._prev_gray = {}, -1, None, None
        return super().run(video_path, out_folder)

    def _motion(self, t, frame_shape):
        if len(t.history) < 3:
            return None
        span = self.WINDOW_S * self.fps
        since = getattr(self, "still_since", -1)
        recent = [h for h in t.history if h[0] >= t.history[-1][0] - span and h[0] > since]
        if len(recent) < 3 or recent[-1][0] - recent[0][0] < 0.5 * span:
            return None
        # robust: compare the mean of the first and last thirds
        k = max(1, len(recent) // 3)
        x0, y0 = np.mean([h[1] for h in recent[:k]]), np.mean([h[2] for h in recent[:k]])
        x1, y1 = np.mean([h[1] for h in recent[-k:]]), np.mean([h[2] for h in recent[-k:]])
        dx, dy = x1 - x0, y1 - y0
        H, W = frame_shape[:2]
        if math.hypot(dx, dy) < self.MIN_MOVE * math.hypot(W, H):
            return None
        n = math.hypot(dx, dy)
        return dx / n, dy / n

    def _cell(self, x, y, frame_shape):
        H, W = frame_shape[:2]
        return min(self.GRID - 1, int(x / W * self.GRID)), min(self.GRID - 1, int(y / H * self.GRID))

    def _learn(self, t, m, frame_shape):
        """Remember where this vehicle was and which way it moved (others are judged against it)."""
        cx, cy = center(t.box)
        samples = self.flow.setdefault(self._cell(cx, cy, frame_shape), [])
        samples.append((cx, cy, m[0], m[1], t.tid))
        if len(samples) > self.MAX_SAMPLES:
            del samples[0]

    def _expected(self, t, frame_shape):
        H, W = frame_shape[:2]
        cx, cy = center(t.box)
        for poly, d in self.zones:
            import cv2
            if cv2.pointPolygonTest((poly * [W, H]).astype(np.float32), (float(cx), float(cy)), False) >= 0:
                return d, "zone"
        # only vehicles that passed close to this one count: a grid cell can straddle a divider,
        # with the opposite carriageway's traffic in one corner (CCTV sample, 40.8 s).
        # "Close" scales with the vehicle's size, so it follows the perspective.
        radius = self.NEAR * max(t.box[2] - t.box[0], t.box[3] - t.box[1])
        gx, gy = self._cell(cx, cy, frame_shape)
        others = {}
        for nx in (gx - 1, gx, gx + 1):
            for ny in (gy - 1, gy, gy + 1):
                for x, y, dx, dy, tid in self.flow.get((nx, ny), ()):
                    if tid != t.tid and math.hypot(x - cx, y - cy) <= radius:
                        v = others.setdefault(tid, [0.0, 0.0, 0])
                        v[0] += dx; v[1] += dy; v[2] += 1
        others = {tid: v for tid, v in others.items() if v[2] >= 2}
        if len(others) < self.MIN_SUPPORT:
            return None, "learning"
        sx = sum(v[0] / v[2] for v in others.values())
        sy = sum(v[1] / v[2] for v in others.values())
        n = math.hypot(sx, sy)
        if n / len(others) < 0.8:           # traffic in this cell isn't consistent (e.g. near a divider)
            return None, "mixed"
        return (sx / n, sy / n), "learned"

    def _camera_motion(self, frame, tracks):
        """Median background feature shift (px/frame, EMA). Vehicles are masked out."""
        import cv2
        gray = cv2.cvtColor(cv2.resize(frame, (480, int(480 * frame.shape[0] / frame.shape[1]))), cv2.COLOR_BGR2GRAY)
        sx = 480 / frame.shape[1]
        prev, self._prev_gray = getattr(self, "_prev_gray", None), gray
        if prev is None or prev.shape != gray.shape:
            return
        mask = np.full(gray.shape, 255, np.uint8)
        for t in tracks:
            x1, y1, x2, y2 = (int(v * sx) for v in t.box)
            mask[max(0, y1):y2, max(0, x1):x2] = 0
        pts = cv2.goodFeaturesToTrack(prev, 150, 0.01, 8, mask=mask)
        if pts is None or len(pts) < 10:
            return
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev, nxt, None)       # forward-backward check
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((back - pts).reshape(-1, 2), axis=1) < 1.0)
        if ok.sum() < 10:
            return
        # one camera movement (shift + zoom + small rotation) must explain most background points;
        # if it does not, the points are noise / moving objects and say nothing about the camera
        M, inl = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=1.5)
        good = pts[ok].reshape(-1, 2)[inl.ravel() == 1] if inl is not None else np.zeros((0, 2))
        # no "points in 3 of 4 quarters" rule: a camera following a vehicle often sees texture only
        # above the horizon (plain road below), as in the handheld sample
        spread = (len(good) >= 10 and np.ptp(np.percentile(good[:, 0], [10, 90])) > 0.4 * gray.shape[1]
                  and np.ptp(np.percentile(good[:, 1], [10, 90])) > 0.1 * gray.shape[0])
        if M is None or inl is None or inl.mean() < 0.5 or inl.sum() < 0.4 * len(pts) or not spread:
            # a camera move shifts the whole picture: agreeing points confined to one area are
            # a vehicle (or noise), not the camera
            shift = 0.0
        else:
            scale = float(np.hypot(M[0, 0], M[1, 0]))
            # a pure zoom barely moves points near the centre: measure it at the frame corners
            corner = abs(scale - 1.0) * 0.5 * float(np.hypot(*gray.shape[::-1]))
            shift = max(corner, float(np.hypot(M[0, 2], M[1, 2]))) / sx
        self.cam_motion = 0.8 * getattr(self, "cam_motion", shift) + 0.2 * shift

    def _scene_cut(self, frame):
        """True when the picture changes abruptly (video edit, PTZ camera moving to another preset)."""
        import cv2
        small = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        prev, self._cut_prev = getattr(self, "_cut_prev", None), small
        if prev is None:
            return False
        return float(np.mean(np.abs(small - prev))) > self.CUT_DIFF

    CUT_DIFF = float(os.getenv("WRONG_LANE_CUT_DIFF", 28))   # mean grey-level change (0-255) that means a new view

    def evaluate(self, frame, tracks):
        if self._scene_cut(frame):
            # new view: motion measured across the cut is meaningless and the learned flow no
            # longer applies (zones from a file are kept: they belong to a fixed view)
            self.flow = {}
            self.still_since = self.frame_no
            self._prev_gray = None
            self.cuts = getattr(self, "cuts", 0) + 1
        self._camera_motion(frame, tracks)
        moving_cam = getattr(self, "cam_motion", 0.0) > self.MOVING_CAM_PX
        if moving_cam:
            # positions recorded while the view moves are not real motion: measure again once it stops
            self.still_since = self.frame_no
        model_wrong = set()
        if self.model is not None and tracks and (moving_cam or self.mode == "model"):
            r = self.model(frame, conf=0.3, device=settings.DEVICE or None, verbose=False)[0]
            wrong_boxes = _boxes(r, set(self.wrong), 0.35)
            right_boxes = _boxes(r, set(range(len(self.model.names))) - set(self.wrong), 0.35)
            if len(right_boxes) < self.MIN_CONTEXT or len(wrong_boxes) >= len(right_boxes):
                wrong_boxes = []
            for b in wrong_boxes:
                # the model may box the whole vehicle or only part of it (rider, delivery box)
                best = max(tracks, key=lambda t: (iou(t.box, b[:4]), ioa(b[:4], rider_zone(t.box, t.vtype))))
                if iou(best.box, b[:4]) > 0.4 or ioa(b[:4], rider_zone(best.box, best.vtype)) > 0.6:
                    model_wrong.add(best.tid)

        obs = {}
        for t in tracks:
            m = self._motion(t, frame.shape)
            direction_hit, cos = False, None
            if m is not None and not moving_cam:
                exp, src = self._expected(t, frame.shape)
                if exp is not None:
                    cos = m[0] * exp[0] + m[1] * exp[1]
                    direction_hit = cos < -0.5
                t.extra["flow_src"] = src
                self._learn(t, m, frame.shape)

            if self.mode == "model":
                hit = t.tid in model_wrong
            elif self.mode == "both":
                hit = direction_hit and t.tid in model_wrong
            elif self.mode == "either":
                # the single-frame model is only trusted when the vehicle's own motion can't be
                # measured (camera moving with it); a vehicle visibly moving WITH the flow is never
                # flagged by the model (it flagged normal traffic on the CCTV sample)
                with_flow = cos is not None and cos > 0.3
                hit = direction_hit or (t.tid in model_wrong and not with_flow)
            else:
                hit = direction_hit
            if hit:
                info = f"against_flow cos={cos:.2f} ({t.extra.get('flow_src')})" if cos is not None else "model"
                obs[t.tid] = Observation(True, 1.0 if cos is None else min(1.0, -cos), t.box, info)
        return obs


# =================================================================
# PHONE USAGE
# =================================================================
class PhoneUsagePipeline(ViolationPipeline):
    module, label = "phone_usage", "Phone Usage"
    vehicle_types = {"car", "motorcycle", "auto", "bus", "truck"}
    min_conf = float(os.getenv("PHONE_CONF", 0.45))

    def setup(self):
        try:
            # retrained weights only: the old phoneusage.pt (mAP50 0.10) missed the sample video,
            # where the COCO person + cell-phone check below works
            self.model = load_first(settings.PHONE_MODEL)
            self.pos = find_classes(self.model, ["phone", "mobile", "call", "text"], negative=False)
            self.neg = find_classes(self.model, ["phone", "mobile", "call", "text"], negative=True)
        except FileNotFoundError:
            self.model, self.pos = None, []
        self.coco = not self.pos
        # optional second opinion (PHONE_CONFIRM_COCO=1): the COCO model must also see a phone inside
        # the flagged area. Off by default: the phone model was trained with hard negatives (hands
        # on the wheel, drinking, talking) and COCO rarely sees a phone held at a rider's ear.
        self.confirm = not self.coco and os.getenv("PHONE_CONFIRM_COCO", "0") == "1"
        if self.confirm:
            self.coco_model = load(settings.PERSON_MODEL)
            self.coco_phone = find_classes(self.coco_model, ["cell phone"])
        if self.coco:       # COCO fallback: a cell phone held at a person's head / chest
            print("⚠ No phone-usage model; using COCO person + cell phone")
            self.model = load(settings.PERSON_MODEL)
            self.phone_ids = find_classes(self.model, ["cell phone"])
            self.person_ids = find_classes(self.model, ["person"])

    def _evidence(self, frame):
        r = self.model(frame, conf=0.25, device=settings.DEVICE or None, verbose=False)[0]
        if not self.coco:
            neg = _boxes(r, self.neg)
            hits = [b for b in _boxes(r, self.pos, self.min_conf) if not _suppressed(b, neg)]
            if hits and self.confirm:
                phones = _boxes(self.coco_model(frame, conf=0.15, classes=self.coco_phone,
                                                device=settings.DEVICE or None, verbose=False)[0],
                                set(self.coco_phone), 0.15)
                def grown(b, k=0.25):
                    w, h = b[2] - b[0], b[3] - b[1]
                    return (b[0] - k * w, b[1] - k * h, b[2] + k * w, b[3] + k * h)
                hits = [b for b in hits if any(ioa(p[:4], grown(b)) > 0.5 for p in phones)]
            return hits
        phones = _boxes(r, self.phone_ids, 0.35)
        out = []
        for p in _boxes(r, self.person_ids, 0.4):
            upper = (p[0], p[1], p[2], p[1] + 0.55 * (p[3] - p[1]))
            for ph in phones:
                if ioa(ph[:4], upper) > 0.6:
                    out.append((*p[:4], ph[4], None))
                    break
        return out

    def evaluate(self, frame, tracks):
        ev = self._evidence(frame)
        linked = self.link_boxes(ev, tracks, min_overlap=0.6) if tracks else {}
        used = {id(b) for bs in linked.values() for b in bs}
        obs = {tid: Observation(True, max(b[4] for b in bs), max(bs, key=lambda b: b[4])[:4])
               for tid, bs in linked.items()}
        for b in ev:          # close-up of a driver: the car itself may not be detected
            if id(b) not in used:
                t = self.orphan(b, "car", frame.shape)
                obs[t.tid] = Observation(True, b[4], b[:4], "close-up")
        return obs


# =================================================================
# SEATBELT
# =================================================================
class SeatbeltPipeline(ViolationPipeline):
    module, label = "seatbelt_violation", "No Seatbelt"
    vehicle_types = {"car", "truck", "bus", "auto"}
    min_conf = float(os.getenv("NO_SEATBELT_CONF", 0.45))

    def setup(self):
        self.model = load_first(settings.SEATBELT_MODEL, settings.SEATBELT_MODEL_FALLBACK)
        self.bad = find_classes(self.model, ["belt"], negative=True)
        self.good = find_classes(self.model, ["belt"], negative=False)
        # trainedmodels/seatbelt_yolov8.pt was trained with its class NAMES swapped: on the sample
        # videos it says "no_seatbelt" for a belted driver and "seatbelt" for an unbelted one
        # (the original script compensated with hard-coded ids). Swap back for that file only.
        if os.path.basename(self.model.ckpt_path or "") == "seatbelt_yolov8.pt" or os.getenv("SEATBELT_SWAP_NAMES") == "1":
            self.bad, self.good = self.good, self.bad
        if not self.bad:
            raise ValueError(f"No 'no seatbelt' class in {self.model.names}")
        self.person_model = load(settings.PERSON_MODEL)
        self.person_ids = find_classes(self.person_model, ["person"]) or [0]
        self.two_wheeler_ids = [c for c, n in self.person_model.names.items() if n in ("motorcycle", "bicycle")]

    def _occupant(self, p, tracks, two_wheelers):
        """True if person box p is someone sitting in a tracked car/truck/bus/auto: almost entirely
        inside the vehicle box with the head below its roof line, and not riding a two-wheeler.
        (Pedestrians standing in front of a car stick out above the roof; on real street footage
        they and motorcyclists were taken for unbelted drivers.)"""
        x1, y1, x2, y2 = p[:4]
        if (y2 - y1) > self.MAX_ASPECT * max(1, x2 - x1):
            return False        # tall thin box: someone standing beside the car, not a seated driver
        lower = (x1, y1 + 0.5 * (y2 - y1), x2, y2)
        if any(ioa(lower, tw[:4]) > 0.3 for tw in two_wheelers):
            return False
        for t in tracks:
            vx1, vy1, vx2, vy2 = t.box
            if ioa(p[:4], t.box) >= self.MIN_INSIDE and y1 >= vy1 - 0.05 * (vy2 - vy1):
                return True
        return False

    # "absence" rule: the retrained model finds belts reliably but saw few unbelted examples
    # (228 boxes), so a clearly visible driver on whom no belt is ever found is also a violation.
    ABSENCE = os.getenv("SEATBELT_ABSENCE_RULE", "1") == "1"
    MIN_PERSON_FRAC = 0.30      # driver must fill >= 30% of the frame height (belt would be visible)
    MIN_VISIBLE = 8             # analysed frames the driver must be seen in
    MAX_BELT_RATIO = 0.15       # a belt found in >= 15% of those frames clears the vehicle
    MIN_INSIDE = 0.85           # share of the person box that must lie inside the vehicle box
    MAX_ASPECT = 2.2            # a seated driver (head to lap) is at most ~2x as tall as wide

    def evaluate(self, frame, tracks):
        r = self.model(frame, conf=0.25, device=settings.DEVICE or None, verbose=False)[0]
        good = _boxes(r, self.good, 0.35)
        bad = [b for b in _boxes(r, self.bad, self.min_conf) if not _suppressed(b, good)]
        evidence = [(b, "no_seatbelt") for b in bad]

        if self.ABSENCE and tracks:
            H = frame.shape[0]
            pr = self.person_model(frame, conf=0.3, classes=self.person_ids + self.two_wheeler_ids,
                                   device=settings.DEVICE or None, verbose=False)[0]
            two_wheelers = _boxes(pr, set(self.two_wheeler_ids), 0.3)
            for p in _dedupe(_boxes(pr, set(self.person_ids), 0.5)):
                if (p[3] - p[1]) < self.MIN_PERSON_FRAC * H:
                    continue
                if not self._occupant(p, tracks, two_wheelers):
                    continue
                torso = (p[0], p[1], p[2], p[1] + 0.75 * (p[3] - p[1]))
                belted = any(ioa(g[:4], torso) > 0.5 for g in good)
                evidence.append(((*p[:4], 0.5, None), "belted" if belted else "no_belt_seen"))

        boxes = [b for b, _ in evidence]
        kinds = {id(b): k for b, k in evidence}
        linked = self.link_boxes(boxes, tracks, min_overlap=0.6) if tracks else {}
        used = {id(b) for bs in linked.values() for b in bs}
        for b in boxes:
            # only the model's own "no_seatbelt" boxes may stand for an undetected car (close-up);
            # "no belt seen" evidence must belong to a person inside a detected vehicle
            if id(b) not in used and kinds[id(b)] == "no_seatbelt":
                t = self.orphan(b, "car", frame.shape)
                linked.setdefault(t.tid, []).append(b)

        obs = {}
        trk = {t.tid: t for t in list(tracks) + self._orphans}
        for tid, bs in linked.items():
            t = trk.get(tid)
            ks = [kinds[id(b)] for b in bs]
            if t is not None and any(k in ("belted", "no_belt_seen") for k in ks):
                t.extra["visible"] = t.extra.get("visible", 0) + 1
                t.extra["belted"] = t.extra.get("belted", 0) + ("belted" in ks)
            hits = [b for b in bs if kinds[id(b)] in ("no_seatbelt", "no_belt_seen")]
            if hits and "belted" not in ks:
                top = max(hits, key=lambda b: b[4])
                if any(kinds[id(b)] == "no_seatbelt" for b in hits) and t is not None:
                    t.extra["explicit"] = True
                obs[tid] = Observation(True, top[4], top[:4], kinds[id(top)])
        return obs

    def accept(self, track):
        if track.extra.get("explicit"):
            return True          # the model itself saw "no_seatbelt"
        seen, belted = track.extra.get("visible", 0), track.extra.get("belted", 0)
        return seen >= self.MIN_VISIBLE and belted / max(seen, 1) < self.MAX_BELT_RATIO

    def describe(self, track):
        seen, belted = track.extra.get("visible", 0), track.extra.get("belted", 0)
        how = "model: no_seatbelt" if track.extra.get("explicit") else "no belt found on visible driver"
        return f"{how}, belt seen in {belted}/{seen} frames"


# =================================================================
# ANPR (every vehicle, not only violators)
# =================================================================
class ANPRPipeline(ViolationPipeline):
    module, label = "anpr", "ANPR"
    read_plates_for_all = True

    def evaluate(self, frame, tracks):
        return {t.tid: Observation(True, 1.0, None) for t in tracks if t.plate_crops}
