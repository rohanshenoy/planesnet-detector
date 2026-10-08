"""
Detect aircraft against sky from ground- or air-based cameras.

This uses an Ultralytics YOLO model. The default COCO weights already include
an 'airplane' class, so it works without any training; fine-tune it with
sky/finetune.py for small, distant aircraft.

Two additions on top of plain YOLO:

  * Tiled inference (--tile): distant aircraft are a few pixels wide and
    vanish when a 4K frame is downscaled to 640 px. The frame is split into
    overlapping tiles at native resolution, plus one full-frame pass for
    large aircraft, and the results are merged with NMS.
  * Temporal confirmation (video): detections are linked into tracks across
    frames and only reported once a track has persisted for --min-hits frames
    and moves consistently. This removes single-frame false positives (birds,
    glare, cloud edges) and prefers smooth, straight motion like an aircraft's
    over the erratic motion of a bird.

Examples:
    python -m sky.detect photo.jpg --tile 640
    python -m sky.detect clip.mp4 --tile 640 --min-hits 5 --out clip_tracked.mp4
"""

import argparse
import json
import os

import numpy as np
import torch
import torchvision

VIDEO_EXTS = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
AIRCRAFT_NAMES = {'airplane', 'aeroplane', 'plane', 'aircraft', 'helicopter', 'drone'}


def make_tiles(width, height, tile, overlap=0.2):
    """Top-left corners (x, y) of overlapping tiles that cover the image."""
    if tile >= width and tile >= height:
        return [(0, 0)]
    step = max(1, int(tile * (1 - overlap)))

    def starts(n):
        if tile >= n:
            return [0]
        s = list(range(0, n - tile, step))
        return s + [n - tile]

    return [(x, y) for y in starts(height) for x in starts(width)]


def merge(boxes, scores, iou=0.5):
    """NMS across detections from overlapping tiles. boxes: (N, 4) xyxy."""
    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    b = torch.as_tensor(np.asarray(boxes), dtype=torch.float32)
    s = torch.as_tensor(np.asarray(scores), dtype=torch.float32)
    keep = torchvision.ops.nms(b, s, iou).numpy()
    return b.numpy()[keep], s.numpy()[keep]


def cut_by_tile_edge(boxes, x, y, tw, th, width, height, margin=2):
    """Mask of tile-local boxes touching a tile edge that is not an image edge.

    Such boxes are truncated objects; thanks to the tile overlap the whole
    object appears in a neighbouring tile or in the full-frame pass.
    """
    b = np.asarray(boxes, np.float32).reshape(-1, 4)
    cut = np.zeros(len(b), bool)
    if x > 0:
        cut |= b[:, 0] <= margin
    if y > 0:
        cut |= b[:, 1] <= margin
    if x + tw < width:
        cut |= b[:, 2] >= tw - margin
    if y + th < height:
        cut |= b[:, 3] >= th - margin
    return cut


def aircraft_class_ids(names, wanted=None):
    """Class ids in the model whose name looks like an aircraft."""
    wanted = {w.lower() for w in wanted} if wanted else AIRCRAFT_NAMES
    ids = [i for i, n in names.items() if n.lower() in wanted]
    if not ids:
        raise ValueError('model has none of the classes %s; got %s' % (sorted(wanted), names))
    return ids


def detect_frame(model, frame, classes, conf=0.25, tile=None, overlap=0.2, imgsz=640):
    """Run YOLO on one BGR/RGB frame, optionally tiled. Returns (boxes, scores)."""
    h, w = frame.shape[:2]
    crops = [(0, 0, frame)]
    if tile:
        crops += [(x, y, frame[y:y + tile, x:x + tile]) for x, y in make_tiles(w, h, tile, overlap)]
    boxes, scores = [], []
    results = model.predict([c for _, _, c in crops], conf=conf, classes=classes,
                            imgsz=tile or imgsz, verbose=False)
    for k, ((x, y, crop), r) in enumerate(zip(crops, results)):
        if r.boxes is None or len(r.boxes) == 0:
            continue
        b = r.boxes.xyxy.cpu().numpy()
        s = r.boxes.conf.cpu().numpy()
        if k > 0:  # tile pass: drop objects cut by an interior tile edge
            keep = ~cut_by_tile_edge(b, x, y, crop.shape[1], crop.shape[0], w, h)
            b, s = b[keep], s[keep]
        boxes.append(b + np.array([x, y, x, y], np.float32))
        scores.append(s)
    if not boxes:
        return merge([], [])
    return merge(np.concatenate(boxes), np.concatenate(scores))


class Track:
    def __init__(self, tid, box, score, frame):
        self.id, self.boxes, self.scores, self.frames = tid, [box], [score], [frame]
        self.misses = 0

    @property
    def centres(self):
        b = np.asarray(self.boxes)
        return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)

    def predict(self):
        """Constant-velocity guess of the box at the next frame."""
        if len(self.boxes) < 2:
            return self.boxes[-1]
        gap = max(self.frames[-1] - self.frames[-2], 1)
        v = (self.boxes[-1] - self.boxes[-2]) / gap
        return self.boxes[-1] + v * (1 + self.misses)

    def straightness(self):
        """Net displacement / path length, in [0, 1]; 1 = perfectly straight."""
        c = self.centres
        path = np.linalg.norm(np.diff(c, axis=0), axis=1).sum()
        return 1.0 if path < 1e-6 else float(np.linalg.norm(c[-1] - c[0]) / path)


class Tracker:
    """Greedy centre-distance tracker with multi-frame confirmation.

    A track is confirmed once it has min_hits detections. Tracks that have
    moved more than min_travel px must also have straightness >= min_straight,
    which rejects the erratic flight of birds and flickering clutter. Slow or
    distant aircraft that barely move are accepted on persistence alone.
    """

    def __init__(self, min_hits=5, max_misses=5, gate=3.0, min_straight=0.7, min_travel=20):
        self.min_hits, self.max_misses, self.gate = min_hits, max_misses, gate
        self.min_straight, self.min_travel = min_straight, min_travel
        self.tracks, self.finished, self.next_id = [], [], 1

    def is_confirmed(self, t):
        if len(t.boxes) < self.min_hits:
            return False
        c = t.centres
        if np.linalg.norm(c[-1] - c[0]) > self.min_travel:
            return t.straightness() >= self.min_straight
        return True

    def update(self, boxes, scores, frame):
        """Feed one frame of detections; returns currently confirmed, matched tracks."""
        boxes = [np.asarray(b, np.float32) for b in boxes]
        unmatched = set(range(len(boxes)))
        pairs = []
        for ti, t in enumerate(self.tracks):
            p = t.predict()
            pc = np.array([(p[0] + p[2]) / 2, (p[1] + p[3]) / 2])
            size = max(p[2] - p[0], p[3] - p[1], 8.0)
            for di in unmatched:
                b = boxes[di]
                d = np.hypot((b[0] + b[2]) / 2 - pc[0], (b[1] + b[3]) / 2 - pc[1])
                if d <= self.gate * size:
                    pairs.append((d, ti, di))
        used_t = set()
        matched = []
        for d, ti, di in sorted(pairs):
            if ti in used_t or di not in unmatched:
                continue
            t = self.tracks[ti]
            t.boxes.append(boxes[di])
            t.scores.append(float(scores[di]))
            t.frames.append(frame)
            t.misses = 0
            used_t.add(ti)
            unmatched.discard(di)
            matched.append(t)
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.misses += 1
        for di in sorted(unmatched):
            self.tracks.append(Track(self.next_id, boxes[di], float(scores[di]), frame))
            self.next_id += 1
        alive = []
        for t in self.tracks:
            (alive if t.misses <= self.max_misses else self.finished).append(t)
        self.tracks = alive
        return [t for t in matched if self.is_confirmed(t)]

    def confirmed_tracks(self):
        return [t for t in self.finished + self.tracks if self.is_confirmed(t)]


def _draw(frame, boxes, labels, color=(0, 0, 255)):
    import cv2
    for b, lab in zip(boxes, labels):
        x0, y0, x1, y1 = [int(v) for v in b]
        pad = max(0, 12 - (x1 - x0)) // 2  # keep tiny aircraft visible
        cv2.rectangle(frame, (x0 - pad, y0 - pad), (x1 + pad, y1 + pad), color, 2)
        cv2.putText(frame, lab, (x0 - pad, y0 - pad - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame


def run_image(model, path, classes, a):
    import cv2
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(path)
    boxes, scores = detect_frame(model, frame, classes, a.conf, a.tile, a.overlap, a.imgsz)
    out = a.out or os.path.splitext(path)[0] + '_detections.jpg'
    cv2.imwrite(out, _draw(frame, boxes, ['%.2f' % s for s in scores]))
    print('%d detections -> %s' % (len(boxes), out))
    return [{'box': [round(float(v), 1) for v in b], 'score': float(s)} for b, s in zip(boxes, scores)]


def run_video(model, path, classes, a):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = a.out or os.path.splitext(path)[0] + '_tracked.mp4'
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    tracker = Tracker(a.min_hits, a.max_misses, min_straight=a.min_straight)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes, scores = detect_frame(model, frame, classes, a.conf, a.tile, a.overlap, a.imgsz)
        live = tracker.update(boxes, scores, i)
        writer.write(_draw(frame, [t.boxes[-1] for t in live], ['#%d' % t.id for t in live]))
        i += 1
    cap.release()
    writer.release()
    tracks = tracker.confirmed_tracks()
    print('%d frames, %d confirmed aircraft tracks -> %s' % (i, len(tracks), out))
    return [{'id': t.id, 'first_frame': t.frames[0], 'last_frame': t.frames[-1],
             'hits': len(t.frames), 'mean_score': float(np.mean(t.scores)),
             'straightness': round(t.straightness(), 3),
             'path': [[round(float(x), 1), round(float(y), 1)] for x, y in t.centres]}
            for t in tracks]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('source', help='image or video file')
    p.add_argument('--weights', default='yolo11n.pt', help='YOLO weights (COCO by default)')
    p.add_argument('--classes', nargs='*', help='class names to keep (default: aircraft-like)')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--tile', type=int, help='tile size for small-object inference, e.g. 640')
    p.add_argument('--overlap', type=float, default=0.2)
    p.add_argument('--min-hits', type=int, default=5, help='frames before a track is reported')
    p.add_argument('--max-misses', type=int, default=5, help='frames a track may go undetected')
    p.add_argument('--min-straight', type=float, default=0.7)
    p.add_argument('--out')
    p.add_argument('--json')
    a = p.parse_args(argv)

    from ultralytics import YOLO
    model = YOLO(a.weights)
    classes = aircraft_class_ids(model.names, a.classes)
    run = run_video if a.source.lower().endswith(VIDEO_EXTS) else run_image
    result = run(model, a.source, classes, a)
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(result, f, indent=2)
    return result


if __name__ == '__main__':
    main()
