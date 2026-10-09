"""
Clip-level evaluation of the sky detector and the multi-frame tracker on AOT.

sky/evaluate.py scores single 640 px crops. This script scores what you would
actually run on video: full 2448x2048 frames, tiled at native resolution, on
continuous clips from held-out flights (same flight split as prepare_aot), and
compares three outputs per frame:

  raw      every detection above the model's threshold
  online   only detections whose track is already confirmed at that frame
           (what sky/detect.py draws live; a track needs --min-hits frames)
  offline  every detection belonging to a track that is confirmed at any
           point in the clip (what a post-processing pass would keep)

Matching and ignore rules are the same as sky/evaluate.py (centre distance;
birds and unidentified objects ignored).

Example:
    python -m sky.evaluate_clips groundtruth.csv data/aot_clips \
        --model single=runs/sky/single/weights/best.pt:0.17 \
        --model temporal=runs/sky/temporal/weights/best.pt:0.18:2
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import temporal
from .detect import AIRCRAFT_NAMES, Tracker, detect_frame
from .evaluate import GT_AIRCRAFT, match_image
from .prepare_aot import AIRCRAFT, fetch, load_groundtruth, split_of


def candidate_clips(flights, length, pad, seed=0):
    """Yield (flight, start) for held-out flights: `length` consecutive labelled
    frames (plus `pad` each side) with an aircraft in at least 80% of them."""
    rng = np.random.default_rng(seed)
    ids = sorted(f for f, fl in flights.items() if fl['split'] == 'val')
    rng.shuffle(ids)
    for fid in ids:
        fl = flights[fid]
        have = set(fl['frames'])
        with_air = sorted(f for f in have if any(o['cls'] in AIRCRAFT for o in fl['objects'].get(f, [])))
        for start in with_air[::max(1, len(with_air) // 8)]:
            if not all(f in have for f in range(start - pad, start + length + pad)):
                continue
            covered = sum(any(o['cls'] in AIRCRAFT for o in fl['objects'].get(f, []))
                          for f in range(start, start + length))
            if covered >= 0.8 * length:
                yield fid, start


def download_clip(flights, fid, start, length, pad, out, workers=16):
    """Cache the clip's frames as greyscale PNGs; returns {frame: path}, or None
    if any frame is missing from the bucket (about 28% of AOT frames are)."""
    import shutil
    import urllib.error
    os.makedirs(out, exist_ok=True)
    frames = list(range(start - pad, start + length + pad))

    def get(f):
        path = os.path.join(out, '%06d.png' % f)
        if not os.path.exists(path):
            cv2.imwrite(path, fetch(fid, flights[fid]['frames'][f]))
        return f, path

    try:
        with ThreadPoolExecutor(workers) as ex:
            return dict(ex.map(get, frames))
    except urllib.error.HTTPError:
        shutil.rmtree(out, ignore_errors=True)
        return None


def ground_truth(fl, frame):
    """(cls, cx, cy, w, h) in full-frame pixels, as sky.evaluate expects."""
    out = []
    for o in fl['objects'].get(frame, []):
        x0, y0, x1, y1 = o['box']
        out.append((o['cls'], (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, 2), max(y1 - y0, 2)))
    return out


def run_model(model, classes, paths, start, length, k, tile, conf):
    """Per-frame (boxes, scores) for frames start..start+length-1."""
    dets = {}
    for f in range(start, start + length):
        cur = cv2.imread(paths[f], cv2.IMREAD_GRAYSCALE)
        if k:
            img = temporal.stack(cv2.imread(paths[f - k], cv2.IMREAD_GRAYSCALE), cur,
                                 cv2.imread(paths[f + k], cv2.IMREAD_GRAYSCALE))
        else:
            img = cv2.cvtColor(cur, cv2.COLOR_GRAY2BGR)
        boxes, scores = detect_frame(model, img, classes, conf=conf, tile=tile)
        dets[f] = (boxes, scores)
    return dets


def outputs(dets, threshold, min_hits, max_misses):
    """raw / online / offline detections per frame, as (score, cx, cy) lists."""
    tracker = Tracker(min_hits=min_hits, max_misses=max_misses)
    raw, online, members = {}, {}, {}
    for f in sorted(dets):
        boxes, scores = dets[f]
        keep = scores >= threshold
        b, s = boxes[keep], scores[keep]
        raw[f] = [(float(si), (bi[0] + bi[2]) / 2, (bi[1] + bi[3]) / 2) for bi, si in zip(b, s)]
        live = tracker.update(b, s, f)
        online[f] = [(t.scores[-1], (t.boxes[-1][0] + t.boxes[-1][2]) / 2,
                      (t.boxes[-1][1] + t.boxes[-1][3]) / 2) for t in live]
    confirmed = tracker.confirmed_tracks()
    offline = {f: [] for f in dets}
    for t in confirmed:
        for f, bx, sc in zip(t.frames, t.boxes, t.scores):
            offline[f].append((sc, (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2))
    return {'raw': raw, 'online': online, 'offline': offline}


def score(per_frame_preds, gts_by_frame):
    tp = fp = n_gt = 0
    records = []
    for f, preds in per_frame_preds.items():
        gts = gts_by_frame[f]
        recs, hit = match_image(preds, gts)
        records += recs
        tp += sum(h is not None for g, h in zip(gts, hit) if g[0] in GT_AIRCRAFT)
        fp += sum(not t for _, t in recs)
        n_gt += sum(g[0] in GT_AIRCRAFT for g in gts)
    return {'recall': tp / max(n_gt, 1), 'precision': tp / max(tp + fp, 1),
            'false_alarms_per_frame': fp / max(len(per_frame_preds), 1),
            'n_aircraft': n_gt, 'n_frames': len(per_frame_preds)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('groundtruth')
    p.add_argument('cache', help='directory for downloaded clip frames')
    p.add_argument('--model', action='append', required=True,
                   help='name=weights:threshold[:k]  (k = temporal frame offset, 0/omitted = single frame)')
    p.add_argument('--clips', type=int, default=6)
    p.add_argument('--length', type=int, default=50)
    p.add_argument('--tile', type=int, default=640)
    p.add_argument('--min-hits', type=int, default=5)
    p.add_argument('--max-misses', type=int, default=5)
    p.add_argument('--val-fraction', type=float, default=0.2)
    p.add_argument('--json')
    a = p.parse_args(argv)

    from ultralytics import YOLO
    specs = []
    for m in a.model:
        name, rest = m.split('=', 1)
        parts = rest.split(':')
        specs.append((name, parts[0], float(parts[1]), int(parts[2]) if len(parts) > 2 else 0))
    pad = max(k for *_, k in specs)

    flights = load_groundtruth(a.groundtruth)
    for fid, fl in flights.items():
        fl['split'] = split_of(fid, a.val_fraction)
    clips, frames, used = [], {}, set()
    for fid, start in candidate_clips(flights, a.length, pad):
        if fid in used:
            continue
        paths = download_clip(flights, fid, start, a.length, pad,
                              os.path.join(a.cache, '%s_%d' % (fid[:12], start)))
        if paths is None:
            continue
        clips.append((fid, start))
        frames[(fid, start)] = paths
        used.add(fid)
        if len(clips) == a.clips:
            break
    print('%d complete clips of %d frames from held-out flights' % (len(clips), a.length), flush=True)
    gts = {(fid, start): {f: ground_truth(flights[fid], f) for f in range(start, start + a.length)}
           for fid, start in clips}

    results = {}
    for name, weights, thr, k in specs:
        model = YOLO(weights)
        classes = [i for i, n in model.names.items() if n.lower() in AIRCRAFT_NAMES]
        pooled = {'raw': {}, 'online': {}, 'offline': {}}
        pooled_gt = {}
        for ci, (fid, start) in enumerate(clips):
            dets = run_model(model, classes, frames[(fid, start)], start, a.length, k, a.tile, conf=min(thr, 0.05))
            outs = outputs(dets, thr, a.min_hits, a.max_misses)
            for kind in pooled:
                for f, preds in outs[kind].items():
                    pooled[kind][(ci, f)] = preds
            for f, g in gts[(fid, start)].items():
                pooled_gt[(ci, f)] = g
            print('  %s: clip %d/%d done' % (name, ci + 1, len(clips)), flush=True)
        results[name] = {kind: score(v, pooled_gt) for kind, v in pooled.items()}
        print(name, json.dumps(results[name], indent=1), flush=True)
    if a.json:
        with open(a.json, 'w') as f:
            json.dump({'clips': [[fid, s] for fid, s in clips], 'length': a.length,
                       'min_hits': a.min_hits, 'results': results}, f, indent=2)
    return results


if __name__ == '__main__':
    main()
