"""
Build YOLO training data from the Airborne Object Tracking (AOT) dataset.

AOT (Amazon Prime Air) is greyscale 2448x2048 video taken from aircraft, with
boxes for airplanes, helicopters, birds, drones and unidentified airborne
objects, most of them only a few pixels across. Full frames are far too large
for those targets, so each sample is a 640x640 crop at native resolution
around a labelled object (plus some crops with nothing in them).

Two datasets with identical crops and labels are written:

  single/    the frame itself (greyscale)
  temporal/  frames t-k, t, t+k as the three channels, each neighbour aligned
             to frame t by phase correlation to cancel the camera's own motion.
             Static scenery comes out grey; anything moving relative to it
             leaves a coloured fringe the detector can learn from.

Train and val are split by flight, so no flight appears in both.

Example:
    curl -o groundtruth.csv https://airborne-obj-detection-challenge-training.s3.amazonaws.com/part1/ImageSets/groundtruth.csv
    python -m sky.prepare_aot groundtruth.csv data/aot --train 1200 --val 300
"""

import argparse
import csv
import hashlib
import json
import os
import re
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import temporal

BUCKET = 'https://airborne-obj-detection-challenge-training.s3.amazonaws.com'
CLASSES = ['airplane', 'helicopter', 'bird', 'drone', 'unknown']
CLASS_OF = {'Airplane': 0, 'Helicopter': 1, 'Bird': 2, 'Flock': 2, 'Drone': 3, 'Airborne': 4}
AIRCRAFT = {0, 1, 3}


def load_groundtruth(path):
    """Return {flight: {'frames': {frame: img_name}, 'objects': {frame: [obj]}}}."""
    flights = defaultdict(lambda: {'frames': {}, 'objects': defaultdict(list)})
    with open(path) as f:
        for r in csv.DictReader(f):
            fl = flights[r['flight_id']]
            frame = int(r['frame'])
            fl['frames'][frame] = r['img_name']
            if r['id']:
                cls = CLASS_OF.get(re.sub(r'\d+$', '', r['id']))
                if cls is None:
                    continue
                fl['objects'][frame].append({
                    'cls': cls,
                    'box': [float(r['gt_left']), float(r['gt_top']),
                            float(r['gt_right']), float(r['gt_bottom'])],
                    'above_horizon': float(r['is_above_horizon'] or 0),
                    'range_m': float(r['range_distance_m']) if r['range_distance_m'] else None,
                })
    return flights


def split_of(flight, val_fraction):
    h = int(hashlib.md5(flight.encode()).hexdigest()[:8], 16) / 0xffffffff
    return 'val' if h < val_fraction else 'train'


def plan_samples(flights, split, n, k, rng, per_flight=6, min_gap=20, negative_fraction=0.15):
    """Choose (flight, frame, kind) samples for one split."""
    pos_pool, neg_pool = [], []
    for fid, fl in flights.items():
        if fl['split'] != split:
            continue
        frames = fl['frames']
        ok = [f for f in frames if f - k in frames and f + k in frames]
        pos = [f for f in ok if any(o['cls'] in AIRCRAFT for o in fl['objects'].get(f, []))]
        if not pos:
            continue
        rng.shuffle(pos)
        chosen = []
        for f in pos:
            if all(abs(f - c) >= min_gap for c in chosen):
                chosen.append(f)
            if len(chosen) == per_flight:
                break
        pos_pool += [(fid, f, 'pos') for f in chosen]
        neg = [f for f in ok if not fl['objects'].get(f)]
        if neg:
            neg_pool.append((fid, neg[rng.integers(len(neg))], 'neg'))
    rng.shuffle(pos_pool)
    rng.shuffle(neg_pool)
    n_neg = min(int(n * negative_fraction), len(neg_pool))
    return pos_pool[:n - n_neg] + neg_pool[:n_neg]


def fetch(flight, img_name):
    url = '%s/part1/Images/%s/%s' % (BUCKET, flight, img_name)
    with urllib.request.urlopen(url, timeout=60) as r:
        buf = np.frombuffer(r.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError('could not decode %s' % url)
    return img


def crop_window(objs, w, h, size, rng):
    """Top-left of a size x size crop containing a random aircraft (or anywhere)."""
    targets = [o for o in objs if o['cls'] in AIRCRAFT]
    if not targets:
        return int(rng.integers(0, w - size + 1)), int(rng.integers(0, h - size + 1))
    x0, y0, x1, y1 = targets[rng.integers(len(targets))]['box']
    m = 32
    lo_x, hi_x = max(0, int(x1) + m - size), min(w - size, int(x0) - m)
    lo_y, hi_y = max(0, int(y1) + m - size), min(h - size, int(y0) - m)
    cx = int(rng.integers(lo_x, hi_x + 1)) if hi_x >= lo_x else int(np.clip((x0 + x1 - size) / 2, 0, w - size))
    cy = int(rng.integers(lo_y, hi_y + 1)) if hi_y >= lo_y else int(np.clip((y0 + y1 - size) / 2, 0, h - size))
    return cx, cy


def crop_labels(objs, x, y, size, min_px=2):
    """YOLO label lines and metadata for objects whose centre lies in the crop."""
    lines, meta = [], []
    for o in objs:
        x0, y0, x1, y1 = o['box']
        if not (x <= (x0 + x1) / 2 < x + size and y <= (y0 + y1) / 2 < y + size):
            continue
        bx0, by0 = max(x0 - x, 0), max(y0 - y, 0)
        bx1, by1 = min(x1 - x, size), min(y1 - y, size)
        bw, bh = max(bx1 - bx0, min_px), max(by1 - by0, min_px)
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        lines.append('%d %.6f %.6f %.6f %.6f' % (o['cls'], cx / size, cy / size, bw / size, bh / size))
        meta.append({'cls': CLASSES[o['cls']], 'box': [bx0, by0, bx1, by1],
                     'above_horizon': o['above_horizon'], 'range_m': o['range_m']})
    return lines, meta


def build_sample(job, flights, out, k, size):
    (fid, frame, kind), split, seed = job
    rng = np.random.default_rng(seed)
    fl = flights[fid]
    imgs = [fetch(fid, fl['frames'][f]) for f in (frame - k, frame, frame + k)]
    h, w = imgs[1].shape
    objs = fl['objects'].get(frame, [])
    x, y = crop_window(objs, w, h, size, rng)
    crops = [im[y:y + size, x:x + size] for im in imgs]
    lines, meta = crop_labels(objs, x, y, size)
    name = '%s_%d_%d_%d' % (fid[:12], frame, x, y)
    cv2.imwrite(os.path.join(out, 'single', 'images', split, name + '.png'), crops[1])
    cv2.imwrite(os.path.join(out, 'temporal', 'images', split, name + '.png'), temporal.stack(*crops))
    for variant in ('single', 'temporal'):
        with open(os.path.join(out, variant, 'labels', split, name + '.txt'), 'w') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))
    return name, split, meta


def write_yaml(out, variant):
    path = os.path.join(out, variant + '.yaml')
    with open(path, 'w') as f:
        f.write('path: %s\ntrain: images/train\nval: images/val\nnames:\n'
                % os.path.abspath(os.path.join(out, variant)))
        for i, c in enumerate(CLASSES):
            f.write('  %d: %s\n' % (i, c))
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('groundtruth', help='AOT part1 groundtruth.csv')
    p.add_argument('out')
    p.add_argument('--train', type=int, default=1200, help='training crops')
    p.add_argument('--val', type=int, default=300, help='validation crops')
    p.add_argument('--val-fraction', type=float, default=0.2, help='fraction of flights held out')
    p.add_argument('--k', type=int, default=2, help='frame offset of the temporal neighbours')
    p.add_argument('--size', type=int, default=640)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--oversample', type=float, default=1.35,
                   help='plan extra samples; some AOT frames are missing from the bucket')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    flights = load_groundtruth(a.groundtruth)
    for fid, fl in flights.items():
        fl['split'] = split_of(fid, a.val_fraction)
    jobs = []
    for split, n in (('train', a.train), ('val', a.val)):
        for d in ('images', 'labels'):
            for v in ('single', 'temporal'):
                os.makedirs(os.path.join(a.out, v, d, split), exist_ok=True)
        jobs += [(s, split, int(rng.integers(2 ** 31))) for s in plan_samples(flights, split, int(n * a.oversample), a.k, rng)]
    print('%d flights; building %d samples' % (len(flights), len(jobs)))

    meta, failed = {}, 0
    with ThreadPoolExecutor(a.workers) as ex:
        futures = [ex.submit(build_sample, j, flights, a.out, a.k, a.size) for j in jobs]
        for i, fut in enumerate(futures):
            try:
                name, split, m = fut.result()
                meta[name] = {'split': split, 'objects': m}
            except Exception as e:  # skip unreadable frames rather than abort
                failed += 1
                print('skip:', e)
            if (i + 1) % 100 == 0:
                print('%d / %d' % (i + 1, len(jobs)), flush=True)
    with open(os.path.join(a.out, 'meta.json'), 'w') as f:
        json.dump(meta, f)
    for v in ('single', 'temporal'):
        print('wrote', write_yaml(a.out, v))
    print('%d samples, %d failed' % (len(meta), failed))


if __name__ == '__main__':
    main()
