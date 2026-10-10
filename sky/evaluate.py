"""
Evaluate an aircraft detector on a YOLO-format dataset (e.g. from prepare_aot).

AOT targets are often under 10 px, where IoU is dominated by one-pixel box
errors, so a prediction counts as a hit when its centre is within
max(min_tol, 0.5 * box size) px of a ground-truth aircraft centre. Aircraft
are airplane, helicopter and drone; predictions that land on a bird or an
unidentified object are ignored rather than counted as false positives.
Any model works as long as its class names are aircraft-like (COCO weights
included), so the off-the-shelf model can be compared with fine-tuned ones.

Example:
    python -m sky.evaluate yolo11n.pt data/aot/single --meta data/aot/meta.json
    python -m sky.evaluate runs/sky/temporal/weights/best.pt data/aot/temporal --meta data/aot/meta.json
"""

import argparse
import glob
import json
import os

import numpy as np

from .detect import AIRCRAFT_NAMES

GT_AIRCRAFT = {0, 1, 3}      # prepare_aot classes: airplane, helicopter, drone
GT_IGNORE = {2, 4}           # bird, unknown


def read_labels(path, size):
    """YOLO label file -> list of (cls, cx, cy, w, h) in pixels."""
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        c, x, y, w, h = line.split()
        out.append((int(c), float(x) * size, float(y) * size, float(w) * size, float(h) * size))
    return out


def match_image(preds, gts, min_tol=8.0):
    """Greedy centre-distance matching for one image.

    preds: list of (score, cx, cy); gts: list of (cls, cx, cy, w, h).
    Returns (records, hit) where records is a list of (score, is_tp) for every
    non-ignored prediction and hit[i] is the best score that matched gt i.
    """
    hit = [None] * len(gts)
    records = []
    for score, px, py in sorted(preds, reverse=True):
        best, best_d = None, np.inf
        for i, (c, gx, gy, w, h) in enumerate(gts):
            d = np.hypot(px - gx, py - gy)
            if d <= max(min_tol, 0.5 * max(w, h)) and d < best_d and (c in GT_IGNORE or hit[i] is None):
                best, best_d = i, d
        if best is None:
            records.append((score, False))
        elif gts[best][0] in GT_IGNORE:
            continue
        else:
            hit[best] = score
            records.append((score, True))
    return records, hit


def average_precision(records, n_gt):
    """All-point interpolated AP plus the precision/recall/threshold at best F1."""
    if n_gt == 0:
        return {'ap': float('nan')}
    r = sorted(records, reverse=True)
    tp = np.cumsum([t for _, t in r]) if r else np.zeros(0)
    fp = np.cumsum([not t for _, t in r]) if r else np.zeros(0)
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1)
    ap, prev_r = 0.0, 0.0
    env = np.maximum.accumulate(precision[::-1])[::-1] if len(r) else precision
    for p_, r_ in zip(env, recall):
        ap += p_ * (r_ - prev_r)
        prev_r = r_
    out = {'ap': float(ap), 'max_recall': float(recall[-1]) if len(r) else 0.0}
    if len(r):
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
        i = int(np.argmax(f1))
        out.update(best_f1=float(f1[i]), precision=float(precision[i]),
                   recall=float(recall[i]), threshold=float(r[i][0]))
    return out


def evaluate(model, data_dir, split='val', meta=None, conf=0.01, imgsz=640, batch=16):
    names = model.names
    keep = [i for i, n in names.items() if n.lower() in AIRCRAFT_NAMES]
    images = sorted(glob.glob(os.path.join(data_dir, 'images', split, '*.png')))
    records, gt_info, n_gt = [], [], 0
    for b in range(0, len(images), batch):
        chunk = images[b:b + batch]
        results = model.predict(chunk, conf=conf, imgsz=imgsz, classes=keep, verbose=False)
        for path, res in zip(chunk, results):
            size = res.orig_shape[0]
            name = os.path.splitext(os.path.basename(path))[0]
            gts = read_labels(os.path.join(data_dir, 'labels', split, name + '.txt'), size)
            xyxy = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
            scores = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.zeros(0)
            preds = [(float(s), (x0 + x1) / 2, (y0 + y1) / 2) for s, (x0, y0, x1, y1) in zip(scores, xyxy)]
            recs, hit = match_image(preds, gts)
            records += recs
            objs = (meta or {}).get(name, {}).get('objects', [{}] * len(gts))
            for (c, _, _, w, h), score, m in zip(gts, hit, objs):
                if c in GT_AIRCRAFT:
                    n_gt += 1
                    gt_info.append({'score': score, 'size': max(w, h),
                                    'above_horizon': m.get('above_horizon')})
    summary = average_precision(records, n_gt)
    summary['n_images'], summary['n_aircraft'] = len(images), n_gt
    t = summary.get('threshold')
    if t is not None:  # recall of subgroups at the best-F1 operating point
        def recall_of(group):
            g = [x for x in gt_info if group(x)]
            return (round(sum(x['score'] is not None and x['score'] >= t for x in g) / len(g), 3)
                    if g else None, len(g))
        summary['recall_by_group'] = {
            'above_horizon (sky background)': recall_of(lambda x: x['above_horizon'] == 1),
            'below_horizon (ground background)': recall_of(lambda x: x['above_horizon'] == -1),
            'tiny (<10 px)': recall_of(lambda x: x['size'] < 10),
            'small+ (>=10 px)': recall_of(lambda x: x['size'] >= 10),
        }
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('weights')
    p.add_argument('data', help='dataset dir containing images/<split> and labels/<split>')
    p.add_argument('--split', default='val')
    p.add_argument('--meta', help='meta.json from prepare_aot, for per-group recall')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--json', help='write the summary here')
    a = p.parse_args(argv)

    from ultralytics import YOLO
    meta = json.load(open(a.meta)) if a.meta else None
    summary = evaluate(YOLO(a.weights), a.data, a.split, meta, imgsz=a.imgsz)
    print(json.dumps(summary, indent=2))
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(summary, f, indent=2)
    return summary


if __name__ == '__main__':
    main()
