"""Re-score sky models on AOT part 2 keeping every prediction: PR curves + per-image hits for the deck."""
import glob, json, os, sys
import numpy as np
from ultralytics import YOLO
from sky.evaluate import read_labels, match_image, average_precision
from sky.detect import AIRCRAFT_NAMES

DATA = 'data/aot_part2'
MODELS = {'coco_yolo11n': ('models/coco_yolo11n.pt', 'single'),
          'yolo11n_1frame': ('models/sky_yolo11n_1frame.pt', 'single'),
          'yolo11n_3frame': ('models/sky_yolo11n_3frame.pt', 'temporal'),
          'yolo11s_3frame': ('models/sky_yolo11s_3frame.pt', 'temporal'),
          'yolo11s_3frame_p13_e50': ('models/sky_yolo11s_3frame_e50.pt', 'temporal')}
out = {}
for name, (w, variant) in MODELS.items():
    model = YOLO(w)
    keep = [i for i, n in model.names.items() if n.lower() in AIRCRAFT_NAMES]
    images = sorted(glob.glob(os.path.join(DATA, variant, 'images', 'val', '*.png')))
    records, n_gt, per_image = [], 0, {}
    for b in range(0, len(images), 16):
        chunk = images[b:b + 16]
        for path, res in zip(chunk, model.predict(chunk, conf=0.01, imgsz=640, classes=keep, verbose=False)):
            size = res.orig_shape[0]
            stem = os.path.splitext(os.path.basename(path))[0]
            gts = read_labels(os.path.join(DATA, variant, 'labels', 'val', stem + '.txt'), size)
            xyxy = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
            sc = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.zeros(0)
            preds = [(float(s), (x0 + x1) / 2, (y0 + y1) / 2) for s, (x0, y0, x1, y1) in zip(sc, xyxy)]
            recs, hit = match_image(preds, gts)
            records += recs
            n_gt += sum(c in (0, 1, 3) for c, *_ in gts)
            per_image[stem] = {'hit': hit, 'gts': [list(g) for g in gts],
                               'boxes': [[float(v) for v in bx] + [float(s)] for bx, s in zip(xyxy, sc) if s >= 0.1]}
    summ = average_precision(records, n_gt)
    r = sorted(records, reverse=True)
    tp = np.cumsum([t for _, t in r]); fp = np.cumsum([not t for _, t in r])
    rec, prec = tp / n_gt, tp / np.maximum(tp + fp, 1)
    env = np.maximum.accumulate(prec[::-1])[::-1]
    grid = np.round(np.arange(0, 0.901, 0.01), 2)
    curve = [float(env[np.searchsorted(rec, g)]) if g <= rec[-1] else None for g in grid]
    out[name] = {'ap': summ['ap'], 'threshold': summ['threshold'], 'n_gt': n_gt, 'recall_grid': grid.tolist(),
                 'precision': curve, 'per_image': per_image}
    print(name, 'AP %.4f' % summ['ap'], flush=True)
json.dump(out, open(sys.argv[1], 'w'))
