"""Collect every number the deck shows from the result files into build/deck_data.json (run from the repo root)."""
import csv, json, os, sys

out = sys.argv[1]
d = {}
R = 'data/aot_part2/results/'
d['part2'] = {n: json.load(open(R + n + '.json')) for n in
              ('coco_yolo11n', 'yolo11n_1frame', 'yolo11n_3frame', 'yolo11s_3frame', 'yolo11n-p2_3frame',
               'yolo11s_3frame_p13_e50')}
rows = list(csv.DictReader(open(R + 's3f_p13_e50_training_results.csv')))
key = lambda s: [c for c in rows[0] if c.strip() == s][0]
d['train50'] = {c: [round(float(r[key(c)]), 4) for r in rows] for c in
                ('epoch', 'train/box_loss', 'val/box_loss', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)')}
d['s2_flying'] = json.load(open('data/s2_flying_eval.json'))
d['s2_airborne'] = json.load(open(os.path.join(os.path.dirname(out), 's2_airborne_eval.json')))
t = open('data/s2_flying_build.log').read()
d['s2_build'] = json.loads(t[t.rindex('\n{\n') + 1:])
cnt = lambda p: len(os.listdir(p)) if os.path.isdir(p) else 0
d['crops'] = {'p1_train': cnt('data/aot_train_p1/temporal/images/train'), 'p1_val': cnt('data/aot_train_p1/temporal/images/val'),
              'p3_train': cnt('data/aot_train_p3/temporal/images/train'), 'part2_test': cnt('data/aot_part2/temporal/images/val')}
# full-frame clips (sky.evaluate_clips + threshold-free AP from its cached detections)
d['clips'] = json.load(open('data/aot_clips/results.json'))
d['clips_ap'] = json.load(open('data/aot_clips/full_frame_ap.json'))
# S2Net: cross-validation by airport and the independent test
d['s2net'] = json.load(open('data/s2_flying/s2net_results.json'))
# recall at matched clutter false-alarm rates on data/s2_airborne, S2Net vs PlaneNet
import numpy as np
from satellite import train_s2
from satellite.s2_model import load as load_s2
from satellite.evaluate_s2 import scores as plane_scores
from satellite.model import load as load_plane
rows, chips = train_s2.read_set('data/s2_airborne')
lab = np.array([r['label'] for r in rows]); pos = np.isin(lab, ['aircraft', 'probable'])
sc = {'S2Net': train_s2.predict(load_s2('models/s2net.pt'), chips, train_s2.examples(rows, chips, with_peaks=False)),
      'PlaneNet': plane_scores(load_plane('models/satellite_planenet.pt'), rows, 'data/s2_airborne', False, 'cpu')}
d['matched_fpr'] = {}
for name, s in sc.items():
    clutter = np.sort(s[lab == 'clutter'])[::-1]
    d['matched_fpr'][name] = {str(f): int((s[pos] > clutter[int(np.floor(f * len(clutter)))]).sum()) for f in (0.05, 0.081, 0.15)}
# the part-1-only 50-epoch run (data vs epochs), once scored
if os.path.exists(R + 'yolo11s_3frame_p1_e50.json'):
    d['part2']['yolo11s_3frame_p1_e50'] = json.load(open(R + 'yolo11s_3frame_p1_e50.json'))
json.dump(d, open(out, 'w'), indent=1)
print('wrote', out)
