"""
Score the satellite classifier on real Sentinel-2 aircraft in flight.

data/s2_airborne/ holds 64x64 chips (10 m pixels, reflectance / 0.3 as 8-bit)
around candidates from a band-offset finder at six busy airports, labelled by
eye: 'aircraft', 'probable', 'clutter' (finder false positives), plus random
'background' chips from the same scenes.

Sentinel-2 captures blue, green and red a fraction of a second apart, so an
aircraft at 10 m resolution appears as three colour copies 3-40 px apart,
wider than the 60 m footprint of a PlanesNet chip. Two inputs are scored:

  as-is       the 60 m (6 px) square around the green copy, upsampled to 20 px
  realigned   red and blue first shifted onto the green copy using the
              finder's blue/red positions (motion compensation), then cropped

Example:
    python -m satellite.evaluate_s2 models/plane.pt data/s2_airborne
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from .model import load
from .train import auc


def chip_input(chip, row, realign, size_px=6):
    """(20, 20, 3) model input from a 64x64 S2 chip centred on the candidate."""
    img = chip.astype(np.float32) / 255.
    if realign and row['blue_x'] != '':
        cx, cy = float(row['x']), float(row['y'])
        for c, (bx, by) in ((0, (row['red_x'], row['red_y'])), (2, (row['blue_x'], row['blue_y']))):
            dy, dx = cy - float(by), cx - float(bx)
            img[..., c] = ndimage.shift(img[..., c], (dy, dx), order=1, mode='nearest')
    h = size_px // 2
    crop = img[32 - h:32 + h, 32 - h:32 + h]
    return np.asarray(Image.fromarray((crop * 255).astype(np.uint8)).resize((20, 20), Image.BILINEAR),
                      dtype=np.float32) / 255.


@torch.no_grad()
def scores(model, rows, root, realign, device):
    xs = [chip_input(np.asarray(Image.open(os.path.join(root, 'chips', r['chip'])).convert('RGB')), r, realign)
          for r in rows]
    x = torch.from_numpy(np.stack(xs).transpose(0, 3, 1, 2)).to(device)
    return torch.sigmoid(model(x)).flatten().cpu().numpy()


def summarise(s, rows):
    lab = np.array([r['label'] for r in rows])
    pos = np.isin(lab, ['aircraft', 'probable'])
    out = {}
    for neg_name, neg in (('vs clutter', lab == 'clutter'), ('vs background', lab == 'background')):
        m = pos | neg
        out['auc ' + neg_name] = round(auc(s[m], pos[m]), 3)
    out['recall@0.5 (aircraft+probable)'] = round(float((s[pos] > 0.5).mean()), 3)
    out['false positive rate@0.5, clutter'] = round(float((s[lab == 'clutter'] > 0.5).mean()), 3)
    out['false positive rate@0.5, background'] = round(float((s[lab == 'background'] > 0.5).mean()), 3)
    out['mean score'] = {k: round(float(s[lab == k].mean()), 3) for k in ('aircraft', 'probable', 'clutter', 'background')}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model')
    p.add_argument('data', help='directory with labels.csv and chips/')
    p.add_argument('--json')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args(argv)

    rows = list(csv.DictReader(open(os.path.join(a.data, 'labels.csv'))))
    model = load(a.model, a.device)
    lab = [r['label'] for r in rows]
    finder = {'finder precision (aircraft only)': round(lab.count('aircraft') / max(sum(r['source'] == 'finder candidate' for r in rows), 1), 3),
              'finder precision (incl. probable)': round((lab.count('aircraft') + lab.count('probable')) /
                                                         max(sum(r['source'] == 'finder candidate' for r in rows), 1), 3)}
    result = {'counts': {k: lab.count(k) for k in ('aircraft', 'probable', 'clutter', 'background')}, 'finder': finder}
    for name, realign in (('as-is', False), ('realigned', True)):
        result[name] = summarise(scores(model, rows, a.data, realign, a.device), rows)
    print(json.dumps(result, indent=2))
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(result, f, indent=2)
    return result


if __name__ == '__main__':
    main()
