"""
Train S2Net on real Sentinel-2 aircraft in flight (data/s2_flying) and score it by airport.

Positives: the aircraft chips of data/s2_flying, a 48x48 window around the green copy.
Negatives: its background chips, twice: centred as cut (random ground, sea, cloud) and
re-centred on the chip's strongest single-band colour peak (a red roof, a blue pool), so
that "something colourful in the middle" is not enough to score as an aircraft.

Scoring: k-fold cross-validation by airport, so every chip is scored by a model that never
saw its airport; then a model trained on all airports is saved and scored on --test sets,
such as data/s2_airborne (2025, other airports, finder clutter labelled by eye).

Example:
    python -m satellite.train_s2 data/s2_flying --out models/s2net.pt --test data/s2_airborne --json s2net.json
"""

import argparse
import csv
import json
import math
import os

import numpy as np
import torch
from PIL import Image
from torch import nn

from .build_s2_flying import colour_excess
from .s2_model import WINDOW, S2Net, save
from .train import auc

CHIP = 64
POSITIVE = ('aircraft', 'probable')


def read_set(root):
    """(rows of labels.csv, uint8 chips (N, 64, 64, 3))."""
    rows = list(csv.DictReader(open(os.path.join(root, 'labels.csv'))))
    chips = np.stack([np.asarray(Image.open(os.path.join(root, 'chips', r['chip'])).convert('RGB')) for r in rows])
    return rows, chips


def crop(chip, cx, cy, size=WINDOW):
    """size x size window centred on (cx, cy), clamped inside the chip."""
    x0 = int(np.clip(cx - size // 2, 0, chip.shape[1] - size))
    y0 = int(np.clip(cy - size // 2, 0, chip.shape[0] - size))
    return chip[y0:y0 + size, x0:x0 + size]


def peak_centre(chip, size=WINDOW):
    """Centre of the strongest single-band colour peak whose window stays inside the chip."""
    s = colour_excess(chip.astype(np.float32) / 255 * 0.3).max(-1)
    lo, hi = size // 2, chip.shape[0] - size // 2
    sub = s[lo:hi + 1, lo:hi + 1]
    y, x = np.unravel_index(np.argmax(sub), sub.shape)
    return int(x + lo), int(y + lo)


def examples(rows, chips, with_peaks=True):
    """(chip index, cx, cy, label, kind) for every example: aircraft centred; background centred and peak-centred."""
    out = []
    for i, r in enumerate(rows):
        c = CHIP // 2
        if r['label'] in POSITIVE:
            out.append((i, c, c, 1, r['label']))
        else:
            out.append((i, c, c, 0, r['label']))
            if with_peaks and r['label'] == 'background':
                out.append((i, *peak_centre(chips[i]), 0, 'background (peak-centred)'))
    return out


def augment(chip, cx, cy, rng, shift=2):
    """Random shift of the window, 90-degree rotation, flips and brightness/contrast."""
    w = crop(chip, cx + rng.integers(-shift, shift + 1), cy + rng.integers(-shift, shift + 1)).astype(np.float32) / 255
    w = np.rot90(w, rng.integers(4))
    if rng.random() < 0.5:
        w = w[:, ::-1]
    w = (w - w.mean()) * rng.uniform(0.8, 1.2) + w.mean() + rng.uniform(-0.05, 0.05)
    return np.clip(w, 0, 1)


def batch_tensor(windows, device):
    return torch.from_numpy(np.ascontiguousarray(np.stack(windows).transpose(0, 3, 1, 2))).float().to(device)


def train(chips, items, epochs=30, batch=128, lr=2e-3, seed=0, device='cpu'):
    """Train S2Net on (chip index, cx, cy, label, kind) items; classes balanced by weighting the positives."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = S2Net().to(device)
    labels = np.array([it[3] for it in items])
    n = len(items)
    steps = epochs * math.ceil(n / batch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((labels == 0).sum() / max((labels == 1).sum(), 1),
                                                           device=device))
    model.train()
    for _ in range(epochs):
        order = rng.permutation(n)
        for b in range(0, n, batch):
            idx = order[b:b + batch]
            x = batch_tensor([augment(chips[items[i][0]], items[i][1], items[i][2], rng) for i in idx], device)
            y = torch.from_numpy(labels[idx]).float().to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return model.eval()


@torch.no_grad()
def predict(model, chips, items, device='cpu', batch=512):
    out = []
    for b in range(0, len(items), batch):
        ws = [crop(chips[i], cx, cy).astype(np.float32) / 255 for i, cx, cy, *_ in items[b:b + batch]]
        out.append(torch.sigmoid(model(batch_tensor(ws, device))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def airport_folds(rows, k):
    """Airports split into k folds with similar numbers of aircraft (largest first, into the smallest fold)."""
    count = {}
    for r in rows:
        count[r['airport']] = count.get(r['airport'], 0) + (r['label'] in POSITIVE)
    folds = [[] for _ in range(k)]
    load = [0] * k
    for ap in sorted(count, key=lambda a: (-count[a], a)):
        j = int(np.argmin(load))
        folds[j].append(ap)
        load[j] += count[ap]
    return folds


def summarise(scores, items, rows, threshold=0.5):
    kinds = np.array([it[4] for it in items])
    y = np.array([it[3] for it in items])
    s = np.asarray(scores)
    out = {'recall@%.1f (aircraft+probable)' % threshold: round(float((s[y == 1] > threshold).mean()), 3)}
    if (kinds == 'aircraft').any() and (kinds == 'probable').any():
        out['recall@%.1f (aircraft only)' % threshold] = round(float((s[kinds == 'aircraft'] > threshold).mean()), 3)
    for neg in sorted(set(kinds[y == 0])):
        m = kinds == neg
        out['auc vs ' + neg] = round(auc(np.r_[s[y == 1], s[m]], np.r_[np.ones((y == 1).sum()), np.zeros(m.sum())]), 3)
        out['false positive rate@%.1f, %s' % (threshold, neg)] = round(float((s[m] > threshold).mean()), 3)
    airports = np.array([rows[it[0]].get('airport', '') for it in items])
    if airports[0]:
        out['recall@%.1f by airport' % threshold] = {a: '%d/%d' % ((s[(y == 1) & (airports == a)] > threshold).sum(),
                                                                  ((y == 1) & (airports == a)).sum())
                                                     for a in sorted(set(airports[y == 1]))}
    out['counts'] = {k: int((kinds == k).sum()) for k in sorted(set(kinds))}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('data', help='training set (labels.csv with an airport column, chips/)')
    p.add_argument('--out', help='save the model trained on all airports here')
    p.add_argument('--test', nargs='*', default=[], help='other test sets to score the final model on')
    p.add_argument('--folds', type=int, default=5, help='cross-validation folds by airport (0 to skip)')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--json')
    a = p.parse_args(argv)

    rows, chips = read_set(a.data)
    items = examples(rows, chips)
    result = {'data': a.data, 'epochs': a.epochs, 'seed': a.seed}
    if a.folds:
        folds = airport_folds(rows, a.folds)
        oof = np.zeros(len(items))
        ap_of = np.array([rows[it[0]]['airport'] for it in items])
        for j, held in enumerate(folds):
            test = np.isin(ap_of, held)
            model = train(chips, [it for it, t in zip(items, test) if not t], a.epochs, seed=a.seed, device=a.device)
            oof[test] = predict(model, chips, [it for it, t in zip(items, test) if t], a.device)
            print('fold %d/%d held out %s' % (j + 1, len(folds), ', '.join(held)), flush=True)
        result['cross-validation by airport'] = {'folds': folds, **summarise(oof, items, rows)}
        print(json.dumps(result['cross-validation by airport'], indent=1), flush=True)
    model = train(chips, items, a.epochs, seed=a.seed, device=a.device)
    if a.out:
        save(model, a.out, trained_on=a.data, epochs=a.epochs)
    for t in a.test:
        trows, tchips = read_set(t)
        titems = examples(trows, tchips, with_peaks=False)
        result['test ' + t] = summarise(predict(model, tchips, titems, a.device), titems, trows)
        print(t, json.dumps(result['test ' + t], indent=1), flush=True)
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(result, f, indent=2)
    return result


if __name__ == '__main__':
    main()
