"""
Train the satellite aircraft classifier for in-flight detection.

PlanesNet is used for aircraft appearance, and every epoch mixes in synthetic
in-flight samples (PlanesNet aircraft composited onto cloud / sea / haze
backgrounds with band-offset ghosting) plus hard negatives. If you have real
labelled in-flight chips, pass them with --airborne-train / --airborne-val;
the checkpoint kept is the one that does best on the in-flight validation set,
not on PlanesNet, because PlanesNet accuracy says little about aircraft over
cloud.

Example:
    python -m satellite.train --planesnet planesnet.json --out models/plane.pt \
        --backgrounds data/cloud_scenes --airborne-val data/airborne_val
"""

import argparse
import os

import numpy as np
import torch
from scipy.stats import rankdata
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset

from . import augment, data
from .model import PlaneNet, save


def auc(scores, labels):
    """Area under the ROC curve (Mann-Whitney U), no sklearn dependency."""
    labels = np.asarray(labels).astype(bool)
    pos, neg = labels.sum(), (~labels).sum()
    if pos == 0 or neg == 0:
        return float('nan')
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    scores, labels = [], []
    for x, y in loader:
        scores.append(torch.sigmoid(model(x.to(device)).flatten()).cpu())
        labels.append(y)
    s, l = torch.cat(scores).numpy(), torch.cat(labels).numpy()
    pred = s > 0.5
    tp = (pred & (l == 1)).sum()
    return {'acc': float((pred == l).mean()), 'auc': auc(s, l),
            'precision': float(tp / max(pred.sum(), 1)),
            'recall': float(tp / max((l == 1).sum(), 1))}


def synthetic_eval_set(cutouts, n, backgrounds, textures=None, seed=1234):
    """Fixed in-flight validation set built from held-out aircraft cut-outs."""
    empty = np.zeros((0, 20, 20, 3), np.uint8)
    ds = data.ChipDataset(empty, np.zeros(0, np.int64), cutouts, n, backgrounds, seed=seed)
    if textures is not None:
        ds.textures = textures
    items = [ds[i] for i in range(len(ds))]
    return torch.utils.data.TensorDataset(torch.stack([x for x, _ in items]),
                                          torch.stack([y for _, y in items]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--planesnet', required=True, help='path to planesnet.json')
    p.add_argument('--out', required=True, help='output checkpoint (.pt)')
    p.add_argument('--backgrounds', help='folder of aircraft-free cloud/sea/land scenes')
    p.add_argument('--airborne-train', help='folder with plane/ and no-plane/ real in-flight chips')
    p.add_argument('--airborne-val', help='held-out real in-flight chips (plane/, no-plane/)')
    p.add_argument('--synthetic-ratio', type=float, default=1.0,
                   help='synthetic samples per epoch as a fraction of real training chips')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args(argv)

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    X, y = data.load_planesnet(a.planesnet)
    perm = rng.permutation(len(X))
    n_val = len(X) // 5
    vi, ti = perm[:n_val], perm[n_val:]
    Xtr, ytr, Xva, yva = X[ti], y[ti], X[vi], y[vi]

    train_cut = data.extract_plane_cutouts(Xtr, ytr)
    val_cut = data.extract_plane_cutouts(Xva, yva)
    print('PlanesNet: %d train / %d val chips; %d / %d clean aircraft cut-outs'
          % (len(Xtr), len(Xva), len(train_cut[0]), len(val_cut[0])))
    bank = data.BackgroundBank(a.backgrounds) if a.backgrounds else None

    n_syn = int(a.synthetic_ratio * len(Xtr))
    train_sets = [data.ChipDataset(Xtr, ytr, train_cut, n_syn, bank, seed=a.seed)]
    if a.airborne_train:
        Xa, ya = data.load_chip_folder(a.airborne_train)
        print('Real in-flight training chips: %d (%d planes)' % (len(Xa), ya.sum()))
        train_sets.append(data.ChipDataset(Xa, ya, seed=a.seed + 1))
    train_loader = DataLoader(ConcatDataset(train_sets), batch_size=a.batch_size, shuffle=True,
                              num_workers=a.workers, worker_init_fn=data.worker_init,
                              persistent_workers=a.workers > 0)

    def loader(ds):
        return DataLoader(ds, batch_size=1024)

    val = {'planesnet': loader(data.ChipDataset(Xva, yva, augment_real=False))}
    if val_cut[0]:
        val['synthetic_airborne'] = loader(synthetic_eval_set(val_cut, max(2000, len(Xva) // 2), bank, Xva[yva == 0]))
    if a.airborne_val:
        Xv, yv = data.load_chip_folder(a.airborne_val)
        val['real_airborne'] = loader(data.ChipDataset(Xv, yv, augment_real=False))
    select = 'real_airborne' if 'real_airborne' in val else (
        'synthetic_airborne' if 'synthetic_airborne' in val else 'planesnet')
    print('Model selection on: %s AUC' % select)

    model = PlaneNet(width=a.width).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs,
                                                steps_per_epoch=len(train_loader))
    loss_fn = nn.BCEWithLogitsLoss()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    best = -1.0
    for epoch in range(a.epochs):
        model.train()
        total = 0.0
        for x, t in train_loader:
            x, t = x.to(a.device), t.to(a.device)
            loss = loss_fn(model(x).flatten(), t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(x)
        metrics = {k: evaluate(model, v, a.device) for k, v in val.items()}
        line = '  '.join('%s acc=%.3f auc=%.3f' % (k, m['acc'], m['auc']) for k, m in metrics.items())
        print('epoch %d/%d loss=%.4f  %s' % (epoch + 1, a.epochs, total / len(train_loader.dataset), line))
        score = metrics[select]['auc']
        if np.isnan(score):  # single-class validation set
            score = metrics[select]['acc']
        if score > best:
            best = score
            save(model, a.out, metrics=metrics, epoch=epoch + 1)
            print('  saved %s (%s auc=%.3f)' % (a.out, select, score))
    return best


if __name__ == '__main__':
    main()
