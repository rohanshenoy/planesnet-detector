import json, sys, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
D = sys.argv[1]
pr = json.load(open(D + '/pr_curves.json'))
meta = json.load(open('data/aot_part2/meta.json'))
for n, v in pr.items():
    g, p = v['recall_grid'], v['precision']
    at = lambda r: p[g.index(r)] if r in g and p[g.index(r)] is not None else None
    print('%-24s AP %.3f  P@R=0.5 %s  P@R=0.7 %s  P@R=0.8 %s  max R %.2f' % (n, v['ap'], at(0.5), at(0.7), at(0.8),
          max(r for r, q in zip(g, p) if q is not None)))
a, b = pr['yolo11s_3frame'], pr['yolo11s_3frame_p13_e50']
ta, tb = a['threshold'], b['threshold']
cands = []
for stem, im in b['per_image'].items():
    objs = meta[stem]['objects']
    for k, (gt, hb) in enumerate(zip(im['gts'], im['hit'])):
        ha = a['per_image'][stem]['hit'][k]
        if gt[0] in (0, 1, 3) and hb is not None and hb >= tb and (ha is None or ha < ta):
            size = max(gt[3], gt[4]); hor = objs[k].get('above_horizon') if k < len(objs) else None
            cands.append((stem, k, size, hor))
print('found by 50 epochs, missed by 12 epochs:', len(cands))
rng = np.random.default_rng(7)
ground = [c for c in cands if c[3] == -1 and c[2] >= 8]; tiny = [c for c in cands if c[2] < 10 and c[3] == 1]
pick = [ground[i] for i in rng.choice(len(ground), 2, replace=False)] + [tiny[i] for i in rng.choice(len(tiny), 2, replace=False)]
plt.rcParams.update({'font.family': 'Arial', 'font.size': 12})
fig, axs = plt.subplots(2, 4, figsize=(12, 6.4))
for j, (stem, k, size, hor) in enumerate(pick):
    im = np.asarray(Image.open('data/aot_part2/single/images/val/%s.png' % stem).convert('L')).astype(np.float32)
    cls, cx, cy, gw, gh = b['per_image'][stem]['gts'][k]
    half = 80
    x0, y0 = int(np.clip(cx - half, 0, 640 - 2 * half)), int(np.clip(cy - half, 0, 640 - 2 * half))
    win = im[y0:y0 + 2 * half, x0:x0 + 2 * half]
    lo, hi = np.percentile(win, 1), np.percentile(win, 99.5)
    win = np.clip((win - lo) / max(hi - lo, 1), 0, 1)
    for i, (m, t, name) in enumerate(((a, ta, '12 epochs, 1,548 crops'), (b, tb, '50 epochs, 10,051 crops'))):
        ax = axs[i, j]
        ax.imshow(win, cmap='gray', interpolation='nearest', extent=(x0, x0 + 2 * half, y0 + 2 * half, y0))
        ax.add_patch(plt.Circle((cx, cy), max(gw, gh) / 2 + 9, fill=False, ec='#1baf7a', lw=2))
        for bx0, by0, bx1, by1, sc in m['per_image'][stem]['boxes']:
            if sc >= t and x0 <= (bx0 + bx1) / 2 <= x0 + 2 * half and y0 <= (by0 + by1) / 2 <= y0 + 2 * half:
                ax.add_patch(plt.Rectangle((bx0 - 2, by0 - 2), bx1 - bx0 + 4, by1 - by0 + 4, fill=False, ec='#eb6834', lw=2))
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_title('%s, %d px' % ('Against ground' if hor == -1 else 'Against sky', round(size)), fontsize=13, color='#2b2b2b')
        if j == 0:
            ax.set_ylabel(name, fontsize=13, color='#2b2b2b')
fig.tight_layout(); fig.savefig(D + '/img/sky_before_after.png', dpi=200, facecolor='white')
print('picked', pick)
