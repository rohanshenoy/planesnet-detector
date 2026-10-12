import csv, sys, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
out = sys.argv[1]
plt.rcParams.update({'font.family': 'Arial', 'font.size': 11, 'text.color': '#2b2b2b'})
rows = list(csv.DictReader(open('data/s2_flying/labels.csv')))
air = [r for r in rows if r['label'] == 'aircraft']
bg = [r for r in rows if r['label'] == 'background']

def chip(r):
    a = np.asarray(Image.open('data/s2_flying/chips/' + r['chip']).convert('RGB')).astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99.5)
    return np.clip((a - lo) / max(hi - lo, 1), 0, 1)

# 1. label offset at CDG: converted label (x) vs aircraft (green copy, chip centre)
cdg = sorted([r for r in air if r['airport'] == 'CDG' and r['match'] == 'file shift' and 25 <= float(r['score']) <= 60],
             key=lambda r: r['chip'])
rng = np.random.default_rng(4); pick = [cdg[i] for i in rng.choice(len(cdg), 4, replace=False)]
fig, axs = plt.subplots(1, 4, figsize=(12, 3.4))
for ax, r in zip(axs, pick):
    ax.imshow(chip(r), interpolation='nearest')
    lx = float(r['label_x']) - (int(r['x']) - 32) - 0.5; ly = float(r['label_y']) - (int(r['y']) - 32) - 0.5
    ax.plot(lx, ly, marker='x', ms=13, mew=3, color='white'); ax.plot(lx, ly, marker='x', ms=11, mew=1.6, color='#d03b3b')
    ax.add_patch(plt.Rectangle((28.5, 28.5), 6, 6, fill=False, ec='#fab219', lw=2))
    ax.annotate('', xy=(29, 31.5), xytext=(lx + 2, ly), arrowprops=dict(arrowstyle='->', color='white', lw=1.6))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('%s  %s' % (r['airport'], r['time'][:10]), fontsize=11, color='#52514e')
fig.tight_layout(); fig.savefig(out + '/s2_label_offset.png', dpi=200, facecolor='white'); plt.close(fig)

# 2. matched aircraft across airports (2 x 6) and backgrounds (1 x 6)
aps = sorted({r['airport'] for r in air})
rng = np.random.default_rng(11)
sel = []
for ap in ['CDG', 'LGW', 'BCN', 'FCO', 'BRU', 'DUB', 'AGP', 'DUS', 'HEL', 'CPH', 'LGW', 'BCN']:
    c = [r for r in air if r['airport'] == ap and float(r['score']) >= 20 and r not in sel]
    sel.append(c[rng.integers(len(c))])
fig, axs = plt.subplots(2, 6, figsize=(12, 4.4))
for ax, r in zip(axs.flat, sel):
    ax.imshow(chip(r), interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(r['airport'], fontsize=11, color='#52514e')
fig.tight_layout(); fig.savefig(out + '/s2_flying_examples.png', dpi=200, facecolor='white'); plt.close(fig)
fig, axs = plt.subplots(1, 6, figsize=(12, 2.4))
for ax, r in zip(axs, [bg[i] for i in rng.choice(len(bg), 6, replace=False)]):
    ax.imshow(chip(r), interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig(out + '/s2_flying_background.png', dpi=200, facecolor='white'); plt.close(fig)
print('ok', [r['chip'] for r in pick])
