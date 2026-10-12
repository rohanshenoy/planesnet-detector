import numpy as np
import torch

from satellite import train_s2
from satellite.s2_model import WINDOW, S2Net, load, save
from satellite.train import auc


def chip_with(rng, blobs):
    """64x64 uint8 RGB ground with 2x2 blobs bright in one band: [(band, x, y)]."""
    c = 60 + rng.normal(0, 3, (64, 64, 3))
    for band, x, y in blobs:
        c[y:y + 2, x:x + 2, band] += 70
    return np.clip(c, 0, 255).astype(np.uint8)


def test_s2net_shape_and_checkpoint_roundtrip(tmp_path):
    m = S2Net().eval()
    x = torch.rand(4, 3, WINDOW, WINDOW)
    assert m(x).shape == (4,)
    save(m, tmp_path / 'm.pt')
    assert torch.allclose(load(tmp_path / 'm.pt')(x), m(x))


def test_crop_clamps_and_peak_centre_finds_the_colour_blob():
    rng = np.random.default_rng(0)
    chip = chip_with(rng, [(0, 36, 27)])                        # one red roof
    assert train_s2.crop(chip, 2, 60).shape == (WINDOW, WINDOW, 3)
    cx, cy = train_s2.peak_centre(chip)
    assert abs(cx - 36.5) <= 1.5 and abs(cy - 27.5) <= 1.5


def test_airport_folds_are_disjoint_and_balanced():
    rows = [{'airport': a, 'label': 'aircraft'} for a, n in (('A', 50), ('B', 40), ('C', 30), ('D', 20), ('E', 10))
            for _ in range(n)] + [{'airport': 'F', 'label': 'background'}]
    folds = train_s2.airport_folds(rows, 2)
    assert sorted(sum(folds, [])) == ['A', 'B', 'C', 'D', 'E', 'F']
    load_ = [sum(n for a, n in (('A', 50), ('B', 40), ('C', 30), ('D', 20), ('E', 10)) if a in f) for f in folds]
    assert abs(load_[0] - load_[1]) <= 10


def test_learns_three_copies_versus_one_coloured_blob():
    rng = np.random.default_rng(1)
    rows, chips = [], []
    for i in range(240):
        d = rng.integers(4, 10)
        if i % 2:   # aircraft: blue, green (centre), red in a line
            blobs, label = [(2, 32 - d, 32), (1, 32, 32), (0, 32 + d, 32)], 'aircraft'
        else:       # clutter: one coloured blob near the centre
            blobs, label = [(int(rng.integers(3)), 32 + int(rng.integers(-3, 4)), 32)], 'background'
        rows.append({'label': label, 'airport': 'X'})
        chips.append(chip_with(rng, blobs))
    chips = np.stack(chips)
    items = train_s2.examples(rows, chips, with_peaks=False)
    # >= ~100 steps: fewer leave batch-norm running statistics unsettled and eval-mode scores collapse
    model = train_s2.train(chips, items[:160], epochs=12, batch=16, seed=0)
    s = train_s2.predict(model, chips, items[160:])
    y = np.array([it[3] for it in items[160:]])
    assert auc(s, y) > 0.9
