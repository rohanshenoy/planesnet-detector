import json

import numpy as np
import pytest
import torch

from satellite import augment, data, detect
from satellite.model import PlaneNet, save, load


def fake_plane_chip(rng, bg=(0.45, 0.45, 0.45)):
    """A white cross-shaped 'aircraft' on a flat tarmac-coloured chip."""
    chip = np.ones((20, 20, 3), np.float32) * np.array(bg, np.float32)
    chip += rng.normal(0, 0.01, chip.shape).astype(np.float32)
    chip[9:11, 4:16] = 0.95   # wings
    chip[5:16, 9:11] = 0.95   # fuselage
    return np.clip(chip, 0, 1)


def fake_planesnet(path, n=200, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i in range(n):
        if i % 2:
            c = fake_plane_chip(rng)
        else:
            c = augment.random_background(rng)
        X.append((c * 255).astype(np.uint8).transpose(2, 0, 1).ravel().tolist())
        y.append(i % 2)
    with open(path, 'w') as f:
        json.dump({'data': X, 'labels': y}, f)


def test_dense_scores_equal_per_window_classification():
    torch.manual_seed(0)
    model = PlaneNet(width=8).eval()
    rng = np.random.default_rng(0)
    img = rng.random((53, 61, 3)).astype(np.float32)
    for stride in (1, 2, 4):
        P = detect.score_map(model, img, stride)
        assert P.shape == ((53 - 20) // stride + 1, (61 - 20) // stride + 1)
        for r in range(0, P.shape[0], 3):
            for c in range(0, P.shape[1], 3):
                chip = img[r * stride:r * stride + 20, c * stride:c * stride + 20]
                with torch.no_grad():
                    ref = torch.sigmoid(model(data.to_tensor(chip)[None])).item()
                assert P[r, c] == pytest.approx(ref, abs=1e-5)


def test_plane_mask_isolates_centre_object():
    rng = np.random.default_rng(1)
    a = augment.plane_mask(fake_plane_chip(rng))
    assert a is not None
    assert a[10, 10] > 0.9 and a[0, 0] < 0.05
    assert augment.plane_mask(np.full((20, 20, 3), 0.5, np.float32)) is None


def test_band_shift_recovers_simulated_motion():
    rng = np.random.default_rng(2)
    chip = fake_plane_chip(rng)
    alpha = augment.plane_mask(chip)
    bg = np.full((20, 20, 3), 0.8, np.float32)
    static = augment.composite(chip, alpha, bg)
    moving = augment.composite(chip, alpha, bg, band_shift=(0.0, 2.0))
    assert np.hypot(*detect.band_shift(static)) < 0.5
    dy, dx = detect.band_shift(moving)
    assert abs(dy) < 0.75 and dx == pytest.approx(4.0, abs=0.75)  # red -2, blue +2


def test_synthetic_samples_are_valid():
    rng = np.random.default_rng(3)
    chip = fake_plane_chip(rng)
    alpha = augment.plane_mask(chip)
    for _ in range(20):
        p = augment.airborne_positive(chip, alpha, rng)
        n = augment.airborne_negative(rng)
        for im in (p, n):
            assert im.shape == (20, 20, 3) and im.dtype == np.float32
            assert 0 <= im.min() and im.max() <= 1


def test_find_peaks_and_static_suppression():
    P = np.zeros((40, 40), np.float32)
    P[10, 10], P[10, 11] = 0.9, 0.8   # one object, two adjacent windows
    P[30, 30] = 0.7
    dets = detect.find_peaks(P, stride=2, threshold=0.5)
    assert [(d['x'], d['y']) for d in dets] == [(30, 30), (70, 70)]
    ref = np.zeros_like(P)
    ref[30, 31] = 0.95                 # second object also present on other dates
    dets = detect.mark_static(dets, [ref, ref.copy()], stride=2)
    assert [d['static'] for d in dets] == [False, True]


def test_train_and_detect_end_to_end(tmp_path):
    from satellite import train
    js = tmp_path / 'planesnet.json'
    fake_planesnet(js, n=300)
    out = tmp_path / 'plane.pt'
    best = train.main(['--planesnet', str(js), '--out', str(out), '--epochs', '4',
                       '--batch-size', '64', '--workers', '0', '--width', '8'])
    assert out.exists() and best > 0.8

    model = load(str(out))
    rng = np.random.default_rng(5)
    scene = augment.fractal_noise(120, rng)[..., None] * 0.2 + 0.7   # cloud deck
    scene = np.repeat(scene, 3, -1).astype(np.float32)
    chip = fake_plane_chip(rng)
    alpha = augment.plane_mask(chip)
    scene[50:70, 30:50] = augment.composite(chip, alpha, scene[50:70, 30:50], (0.0, 1.5))
    dets = detect.detect(model, (scene * 255).astype(np.uint8), threshold=0.5)
    assert any(abs(d['x'] - 40) <= 4 and abs(d['y'] - 60) <= 4 for d in dets)
    hit = min(dets, key=lambda d: abs(d['x'] - 40) + abs(d['y'] - 60))
    assert hit['moving']


def test_checkpoint_roundtrip(tmp_path):
    m = PlaneNet(width=8).eval()
    save(m, tmp_path / 'm.pt')
    m2 = load(str(tmp_path / 'm.pt'))
    x = torch.rand(2, 3, 20, 20)
    with torch.no_grad():
        assert torch.allclose(m(x), m2(x))


def test_distractor_negatives_are_valid():
    rng = np.random.default_rng(6)
    tex = np.full((20, 20, 3), 0.45, np.float32)
    for _ in range(10):
        m = augment.random_blob(rng)
        assert m[10, 10] > 0.5 and m[0, 0] < 0.5
        im = augment.airborne_distractor(tex, rng)
        assert im.shape == (20, 20, 3) and 0 <= im.min() and im.max() <= 1
