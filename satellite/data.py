"""
Data loading for satellite aircraft chips.

Sources:
  * PlanesNet JSON (the original tarmac-heavy dataset).
  * Optional chip folders laid out as <root>/plane/*.png and <root>/no-plane/*.png,
    e.g. real in-flight aircraft you have labelled yourself.
  * Optional background scenes (any images of cloud, sea, open country with no
    aircraft) from which 20x20 backgrounds and hard negatives are cropped.
  * Synthetic in-flight samples generated on the fly by satellite.augment.
"""

import glob
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import augment

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


def load_planesnet(path):
    """Return (X uint8 (N, 20, 20, 3), y int64 (N,)) from planesnet.json."""
    with open(path) as f:
        d = json.load(f)
    X = np.asarray(d['data'], dtype=np.uint8).reshape(-1, 3, 20, 20).transpose(0, 2, 3, 1)
    return np.ascontiguousarray(X), np.asarray(d['labels'], dtype=np.int64)


def _images(folder):
    return sorted(p for p in glob.glob(os.path.join(folder, '*'))
                  if p.lower().endswith(IMAGE_EXTS))


def read_rgb(path):
    return np.asarray(Image.open(path).convert('RGB'))


def load_chip_folder(root, size=augment.CHIP):
    """Load <root>/plane and <root>/no-plane chips, resized to size x size."""
    X, y = [], []
    for label, sub in ((1, 'plane'), (0, 'no-plane')):
        for p in _images(os.path.join(root, sub)):
            im = Image.open(p).convert('RGB')
            if im.size != (size, size):
                im = im.resize((size, size), Image.BILINEAR)
            X.append(np.asarray(im))
            y.append(label)
    if not X:
        raise ValueError('no chips found under %s/{plane,no-plane}' % root)
    return np.stack(X), np.asarray(y, dtype=np.int64)


class BackgroundBank:
    """Random 20x20 crops from large aircraft-free scenes (cloud, sea, ...)."""

    def __init__(self, folder):
        self.scenes = [read_rgb(p).astype(np.float32) / 255. for p in _images(folder)]
        self.scenes = [s for s in self.scenes if min(s.shape[:2]) >= augment.CHIP]
        if not self.scenes:
            raise ValueError('no usable background images in %s' % folder)

    def sample(self, rng):
        s = self.scenes[rng.integers(len(self.scenes))]
        i = rng.integers(s.shape[0] - augment.CHIP + 1)
        j = rng.integers(s.shape[1] - augment.CHIP + 1)
        return s[i:i + augment.CHIP, j:j + augment.CHIP].copy()


def extract_plane_cutouts(X, y):
    """Plane chips (float) and alpha masks for every positive that segments cleanly."""
    planes, alphas = [], []
    for chip in X[y == 1]:
        f = chip.astype(np.float32) / 255.
        a = augment.plane_mask(f)
        if a is not None:
            planes.append(f)
            alphas.append(a)
    return planes, alphas


class ChipDataset(Dataset):
    """Real chips plus n_synthetic on-the-fly synthetic in-flight samples.

    Synthetic samples are half positives (a real aircraft cut-out composited
    onto an airborne background) and half hard negatives (background only, or
    a blob of real ground texture pasted onto the background).
    """

    def __init__(self, X, y, cutouts=None, n_synthetic=0, backgrounds=None,
                 augment_real=True, seed=0):
        self.X, self.y = X, y
        self.planes, self.alphas = cutouts if cutouts else ([], [])
        self.n_synthetic = n_synthetic if self.planes else 0
        self.backgrounds = backgrounds
        self.augment_real = augment_real
        self.textures = X[y == 0] if len(X) else X  # ground texture for distractor negatives
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.X) + self.n_synthetic

    def _background(self):
        if self.backgrounds is not None and self.rng.random() < 0.7:
            return self.backgrounds.sample(self.rng)
        return augment.random_background(self.rng)

    def __getitem__(self, idx):
        rng = self.rng
        if idx < len(self.X):
            img = self.X[idx].astype(np.float32) / 255.
            label = int(self.y[idx])
            if self.augment_real:
                img = augment.geometric(img, rng)
                if rng.random() < 0.5:
                    img = augment.photometric(img, rng)
        elif idx % 2 == 0:
            k = rng.integers(len(self.planes))
            img = augment.airborne_positive(self.planes[k], self.alphas[k], rng,
                                            self._background())
            label = 1
        elif len(self.textures) and rng.random() < 0.3:
            tex = self.textures[rng.integers(len(self.textures))].astype(np.float32) / 255.
            img = augment.airborne_distractor(tex, rng, self._background())
            label = 0
        else:
            img = augment.airborne_negative(rng, self._background())
            label = 0
        return to_tensor(img), torch.tensor(label, dtype=torch.float32)


def to_tensor(img):
    """(H, W, 3) float in [0, 1] -> (3, H, W) tensor."""
    return torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32).transpose(2, 0, 1))


def worker_init(worker_id):
    ds = torch.utils.data.get_worker_info().dataset
    ds.rng = np.random.default_rng(torch.initial_seed() % 2 ** 32)
