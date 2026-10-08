"""
Synthetic augmentation that moves PlanesNet aircraft off the tarmac and into
the air.

PlanesNet positives are almost all parked or taxiing aircraft on concrete. An
aircraft in flight seen from orbit sits over cloud, sea, haze or open land, and
because the colour bands of push-broom / multi-array sensors (PlanetScope,
Sentinel-2) are captured a fraction of a second apart, a fast-moving aircraft
shows up as spatially offset colour copies ("rainbow ghosting"). The helpers
below cut the aircraft out of a PlanesNet chip, paste it onto an airborne-style
background and optionally simulate that band offset, haze and contrails.

All images are float32 arrays in [0, 1] with shape (H, W, 3).
"""

import numpy as np
from scipy import ndimage

CHIP = 20


def fractal_noise(size, rng, octaves=4, persistence=0.55):
    """Smooth multi-octave value noise in [0, 1] of shape (size, size)."""
    out = np.zeros((size, size), dtype=np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        cells = 2 ** (o + 1)
        coarse = rng.random((cells + 1, cells + 1)).astype(np.float32)
        out += amp * ndimage.zoom(coarse, size / (cells + 1), order=1)[:size, :size]
        total += amp
        amp *= persistence
    out /= total
    out -= out.min()
    return out / max(out.max(), 1e-6)


def random_background(rng, size=CHIP, kind=None):
    """Procedural background of the kind an airborne aircraft is seen against.

    kind is one of 'cloud', 'sea', 'land', 'haze' (random if None).
    """
    kind = kind or rng.choice(['cloud', 'sea', 'land', 'haze'])
    n = fractal_noise(size, rng)
    if kind == 'cloud':
        base = rng.uniform(0.65, 0.95)
        grey = base - 0.25 * rng.random() * n
        rgb = np.stack([grey, grey, grey + rng.uniform(0, 0.04)], -1)
    elif kind == 'sea':
        color = np.array([rng.uniform(0.05, 0.15), rng.uniform(0.12, 0.25),
                          rng.uniform(0.18, 0.35)], dtype=np.float32)
        rgb = color + 0.05 * (n[..., None] - 0.5)
        if rng.random() < 0.3:  # sun glint / whitecaps
            glint = (rng.random((size, size)) > 0.985).astype(np.float32)
            rgb = rgb + 0.6 * ndimage.gaussian_filter(glint, 0.5)[..., None]
    elif kind == 'land':
        color = np.array([rng.uniform(0.2, 0.45), rng.uniform(0.25, 0.45),
                          rng.uniform(0.15, 0.3)], dtype=np.float32)
        rgb = color * (0.7 + 0.6 * n[..., None])
    else:  # thin haze or broken cloud over darker ground
        ground = random_background(rng, size, rng.choice(['sea', 'land']))
        cloud = random_background(rng, size, 'cloud')
        a = rng.uniform(0.3, 0.8) * n[..., None]
        rgb = (1 - a) * ground + a * cloud
    return np.clip(rgb, 0, 1).astype(np.float32)


def plane_mask(chip, thresh=0.12, min_px=6, max_px=250):
    """Estimate a soft alpha mask for the aircraft in a centred PlanesNet chip.

    The local background is taken as the median of the chip border; pixels
    that differ from it by more than thresh are foreground, and only the
    connected component closest to the centre is kept. Returns None when no
    clean single aircraft can be isolated.
    """
    border = np.concatenate([chip[0], chip[-1], chip[:, 0], chip[:, -1]])
    bg = np.median(border, axis=0)
    diff = np.linalg.norm(chip - bg, axis=-1)
    fg = diff > thresh
    labels, n = ndimage.label(fg)
    if n == 0:
        return None
    h, w = fg.shape
    yy, xx = np.mgrid[:h, :w]
    best, best_d = None, np.inf
    for lab in range(1, n + 1):
        m = labels == lab
        d = np.hypot(yy[m].mean() - h / 2, xx[m].mean() - w / 2)
        if d < best_d:
            best, best_d = m, d
    if best_d > 4 or not (min_px <= best.sum() <= max_px):
        return None
    best = ndimage.binary_closing(best, iterations=1)
    return np.clip(ndimage.gaussian_filter(best.astype(np.float32), 0.6) * 1.5, 0, 1)


def shift_image(img, dy, dx):
    """Sub-pixel shift with edge replication."""
    return ndimage.shift(img, (dy, dx) + (0,) * (img.ndim - 2), order=1, mode='nearest')


def composite(plane, alpha, background, band_shift=(0.0, 0.0)):
    """Paste plane (with alpha) onto background.

    band_shift=(dy, dx) simulates inter-band acquisition delay: the red band is
    displaced by -shift and the blue band by +shift relative to green, so a
    moving aircraft leaves offset colour copies while the static background
    stays aligned.
    """
    out = np.empty_like(background)
    dy, dx = band_shift
    for c, k in enumerate((-1, 0, 1)):
        p = shift_image(plane[..., c], k * dy, k * dx)
        a = shift_image(alpha, k * dy, k * dx)
        out[..., c] = a * p + (1 - a) * background[..., c]
    return out


def add_contrail(img, rng, through_centre=False, behind=None):
    """Draw a faint linear contrail. If behind=(angle) it trails the centre."""
    h, w, _ = img.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    angle = rng.uniform(0, np.pi) if behind is None else behind
    if through_centre or behind is not None:
        cy, cx = h / 2, w / 2
    else:
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
    dy, dx = np.sin(angle), np.cos(angle)
    dist = np.abs((yy - cy) * dx - (xx - cx) * dy)
    line = np.exp(-(dist / rng.uniform(0.5, 1.2)) ** 2)
    if behind is not None:  # only on the side opposite the direction of travel
        along = (yy - cy) * dy + (xx - cx) * dx
        line *= (along < -2).astype(np.float32)
    strength = rng.uniform(0.15, 0.4)
    return np.clip(img + strength * line[..., None] * (1 - img), 0, 1)


def photometric(img, rng):
    """Haze, blur, brightness/contrast jitter and sensor noise."""
    if rng.random() < 0.5:
        a = rng.uniform(0.0, 0.5)
        img = (1 - a) * img + a * rng.uniform(0.6, 0.9)
    if rng.random() < 0.3:
        img = ndimage.gaussian_filter(img, (rng.uniform(0.3, 0.8),) * 2 + (0,))
    img = (img - 0.5) * rng.uniform(0.75, 1.25) + 0.5 + rng.uniform(-0.1, 0.1)
    img = img + rng.normal(0, rng.uniform(0.0, 0.03), img.shape)
    return np.clip(img, 0, 1).astype(np.float32)


def geometric(img, rng):
    """Random 90-degree rotation and flips (aircraft heading is arbitrary)."""
    img = np.rot90(img, rng.integers(4))
    if rng.random() < 0.5:
        img = img[:, ::-1]
    return np.ascontiguousarray(img)


def airborne_positive(plane_chip, alpha, rng, background=None):
    """Turn one PlanesNet plane chip into a synthetic in-flight sample."""
    bg = background if background is not None else random_background(rng)
    plane = plane_chip
    if rng.random() < 0.5:  # aircraft over a white cloud is low contrast; vary it
        plane = np.clip(plane * rng.uniform(0.7, 1.1), 0, 1)
    shift = (0.0, 0.0)
    if rng.random() < 0.6:
        mag, ang = rng.uniform(0.5, 3.0), rng.uniform(0, 2 * np.pi)
        shift = (mag * np.sin(ang), mag * np.cos(ang))
    img = composite(plane, alpha, bg, shift)
    if rng.random() < 0.25:
        img = add_contrail(img, rng, behind=rng.uniform(0, 2 * np.pi))
    return photometric(geometric(img, rng), rng)


def airborne_negative(rng, background=None):
    """A hard negative: cloud / sea / land / contrail with no aircraft."""
    img = background if background is not None else random_background(rng)
    if rng.random() < 0.2:
        img = add_contrail(img, rng, through_centre=rng.random() < 0.5)
    return photometric(geometric(img, rng), rng)
