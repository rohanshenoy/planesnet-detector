"""
Detect aircraft across a full satellite scene.

Replaces the old per-window sliding loop with a single fully convolutional
pass (see satellite.model), then applies two cues that matter for aircraft in
flight:

  * Temporal persistence (--references): give co-registered images of the same
    area from other dates. A detection that also fires at the same spot in
    most reference images is a static object (building, tank, moored boat,
    parked aircraft) and is suppressed; aircraft in flight are never in the
    same place twice.
  * Band offset: the colour bands are captured a fraction of a second apart,
    so a moving aircraft appears as offset red/blue copies. The red-to-blue
    displacement is measured for each detection; a non-zero shift marks it as
    moving and gives its heading. Speed = shift (px) * GSD (m/px) / inter-band
    delay (s), which depends on the sensor.

Example:
    python -m satellite.detect models/plane.pt scenes/scene_1.png \
        --references scenes/scene_1_may.png scenes/scene_1_jun.png --json dets.json
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage

from .model import WINDOW, STRIDE, load

HALF = WINDOW // 2


@torch.no_grad()
def score_map(model, img, stride=2, device='cpu'):
    """Aircraft probability for every WINDOW x WINDOW window at the given stride.

    img: (H, W, 3) float in [0, 1]. stride must divide 4 (1, 2 or 4).
    Returns P with P[r, c] = score of the window whose top-left corner is
    (r * stride, c * stride).
    """
    if STRIDE % stride:
        raise ValueError('stride must be 1, 2 or 4')
    h, w = img.shape[:2]
    if h < WINDOW or w < WINDOW:
        raise ValueError('image smaller than %d px' % WINDOW)
    P = np.zeros(((h - WINDOW) // stride + 1, (w - WINDOW) // stride + 1), np.float32)
    x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1), np.float32))[None].to(device)
    k = STRIDE // stride
    for oy in range(0, STRIDE, stride):
        for ox in range(0, STRIDE, stride):
            sub = x[:, :, oy:, ox:]
            if sub.shape[2] < WINDOW or sub.shape[3] < WINDOW:
                continue
            prob = torch.sigmoid(model(sub))[0, 0].cpu().numpy()
            P[oy // stride::k, ox // stride::k][:prob.shape[0], :prob.shape[1]] = prob
    return P


def find_peaks(P, stride, threshold=0.5, min_dist=WINDOW // 2):
    """Local maxima of P above threshold, at least min_dist px apart.

    Returns a list of dicts with the window centre (x, y) in pixels and score.
    """
    r = max(1, int(np.ceil(min_dist / stride)))
    local_max = P == ndimage.maximum_filter(P, size=2 * r + 1, mode='constant')
    ys, xs = np.nonzero(local_max & (P > threshold))
    order = np.argsort(-P[ys, xs], kind='stable')
    kept = np.zeros((0, 2))
    dets = []
    for idx in order:  # plateaus give several equal maxima; keep them min_dist apart
        c = np.array([ys[idx], xs[idx]]) * stride + HALF
        if len(kept) and np.hypot(*(kept - c).T).min() < min_dist:
            continue
        kept = np.vstack([kept, c])
        dets.append({'x': int(c[1]), 'y': int(c[0]), 'score': float(P[ys[idx], xs[idx]])})
    return dets


def band_shift(chip, max_shift=6):
    """Displacement (dy, dx) of the blue band relative to the red band, in px.

    Estimated by cross-correlating the mean-removed red and blue channels.
    A static object gives ~(0, 0); an aircraft in flight gives a shift along
    its direction of travel (the sign depends on the sensor's band order).
    """
    r = chip[..., 0] - chip[..., 0].mean()
    b = chip[..., 2] - chip[..., 2].mean()
    if r.std() < 1e-3 or b.std() < 1e-3:
        return 0.0, 0.0
    win = np.outer(np.hanning(chip.shape[0]), np.hanning(chip.shape[1]))
    R, B = np.fft.fft2(r * win), np.fft.fft2(b * win)
    corr = np.fft.fftshift(np.real(np.fft.ifft2(B * np.conj(R))))
    cy, cx = np.array(corr.shape) // 2
    sub = corr[cy - max_shift:cy + max_shift + 1, cx - max_shift:cx + max_shift + 1]
    py, px = np.unravel_index(np.argmax(sub), sub.shape)

    def refine(c, m, p):  # parabolic sub-pixel refinement
        d = c - 2 * m + p
        return 0.0 if d == 0 else 0.5 * (c - p) / d

    dy, dx = float(py - max_shift), float(px - max_shift)
    if 0 < py < sub.shape[0] - 1:
        dy += refine(sub[py - 1, px], sub[py, px], sub[py + 1, px])
    if 0 < px < sub.shape[1] - 1:
        dx += refine(sub[py, px - 1], sub[py, px], sub[py, px + 1])
    return dy, dx


def add_motion(dets, img, min_shift=1.0, size=WINDOW + 8):
    """Annotate detections with band offset, heading and a moving flag."""
    h, w = img.shape[:2]
    for d in dets:
        y0 = int(np.clip(d['y'] - size // 2, 0, max(h - size, 0)))
        x0 = int(np.clip(d['x'] - size // 2, 0, max(w - size, 0)))
        dy, dx = band_shift(img[y0:y0 + size, x0:x0 + size])
        d['band_shift'] = [round(dy, 2), round(dx, 2)]
        d['moving'] = bool(np.hypot(dy, dx) >= min_shift)
        d['heading_deg'] = round(float(np.degrees(np.arctan2(dx, -dy)) % 360), 1)
    return dets


def mark_static(dets, ref_maps, stride, threshold=0.5, radius=6, min_fraction=0.5):
    """Flag detections that also fire in most co-registered reference images."""
    if not ref_maps:
        return dets
    r = max(1, int(round(radius / stride)))
    for d in dets:
        row, col = (d['y'] - HALF) // stride, (d['x'] - HALF) // stride
        hits = 0
        for P in ref_maps:
            patch = P[max(row - r, 0):row + r + 1, max(col - r, 0):col + r + 1]
            hits += patch.size > 0 and patch.max() > threshold
        d['ref_hits'] = int(hits)
        d['static'] = hits >= min_fraction * len(ref_maps)
    return dets


def draw(img_u8, dets):
    im = Image.fromarray(img_u8)
    dr = ImageDraw.Draw(im)
    for d in dets:
        color = (255, 200, 0) if d.get('static') else (255, 0, 0)
        x, y = d['x'], d['y']
        dr.rectangle([x - HALF, y - HALF, x + HALF, y + HALF], outline=color, width=2)
        if d.get('moving'):  # arrow along the measured band offset
            a = np.radians(d['heading_deg'])
            dr.line([x, y, x + 14 * np.sin(a), y - 14 * np.cos(a)], fill=(0, 255, 255), width=2)
    return im


def read_image(path):
    return np.asarray(Image.open(path).convert('RGB'))


def detect(model, img_u8, references=(), stride=2, threshold=0.5, device='cpu',
           keep_static=False):
    img = img_u8.astype(np.float32) / 255.
    P = score_map(model, img, stride, device)
    dets = add_motion(find_peaks(P, stride, threshold), img)
    refs = []
    for ref in references:
        if ref.shape != img_u8.shape:
            raise ValueError('reference images must be co-registered and the same size')
        refs.append(score_map(model, ref.astype(np.float32) / 255., stride, device))
    dets = mark_static(dets, refs, stride, threshold)
    if not keep_static:
        dets = [d for d in dets if not d.get('static')]
    return dets


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model', help='checkpoint from satellite.train (.pt)')
    p.add_argument('image')
    p.add_argument('--out', help='annotated output image (default <image>_detections.png)')
    p.add_argument('--json', help='write detections as JSON')
    p.add_argument('--references', nargs='*', default=[],
                   help='co-registered images of the same area at other times')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--stride', type=int, default=2, choices=[1, 2, 4])
    p.add_argument('--keep-static', action='store_true',
                   help='keep (yellow) detections that persist across reference images')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args(argv)

    model = load(a.model, a.device)
    img = read_image(a.image)
    dets = detect(model, img, [read_image(r) for r in a.references], a.stride,
                  a.threshold, a.device, a.keep_static)
    out = a.out or os.path.splitext(a.image)[0] + '_detections.png'
    draw(img, dets).save(out)
    print('%d detections (%d moving) -> %s' % (len(dets), sum(d['moving'] for d in dets), out))
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(dets, f, indent=2)
    return dets


if __name__ == '__main__':
    main()
