"""
Stack frames t-k, t, t+k into one 3-channel image for temporal detection.

Each neighbour is aligned to frame t by phase correlation (a global
translation), which cancels most of the camera's own motion over a fraction of
a second. After alignment static scenery is identical in all three channels
and renders grey, while an object moving relative to it leaves a coloured
fringe that a detector trained on such stacks (prepare_aot temporal/) uses.
"""

import cv2
import numpy as np


def to_gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def align(ref, img):
    """Translate img onto ref (both greyscale) to remove camera motion."""
    win = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(ref.astype(np.float32), img.astype(np.float32), win)
    m = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(img, m, ref.shape[::-1], borderMode=cv2.BORDER_REPLICATE)


def match_brightness(ref, img):
    """Gain/offset-match img to ref so exposure changes between frames don't tint the stack."""
    r, i = ref.astype(np.float32), img.astype(np.float32)
    out = (i - i.mean()) * (r.std() / max(i.std(), 1e-3)) + r.mean()
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def stack(prev, cur, nxt):
    """(H, W, 3) uint8 with channels (prev, cur, next) as written by cv2 (B, G, R)."""
    prev, cur, nxt = to_gray(prev), to_gray(cur), to_gray(nxt)
    return np.dstack([match_brightness(cur, align(cur, prev)), cur,
                      match_brightness(cur, align(cur, nxt))])
