import numpy as np
import pytest

from sky.detect import Tracker, aircraft_class_ids, cut_by_tile_edge, make_tiles, merge


def test_tiles_cover_image():
    w, h, t = 1920, 1080, 640
    tiles = make_tiles(w, h, t, 0.2)
    cover = np.zeros((h, w), bool)
    for x, y in tiles:
        assert 0 <= x <= w - t and 0 <= y <= h - t
        cover[y:y + t, x:x + t] = True
    assert cover.all()
    assert make_tiles(500, 400, 640) == [(0, 0)]


def test_merge_removes_tile_duplicates():
    boxes = [[10, 10, 30, 30], [11, 10, 31, 30], [100, 100, 120, 120]]
    b, s = merge(boxes, [0.9, 0.8, 0.5])
    assert len(b) == 2 and s[0] == pytest.approx(0.9)


def test_class_lookup():
    names = {0: 'person', 4: 'airplane', 14: 'bird'}
    assert aircraft_class_ids(names) == [4]
    with pytest.raises(ValueError):
        aircraft_class_ids({0: 'person'})


def box(cx, cy, s=6):
    return [cx - s, cy - s, cx + s, cy + s]


def test_tracker_confirms_straight_mover_rejects_blips_and_erratic():
    rng = np.random.default_rng(0)
    tr = Tracker(min_hits=5, max_misses=2)
    for f in range(20):
        dets = [box(50 + 8 * f, 300 - 2 * f)]                       # aircraft: straight line
        dets.append(box(600 + rng.uniform(-30, 30), 400 + rng.uniform(-30, 30)))  # bird: erratic
        if f == 7:
            dets.append(box(900, 50))                                 # one-frame glint
        tr.update(dets, [0.6] * len(dets), f)
    confirmed = tr.confirmed_tracks()
    assert len(confirmed) == 1
    c = confirmed[0].centres
    assert c[0][0] == pytest.approx(50) and confirmed[0].straightness() > 0.99


def test_tracker_bridges_missed_frames():
    tr = Tracker(min_hits=4, max_misses=3)
    for f in range(10):
        if f in (3, 4):
            tr.update([], [], f)
            continue
        tr.update([box(100 + 10 * f, 100)], [0.5], f)
    assert len(tr.confirmed_tracks()) == 1
    assert len(tr.confirmed_tracks()[0].frames) == 8


def test_cut_by_tile_edge_only_flags_interior_edges():
    # tile at (0, 0) of a 1000x1000 image, size 640: right/bottom edges are interior
    boxes = [[100, 100, 200, 200], [500, 100, 639.5, 200], [0, 0, 50, 50]]
    assert cut_by_tile_edge(boxes, 0, 0, 640, 640, 1000, 1000).tolist() == [False, True, False]
    # last tile (360, 360): its right/bottom edges are the image edges
    assert not cut_by_tile_edge([[500, 500, 640, 640]], 360, 360, 640, 640, 1000, 1000)[0]
