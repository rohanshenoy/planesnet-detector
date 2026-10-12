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


def test_centre_distance_matching_and_ap():
    from sky.evaluate import average_precision, match_image
    gts = [(0, 100, 100, 6, 6), (1, 300, 300, 20, 10), (2, 500, 500, 6, 6)]  # plane, heli, bird
    preds = [(0.9, 103, 101), (0.8, 100, 100), (0.7, 505, 500), (0.6, 900, 900), (0.5, 306, 302)]
    records, hit = match_image(preds, gts)
    # 0.9 hits the plane, 0.8 is a duplicate (FP), bird hit is ignored, 0.6 FP, 0.5 hits heli
    assert records == [(0.9, True), (0.8, False), (0.6, False), (0.5, True)]
    assert hit == [0.9, 0.5, None]
    s = average_precision(records, n_gt=2)
    assert s['max_recall'] == 1.0
    assert s['ap'] == pytest.approx(0.5 * 1.0 + 0.5 * 0.5)


def test_clip_outputs_raw_online_offline():
    from sky.evaluate_clips import outputs, score
    dets, gts = {}, {}
    for f in range(10):
        b = [box(100 + 5 * f, 100)]           # aircraft, every frame
        s = [0.9]
        if f == 4:
            b.append(box(400, 400)); s.append(0.8)   # one-frame false alarm
        dets[f] = (np.array(b, np.float32), np.array(s, np.float32))
        gts[f] = [(0, 100 + 5 * f, 100, 12, 12)]
    out = outputs(dets, threshold=0.5, min_hits=3, max_misses=2)
    raw, online, offline = (score(out[k], gts) for k in ('raw', 'online', 'offline'))
    assert raw['recall'] == 1.0 and raw['false_alarms_per_frame'] == pytest.approx(0.1)
    assert online['recall'] == pytest.approx(0.8)      # frames 0-1 wait for confirmation
    assert online['false_alarms_per_frame'] == 0
    assert offline['recall'] == 1.0 and offline['precision'] == 1.0


def test_prepare_aot_reads_any_part_and_part2_column_order(tmp_path):
    from sky.prepare_aot import image_url, load_groundtruth
    assert image_url('part2', 'abc', 'f.png').endswith('/part2/Images/abc/f.png')
    # part2's groundtruth.csv orders its columns differently and spells one bird 'BIrd'
    csv_path = tmp_path / 'gt.csv'
    csv_path.write_text(
        ',time,id,flight_id,img_name,frame,range_distance_m,is_above_horizon,size_width,size_height,'
        'gt_left,gt_top,gt_right,gt_bottom\n'
        '0,1,Airplane1,fl,a.png,3,900.0,1.0,2448,2048,10,20,16,26\n'
        '1,1,BIrd2,fl,a.png,3,50.0,-1.0,2448,2048,100,200,104,204\n')
    fl = load_groundtruth(str(csv_path))['fl']
    assert fl['frames'] == {3: 'a.png'}
    assert [o['cls'] for o in fl['objects'][3]] == [0, 2]


def test_split_is_stable_across_parts_and_exclusion(tmp_path):
    from sky.prepare_aot import flight_ids, split_of
    # the split is a hash of the flight id alone, so a flight shared by two parts keeps its side
    val = sum(split_of('%032x' % (i * 7919), 0.2) == 'val' for i in range(2000))
    assert 300 < val < 500
    p = tmp_path / 'gt.csv'
    p.write_text(',time,flight_id,img_name,frame,id\n0,1,f1,a.png,1,\n1,2,f2,b.png,1,\n2,3,f1,c.png,2,\n')
    assert flight_ids(str(p)) == {'f1', 'f2'}


def test_fetch_retries_transient_errors_but_not_missing_frames(monkeypatch):
    import io
    import urllib.error
    import cv2
    from sky import prepare_aot
    png = cv2.imencode('.png', np.full((4, 6), 7, np.uint8))[1].tobytes()
    calls = []

    def flaky(url, timeout):
        calls.append(url)
        if len(calls) < 3:
            raise ConnectionResetError(54, 'Connection reset by peer')
        return io.BytesIO(png)
    monkeypatch.setattr(prepare_aot.urllib.request, 'urlopen', flaky)
    monkeypatch.setattr(prepare_aot.time, 'sleep', lambda s: None)
    assert prepare_aot.fetch('part1', 'fl', 'a.png').shape == (4, 6) and len(calls) == 3

    def missing(url, timeout):
        calls.append(url)
        raise urllib.error.HTTPError(url, 404, 'Not Found', {}, None)
    calls.clear()
    monkeypatch.setattr(prepare_aot.urllib.request, 'urlopen', missing)
    with pytest.raises(urllib.error.HTTPError):
        prepare_aot.fetch('part1', 'fl', 'a.png')
    assert len(calls) == 1


def test_download_clip_fetches_frames_from_the_given_part(tmp_path, monkeypatch):
    from sky import evaluate_clips
    calls = []

    def fake_fetch(part, flight, img_name):
        calls.append((part, flight, img_name))
        return np.zeros((4, 4), np.uint8)
    monkeypatch.setattr(evaluate_clips, 'fetch', fake_fetch)
    flights = {'fl': {'frames': {f: 'f%d.png' % f for f in range(10)}}}
    paths = evaluate_clips.download_clip(flights, 'fl', 3, 4, 1, str(tmp_path / 'clip'), workers=2, part='part2')
    assert sorted(paths) == list(range(2, 8))
    assert {c[0] for c in calls} == {'part2'} and len(calls) == 6
