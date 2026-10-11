import datetime

import numpy as np
import pytest

from satellite import build_s2_flying as b


def test_mosaic_centre_is_the_airport_and_blocks_tile_the_grid():
    lat, lon = b.AIRPORTS['AGP']
    geom = b.mosaic_geometry(lat, lon)
    lon0, lat0, w, h = geom
    # about 10 m pixels: 0.15 deg of longitude at 36.7 N is ~13.4 km, 0.10 deg of latitude ~11.1 km
    assert w == pytest.approx(1340, abs=5) and h == pytest.approx(1113, abs=5)
    assert b.mosaic_to_lonlat(b.GRID * h / 2, b.GRID * w / 2, geom) == pytest.approx((lon, lat))
    assert b.mosaic_to_lonlat(0, 0, geom) == pytest.approx((lon0, lat0 + b.GRID * b.BLOCK_H))
    # the start of block (row 2, col 3) is that block's north-west corner
    assert b.mosaic_to_lonlat(2 * h, 3 * w, geom) == pytest.approx(
        (lon0 + 3 * b.BLOCK_W, lat0 + (b.GRID - 2) * b.BLOCK_H))
    # labels are pixel indices: their centre is half a pixel in
    assert b.label_lonlat(0, 0, geom) == pytest.approx(b.mosaic_to_lonlat(0.5, 0.5, geom))


def test_parse_time_and_labels(tmp_path):
    assert b.parse_time('AGP/2020-01-19-11-11-01.txt') == datetime.datetime(
        2020, 1, 19, 11, 11, 1, tzinfo=datetime.timezone.utc)
    f = tmp_path / 'x.txt'
    f.write_text('5758 5225\n2264 698\n\n')
    assert b.read_labels(str(f)) == [(5758, 5225), (2264, 698)]


def item(id_, when, lon0, lat0, lon1, lat1, baseline):
    ring = [[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]]
    return {'id': id_, 'geometry': {'type': 'Polygon', 'coordinates': [ring]},
            'properties': {'datetime': when, 's2:processing_baseline': baseline}}


def test_pick_item_needs_footprint_and_time_and_prefers_original_processing():
    when = datetime.datetime(2020, 1, 19, 11, 11, 1, tzinfo=datetime.timezone.utc)
    items = [item('reprocessed', '2020-01-19T11:11:01.395Z', 0, 0, 1, 1, '05.00'),
             item('original', '2020-01-19T11:11:01.398Z', 0, 0, 1, 1, '02.13'),
             item('next tile', '2020-01-19T11:11:01.398Z', 1, 0, 2, 1, '02.13'),
             item('other pass', '2020-01-24T11:11:01Z', 0, 0, 1, 1, '02.13')]
    assert b.pick_item(items, 0.5, 0.5, when)['id'] == 'original'
    assert b.pick_item(items, 1.5, 0.5, when)['id'] == 'next tile'
    assert b.pick_item(items, 2.5, 0.5, when) is None
    multi = {'type': 'MultiPolygon', 'coordinates': [item('', '', 0, 0, 1, 1, '')['geometry']['coordinates'],
                                                     item('', '', 5, 5, 6, 6, '')['geometry']['coordinates']]}
    assert b.point_in_polygon(5.5, 5.5, multi) and not b.point_in_polygon(3, 3, multi)


def test_reflectance_scaling_matches_s2_airborne():
    # processing baseline >= 04.00 stores reflectance + 0.1; STAC gives offset -0.1
    refl = b.dn_to_reflectance(np.array([1000, 2500, 4000, 7000]), 1e-4, -0.1)
    assert refl == pytest.approx([0.0, 0.15, 0.3, 0.6], abs=1e-6)
    assert b.to_uint8(refl).tolist() == [0, 128, 255, 255]


def scene_with_aircraft(rng, copies, size=120):
    """Reflectance (size, size, 3) R, G, B: noisy ground plus 2x2 blobs that are bright in one band only."""
    refl = 0.1 + 0.004 * rng.standard_normal((size, size, 3)) + np.linspace(0, 0.05, size)[None, :, None]
    for band, (x, y) in copies:
        refl[y:y + 2, x:x + 2, band] += 0.08
    return refl.astype(np.float32)


def test_find_triplets_needs_three_collinear_copies_with_green_between():
    rng = np.random.default_rng(0)
    R, G, B = 0, 1, 2
    refl = scene_with_aircraft(rng, [
        (B, (40, 60)), (G, (50, 62)), (R, (60, 64)),          # aircraft: blue -> green -> red in a line
        (B, (85, 20)), (G, (95, 45)), (R, (105, 20)),         # green far off the blue-red line
        (G, (20, 100)), (R, (30, 100)), (B, (40, 100)),       # in a line, but green is not between
        (R, (100, 100))])                                     # a lone red roof
    found = b.find_triplets(refl)
    assert len(found) == 1
    t = found[0]
    assert np.hypot(t['g'][0] - 50.5, t['g'][1] - 62.5) <= 1.5
    assert np.hypot(t['b'][0] - 40.5, t['b'][1] - 60.5) <= 1.5 and np.hypot(t['r'][0] - 60.5, t['r'][1] - 64.5) <= 1.5


def test_find_triplets_ignores_no_data_edges():
    rng = np.random.default_rng(1)
    refl = scene_with_aircraft(rng, [(2, (40, 60)), (1, (50, 62)), (0, (60, 64))])
    valid = np.ones(refl.shape[:2], bool)
    valid[:, 90:] = False
    refl[~valid] = 0
    assert len(b.find_triplets(refl, valid)) == 1


def sig(x, y, score=10.0):
    return {'b': (x - 8, y), 'g': (x, y), 'r': (x + 8, y), 'score': score}


def test_vote_shift_needs_two_labels_to_agree():
    labels = [(100, 100), (300, 200), (500, 50)]
    trips = [sig(120, 97), sig(321, 97 + 100),               # labels 0 and 1, both shifted by about (+20, -3)
             sig(530, 120), sig(60, 140)]                     # clutter near labels 2 and 0
    assert b.vote_shift(labels, trips) == pytest.approx((20.5, -3.0))
    # one label alone, or labels whose nearby signatures disagree, give no vote
    assert b.vote_shift([(100, 100)], [sig(120, 97)]) is None
    assert b.vote_shift([(100, 100), (300, 200)], [sig(120, 97), sig(260, 260)]) is None


def test_match_at_uses_each_signature_once_within_tolerance():
    labels = [(100, 100), (300, 200), (500, 50)]
    trips = [sig(120, 97), sig(321, 197), sig(530, 120), sig(119, 99)]
    assert b.match_at(labels, trips, (20, -3)) == {0: 0, 1: 1}         # label 2 has nothing at the shift
    assert b.match_at([(100, 100), (101, 100)], [sig(120, 97)], (20, -3)) == {0: 0}


def test_airport_shift_needs_most_files_to_agree():
    consistent = [(20, 0), (19, 1), (21, -1), (3, -3), (20, 0)]
    shift, agree = b.airport_shift(consistent)
    assert shift == pytest.approx((20, 0)) and agree == pytest.approx(0.8)
    scattered = [(-52, 8), (-123, 49), (4, -8), (-41, 48), (-140, 8), (112, 94)]
    assert b.airport_shift(scattered)[0] is None
    assert b.airport_shift([(20, 0), (20, 1)])[0] is None             # too few files to tell


def test_assign_uses_airport_offset_drops_weak_and_duplicate_matches():
    def rec(time, labels_px, trips, labels_ll):
        return {'airport': 'CDG', 'time': time, 'labels': labels_ll,
                'groups': [{'item': 'x', 'label_index': list(range(len(labels_px))), 'label_px': labels_px,
                            'triplets': trips, 'vote': b.vote_shift(labels_px, trips)}]}
    ll = [(2.50, 49.00), (2.60, 49.05)]
    files = [rec('2020-01-0%dT10:00:00' % d, [(100, 100), (400, 300)], [sig(120, 100, 30), sig(420, 300, 30)], ll)
             for d in (1, 2, 3)]
    # a file with one label: matched at the airport offset; its weak signature next to it is ignored
    files.append(rec('2020-01-04T10:00:00', [(200, 200)], [sig(221, 202, 40), sig(219, 199, 9)], [(2.55, 49.02)]))
    # a second file of the 2020-01-01 pass repeating the first label: a duplicate, kept once
    files.append(rec('2020-01-01T10:00:15', [(700, 700)], [sig(720, 700, 20)], [(2.50, 49.0001)]))
    shifts, matches = b.assign(files)
    assert shifts['CDG'][0] == pytest.approx((20, 0))
    kinds = sorted((m[0]['time'][:10], m[4]) for m in matches)
    assert kinds == [('2020-01-01', 'file shift'), ('2020-01-01', 'file shift'), ('2020-01-02', 'file shift'),
                     ('2020-01-02', 'file shift'), ('2020-01-03', 'file shift'), ('2020-01-03', 'file shift'),
                     ('2020-01-04', 'airport shift')]
    single = next(m for m in matches if m[4] == 'airport shift')
    assert single[1]['triplets'][single[3]]['score'] == 40
