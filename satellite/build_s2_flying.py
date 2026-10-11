"""
Build a Sentinel-2 test set of aircraft in flight, guided by the "Flying Airplanes" labels.

Pamplona Segundo et al. (2021), "Measuring economic activity from space: a case study
using flying airplanes and COVID-19" (IEEE JSTARS), labelled 1,782 airplanes in flight in
Sentinel-2 images around 15 European airports, January to June 2020. The labels are in
https://github.com/maups/covid19-custom-script-contest (MIT licence) as
annotations/<AIRPORT>/<YYYY-MM-DD-HH-MM-SS>.txt, one "row col" line per airplane, in
pixels of a mosaic their download.py builds: a 7x7 grid of 0.15 x 0.10 degree blocks
centred on the airport, each resampled to about 10 m pixels.

Converted to ground positions with that geometry, the labels miss their aircraft: by one
offset per airport (about 200 m at CDG, 800 m at AGP), or inconsistently (IST). The cause
could not be found without the authors' own images, so the labels are used as hints:

  pass 1  around each label, find aircraft signatures in the public Sentinel-2 L2A COGs on
          AWS: blue, green and red copies in a line, green between, 3-45 px apart (the bands
          are recorded a fraction of a second apart, so a moving aircraft is seen three times).
          Each file votes for the shift that puts two or more of its labels on a signature.
  pass 2  each airport's offset is the shift most of its files agree on; airports without
          one are dropped. Labels are matched to a signature at that offset only.
  pass 3  chips in the format of data/s2_airborne: 64x64 px at 10 m centred on the green
          copy, B04/B03/B02 as reflectance / 0.3 in 8 bits, with the blue and red copy
          positions; plus random background chips at least 1 km from any label.

Kept aircraft are those the signature finder sees, so the set leans towards clear ones,
the same bias as data/s2_airborne. The labels are not exhaustive, so a few background chips
may hold an unlabelled aircraft. Where an acquisition was processed more than once, the
original processing is used.

Example:
    git clone --depth 1 https://github.com/maups/covid19-custom-script-contest.git
    python -m satellite.build_s2_flying covid19-custom-script-contest/annotations data/s2_flying
    python -m satellite.evaluate_s2 models/plane.pt data/s2_flying

    # chips are not committed; re-cut them from the committed labels.csv
    python -m satellite.build_s2_flying data/s2_flying --chips-from-labels
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from scipy import ndimage

# airport centres, from download.py
AIRPORTS = {
    'AGP': (36.67166398, -4.492831362), 'AMS': (52.312141474163397, 4.757732975709088),
    'ARN': (59.6512, 17.9178), 'ATH': (37.882337, 23.875459),
    'BCN': (41.296657915514594, 2.085735972162202), 'BRU': (50.9008, 4.4840),
    'CDG': (49.006976345624935, 2.559786183921451), 'CPH': (55.620750, 12.650462),
    'DUB': (53.4264, -6.2499), 'DUS': (51.2870, 6.7667),
    'FCO': (41.807101001429608, 12.253597669510512), 'FRA': (50.033333, 8.570556),
    'HEL': (60.3170, 24.9580), 'IST': (40.982555, 28.820829),
    'LGW': (51.153047759076003, -0.185395073719614), 'LHR': (51.469716296645025, -0.458042395268015),
    'LIS': (38.7675, -8.8028), 'LTN': (51.8715, -0.3677),
    'MAD': (40.493475736010851, -3.566288968675247), 'MAN': (53.3523, -2.2717),
    'MUC': (48.353443982011179, 11.781958961375517), 'MXP': (45.62995905, 8.723056),
    'ORY': (48.7262, 2.3652), 'OSL': (60.197552, 11.100415), 'PMI': (39.5510, 2.7367),
    'STN': (51.904949, 0.202641), 'TXL': (52.5558, 13.2860), 'VIE': (48.1062, 16.5685),
    'WAW': (52.1672369, 20.9678911), 'ZRH': (47.451542, 8.564572),
}
BLOCK_W, BLOCK_H, GRID = 0.15, 0.10, 7   # degrees of longitude, latitude; blocks per side
PIXEL_M = 10.0
CHIP = 64
STAC_ROOT = 'https://earth-search.aws.element84.com/v1'
COLLECTION = 'sentinel-2-l2a'
MAX_TIME_DIFF_S = 30
MIN_BACKGROUND_DIST_M = 1000.0
SEARCH_RADIUS = 150      # px: converted labels were seen up to about 1 km from their aircraft
MATCH_TOL = 5            # px: a matched signature lies within this of label + shift
AIRPORT_TOL = 15         # px: files whose voted shift is this close to their airport's offset agree with it
AIRPORT_MIN_FILES = 3    # an airport's offset needs votes from at least this many files...
AIRPORT_MIN_AGREE = 0.6  # ...and this fraction of them within AIRPORT_TOL of it
AIRPORT_MATCH_TOL = 8    # px: tolerance when matching at the airport offset rather than a file's own vote
MIN_MATCH_SCORE = 15.0   # weaker matched signatures were mostly faint clutter or cloud edges when checked by eye
DUPLICATE_DIST_M = 100.0  # the same aircraft labelled in two overlapping files of one pass
MIN_SEP, MAX_SEP = 3, 45  # px between the blue and red copies
PEAK_SIGMA = 5.0         # copy peaks, in robust standard deviations of the colour excess

os.environ.setdefault('AWS_NO_SIGN_REQUEST', 'YES')
os.environ.setdefault('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
os.environ.setdefault('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.tif')
os.environ.setdefault('GDAL_HTTP_MAX_RETRY', '4')
os.environ.setdefault('GDAL_HTTP_RETRY_DELAY', '2')
os.environ.setdefault('GDAL_CACHEMAX', '1024')


def geo_dist(lat1, lon1, lat2, lon2):
    """Haversine distance in metres, exactly as download.py computes it."""
    r = 6378.137
    d_lat = np.radians(lat2) - np.radians(lat1)
    d_lon = np.radians(lon2) - np.radians(lon1)
    a = np.sin(d_lat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(d_lon / 2) ** 2
    return r * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) * 1000.0


def mosaic_geometry(lat, lon):
    """(lon0, lat0, block_w_px, block_h_px) of download.py's mosaic around an airport.

    lon0, lat0 is the south-west corner; the block pixel size uses download.py's rounding.
    """
    la, lo = lat - 0.5 * BLOCK_H, lon - 0.5 * BLOCK_W
    w = int((geo_dist(la, lo, la, lo + BLOCK_W) + geo_dist(la + BLOCK_H, lo, la + BLOCK_H, lo + BLOCK_W))
            / PIXEL_M / 2.0 + 0.5)
    h = int((geo_dist(la, lo, la + BLOCK_H, lo) + geo_dist(la, lo + BLOCK_W, la + BLOCK_H, lo + BLOCK_W))
            / PIXEL_M / 2.0 + 0.5)
    return lon - GRID / 2.0 * BLOCK_W, lat - GRID / 2.0 * BLOCK_H, w, h


def mosaic_to_lonlat(y, x, geom):
    """Continuous mosaic coordinates (pixel corners on integers, row 0 at the north) to lon, lat.

    Every block spans BLOCK_W x BLOCK_H degrees in w x h pixels, so the whole mosaic is one
    regular longitude/latitude grid.
    """
    lon0, lat0, w, h = geom
    return lon0 + x * BLOCK_W / w, lat0 + GRID * BLOCK_H - y * BLOCK_H / h


def label_lonlat(row, col, geom):
    """Longitude, latitude of the centre of labelled pixel (row, col)."""
    return mosaic_to_lonlat(row + 0.5, col + 0.5, geom)


def parse_time(name):
    """'2020-01-19-11-11-01.txt' -> UTC datetime."""
    return datetime.datetime.strptime(os.path.basename(name)[:19], '%Y-%m-%d-%H-%M-%S').replace(
        tzinfo=datetime.timezone.utc)


def read_labels(path):
    rows = []
    for line in open(path):
        if line.strip():
            r, c = (int(v) for v in line.split())
            rows.append((r, c))
    return rows


def point_in_polygon(lon, lat, geometry):
    """Point in a GeoJSON Polygon or MultiPolygon (outer rings only)."""
    polys = geometry['coordinates'] if geometry['type'] == 'MultiPolygon' else [geometry['coordinates']]
    for poly in polys:
        ring = poly[0]
        inside = False
        for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
            if (y1 > lat) != (y2 > lat) and lon < x1 + (lat - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
        if inside:
            return True
    return False


def baseline(item):
    try:
        return float(item['properties'].get('s2:processing_baseline', 99))
    except (TypeError, ValueError):
        return 99.0


def pick_item(items, lon, lat, when):
    """Item covering lon/lat acquired within MAX_TIME_DIFF_S of `when`; original processing first."""
    ok = []
    for it in items:
        t = datetime.datetime.fromisoformat(it['properties']['datetime'].replace('Z', '+00:00'))
        if abs((t - when).total_seconds()) <= MAX_TIME_DIFF_S and point_in_polygon(lon, lat, it['geometry']):
            ok.append(it)
    return min(ok, key=lambda it: (baseline(it), it['id'])) if ok else None


def search(bbox, when, retries=4):
    t0 = (when - datetime.timedelta(seconds=120)).strftime('%Y-%m-%dT%H:%M:%SZ')
    t1 = (when + datetime.timedelta(seconds=120)).strftime('%Y-%m-%dT%H:%M:%SZ')
    body = json.dumps({'collections': [COLLECTION], 'bbox': list(bbox), 'datetime': '%s/%s' % (t0, t1),
                       'limit': 100}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(STAC_ROOT + '/search', data=body, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)['features']
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def fetch_item(item_id, retries=4):
    url = '%s/collections/%s/items/%s' % (STAC_ROOT, COLLECTION, item_id)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def dn_to_reflectance(dn, scale, offset):
    return dn.astype(np.float32) * scale + offset


def to_uint8(refl):
    """Reflectance -> 8 bits as reflectance / 0.3, the scaling of data/s2_airborne."""
    return np.clip(np.round(refl / 0.3 * 255), 0, 255).astype(np.uint8)


class Scene:
    """Open red/green/blue COGs of one STAC item; reads windows in tile pixel coordinates."""

    def __init__(self, item):
        import rasterio
        self.id = item['id']
        self.bands = []
        for name in ('red', 'green', 'blue'):
            asset = item['assets'][name]
            rb = (asset.get('raster:bands') or [{}])[0]
            ds = rasterio.open(asset['href'])
            self.bands.append((ds, rb.get('scale', 1e-4), rb.get('offset', 0.0)))
        self.crs = self.bands[0][0].crs

    def pixel(self, lon, lat):
        """Continuous tile pixel coordinates (col, row) of lon/lat."""
        from rasterio.warp import transform
        xs, ys = transform('EPSG:4326', self.crs, [lon], [lat])
        col, row = ~self.bands[0][0].transform * (xs[0], ys[0])
        return col, row

    def read(self, col0, row0, w, h):
        """Reflectance (h, w, 3) as R, G, B for the window at tile pixel (col0, row0), and its valid-pixel mask."""
        from rasterio.windows import Window
        win = Window(col0, row0, w, h)
        refl, valid = [], np.ones((h, w), bool)
        for ds, scale, offset in self.bands:
            dn = ds.read(1, window=win, boundless=True, fill_value=0)
            valid &= dn != 0
            refl.append(dn_to_reflectance(dn, scale, offset))
        return np.stack(refl, -1), valid

    def chip_at(self, x, y, size=CHIP):
        """uint8 chip with tile pixel (x, y) at index (size // 2, size // 2), or None if it touches no-data."""
        refl, valid = self.read(int(x) - size // 2, int(y) - size // 2, size, size)
        return None if (~valid).mean() > 0.01 else to_uint8(refl)

    def close(self):
        for ds, _, _ in self.bands:
            ds.close()


def colour_excess(refl, size=11):
    """How far each pixel stands out in one band over the other two, in robust standard deviations.

    Sentinel-2 records blue, green and red a fraction of a second apart, so a moving aircraft leaves a
    copy that stands out in one band only. Scene structure is removed first with a median filter.
    """
    hp = refl - ndimage.median_filter(refl, size=(size, size, 1))
    e = hp - (hp.sum(-1, keepdims=True) - hp) / 2
    e = ndimage.gaussian_filter(e, (0.7, 0.7, 0))
    med = np.median(e, (0, 1))
    mad = np.median(np.abs(e - med), (0, 1)) * 1.4826 + 1e-6
    return (e - med) / mad


def _peaks(s, thresh):
    m = (s == ndimage.maximum_filter(s, size=5)) & (s >= thresh)
    ys, xs = np.nonzero(m)
    return np.c_[xs, ys].astype(float), s[ys, xs]


def find_triplets(refl, valid=None, thresh=PEAK_SIGMA):
    """Aircraft signatures: blue, green and red copies in a line with green between, 3-45 px from blue to red.

    Returns [{'b', 'g', 'r': (x, y), 'score'}], best first; score is the weakest copy in robust standard
    deviations. Each copy is used by at most one signature.
    """
    refl = refl.copy()
    if valid is not None and not valid.all():
        refl[~valid] = np.median(refl[valid], 0) if valid.any() else 0
    s = colour_excess(refl)
    if valid is not None and not valid.all():
        s[ndimage.binary_dilation(~valid, iterations=6)] = 0
    (pr, vr), (pg, vg), (pb, vb) = (_peaks(s[..., c], thresh) for c in range(3))
    cands = []
    for gi, g in enumerate(pg):
        bs = np.nonzero(np.hypot(*(pb - g).T) <= MAX_SEP)[0] if len(pb) else []
        rs = np.nonzero(np.hypot(*(pr - g).T) <= MAX_SEP)[0] if len(pr) else []
        for bi in bs:
            for ri in rs:
                v = pr[ri] - pb[bi]
                n = np.hypot(*v)
                if not MIN_SEP <= n <= MAX_SEP:
                    continue
                w = g - pb[bi]
                t = w @ v / n ** 2
                off_line = abs(v[0] * w[1] - v[1] * w[0]) / n
                if 0.25 <= t <= 0.75 and off_line <= max(2.0, 0.2 * n):
                    cands.append((min(vb[bi], vg[gi], vr[ri]), bi, gi, ri))
    out, used = [], set()
    for score, bi, gi, ri in sorted(cands, reverse=True):
        keys = {('b', bi), ('g', gi), ('r', ri)}
        if keys & used:
            continue
        used |= keys
        out.append({'b': tuple(pb[bi]), 'g': tuple(pg[gi]), 'r': tuple(pr[ri]), 'score': float(score)})
    return out


def _offsets(labels, triplets):
    L = np.asarray(labels, float).reshape(-1, 2)
    G = np.array([t['g'] for t in triplets], float).reshape(-1, 2)
    return G[None] - L[:, None]                               # (labels, signatures, 2)


def vote_shift(labels, triplets, radius=None, tol=MATCH_TOL):
    """The shift that puts the most labels within `tol` of a signature's green copy, if two or more agree.

    Within one acquisition the converted labels are off by one shared shift, so a shift supported by
    several labels is very unlikely to come from clutter.
    """
    if not len(labels) or not len(triplets):
        return None
    radius = SEARCH_RADIUS if radius is None else radius
    D = _offsets(labels, triplets)
    best, best_key = None, (1, 0.0)
    for i, j in zip(*np.nonzero(np.hypot(D[..., 0], D[..., 1]) <= radius)):
        err = np.hypot(*(D - D[i, j]).transpose(2, 0, 1))
        key = (int((err.min(1) <= tol).sum()), -float(np.hypot(*D[i, j])))
        if key > best_key:
            best, best_key = D[i, j], key
    if best is None:
        return None
    err = np.hypot(*(D - best).transpose(2, 0, 1))
    return tuple(np.median([D[i, err[i].argmin()] for i in range(len(D)) if err[i].min() <= tol], 0))


def match_at(labels, triplets, shift, tol=MATCH_TOL):
    """{label: signature} for signatures within `tol` of label + shift; each signature used once, nearest first."""
    if not len(labels) or not len(triplets):
        return {}
    err = np.hypot(*(_offsets(labels, triplets) - np.asarray(shift, float)).transpose(2, 0, 1))
    out, taken = {}, set()
    for i in np.argsort(err.min(1)):
        j = int(err[i].argmin())
        if err[i, j] <= tol and j not in taken:
            out[int(i)] = j
            taken.add(j)
    return out


def airport_shift(file_shifts, tol=AIRPORT_TOL, min_files=AIRPORT_MIN_FILES, min_agree=AIRPORT_MIN_AGREE):
    """(shift, fraction of files agreeing) from the shifts voted in each file; shift is None if inconsistent.

    The label conversion is off by one offset per airport, so most files must agree within `tol`.
    """
    S = np.asarray(file_shifts, float).reshape(-1, 2)
    if len(S) < min_files:
        return None, 0.0
    near = np.hypot(*(S[:, None] - S[None]).transpose(2, 0, 1)) <= tol
    centre = int(near.sum(1).argmax())
    agree = near[centre].mean()
    if agree < min_agree:
        return None, float(agree)
    return tuple(np.median(S[near[centre]], 0)), float(agree)


def minimal_item(it):
    keep = {k: it['properties'].get(k) for k in ('datetime', 's2:processing_baseline')}
    return {'id': it['id'], 'geometry': it['geometry'], 'properties': keep,
            'assets': {b: {'href': it['assets'][b]['href'], 'raster:bands': it['assets'][b].get('raster:bands')}
                       for b in ('red', 'green', 'blue')}}


def collect(path, airport):
    """Pass 1: label positions and nearby aircraft signatures for one annotation file, per scene."""
    lat_c, lon_c = AIRPORTS[airport]
    geom = mosaic_geometry(lat_c, lon_c)
    when = parse_time(path)
    labels = [label_lonlat(r, c, geom) for r, c in read_labels(path)]
    lon0, lat0 = geom[0], geom[1]
    bbox = (lon0, lat0, lon0 + GRID * BLOCK_W, lat0 + GRID * BLOCK_H)
    items = [minimal_item(it) for it in search(bbox, when)]
    rec = {'airport': airport, 'time': when.isoformat(), 'bbox': bbox, 'items': items, 'labels': labels,
           'no_scene': 0, 'groups': []}
    groups = {}
    for i, (lon, lat) in enumerate(labels):
        it = pick_item(items, lon, lat, when)
        if it is None:
            rec['no_scene'] += 1
        else:
            groups.setdefault(it['id'], (it, []))[1].append(i)
    half = SEARCH_RADIUS + MAX_SEP
    for item, idx in groups.values():
        sc = Scene(item)
        try:
            px = [sc.pixel(*labels[i]) for i in idx]
            triplets = []
            for x, y in px:
                x0, y0 = int(round(x)) - half, int(round(y)) - half
                refl, valid = sc.read(x0, y0, 2 * half, 2 * half)
                for t in find_triplets(refl, valid):
                    t = dict(t, **{k: (t[k][0] + x0, t[k][1] + y0) for k in 'bgr'})
                    if all(np.hypot(t['g'][0] - u['g'][0], t['g'][1] - u['g'][1]) > 2 for u in triplets):
                        triplets.append(t)
        finally:
            sc.close()
        rec['groups'].append({'item': item['id'], 'label_index': idx, 'label_px': px, 'triplets': triplets,
                              'vote': vote_shift(px, triplets)})
    return rec


def assign(records):
    """Pass 2 (offline): one offset per airport from the files' votes, then matches at that offset.

    A file whose own vote agrees with its airport's offset (within AIRPORT_TOL) is matched at its own,
    sharper shift with MATCH_TOL; any other file at AIRPORT_MATCH_TOL around the airport offset. Airports
    without a consistent offset are dropped. Only signatures scoring MIN_MATCH_SCORE or more are matched,
    and an aircraft labelled in two files of the same pass is kept once, at its stronger match.
    Returns ({airport: (shift, agree, files voting)}, matches).
    """
    votes = {}
    for rec in records:
        for g in rec['groups']:
            if g['vote'] is not None:
                votes.setdefault(rec['airport'], []).append(g['vote'])
    shifts = {ap: airport_shift(v) + (len(v),) for ap, v in votes.items()}
    matches = []                                     # (record, group, label k, triplet j, kind, shift)
    for rec in records:
        ap_shift = shifts.get(rec['airport'], (None,))[0]
        if ap_shift is None:
            continue
        for g in rec['groups']:
            own = g['vote']
            if own is not None and np.hypot(own[0] - ap_shift[0], own[1] - ap_shift[1]) <= AIRPORT_TOL:
                shift, tol, kind = own, MATCH_TOL, 'file shift'
            else:
                shift, tol, kind = ap_shift, AIRPORT_MATCH_TOL, 'airport shift'
            strong = [j for j, t in enumerate(g['triplets']) if t['score'] >= MIN_MATCH_SCORE]
            for k, j in match_at(g['label_px'], [g['triplets'][j] for j in strong], shift, tol).items():
                matches.append((rec, g, k, strong[j], kind, shift))
    kept = []
    for m in sorted(matches, key=lambda m: -m[1]['triplets'][m[3]]['score']):
        rec, g, k = m[:3]
        lon, lat = rec['labels'][g['label_index'][k]]
        if not any(r['airport'] == rec['airport'] and r['time'][:10] == rec['time'][:10]
                   and geo_dist(lat, lon, *r['labels'][h['label_index'][kk]][::-1]) < DUPLICATE_DIST_M
                   for r, h, kk, *_ in kept):
            kept.append(m)
    return shifts, kept


def cut(rec, groups, out, n_background):
    """Pass 3: chips for one file's matched labels ({group index: [(k, j, kind, shift)]}) and backgrounds."""
    when = datetime.datetime.fromisoformat(rec['time'])
    stamp = when.strftime('%Y%m%d%H%M%S')
    airport, labels, bbox = rec['airport'], rec['labels'], rec['bbox']
    items = {it['id']: it for it in rec['items']}
    rows, scenes, counts = [], {}, {'no-data': 0}

    def scene_of(item_id):
        if item_id not in scenes:
            scenes[item_id] = Scene(items[item_id])
        return scenes[item_id]

    def write(img, name, sc, label, source, extra):
        Image.fromarray(img).save(os.path.join(out, 'chips', name))
        row = {'chip': name, 'scene': sc.id, 'label': label, 'on_runway': '', 'source': source,
               'x': '', 'y': '', 'blue_x': '', 'blue_y': '', 'red_x': '', 'red_y': '', 'airport': airport,
               'time': rec['time'], 'lon': '', 'lat': '', 'label_x': '', 'label_y': '',
               'match': '', 'shift_x': '', 'shift_y': '', 'score': ''}
        row.update(extra)
        rows.append(row)

    try:
        for gi, ms in groups.items():
            g = rec['groups'][gi]
            sc = scene_of(g['item'])
            for k, j, kind, shift in ms:
                t, i = g['triplets'][j], g['label_index'][k]
                gx, gy = int(t['g'][0]), int(t['g'][1])
                img = sc.chip_at(gx, gy)
                if img is None:
                    counts['no-data'] += 1
                    continue
                write(img, '%s_%s_p%02d.png' % (airport, stamp, i), sc, 'aircraft', 'flying-airplanes label', {
                    'x': gx, 'y': gy, 'blue_x': '%.0f' % t['b'][0], 'blue_y': '%.0f' % t['b'][1],
                    'red_x': '%.0f' % t['r'][0], 'red_y': '%.0f' % t['r'][1],
                    'lon': '%.6f' % labels[i][0], 'lat': '%.6f' % labels[i][1],
                    'label_x': '%.1f' % g['label_px'][k][0], 'label_y': '%.1f' % g['label_px'][k][1],
                    'match': kind, 'shift_x': '%.1f' % shift[0], 'shift_y': '%.1f' % shift[1],
                    'score': '%.1f' % t['score']})
        rng = np.random.default_rng(int(hashlib.md5((airport + stamp).encode()).hexdigest()[:8], 16))
        n_bg, tries = 0, 0
        while n_bg < n_background and tries < 20 * n_background:
            tries += 1
            lon = bbox[0] + rng.random() * (bbox[2] - bbox[0])
            lat = bbox[1] + rng.random() * (bbox[3] - bbox[1])
            if any(geo_dist(lat, lon, la, lo) < MIN_BACKGROUND_DIST_M for lo, la in labels):
                continue
            it = pick_item(rec['items'], lon, lat, when)
            if it is None:
                continue
            sc = scene_of(it['id'])
            col, row = sc.pixel(lon, lat)
            img = sc.chip_at(int(round(col)), int(round(row)))
            if img is None:
                continue
            write(img, '%s_%s_bg%02d.png' % (airport, stamp, n_bg), sc, 'background', 'random', {
                'x': int(round(col)), 'y': int(round(row)), 'lon': '%.6f' % lon, 'lat': '%.6f' % lat})
            n_bg += 1
    finally:
        for sc in scenes.values():
            sc.close()
    return rows, counts


def chips_from_labels(out, workers):
    """Re-cut every chip in <out>/labels.csv from its scene: no label repo or signature search needed."""
    by_scene = {}
    for r in csv.DictReader(open(os.path.join(out, 'labels.csv'))):
        by_scene.setdefault(r['scene'], []).append(r)
    os.makedirs(os.path.join(out, 'chips'), exist_ok=True)

    def one(scene_id, rows):
        sc = Scene(minimal_item(fetch_item(scene_id)))
        missing = 0
        try:
            for r in rows:
                img = sc.chip_at(int(r['x']), int(r['y']))
                if img is None:
                    missing += 1
                else:
                    Image.fromarray(img).save(os.path.join(out, 'chips', r['chip']))
        finally:
            sc.close()
        return len(rows) - missing, missing

    results, failed = run_pool(one, list(by_scene.items()), workers, 'chips')
    print('%d chips written, %d not readable, %d scenes failed' % (
        sum(r[0] for r in results), sum(r[1] for r in results), len(failed)))


def run_pool(fn, jobs, workers, what):
    out, failed = [], []
    with ThreadPoolExecutor(workers) as ex:
        futures = [ex.submit(fn, *j) for j in jobs]
        for i, fut in enumerate(futures):
            try:
                out.append(fut.result())
            except Exception as e:  # a failed search or read loses one file, not the run
                failed.append(jobs[i][0] if isinstance(jobs[i][0], str) else i)
                print('failed:', e, flush=True)
            if (i + 1) % 200 == 0:
                print('%s: %d / %d' % (what, i + 1, len(jobs)), flush=True)
    return out, failed


def main(argv=None):
    global SEARCH_RADIUS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('annotations', nargs='?', help='annotations/ directory of maups/covid19-custom-script-contest')
    p.add_argument('out')
    p.add_argument('--background-per-file', type=int, default=1,
                   help='random background chips per annotation file (labelled or empty)')
    p.add_argument('--airports', nargs='*', help='only these airports (default: all labelled)')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--search-radius', type=int, default=SEARCH_RADIUS,
                   help='px around each converted label searched for its aircraft (pass 1)')
    p.add_argument('--reuse-candidates', action='store_true',
                   help='skip pass 1 and reuse <out>/candidates.json (to re-tune matching without downloads)')
    p.add_argument('--chips-from-labels', action='store_true',
                   help='only re-cut the chips listed in <out>/labels.csv (e.g. after cloning the repo)')
    a = p.parse_args(argv)

    if a.chips_from_labels:
        return chips_from_labels(a.out, a.workers)
    SEARCH_RADIUS = a.search_radius
    os.makedirs(os.path.join(a.out, 'chips'), exist_ok=True)
    cand_path = os.path.join(a.out, 'candidates.json')
    if a.reuse_candidates:
        records = json.load(open(cand_path))
    else:
        jobs = []
        for airport in sorted(os.listdir(a.annotations)):
            if airport not in AIRPORTS or (a.airports and airport not in a.airports):
                continue
            for f in sorted(os.listdir(os.path.join(a.annotations, airport))):
                if f.endswith('.txt'):
                    jobs.append((os.path.join(a.annotations, airport, f), airport))
        print('%d annotation files' % len(jobs), flush=True)
        records, failed = run_pool(collect, jobs, a.workers, 'pass 1, signatures')
        if failed:
            print('%d files failed in pass 1' % len(failed))
        with open(cand_path, 'w') as f:
            json.dump(records, f)

    shifts, matches = assign(records)
    per_file = {}
    for rec, g, k, j, kind, shift in matches:
        gi = next(n for n, h in enumerate(rec['groups']) if h is g)
        per_file.setdefault(id(rec), (rec, {}))[1].setdefault(gi, []).append((k, j, kind, shift))
    jobs = [(rec, per_file.get(id(rec), (rec, {}))[1], a.out, a.background_per_file) for rec in records]
    results, failed = run_pool(cut, jobs, a.workers, 'pass 3, chips')
    rows = [r for rs, _ in results for r in rs]

    rows.sort(key=lambda r: r['chip'])
    with open(os.path.join(a.out, 'labels.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by_airport = {}
    for rec in records:
        st = by_airport.setdefault(rec['airport'], {'labels': 0, 'matched': 0})
        st['labels'] += len(rec['labels'])
    for row in rows:
        if row['label'] == 'aircraft':
            by_airport[row['airport']]['matched'] += 1
    for ap, st in by_airport.items():
        sh = shifts.get(ap, (None, 0.0, 0))
        st['offset px'] = None if sh[0] is None else [round(v, 1) for v in sh[0]]
        st['files agreeing'] = '%.0f%% of %d' % (100 * sh[1], sh[2])
    summary = {'files': len(records), 'failed files': len(failed),
               'aircraft': sum(r['label'] == 'aircraft' for r in rows),
               'background': sum(r['label'] == 'background' for r in rows),
               'match kinds': {k: sum(r['match'] == k for r in rows) for k in ('file shift', 'airport shift')},
               'by airport': by_airport}
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    main()
