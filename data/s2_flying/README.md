# Sentinel-2 aircraft-in-flight test set, label-guided

927 aircraft in flight at 10 European airports, plus 2,134 background chips, for
`python -m satellite.evaluate_s2`. It is 49 times the size of `data/s2_airborne` (19 aircraft).

- **Source of the labels:** the "Flying Airplanes" labels of Pamplona Segundo et al. (2021),
  [maups/covid19-custom-script-contest](https://github.com/maups/covid19-custom-script-contest) (MIT licence):
  1,782 airplanes in Sentinel-2 images around 15 airports, January to June 2020.
- **Imagery:** Sentinel-2 L2A, bands B04/B03/B02 at 10 m, public COGs on AWS found through Earth Search.
  Contains modified Copernicus Sentinel data (2020).
- **Chips:** 64x64 px at 10 m centred on the aircraft's green copy, 8-bit PNG of reflectance / 0.3, the format of
  `data/s2_airborne`. Chips are not committed; re-cut them from `labels.csv` (a few minutes):

  ```bash
  python -m satellite.build_s2_flying data/s2_flying --chips-from-labels
  ```

## How it was built

`satellite/build_s2_flying.py`. Converted to ground positions with the label authors' mosaic geometry, the labels
miss their aircraft by one offset per airport, for a reason that could not be found without the authors' own
images. So the labels are used as hints:

1. Around each label (150 px radius), find aircraft signatures: blue, green and red copies in a line with green
   between, 3 to 45 px apart. Each image votes for the shift that puts two or more of its labels on a signature.
2. Each airport's offset is the shift most of its images agree on (at least 60%, within 15 px). Labels are matched to
   a signature at that offset only, scoring at least 15. An aircraft labelled in two overlapping images of one
   pass is kept once.
3. Chips are cut around each match, plus one random background chip per label file, at least 1 km from any label.

| Airport | Labels | Kept | Offset (px, x right / y down) | Images agreeing |
| --- | --- | --- | --- | --- |
| AGP Malaga | 71 | 53 | (-26.5, -76.5) | 93% of 15 |
| BCN Barcelona | 151 | 139 | (-26.4, -3.5) | 100% of 25 |
| BRU Brussels | 115 | 82 | (10.8, -3.1) | 100% of 26 |
| CDG Paris | 239 | 195 | (19.3, 0.1) | 98% of 43 |
| CPH Copenhagen | 58 | 36 | (-84.3, 123.7) | 60% of 10 |
| DUB Dublin | 73 | 66 | (-121.6, -20.4) | 93% of 15 |
| DUS Dusseldorf | 104 | 48 | (10.7, -2.0) | 100% of 20 |
| FCO Rome | 150 | 124 | (-25.3, -88.7) | 97% of 32 |
| HEL Helsinki | 41 | 31 | (-76.6, -56.9) | 100% of 7 |
| LGW London Gatwick | 187 | 153 | (-22.3, -11.7) | 95% of 43 |
| AMS, ARN, ATH, FRA, IST | 593 | 0 | no consistent offset within 150 px | 12-33% |

AMS and FRA have offsets beyond the 150 px search: with `--search-radius 400` they give (-4, -158) and (-188, 113),
95 and 84 aircraft, but only 61-64% of images agree and about 80% of a checked sample (25 of 32) were clear aircraft,
so they are left out. ARN, ATH and IST were not tried with a wider search. CPH only just passes, with an offset at the
edge of the search radius.

## Quality and biases

- **Checked by eye:** in two random samples of the final set (80 chips, all 10 airports), 72 (90%) show a clear
  three-colour aircraft signature on the centre; the other 8 are faint or doubtful.
- **Clear aircraft only:** a match needs a strong signature, so faint aircraft are under-represented and recall on
  this set overstates recall on all aircraft. `data/s2_airborne` has the same bias.
- **Labels are not exhaustive:** a few background chips may hold an unlabelled aircraft.
- **January to June 2020**, partly during COVID-19 lockdowns.

## Columns of labels.csv

`chip`, `scene` (STAC item id), `label` (`aircraft` / `background`), `source`, `x`, `y` (tile pixel of the chip
centre: the green copy for aircraft), `blue_x`, `blue_y`, `red_x`, `red_y` (copy positions, tile pixels), `airport`,
`time` (acquisition, from the label file name), `lon`, `lat` (the converted label, or the background point),
`label_x`, `label_y` (converted label, tile pixels), `match` (`file shift`: the image's own vote; `airport shift`: the
airport's offset), `shift_x`, `shift_y`, `score` (weakest copy, in robust standard deviations). `on_runway` is unused.
