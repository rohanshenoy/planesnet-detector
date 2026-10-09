# Sentinel-2 aircraft-in-flight test set

A small, hand-labelled test set of aircraft seen from orbit, used by `python -m satellite.evaluate_s2`.

- **Source:** Sentinel-2 L2A, bands B04/B03/B02 (10 m), public COGs at
  `sentinel-cogs.s3.us-west-2.amazonaws.com`, 10 clear scenes (June to August 2025) around
  Heathrow, Gatwick, Paris CDG, Frankfurt, Chicago O'Hare and New York JFK.
- **Candidates:** a band-offset finder. Sentinel-2 records blue, green and red a fraction of a second apart,
  so a moving aircraft shows as blue, green and red copies in a line; the finder looks for blue and red peaks
  3 to 40 px apart with a green peak between them and no co-located static object.
- **Labels (by eye):** `aircraft` (14, of which 2 on a runway mid take-off), `probable` (5, faint),
  `clutter` (62 finder false positives: cars, roofs, stadium lights), `background` (250 random chips, 25 per scene).
- **Chips:** 64x64 px at 10 m, centred on the candidate (the green copy), saved as 8-bit PNG of
  reflectance / 0.3. `labels.csv` keeps scene id, pixel position in the read window, and the blue/red copy positions.

Small by design (19 positives): treat every metric from it as having wide error bars.
