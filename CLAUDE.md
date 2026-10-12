# CLAUDE.md

Detecting aircraft **in flight**. The original repo (root `model.py`, `train.py`, `detector.py`, TF1/TFLearn)
classifies PlanesNet chips, which are almost all parked aircraft on tarmac; those files are kept for reference
only and do not install on current Python. All active code is PyTorch, in two pipelines:

- `satellite/`: looking down from orbit (Planet / Sentinel-2 style RGB).
- `sky/`: cameras looking at aircraft against sky or ground (AOT dataset, YOLO).

The full write-up, with figures, results and loss curves, is `docs/report.md`.

## Commands

```bash
pip install -r requirements.txt
python -m pytest -q                      # ~1 min, synthetic data only, no downloads

# Satellite
python -m satellite.train --planesnet planesnet.json --out models/plane.pt [--backgrounds DIR] [--airborne-val DIR]
python -m satellite.build_s2_flying data/s2_flying --chips-from-labels    # chips are not committed
python -m satellite.evaluate_s2 models/plane.pt data/s2_flying
python -m satellite.train_s2 data/s2_flying --out models/s2net.pt --test data/s2_airborne   # ~1.5 h on CPU
python -m satellite.detect models/plane.pt scene.png [--references other_dates.png ...] [--json dets.json]

# Sky
curl -o groundtruth.csv https://airborne-obj-detection-challenge-training.s3.amazonaws.com/part1/ImageSets/groundtruth.csv
python -m sky.prepare_aot groundtruth.csv data/aot --train 1500 --val 400      # writes data/aot/{single,temporal}
python -m sky.prepare_aot part2.csv data/aot_part2 --part part2 --train 0 --val 1500 --val-fraction 1   # test set
python -m sky.finetune data/aot/temporal.yaml --temporal --imgsz 640 --epochs 50
python -m sky.evaluate runs/sky/train/weights/best.pt data/aot/temporal --meta data/aot/meta.json
python -m sky.evaluate_clips groundtruth.csv data/aot_clips --model temporal=best.pt:0.18:2
python -m sky.detect clip.mp4 --weights best.pt --temporal 2 --tile 640
```

## Layout

| Path | What it does |
| --- | --- |
| `satellite/model.py` | `PlaneNet`: unpadded convs + aligned pools, so a full scene scores every 20 px window at stride 4 in one pass |
| `satellite/augment.py` | Plane cut-outs (`plane_mask`), procedural cloud/sea/land backgrounds, band-offset ghosting, contrails, distractor negatives |
| `satellite/data.py` | PlanesNet JSON loader, `ChipDataset` mixing real chips with on-the-fly synthetic in-flight samples |
| `satellite/train.py` | Training; model selection on in-flight validation (real if `--airborne-val`, else synthetic), not PlanesNet |
| `satellite/build_s2_flying.py` | Label-guided Sentinel-2 test set (`data/s2_flying`): converts the Flying Airplanes labels, finds three-colour signatures near them, one offset per airport; `--chips-from-labels` re-cuts chips |
| `satellite/s2_model.py`, `satellite/train_s2.py` | S2Net: 48x48 px Sentinel-2 classifier trained on `data/s2_flying`, cross-validated by airport, tested on `data/s2_airborne` |
| `satellite/evaluate_s2.py` | Scores PlaneNet on a Sentinel-2 test set (`data/s2_airborne`, `data/s2_flying`) |
| `satellite/detect.py` | Dense scoring, peak finding, static-object suppression across dates, band-offset motion and heading |
| `sky/prepare_aot.py` | AOT to YOLO crops (640 px, native resolution), flight-level split, single and temporal variants |
| `sky/temporal.py` | Align frames t-k and t+k to t (phase correlation), brightness-match, stack as 3 channels |
| `sky/finetune.py` | Ultralytics training wrapper; `--temporal` disables hue/saturation jitter |
| `sky/models/yolo11-p2.yaml` | YOLO11 with an extra stride-4 head for 4-10 px targets; use as `yolo11n-p2.yaml` + `--pretrained yolo11n.pt` |
| `sky/evaluate.py` | Crop-level scoring with centre-distance matching |
| `sky/evaluate_clips.py` | Clip-level scoring of raw vs tracker-confirmed detections on full frames |
| `sky/detect.py` | Image/video inference: tiling, tile-edge de-duplication, `Tracker` (multi-frame confirmation) |
| `report_figs/` | Figures used in `docs/report.md` (made by one-off scripts that are not in the repo) |

## Conventions that matter

- **Evaluation matches by centre distance, not IoU** (hit if within max(8 px, half the box)). AOT aircraft are often
  under 10 px, where IoU is dominated by 1 px errors. Birds and AOT `Airborne` (unknown) objects are ignored, never
  counted as false positives. Keep this when adding metrics, or numbers stop being comparable with the report.
- **AOT splits are by flight** (`split_of`: md5 of flight id, 20% val). Never split by frame: neighbouring frames are
  near-duplicates and leak.
- **AOT parts share flights**: part3 has 352 of part1's flights and 364 of part2's; part1 and part2 share none. Part2 is
  the test set for models trained on part1 (`data/aot_part2`). When training on part3, pass
  `--exclude-flights part2.csv` and keep `--val-fraction 0.2`, so shared flights stay on the same side as in part1.
- **Temporal stacks** are written by OpenCV, so channels are (B, G, R) = (t-k, t, t+k). Training and inference must
  both go through `sky/temporal.stack`. Any colour augmentation that mixes channels breaks them.
- **PlaneNet must stay unpadded** with pool-aligned strides; `tests/test_satellite.py` checks dense scores equal
  per-window scores. Padding would silently change full-scene results.
- Class ids from `prepare_aot`: 0 airplane, 1 helicopter, 2 bird, 3 drone, 4 unknown. Aircraft = {0, 1, 3}.

## Data (not in the repo)

- **PlanesNet**: `planesnet.json` from Kaggle (`rhammell/planesnet`). Kaggle is blocked in the Claude cloud
  environment; it was supplied on the `data` branch as `archive.zip`.
- **AOT**: public S3, no account; `groundtruth.csv` per part at `<part>/ImageSets/groundtruth.csv` (part2 orders its
  columns differently, which `csv.DictReader` handles). About 28% of part1 frames are missing (HTTP 404);
  `prepare_aot` over-plans by 35% and `evaluate_clips` retries other clips.
- **Sentinel-2** (airborne satellite test set): public COGs at `sentinel-cogs.s3.us-west-2.amazonaws.com`,
  read by window with rasterio. The STAC search API was blocked in the cloud environment; locally, Earth Search
  (`earth-search.aws.element84.com/v1`, collection `sentinel-2-l2a`) works and covers 2020. Its `raster:bands` give the
  scale and offset (baseline >= 04.00 stores reflectance + 0.1).
- Model weights (`*.pt`), `runs/` and data dirs are git-ignored.

## Gotchas

- Training on CPU is slow (`yolo11n` at 640 px: about 6 min per epoch on 1,500 crops with 4 cores). Use a GPU for
  anything beyond smoke tests.
- `sky/detect.py --temporal K` delays output by K frames; reported frame numbers are shifted back to source frames.
- Small networks with batch norm need about 100+ training steps before eval-mode scores mean anything; with fewer,
  running statistics are unsettled and every input gets the same score (seen in `tests/test_s2_model.py`).
- PyTorch on CPU uses every core by default; alongside other jobs set `OMP_NUM_THREADS=4` or it thrashes.
- Untrained YOLO heads (the P2 layers) need more epochs than the transferred ones; compare architectures at equal
  and sufficient epochs.
