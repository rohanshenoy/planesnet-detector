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
python -m satellite.detect models/plane.pt scene.png [--references other_dates.png ...] [--json dets.json]

# Sky
curl -o groundtruth.csv https://airborne-obj-detection-challenge-training.s3.amazonaws.com/part1/ImageSets/groundtruth.csv
python -m sky.prepare_aot groundtruth.csv data/aot --train 1500 --val 400      # writes data/aot/{single,temporal}
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
| `satellite/detect.py` | Dense scoring, peak finding, static-object suppression across dates, band-offset motion and heading |
| `sky/prepare_aot.py` | AOT to YOLO crops (640 px, native resolution), flight-level split, single and temporal variants |
| `sky/temporal.py` | Align frames t-k and t+k to t (phase correlation), brightness-match, stack as 3 channels |
| `sky/finetune.py` | Ultralytics training wrapper; `--temporal` disables hue/saturation jitter |
| `sky/models/yolo11-p2.yaml` | YOLO11 with an extra stride-4 head for 4-10 px targets; use as `yolo11n-p2.yaml` + `--pretrained yolo11n.pt` |
| `sky/evaluate.py` | Crop-level scoring with centre-distance matching |
| `sky/evaluate_clips.py` | Clip-level scoring of raw vs tracker-confirmed detections on full frames |
| `sky/detect.py` | Image/video inference: tiling, tile-edge de-duplication, `Tracker` (multi-frame confirmation) |
| `report_figs/` | Figures used in `docs/report.md` |

## Conventions that matter

- **Evaluation matches by centre distance, not IoU** (hit if within max(8 px, half the box)). AOT aircraft are often
  under 10 px, where IoU is dominated by 1 px errors. Birds and AOT `Airborne` (unknown) objects are ignored, never
  counted as false positives. Keep this when adding metrics, or numbers stop being comparable with the report.
- **AOT splits are by flight** (`split_of`: md5 of flight id, 20% val). Never split by frame: neighbouring frames are
  near-duplicates and leak.
- **Temporal stacks** are written by OpenCV, so channels are (B, G, R) = (t-k, t, t+k). Training and inference must
  both go through `sky/temporal.stack`. Any colour augmentation that mixes channels breaks them.
- **PlaneNet must stay unpadded** with pool-aligned strides; `tests/test_satellite.py` checks dense scores equal
  per-window scores. Padding would silently change full-scene results.
- Class ids from `prepare_aot`: 0 airplane, 1 helicopter, 2 bird, 3 drone, 4 unknown. Aircraft = {0, 1, 3}.

## Data (not in the repo)

- **PlanesNet**: `planesnet.json` from Kaggle (`rhammell/planesnet`). Kaggle is blocked in the Claude cloud
  environment; it was supplied on the `data` branch as `archive.zip`.
- **AOT**: public S3, no account. About 28% of frames listed in `groundtruth.csv` are missing (HTTP 404);
  `prepare_aot` over-plans by 35% and `evaluate_clips` retries other clips.
- **Sentinel-2** (airborne satellite test set): public COGs at `sentinel-cogs.s3.us-west-2.amazonaws.com`,
  read by window with rasterio. The STAC search API was blocked; scenes are found by listing the bucket per MGRS tile.
- Model weights (`*.pt`), `runs/` and data dirs are git-ignored.

## Gotchas

- Training on CPU is slow (`yolo11n` at 640 px: about 6 min per epoch on 1,500 crops with 4 cores). Use a GPU for
  anything beyond smoke tests.
- `sky/detect.py --temporal K` delays output by K frames; reported frame numbers are shifted back to source frames.
- Untrained YOLO heads (the P2 layers) need more epochs than the transferred ones; compare architectures at equal
  and sufficient epochs.
