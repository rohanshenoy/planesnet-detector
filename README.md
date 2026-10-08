# planesnet-detector
This repository contains scripts that enable the automatic detection of aircraft in [Planet](https://www.planet.com/) imagery using machine learning techniques. Included are files which define a machine learning model, train it using the Planesnet dataset, and apply it across an entire image scene to highlight aircraft detections.

## Detecting aircraft in flight

PlanesNet is almost entirely parked and taxiing aircraft on tarmac, so a model trained on it alone learns
"white cross on grey concrete next to a terminal" and does poorly on aircraft in the air. This repository now has
two pipelines for airborne aircraft, written in PyTorch:

| | `satellite/` | `sky/` |
|---|---|---|
| Viewpoint | Looking down from orbit (Planet, Sentinel-2) | Ground or airborne camera looking at the sky |
| Model | Small fully convolutional CNN on 20x20 chips | Ultralytics YOLO (COCO `airplane` class out of the box) |
| Closing the domain gap | PlanesNet aircraft composited onto cloud / sea / haze backgrounds, simulated band-offset ghosting, contrails, hard negatives; real in-flight chips if you have them | Fine-tune on sky imagery (e.g. Airborne Object Tracking dataset) with birds / glare / clouds as negatives |
| Time-series cue | Static objects present in co-registered images from other dates are suppressed; inter-band offset flags moving aircraft and gives heading | Multi-frame tracking: a detection is reported only after it persists and moves smoothly, rejecting one-frame clutter and erratic birds |
| Small targets | Dense scoring at 1-4 px stride in one pass | Tiled inference at native resolution, merged with NMS |

```bash
pip install -r requirements.txt
python -m pytest            # runs on synthetic data, no downloads needed
```

### Satellite: aircraft in flight seen from orbit

```bash
# Train. Model selection uses the in-flight validation set, not PlanesNet accuracy.
#   --backgrounds     aircraft-free cloud/sea/land scenes for backgrounds and hard negatives (optional)
#   --airborne-train  real in-flight chips as plane/ and no-plane/ folders (optional)
#   --airborne-val    held-out real in-flight chips (optional, strongly recommended)
python -m satellite.train --planesnet planesnet.json --out models/plane.pt \
    --backgrounds data/cloud_sea_scenes \
    --airborne-train data/airborne_train --airborne-val data/airborne_val

# Detect. --references are co-registered images of the same area on other dates.
python -m satellite.detect models/plane.pt scenes/scene_1.png \
    --references scene_1_2024-05.png scene_1_2024-06.png --json dets.json
```

Each detection in the JSON has its centre, score, `band_shift` (red-to-blue displacement in px), `moving`
and `heading_deg`. Speed is `|band_shift| * GSD / inter-band delay`, which depends on the sensor.

Synthetic compositing is a starting point, not a substitute for real data: label a few hundred real
in-flight chips (search scenes under busy flight corridors, matched against ADS-B tracks from OpenSky) and
pass them as `--airborne-val` so you measure what you actually care about.

### Sky: aircraft seen from the ground or another aircraft

```bash
# Works immediately with COCO weights
python -m sky.detect photo.jpg --tile 640
python -m sky.detect clip.mp4 --tile 640 --min-hits 5 --json tracks.json

# Fine-tune on your own sky data (YOLO format, see sky/finetune.py)
python -m sky.finetune data/sky_aircraft.yaml --imgsz 1280 --epochs 100
python -m sky.detect clip.mp4 --weights runs/sky/train/weights/best.pt --tile 1280
```

### Results on real airborne footage (AOT)

`sky/prepare_aot.py` builds 640 px native-resolution crops from the
[Airborne Object Tracking](https://registry.opendata.aws/airborne-object-tracking/) dataset (part 1),
split by flight. Trained for 12 epochs from `yolo11n.pt` on CPU (1,548 crops, 1,357 aircraft) and scored with
`sky/evaluate.py` on 287 crops / 260 aircraft from held-out flights. A hit is a prediction centre within
max(8 px, half the box) of an airplane, helicopter or drone; hits on birds and unidentified objects are ignored.

| | COCO `yolo11n` | Fine-tuned, single frame | Fine-tuned, 3 stacked frames |
|---|---|---|---|
| Average precision | 0.11 | 0.55 | **0.58** |
| Precision / recall at best F1 | 52% / 12% | 58% / 57% | **70% / 58%** |
| Recall, sky background | 17% | 75% | 74% |
| Recall, ground background | 2% | 28% | 30% |
| Recall, targets < 10 px | 0% | 26% | 28% |

The temporal model sees frames t-2, t, t+2 aligned and stacked as channels (`sky/temporal.py`); at the
same recall it makes about 40% fewer false detections. Single training run each, so small differences
(a few points of recall in a subgroup) are within noise.

```bash
curl -o groundtruth.csv https://airborne-obj-detection-challenge-training.s3.amazonaws.com/part1/ImageSets/groundtruth.csv
python -m sky.prepare_aot groundtruth.csv data/aot --train 1500 --val 400
python -m sky.finetune data/aot/temporal.yaml --temporal --imgsz 640 --epochs 50
python -m sky.evaluate runs/sky/train/weights/best.pt data/aot/temporal --meta data/aot/meta.json
python -m sky.detect clip.mp4 --weights runs/sky/train/weights/best.pt --temporal 2 --tile 640
```

## Original PlanesNet detector (legacy)

The original TensorFlow 1 / TFLearn scripts (`model.py`, `train.py`, `detector.py`) are kept below for
reference. They require Python 3.5-era TensorFlow and no longer install on current Python.

### Methodology
[PlanesNet](https://www.kaggle.com/rhammell/planesnet) is a labeled training dataset consiting of image chips extracted from Planet satellite imagery. It contains thousands of 20x20 pixel RGB image chips labeled with either a "plane" or "no-plane" classification. Machine learning models can be trained against this data to classify any given input chip into either one of these classes. 

With an accurately trained model, this classification process can be extended to a full Planet image scene by using a sliding window technique. A 20x20 pixel window is moved across each pixel position in the image, extracted, and classified by the model. Neighboring window poistions that are classified as "plane" are then clustered into a single detection. These detections are highlighted with a bounding box in a copy of the original Planet scene.

See an example of the results below. 
<p align="center">
<img src="img/input.jpg" width="400">
<img src="img/detections.jpg" width="400">
</p>

[Additional Results](https://imgur.com/a/vYnQw)

### Setup
Python 3.5+ is required for compatability with all required modules

```bash
# Clone this repository
git clone https://github.com/rhammell/planesnet-detector.git

# Go into the repository
cd planesnet-detector

# Install required modules
pip install -r requirements.txt
```

### Model
A convolutional neural network (CNN) is defined within the `model.py` module using the [TFLearn](http://tflearn.org/) library. This model supports the 20x20x3 input dimensions of the PlanesNet image data.

### Training
The defined CNN can be trained with the JSON version of the PlanesNet dataset and saved to a Tensorflow .tfl file for later use. Train the model by running `train.py` and passing the path to `planesnet.json` and the path to the output .tfl file as arguments.

```bash
# Train the model
mkdir models
python train.py "planesnet.json" "models/model.tfl"
```

The latest version of `planesnet.json` is available through the [PlanesNet](https://www.kaggle.com/rhammell/planesnet) Kaggle page, which has further information describing the dataset layout. 

### Detector
A trained model can be applied across entire images using the sliding window detector function `detector.py`, which takes the model file path, input image path, and optional output image path as arguments. The output image will cluster positive detections and draw a bounding box around their center point. 

Example images are contained in the `scenes` directory. 
```bash
# Run on demo image with default output path
python detector.py "models/model.tfl" "scenes/scene_1.png"

# Run on demo image with defined output path
python detector.py "models/model.tfl" "scenes/scene_1.png" "scenes/scene_1_detections.png"
```
