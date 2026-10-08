"""
Fine-tune a YOLO model on sky imagery of aircraft.

The dataset must be in Ultralytics YOLO format, described by a YAML file:

    path: data/sky_aircraft
    train: images/train
    val: images/val
    names: {0: airplane}

with one .txt label file per image (class cx cy w h, normalised).

Useful public sources of aircraft-against-sky imagery:
  * Airborne Object Tracking (AOT) dataset, Amazon Prime Air: long sequences
    of distant aircraft, helicopters and birds seen from aircraft-mounted
    cameras (labels are bounding boxes; convert to YOLO format).
  * Your own footage, labelled with a tool such as CVAT or Label Studio.
Add sky frames with birds, contrails, sun glare and cloud edges but no
aircraft as background images (images with empty label files): they are the
hard negatives that cut false positives.

Example:
    python -m sky.finetune data/sky_aircraft.yaml --epochs 100 --imgsz 1280
"""

import argparse


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('data', help='dataset YAML')
    p.add_argument('--weights', default='yolo11n.pt', help='starting weights')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--imgsz', type=int, default=1280,
                   help='train at high resolution so distant aircraft keep their pixels')
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--device', default=None)
    p.add_argument('--project', default='runs/sky')
    p.add_argument('--name', default='train')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--close-mosaic', type=int, default=10, help='final epochs without mosaic')
    p.add_argument('--temporal', action='store_true',
                   help='images are prepare_aot temporal stacks: disable hue/saturation '
                        'jitter, which would scramble the time channels')
    a = p.parse_args(argv)

    from ultralytics import YOLO
    model = YOLO(a.weights)
    colour = dict(hsv_h=0.0, hsv_s=0.0) if a.temporal else {}
    return model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                       device=a.device, project=a.project, name=a.name, workers=a.workers,
                       close_mosaic=a.close_mosaic,
                       **colour,
                       # sky has no canonical "up" for distant aircraft; keep scale jitter mild
                       # so tiny targets are not shrunk out of existence
                       fliplr=0.5, flipud=0.2, degrees=10, scale=0.3, mosaic=1.0)


if __name__ == '__main__':
    main()
