# Model Setup — MobileNet-SSD (Path 2: Object Detection)

The `models/` folder should contain two files:

```
models/
├── MobileNetSSD_deploy.prototxt      (~44 KB  — network architecture)
└── MobileNetSSD_deploy.caffemodel    (~23 MB  — pre-trained weights)
```

**These are already bundled in this project.** You don't need to do
anything extra to run `python3 main.py detect ...`.

## If the files are missing (or you deleted them)

### Option A — automatic
```bash
python3 download_models.py
```

### Option B — manual download
Download both files from the official public release and place them in
`models/`:
- https://github.com/chuanqi305/MobileNet-SSD/blob/master/deploy.prototxt
- https://github.com/chuanqi305/MobileNet-SSD/blob/master/mobilenet_iter_73000.caffemodel

(Use the "Download raw file" / "Save As" option — not the HTML preview page.)

## About this model
- **Architecture:** MobileNet v3 backbone + Single Shot Detector (SSD) head
- **Trained on:** PASCAL VOC 2007/2012 (20 object classes: person, car,
  dog, bicycle, bus, cat, chair, etc. — see `VOC_CLASSES` in
  `object_detector.py` for the full list)
- **Why this model:** Small, fast, CPU-friendly — ideal for a "basic"
  implementation milestone without needing a GPU or large downloads,
  exactly matching the deck's "Transfer Learning" philosophy: reuse an
  already-trained model instead of training from scratch.

## Note on detection results
This model only recognizes the 20 PASCAL VOC classes. If you test it on a
photo containing objects outside that list (e.g. a laptop, a phone), it
will report 0 accepted detections — that's expected model behavior, not
a bug in this project's code.
