# 👁️ DecodeLabs AI Project 4 — Image / Text Recognition (Basic)

**"Building the Machine's Optic Nerve" — The DecodeLabs Architect's Playbook for Image & Text Recognition**
**Industrial Training Kit | Batch 2026 | Optional Mastery Phase**

A Python implementation of both recognition paths described in the
Project 4 brief — **OCR** and **Object Detection** — built on pre-trained
models (Transfer Learning), full image pre-processing, and an
80%-confidence "Gatekeeper" filter.

---

## 📌 What This Project Implements

| Spec Requirement (from the brief) | Status | Where |
|---|---|---|
| Use a pre-trained model or simple library | ✅ | `pytesseract` (Tesseract OCR) + `MobileNet-SSD` (cv2.dnn) |
| Perform recognition on sample input | ✅ | `sample_images/sample_text.jpg` + your own photos |
| Display the output clearly | ✅ | Annotated images in `output/`, CLI printout, JSON mode |
| Using AI libraries, understanding model outputs | ✅ | Confidence scores, bounding boxes, PSM tuning |

### The 4 Gatekeeper Rules (validated automatically)
| # | Rule | Implementation |
|---|---|---|
| 1 | **Library Integration** | Clean `pytesseract` / `cv2.dnn` imports (`validate_gatekeeper.py`) |
| 2 | **Pre-Processing Integrity** | Grayscale → Gaussian Blur → Deskew → Otsu Adaptive Thresholding (`preprocessing.py`) |
| 3 | **Accuracy Benchmarking** | Minimum validated confidence ≥ 80% enforced everywhere |
| 4 | **Visual Confirmation** | Annotated bounding-box / OCR images saved to `output/` |

Run `python3 validate_gatekeeper.py` to check all four automatically.

---

## 🧭 Choose Your Execution Path

| | Path 1: OCR | Path 2: Object Detection |
|---|---|---|
| **Objective** | Extract machine-readable text strings | Identify & locate physical entities |
| **Library** | `pytesseract` (Google Tesseract) | `cv2.dnn` + MobileNet-SSD |
| **Pre-processing** | Grayscale → blur → deskew → adaptive threshold | 4D blob construction (`blobFromImage`) |
| **Output** | Formatted text + per-word confidence | `(x, y, w, h)` bounding boxes + confidence |

---

## 📁 Project Structure

```
DecodeLabs_AI_Project4_VisionRecognition/
├── main.py                    # Unified CLI — choose ocr / detect
├── app.py                      # Streamlit web app (visual interface, 4 tabs)
├── preprocessing.py            # Grayscale, blur, deskew, Otsu thresholding
├── ocr_engine.py                # Path 1: pytesseract OCR pipeline
├── object_detector.py           # Path 2: MobileNet-SSD object detection
├── download_models.py           # Fetches MobileNet-SSD weights (already bundled)
├── validate_gatekeeper.py        # Automated check of all 4 Gatekeeper Rules
├── MODEL_SETUP.md                # Manual model download instructions
├── requirements.txt
├── sample_images/
│   └── sample_text.jpg           # Synthetic test image (rotated + noisy text)
├── models/
│   ├── MobileNetSSD_deploy.prototxt
│   └── MobileNetSSD_deploy.caffemodel
├── output/                        # Annotated results land here
└── tests/
    └── test_pipeline.py            # 10 pytest cases
```

---

## 🖥️ Web App (recommended — visual interface, no terminal-reading needed)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` with 4 tabs:
- **📝 Path 1: OCR** — upload an image, pick a PSM mode, see recognized text + confidence overlay
- **📦 Path 2: Object Detection** — upload a photo, adjust the confidence slider, see bounding boxes live
- **🔬 Pre-Processing Pipeline** — visualizes all 4 preprocessing stages side-by-side
- **✅ Gatekeeper Validation** — one-click check of all 4 milestone rules with pass/fail indicators

---

## 🚀 Getting Started (CLI)

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Install the Tesseract OCR engine (system-level, not pip)
`pytesseract` is just a Python wrapper — it needs the actual Tesseract
binary installed separately:

| OS | Command |
|---|---|
| Windows | Download installer: https://github.com/UB-Mannheim/tesseract/wiki |
| macOS | `brew install tesseract` |
| Linux | `sudo apt install tesseract-ocr` |

**Windows note:** if `pytesseract` can't find the binary, set it explicitly:
```powershell
$env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
```
(or edit the path directly at the top of `ocr_engine.py`)

### 3. Run OCR on the included sample
```bash
python3 main.py ocr --image sample_images/sample_text.jpg
```
```
Recognized text:
  DecodeLabs Project 4 Optical Character Recognition Test Batch 2026 - Industrial Training Kit

Average confidence    : 93.77%
80% Gatekeeper passed  : True
Annotated output       : output/ocr_annotated.jpg
```

### 4. Run Object Detection on your own photo
```bash
python3 main.py detect --image path/to/your_photo.jpg
```
(Model files are already bundled — no extra download needed. Works best
on photos containing people, cars, dogs, or other PASCAL VOC classes —
see `MODEL_SETUP.md` for the full 20-class list.)

### 5. Validate against all 4 Gatekeeper Rules
```bash
python3 validate_gatekeeper.py --image sample_images/sample_text.jpg
```

### 6. Run the test suite
```bash
pytest tests/ -v
```

---

## 🔬 Tuning OCR for Different Layouts (PSM modes)

| Mode | Use case |
|---|---|
| `--psm 3` | Fully automatic — default, varied layouts |
| `--psm 6` | Single uniform block of text (book pages) |
| `--psm 7` | Single text line (number plates, headers) |
| `--psm 11` | Sparse, scattered text (invoices) |

```bash
python3 main.py ocr --image your_invoice.jpg --psm 11
```

---

## 🧠 Key Concepts Demonstrated

- **The IPO Model applied to vision:** an image is a 3D array
  (Height × Width × 3 color channels), each pixel 0–255.
- **Transfer Learning:** both paths reuse pre-trained models (Tesseract's
  LSTM engine, MobileNet trained on ImageNet/PASCAL VOC) instead of
  training from scratch — "why train an AI from scratch when you can
  download a degree?"
- **Otsu's adaptive thresholding:** dynamically computes the optimal
  black/white cutoff per image, rather than a hardcoded value.
- **Softmax / confidence scores:** the model never "knows" an answer —
  it outputs a probability distribution, and the 80% gate filters out
  low-confidence guesses to prevent "confident hallucinations."
- **Coordinate scaling:** the network outputs normalized (0–1)
  coordinates; these are multiplied by the real image's pixel
  width/height to draw actual bounding boxes.

---

## ⚠️ Limitations (by design, per the "Basic" scope of this milestone)

- Object detection only recognizes 20 PASCAL VOC classes (no custom
  object training).
- OCR accuracy depends heavily on image quality — very low-light or
  heavily skewed (>45°) images may still fail deskewing.
- This is a **basic/optional milestone** — it intentionally does not
  cover model fine-tuning, custom dataset training, or GPU acceleration.

---

## 📞 Contact — DecodeLabs

📞 +91 89330 06408
✉️ decodelabs.tech@gmail.com
🌐 www.decodelabs.tech
📍 Greater Lucknow, India
