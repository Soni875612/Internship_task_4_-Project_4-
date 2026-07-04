# 👁️ AI Vision Recognition — OCR & Object Detection

**"Building the Machine's Optic Nerve"** — DecodeLabs Industrial Training Kit · Project 4 · Batch 2026

A Python computer-vision system that implements **two recognition paths** using pre-trained models (Transfer Learning) — **Optical Character Recognition (OCR)** and **Object Detection** — wrapped in a full image pre-processing pipeline and an **80% "Gatekeeper" confidence filter** that rejects low-confidence results automatically.

---

## 📌 What This Project Does

| Path | Task | Technology |
|---|---|---|
| **Path 1 — OCR** | Extracts machine-readable text from an image | `pytesseract` (Tesseract OCR engine) |
| **Path 2 — Object Detection** | Detects & locates 20 object classes (person, car, dog, chair, etc.) | `MobileNet-SSD` via `cv2.dnn` |

Both paths enforce an **80% Gatekeeper Rule** — any detection or recognized word below 80% confidence is automatically rejected, so the system never silently reports low-quality results.

---

## ✅ The 4 Gatekeeper Rules

| # | Rule | Where it's implemented |
|---|---|---|
| 1 | **Library Integration** | Clean `pytesseract` / `cv2.dnn` imports |
| 2 | **Pre-Processing Integrity** | Grayscale → Gaussian Blur → Deskew → Otsu Adaptive Thresholding (`preprocessing.py`) |
| 3 | **Accuracy Benchmarking** | Minimum validated confidence ≥ 80% enforced everywhere |
| 4 | **Visual Confirmation** | Annotated bounding-box / OCR images saved to `output/` |

Run `python3 validate_gatekeeper.py` to check all four automatically, or use the **"Gatekeeper Validation"** tab in the web app.

---

## 🖼️ Screenshots


| OCR Tab | Object Detection Tab |

<img width="1918" height="911" alt="image" src="https://github.com/user-attachments/assets/9c4b47bb-1764-4e3c-bde0-a6522cc6e5ae" />

<img width="1891" height="697" alt="image" src="https://github.com/user-attachments/assets/b7dd07c8-65e0-41a4-8d7c-cca28416ca0a" />


https://github.com/user-attachments/assets/9aa9392e-48be-4881-ba42-df7b1f70ed63

| Pre-Processing Pipeline | Gatekeeper Validation |

<img width="1828" height="748" alt="image" src="https://github.com/user-attachments/assets/d8c56ec2-b851-4877-8293-ace6ac9b25c3" />

<img width="1860" height="648" alt="image" src="https://github.com/user-attachments/assets/77f0f362-cb5d-4a14-b1ca-73876dfd46c8" />

---

## 🌐 Web App — 4 Tabs

Launch with `streamlit run app.py`:

1. **📝 Path 1: OCR** — Upload an image (or use the bundled sample), choose a Tesseract Page Segmentation Mode, and extract text with per-word confidence scores
2. **📦 Path 2: Object Detection** — Upload a photo and detect objects with an adjustable confidence slider
3. **🔬 Pre-Processing Pipeline** — Visualizes the exact 4-step pipeline: Grayscale → Gaussian Blur → Deskew → Otsu Binary Threshold
4. **✅ Gatekeeper Validation** — One-click automated check of all 4 Gatekeeper Rules

---

## 📁 Project Structure

```
DecodeLabs_AI_Project4_VisionRecognition/
├── app.py                    # Streamlit web app (4 tabs)
├── main.py                   # Unified CLI — choose ocr / detect
├── preprocessing.py          # Grayscale, blur, deskew, Otsu thresholding
├── ocr_engine.py             # Path 1: pytesseract OCR pipeline
├── object_detector.py        # Path 2: MobileNet-SSD object detection
├── download_models.py        # Fetches MobileNet-SSD weights
├── validate_gatekeeper.py    # Automated check of all 4 Gatekeeper Rules
├── MODEL_SETUP.md            # Manual model download instructions
├── requirements.txt
├── sample_images/
│   └── sample_text.jpg       # Synthetic test image (rotated + noisy text)
├── models/
│   ├── MobileNetSSD_deploy.prototxt
│   └── MobileNetSSD_deploy.caffemodel
├── output/                   # Annotated results saved here
└── tests/
    └── test_pipeline.py
```

---

## ⚙️ Installation & Usage

**1. Clone the repository**
```bash
git clone https://github.com/Soni875612/<repo-name>.git
cd DecodeLabs_AI_Project4_VisionRecognition
```

**2. Install Python dependencies**
```bash
pip install -r requirements.txt
```
> ⚠️ Install **only** `opencv-python-headless` (already pinned in `requirements.txt`). Do not also install `opencv-python`, `opencv-contrib-python`, or `opencv-contrib-python-headless` in the same environment — having more than one causes a broken `cv2.dnn` module.

**3. Install Tesseract OCR (system-level dependency, not via pip)**
- **Windows:** [UB-Mannheim Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS:** `brew install tesseract`
- **Linux:** `sudo apt install tesseract-ocr`

**4. Run the web app**
```bash
python -m streamlit run app.py
```

**5. Or use the CLI**
```bash
python3 main.py ocr --image sample_images/sample_text.jpg --psm 3
python3 main.py detect --image path/to/photo.jpg --confidence 0.80
```

**6. Validate the Gatekeeper Rules**
```bash
python3 validate_gatekeeper.py
```

---

## 🧠 How It Works

**Path 1 — OCR Pipeline:**
```
Image → Grayscale → Gaussian Blur → Deskew → Otsu Adaptive Threshold → Tesseract OCR → 80% Confidence Filter → Text Output
```

**Path 2 — Object Detection Pipeline:**
```
Image → blobFromImage (300×300, mean subtraction) → MobileNet-SSD Forward Pass → Coordinate Scaling → 80% Confidence Gate → Bounding Boxes
```

The Object Detector recognizes 20 PASCAL VOC classes: person, car, dog, cat, bus, bicycle, chair, and more. Objects outside this list (e.g. a laptop or phone) will correctly report zero detections — that's expected model behavior, not a bug.

---

## 🛠️ Tech Stack

`Python` · `OpenCV (cv2.dnn)` · `Tesseract OCR (pytesseract)` · `MobileNet-SSD` · `NumPy` · `Pillow` · `Streamlit` · `pytest`

---

## 🎯 Key Learnings

- Using pre-trained models via Transfer Learning instead of training from scratch
- Building a systematic image pre-processing pipeline (grayscale, blur, deskew, adaptive thresholding)
- Understanding and tuning confidence thresholds to filter unreliable AI output
- Working with `cv2.dnn` for Caffe-based deep learning inference
- Designing automated "Gatekeeper" validation checks for an ML pipeline
- Shipping a multi-path computer vision system through both CLI and web interfaces

---

## 👩‍💻 Author

**Soni**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/soni-devi-131a9938b/)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/Soni875612)
[![LeetCode](https://img.shields.io/badge/LeetCode-Profile-orange?logo=leetcode)](https://leetcode.com/u/soni_2007/)

- 🔗 **LinkedIn:** [linkedin.com/in/soni-devi-131a9938b](https://www.linkedin.com/in/soni-devi-131a9938b/)
- 💻 **GitHub:** [github.com/Soni875612](https://github.com/Soni875612)
- 🧩 **LeetCode:** [leetcode.com/u/soni_2007](https://leetcode.com/u/soni_2007/)

---

## 📄 License

Developed as part of the **DecodeLabs Industrial Training Kit**. Free to use for learning and reference purposes.
