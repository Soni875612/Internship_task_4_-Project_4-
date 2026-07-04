"""
DecodeLabs - Project 4: Image/Text Recognition (Basic)
=======================================================
ocr_engine.py — "Path 1: Optical Character Recognition (OCR)"

Engine: pytesseract (Python wrapper for Google's Tesseract — a
convolutional + bi-directional LSTM pipeline for reading text sequences).

Implements:
    - Full preprocessing pipeline (preprocessing.py) before OCR
    - Configurable Page Segmentation Mode (PSM):
          --psm 3  : Fully automatic (default, varied layouts)
          --psm 6  : Single uniform block of text (book pages)
          --psm 7  : Single text line (number plates / headers)
          --psm 11 : Sparse, scattered text (invoices)
    - Word-level confidence scores
    - The Gatekeeper Rule's 80% Confidence Filter — low-confidence words
      are flagged rather than silently trusted (avoids "confident
      hallucinations")
"""

from __future__ import annotations

import os
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from preprocessing import full_pipeline

CONFIDENCE_THRESHOLD = 80.0  # Gatekeeper Rule #3: 80% minimum standard

# --- Windows setup helper -------------------------------------------------
# pytesseract needs to find the Tesseract OCR *binary* (not just the pip
# package). On Windows this is usually NOT on PATH after installing via the
# UB-Mannheim installer. Set the path explicitly via an environment
# variable if needed:
#
#   set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe   (cmd)
#   $env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe" (PowerShell)
#
# Or just uncomment and edit the line below.
_tess_cmd = os.environ.get("TESSERACT_CMD")
if _tess_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tess_cmd
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# ---------------------------------------------------------------------------


class OCREngine:
    """OCR pipeline: preprocess -> recognize -> filter by confidence."""

    def __init__(self, psm: int = 3, lang: str = "eng"):
        self.psm = psm
        self.lang = lang

    def _tess_config(self) -> str:
        return f"--oem 3 --psm {self.psm}"

    def recognize(self, image_path: str) -> dict:
        """Runs the full pipeline on an image file and returns structured
        OCR results including per-word confidence."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at: {image_path}")

        stages = full_pipeline(image)
        processed = stages["binary"]

        raw_data = pytesseract.image_to_data(
            processed, lang=self.lang, config=self._tess_config(), output_type=Output.DICT
        )

        words = []
        for i, text in enumerate(raw_data["text"]):
            text = text.strip()
            conf = float(raw_data["conf"][i])
            if text and conf >= 0:  # tesseract uses -1 for non-text regions
                words.append(
                    {
                        "text": text,
                        "confidence": conf,
                        "passed_threshold": conf >= CONFIDENCE_THRESHOLD,
                        "bbox": (
                            raw_data["left"][i],
                            raw_data["top"][i],
                            raw_data["width"][i],
                            raw_data["height"][i],
                        ),
                    }
                )

        accepted = [w for w in words if w["passed_threshold"]]
        rejected = [w for w in words if not w["passed_threshold"]]

        full_text = " ".join(w["text"] for w in accepted)
        avg_conf = round(sum(w["confidence"] for w in words) / len(words), 2) if words else 0.0

        return {
            "image_path": image_path,
            "psm_used": self.psm,
            "otsu_threshold": stages["otsu_threshold"],
            "words": words,
            "accepted_words": accepted,
            "rejected_words": rejected,
            "recognized_text": full_text,
            "average_confidence": avg_conf,
            "meets_gatekeeper_standard": avg_conf >= CONFIDENCE_THRESHOLD,
            "stages": stages,  # for visual debugging / saving intermediate images
        }

    @staticmethod
    def annotate(image: np.ndarray, words: list[dict]) -> np.ndarray:
        """Draws bounding boxes + confidence labels — green for words that
        passed the 80% gate, red for rejected words. Satisfies Gatekeeper
        Rule #4: Visual Confirmation."""
        annotated = image.copy()
        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        for w in words:
            x, y, wd, ht = w["bbox"]
            color = (0, 200, 0) if w["passed_threshold"] else (0, 0, 220)
            cv2.rectangle(annotated, (x, y), (x + wd, y + ht), color, 2)
            label = f"{w['text']} ({w['confidence']:.0f}%)"
            cv2.putText(
                annotated, label, (x, max(y - 6, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )
        return annotated


def run_ocr(image_path: str, psm: int = 3, output_dir: str = "output") -> dict:
    """Convenience wrapper: recognize + save annotated output image."""
    engine = OCREngine(psm=psm)
    result = engine.recognize(image_path)

    os.makedirs(output_dir, exist_ok=True)
    annotated = engine.annotate(result["stages"]["original"], result["words"])
    out_path = os.path.join(output_dir, "ocr_annotated.jpg")
    cv2.imwrite(out_path, annotated)
    result["annotated_output_path"] = out_path

    # Also save the binary (preprocessed) image for pipeline transparency
    binary_path = os.path.join(output_dir, "ocr_preprocessed_binary.jpg")
    cv2.imwrite(binary_path, result["stages"]["binary"])
    result["binary_output_path"] = binary_path

    return result


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python3 ocr_engine.py <image_path> [psm]")
        sys.exit(1)

    img_path = sys.argv[1]
    psm_mode = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    res = run_ocr(img_path, psm=psm_mode)
    print(f"\nRecognized Text:\n{res['recognized_text']}\n")
    print(f"Average Confidence : {res['average_confidence']}%")
    print(f"Meets 80% Standard : {res['meets_gatekeeper_standard']}")
    print(f"Accepted words     : {len(res['accepted_words'])}")
    print(f"Rejected words     : {len(res['rejected_words'])}")
    print(f"Annotated image    : {res['annotated_output_path']}")
