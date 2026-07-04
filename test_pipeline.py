"""
DecodeLabs - Project 4: Image/Text Recognition (Basic)
=======================================================
tests/test_pipeline.py — pytest suite for preprocessing + OCR engine.

Run: pytest tests/
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import pytest

from preprocessing import to_grayscale, denoise, deskew, adaptive_threshold, full_pipeline
from ocr_engine import OCREngine, run_ocr

SAMPLE_IMAGE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sample_images", "sample_text.jpg",
)


@pytest.fixture
def sample_image():
    img = cv2.imread(SAMPLE_IMAGE)
    assert img is not None, f"Sample image not found at {SAMPLE_IMAGE}"
    return img


def test_grayscale_reduces_to_single_channel(sample_image):
    gray = to_grayscale(sample_image)
    assert len(gray.shape) == 2


def test_grayscale_idempotent(sample_image):
    gray = to_grayscale(sample_image)
    gray_again = to_grayscale(gray)
    assert np.array_equal(gray, gray_again)


def test_denoise_preserves_shape(sample_image):
    gray = to_grayscale(sample_image)
    blurred = denoise(gray)
    assert blurred.shape == gray.shape


def test_deskew_preserves_shape(sample_image):
    gray = to_grayscale(sample_image)
    blurred = denoise(gray)
    deskewed = deskew(blurred)
    assert deskewed.shape == blurred.shape


def test_adaptive_threshold_is_binary(sample_image):
    gray = to_grayscale(sample_image)
    binary, t_value = adaptive_threshold(gray)
    unique_values = set(np.unique(binary).tolist())
    assert unique_values.issubset({0, 255})
    assert 0 <= t_value <= 255


def test_full_pipeline_returns_all_stages(sample_image):
    stages = full_pipeline(sample_image)
    for key in ("original", "grayscale", "blurred", "deskewed", "binary", "otsu_threshold"):
        assert key in stages


def test_ocr_engine_recognizes_known_text():
    engine = OCREngine(psm=3)
    result = engine.recognize(SAMPLE_IMAGE)
    # The sample image contains "DecodeLabs" and "2026" — check they're recognized
    assert "decodelabs" in result["recognized_text"].lower()
    assert "2026" in result["recognized_text"]


def test_ocr_confidence_threshold_flagging():
    engine = OCREngine(psm=3)
    result = engine.recognize(SAMPLE_IMAGE)
    for word in result["words"]:
        expected_flag = word["confidence"] >= 80.0
        assert word["passed_threshold"] == expected_flag


def test_run_ocr_saves_output_files(tmp_path):
    output_dir = str(tmp_path)
    result = run_ocr(SAMPLE_IMAGE, output_dir=output_dir)
    assert os.path.exists(result["annotated_output_path"])
    assert os.path.exists(result["binary_output_path"])


def test_ocr_meets_gatekeeper_standard_on_clean_sample():
    engine = OCREngine(psm=3)
    result = engine.recognize(SAMPLE_IMAGE)
    assert result["average_confidence"] >= 80.0
    assert result["meets_gatekeeper_standard"] is True
