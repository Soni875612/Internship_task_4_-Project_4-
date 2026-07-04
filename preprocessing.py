"""
DecodeLabs - Project 4: Image/Text Recognition (Basic)
=======================================================
preprocessing.py — "The Logic Skeleton: Systematic Image Pre-Processing"

Implements the exact 3-step pipeline from the training deck:
    Step 1: Grayscale Conversion  -> collapse 3D RGB matrix to 1D intensity
    Step 2: Gaussian Blur         -> remove micro-noise / artifact noise
    Step 3: Deskewing             -> snap tilted text to horizontal baseline

Followed by Adaptive Thresholding (Otsu's Method):
    IF pixel_intensity >= T THEN pixel = 255 (white)
    IF pixel_intensity <  T THEN pixel = 0   (black)

Every function returns a numpy array (OpenCV BGR/gray convention) so steps
can be chained or inspected individually.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Step 1: Collapse the 3-channel (B, G, R) matrix into a single
    intensity channel. Removes color data that is irrelevant for reading
    text or detecting shape/edge structure."""
    if len(image.shape) == 2:
        return image  # already grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray, kernel_size: tuple[int, int] = (5, 5)) -> np.ndarray:
    """Step 2: Gaussian Blur — smooths micro-imperfections and sensor
    noise before edge/character detection, without destroying large
    structural edges."""
    return cv2.GaussianBlur(gray, kernel_size, 0)


def deskew(gray: np.ndarray) -> np.ndarray:
    """Step 3: Deskewing — detects the dominant rotation angle of text
    lines via minAreaRect over thresholded foreground pixels, then
    rotates the image back to a horizontal baseline.
    """
    # Invert + threshold to isolate foreground (text/objects) as white
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))

    if coords.shape[0] < 10:
        return gray  # not enough foreground pixels to estimate skew reliably

    angle = cv2.minAreaRect(coords)[-1]

    # cv2.minAreaRect returns angles in [-90, 0); normalize to a small
    # correction angle rather than a full 90-degree flip.
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def adaptive_threshold(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Otsu's adaptive thresholding — forces every pixel to a binary
    decision (0 or 255), automatically computing the optimal cutoff
    intensity T rather than a hardcoded value (the deck's example used
    T=88, but Otsu computes this dynamically per-image for robustness).

    Returns (binary_image, computed_threshold_value).
    """
    t_value, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, t_value


def full_pipeline(image: np.ndarray) -> dict:
    """Runs the complete deck-specified pipeline and returns every
    intermediate stage — useful for visual confirmation / debugging
    (Gatekeeper Rule #2: Pre-Processing Integrity)."""
    gray = to_grayscale(image)
    blurred = denoise(gray)
    deskewed = deskew(blurred)
    binary, t_value = adaptive_threshold(deskewed)

    return {
        "original": image,
        "grayscale": gray,
        "blurred": blurred,
        "deskewed": deskewed,
        "binary": binary,
        "otsu_threshold": t_value,
    }
