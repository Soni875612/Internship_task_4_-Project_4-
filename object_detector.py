"""
DecodeLabs - Project 4: Image/Text Recognition (Basic)
=======================================================
object_detector.py — "Path 2: Object Detection with MobileNet-SSD"

Backbone: MobileNet-SSD (Single Shot Detector)
    - Depthwise separable convolutions filter input channels separately
    - Optimized for high-speed, real-time inference on edge devices

Pipeline (per the deck):
    1. Blob Construction : cv2.dnn.blobFromImage — mean subtraction +
                            scale to the network's 300x300 input dims
    2. Forward Pass       : single shot through the network
    3. Coordinate Scaling : normalized (0-1) output coords are multiplied
                            by the original image's actual pixel W/H
    4. The 80% Confidence Gate (Gatekeeper Rule #3):
           if confidence >= 0.80: draw_box_and_label()
           else:                  drop_detection()

Model files (not bundled — see MODEL_SETUP.md):
    - MobileNetSSD_deploy.prototxt   (network architecture)
    - MobileNetSSD_deploy.caffemodel (pre-trained weights, ~23 MB)
Both are the standard chuanqi305/MobileNet-SSD release, trained on the
20-class PASCAL VOC dataset (person, car, dog, chair, etc.).
"""

from __future__ import annotations

import os
import cv2
import numpy as np

CONFIDENCE_THRESHOLD = 0.80  # Gatekeeper Rule #3: the "80% Gate"

# PASCAL VOC 20 classes + background (index 0), as used by this model release
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]


class ObjectDetector:
    def __init__(self, prototxt_path: str, model_path: str):
        if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
            raise FileNotFoundError(
                "Model files not found. Run `python3 download_models.py` first, "
                "or see MODEL_SETUP.md for manual download instructions.\n"
                f"Expected: {prototxt_path}\n"
                f"Expected: {model_path}"
            )
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    def detect(self, image_path: str, confidence_threshold: float = CONFIDENCE_THRESHOLD) -> dict:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at: {image_path}")

        (h, w) = image.shape[:2]

        # Step 1: Blob Construction — mean subtraction + scale to 300x300
        blob = cv2.dnn.blobFromImage(
            image, scalefactor=0.007843, size=(300, 300),
            mean=(127.5, 127.5, 127.5), swapRB=False, crop=False,
        )

        # Step 2: Forward pass through the network
        self.net.setInput(blob)
        detections = self.net.forward()

        accepted, rejected = [], []

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            class_id = int(detections[0, 0, i, 1])
            label = VOC_CLASSES[class_id] if class_id < len(VOC_CLASSES) else f"class_{class_id}"

            # Step 3: Coordinate Scaling — normalized -> actual pixel coords
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype(int)

            record = {
                "label": label,
                "confidence": round(confidence * 100, 2),
                "bbox": (int(start_x), int(start_y), int(end_x), int(end_y)),
            }

            # Step 4: The 80% Confidence Gate
            if confidence >= confidence_threshold:
                accepted.append(record)
            else:
                rejected.append(record)

        return {
            "image_path": image_path,
            "image_shape": (h, w),
            "accepted_detections": accepted,
            "rejected_detections": rejected,
            "meets_gatekeeper_standard": len(accepted) > 0,
            "original_image": image,
        }

    @staticmethod
    def annotate(image: np.ndarray, detections: list[dict], color=(0, 200, 0)) -> np.ndarray:
        """Draws bounding boxes + label + confidence. Satisfies Gatekeeper
        Rule #4: Visual Confirmation."""
        annotated = image.copy()
        for det in detections:
            (x1, y1, x2, y2) = det["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{det['label']}: {det['confidence']:.1f}%"
            y_label = max(y1 - 8, 12)
            cv2.putText(
                annotated, label, (x1, y_label),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )
        return annotated


def run_detection(
    image_path: str,
    prototxt_path: str = "models/MobileNetSSD_deploy.prototxt",
    model_path: str = "models/MobileNetSSD_deploy.caffemodel",
    output_dir: str = "output",
) -> dict:
    """Convenience wrapper: detect + save annotated output image."""
    detector = ObjectDetector(prototxt_path, model_path)
    result = detector.detect(image_path)

    os.makedirs(output_dir, exist_ok=True)
    annotated = detector.annotate(result["original_image"], result["accepted_detections"])
    out_path = os.path.join(output_dir, "detection_annotated.jpg")
    cv2.imwrite(out_path, annotated)
    result["annotated_output_path"] = out_path

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 object_detector.py <image_path>")
        sys.exit(1)

    res = run_detection(sys.argv[1])
    print(f"\nAccepted detections (>= 80% confidence): {len(res['accepted_detections'])}")
    for d in res["accepted_detections"]:
        print(f"  - {d['label']}: {d['confidence']}% at {d['bbox']}")
    print(f"\nRejected (< 80%): {len(res['rejected_detections'])}")
    print(f"Annotated image saved to: {res['annotated_output_path']}")
