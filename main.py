"""
DecodeLabs - Project 4: Image/Text Recognition (Basic)
=======================================================
main.py — Unified CLI: "The Perception Matrix: Choose Your Execution Path"

Usage:
    python3 main.py ocr    --image sample_images/sample_text.jpg [--psm 3]
    python3 main.py detect --image path/to/photo.jpg [--confidence 0.80]
"""

import argparse
import json
import sys

from ocr_engine import run_ocr
from object_detector import run_detection


def cmd_ocr(args):
    result = run_ocr(args.image, psm=args.psm, output_dir=args.output)

    print("\n" + "=" * 60)
    print("PATH 1: OCR (pytesseract)")
    print("=" * 60)
    print(f"Image                : {args.image}")
    print(f"PSM mode              : {result['psm_used']}")
    print(f"Otsu threshold (T)     : {result['otsu_threshold']:.1f}")
    print(f"\nRecognized text:\n  {result['recognized_text']}\n")
    print(f"Average confidence    : {result['average_confidence']}%")
    print(f"80% Gatekeeper passed  : {result['meets_gatekeeper_standard']}")
    print(f"Accepted / Rejected    : {len(result['accepted_words'])} / {len(result['rejected_words'])}")
    print(f"Annotated output       : {result['annotated_output_path']}")
    print(f"Preprocessed (binary)  : {result['binary_output_path']}")

    if args.json:
        summary = {k: v for k, v in result.items() if k != "stages"}
        print("\n--- JSON summary ---")
        print(json.dumps(summary, indent=2, default=str))


def cmd_detect(args):
    result = run_detection(
        args.image,
        prototxt_path=args.prototxt,
        model_path=args.model,
        output_dir=args.output,
    )

    print("\n" + "=" * 60)
    print("PATH 2: Object Detection (MobileNet-SSD)")
    print("=" * 60)
    print(f"Image                     : {args.image}")
    print(f"Confidence threshold       : {CONFIDENCE_DISPLAY(args)}")
    print(f"\nAccepted detections (>= threshold): {len(result['accepted_detections'])}")
    for d in result["accepted_detections"]:
        print(f"  - {d['label']:<15} {d['confidence']:>6.2f}%  bbox={d['bbox']}")
    print(f"\nRejected detections (< threshold) : {len(result['rejected_detections'])}")
    print(f"80% Gatekeeper passed      : {result['meets_gatekeeper_standard']}")
    print(f"Annotated output           : {result['annotated_output_path']}")

    if args.json:
        summary = {k: v for k, v in result.items() if k != "original_image"}
        print("\n--- JSON summary ---")
        print(json.dumps(summary, indent=2, default=str))


def CONFIDENCE_DISPLAY(args):
    return "80% (default)" if not hasattr(args, "confidence") else f"{args.confidence * 100:.0f}%"


def build_parser():
    parser = argparse.ArgumentParser(
        description="DecodeLabs Project 4 — Image/Text Recognition (Basic)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ocr_p = sub.add_parser("ocr", help="Path 1: Optical Character Recognition")
    ocr_p.add_argument("--image", required=True, help="Path to input image")
    ocr_p.add_argument("--psm", type=int, default=3, help="Tesseract Page Segmentation Mode")
    ocr_p.add_argument("--output", default="output", help="Output directory")
    ocr_p.add_argument("--json", action="store_true", help="Also print JSON summary")
    ocr_p.set_defaults(func=cmd_ocr)

    det_p = sub.add_parser("detect", help="Path 2: Object Detection")
    det_p.add_argument("--image", required=True, help="Path to input image")
    det_p.add_argument("--prototxt", default="models/MobileNetSSD_deploy.prototxt")
    det_p.add_argument("--model", default="models/MobileNetSSD_deploy.caffemodel")
    det_p.add_argument("--confidence", type=float, default=0.80)
    det_p.add_argument("--output", default="output", help="Output directory")
    det_p.add_argument("--json", action="store_true", help="Also print JSON summary")
    det_p.set_defaults(func=cmd_detect)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    parsed_args = parser.parse_args()
    try:
        parsed_args.func(parsed_args)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
