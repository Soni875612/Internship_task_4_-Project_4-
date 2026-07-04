"""
DecodeLabs - Project 4: Image/Text Recognition (Basic)
=======================================================
validate_gatekeeper.py — "The Gatekeeper Rule: Milestone Validation"

Automatically checks the four uncompromising technical validations
required to complete Project 4:

    1. Library Integration      -> pytesseract / cv2.dnn import without error
    2. Pre-Processing Integrity -> Grayscale + Adaptive Thresholding executed
    3. Accuracy Benchmarking    -> minimum validated confidence >= 80%
    4. Visual Confirmation      -> annotated output image generated on disk

Run:
    python3 validate_gatekeeper.py --image sample_images/sample_text.jpg
"""

import argparse
import os
import sys

CHECKMARK = "\u2705"
CROSS = "\u274c"


def check_1_library_integration() -> tuple[bool, str]:
    try:
        import cv2  # noqa: F401
        import pytesseract  # noqa: F401
        return True, "cv2.dnn and pytesseract both import cleanly."
    except ImportError as e:
        return False, f"Import failed: {e}"


def check_2_preprocessing(image_path: str) -> tuple[bool, dict]:
    import cv2
    from preprocessing import full_pipeline

    image = cv2.imread(image_path)
    if image is None:
        return False, {"error": f"Could not read {image_path}"}

    stages = full_pipeline(image)
    ok = (
        stages["grayscale"] is not None
        and stages["binary"] is not None
        and stages["otsu_threshold"] > 0
    )
    return ok, {"otsu_threshold": stages["otsu_threshold"]}


def check_3_accuracy_benchmark(image_path: str) -> tuple[bool, dict]:
    from ocr_engine import run_ocr

    result = run_ocr(image_path, output_dir="output")
    return result["meets_gatekeeper_standard"], {
        "average_confidence": result["average_confidence"],
        "accepted_words": len(result["accepted_words"]),
        "rejected_words": len(result["rejected_words"]),
    }


def check_4_visual_confirmation() -> tuple[bool, dict]:
    expected = ["output/ocr_annotated.jpg", "output/ocr_preprocessed_binary.jpg"]
    existing = [p for p in expected if os.path.exists(p)]
    return len(existing) == len(expected), {"files": existing}


def main():
    parser = argparse.ArgumentParser(description="Validate Project 4 Gatekeeper Rules")
    parser.add_argument("--image", default="sample_images/sample_text.jpg")
    args = parser.parse_args()

    print("=" * 65)
    print("PROJECT 4 — GATEKEEPER RULE VALIDATION")
    print("=" * 65)

    results = []

    ok, msg = check_1_library_integration()
    results.append(ok)
    print(f"\n1. Library Integration        {CHECKMARK if ok else CROSS}")
    print(f"   {msg}")

    ok, detail = check_2_preprocessing(args.image)
    results.append(ok)
    print(f"\n2. Pre-Processing Integrity   {CHECKMARK if ok else CROSS}")
    print(f"   Otsu threshold computed: {detail.get('otsu_threshold')}")

    ok, detail = check_3_accuracy_benchmark(args.image)
    results.append(ok)
    print(f"\n3. Accuracy Benchmarking      {CHECKMARK if ok else CROSS}")
    print(f"   Average confidence: {detail.get('average_confidence')}% "
          f"(need >= 80%) | accepted={detail.get('accepted_words')} "
          f"rejected={detail.get('rejected_words')}")

    ok, detail = check_4_visual_confirmation()
    results.append(ok)
    print(f"\n4. Visual Confirmation        {CHECKMARK if ok else CROSS}")
    print(f"   Output files found: {detail.get('files')}")

    print("\n" + "=" * 65)
    if all(results):
        print(f"{CHECKMARK} ALL 4 GATEKEEPER RULES PASSED — Project 4 milestone complete.")
    else:
        print(f"{CROSS} {results.count(False)} rule(s) failed. Review output above.")
    print("=" * 65)

    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
