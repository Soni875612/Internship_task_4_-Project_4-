"""
DecodeLabs - Project 4: Image/Text Recognition (Basic)
=======================================================
download_models.py — fetches the pre-trained MobileNet-SSD model files
required for Path 2 (Object Detection).

This is the "Transfer Learning" step from the deck: "Why train an AI from
scratch when you can download a degree?" These weights were already
trained on millions of PASCAL VOC images — we just plug in our task.

Source: chuanqi305/MobileNet-SSD (Caffe implementation), the standard
public release used with OpenCV's cv2.dnn module.
"""

import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

FILES = {
    "MobileNetSSD_deploy.prototxt": (
        "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
    ),
    "MobileNetSSD_deploy.caffemodel": (
        "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel"
    ),
}


def download(url: str, dest: str):
    print(f"Downloading {os.path.basename(dest)} ...")
    urllib.request.urlretrieve(url, dest)
    size_kb = os.path.getsize(dest) / 1024
    print(f"  -> saved to {dest} ({size_kb:.1f} KB)")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for filename, url in FILES.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest):
            print(f"{filename} already exists — skipping.")
            continue
        try:
            download(url, dest)
        except Exception as e:
            print(f"FAILED to download {filename}: {e}")
            print("See MODEL_SETUP.md for manual download instructions.")

    print("\nDone. Model files are in:", MODELS_DIR)


if __name__ == "__main__":
    main()
