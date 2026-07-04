"""
DecodeLabs - Project 4 (Advanced): Image/Text Recognition
Streamlit App — "The Machine's Optic Nerve" — Visual Interface

Run:
    streamlit run app.py
"""

import os
import sys
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import full_pipeline
from ocr_engine import OCREngine, CONFIDENCE_THRESHOLD as OCR_THRESHOLD
from object_detector import ObjectDetector, CONFIDENCE_THRESHOLD as DET_THRESHOLD

st.set_page_config(
    page_title="DecodeLabs | Vision Recognition",
    page_icon="👁️",
    layout="wide",
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
PROTOTXT = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.prototxt")
CAFFEMODEL = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.caffemodel")


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv_img: np.ndarray) -> Image.Image:
    if len(cv_img.shape) == 2:
        return Image.fromarray(cv_img)
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


@st.cache_resource
def load_detector():
    if not (os.path.exists(PROTOTXT) and os.path.exists(CAFFEMODEL)):
        return None
    return ObjectDetector(PROTOTXT, CAFFEMODEL)


# --------------------------------------------------------------------------- #
# HEADER
# --------------------------------------------------------------------------- #
st.title("👁️ AI Vision Recognition — OCR & Object Detection")
st.caption(
    "DecodeLabs · Project 4 (Advanced) · Building the Machine's Optic Nerve · "
    "Pre-trained models, full pre-processing pipeline, 80% Gatekeeper confidence filter"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Path 1: OCR",
    "📦 Path 2: Object Detection",
    "🔬 Pre-Processing Pipeline",
    "✅ Gatekeeper Validation",
])

# --------------------------------------------------------------------------- #
# TAB 1 — OCR
# --------------------------------------------------------------------------- #
with tab1:
    st.subheader("Optical Character Recognition (pytesseract)")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        uploaded_ocr = st.file_uploader(
            "Upload an image with text", type=["jpg", "jpeg", "png"], key="ocr_upload"
        )
        use_sample = st.checkbox("Use bundled sample image instead", value=uploaded_ocr is None)
        psm = st.selectbox(
            "Page Segmentation Mode (PSM)",
            options=[3, 6, 7, 11],
            format_func=lambda x: {
                3: "3 — Fully automatic (default)",
                6: "6 — Single uniform block (book page)",
                7: "7 — Single text line (headers/plates)",
                11: "11 — Sparse scattered text (invoices)",
            }[x],
        )
        run_ocr_btn = st.button("Run OCR", type="primary", key="run_ocr")

    with col_right:
        st.info(
            "💡 **Pipeline:** Grayscale → Gaussian Blur → Deskew → Otsu "
            f"Adaptive Threshold → Tesseract OCR → **{OCR_THRESHOLD:.0f}% confidence filter**"
        )

    if run_ocr_btn:
        if use_sample or uploaded_ocr is None:
            image_path = os.path.join("sample_images", "sample_text.jpg")
            input_image = cv2.imread(image_path)
        else:
            pil_img = Image.open(uploaded_ocr)
            input_image = pil_to_cv2(pil_img)
            image_path = None

        if input_image is None:
            st.error("Could not load the image.")
        else:
            with st.spinner("Running OCR pipeline..."):
                t0 = time.time()
                engine = OCREngine(psm=psm)

                if image_path:
                    result = engine.recognize(image_path)
                else:
                    tmp_path = "output/_tmp_ocr_input.jpg"
                    os.makedirs("output", exist_ok=True)
                    cv2.imwrite(tmp_path, input_image)
                    result = engine.recognize(tmp_path)

                annotated = engine.annotate(result["stages"]["original"], result["words"])
                elapsed = time.time() - t0

            c1, c2 = st.columns(2)
            with c1:
                st.image(cv2_to_pil(result["stages"]["original"]), caption="Original", use_container_width=True)
            with c2:
                st.image(cv2_to_pil(annotated), caption="Annotated (green ≥80%, red <80%)", use_container_width=True)

            st.markdown("### Recognized Text")
            st.code(result["recognized_text"] or "(no text recognized)")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Average Confidence", f"{result['average_confidence']:.1f}%")
            m2.metric("Otsu Threshold", f"{result['otsu_threshold']:.0f}")
            m3.metric("Accepted Words", len(result["accepted_words"]))
            m4.metric("Processing Time", f"{elapsed:.2f}s")

            if result["meets_gatekeeper_standard"]:
                st.success("✅ Passed the 80% Gatekeeper confidence standard.")
            else:
                st.warning("⚠️ Below the 80% Gatekeeper confidence standard — try a clearer image or different PSM mode.")

            with st.expander("Word-level confidence breakdown"):
                for w in result["words"]:
                    icon = "🟢" if w["passed_threshold"] else "🔴"
                    st.write(f"{icon} `{w['text']}` — {w['confidence']:.1f}%")

# --------------------------------------------------------------------------- #
# TAB 2 — OBJECT DETECTION
# --------------------------------------------------------------------------- #
with tab2:
    st.subheader("Object Detection (MobileNet-SSD)")

    detector = load_detector()
    if detector is None:
        st.error(
            "Model files not found in `models/`. Run `python3 download_models.py` "
            "or see MODEL_SETUP.md."
        )
    else:
        col_left, col_right = st.columns([1, 1])
        with col_left:
            uploaded_det = st.file_uploader(
                "Upload a photo (people, cars, dogs, etc.)", type=["jpg", "jpeg", "png"], key="det_upload"
            )
            conf_slider = st.slider(
                "Confidence threshold", min_value=0.10, max_value=0.95,
                value=DET_THRESHOLD, step=0.05,
            )
            run_det_btn = st.button("Run Detection", type="primary", key="run_det")

        with col_right:
            st.info(
                "💡 **Pipeline:** blobFromImage (300×300, mean subtraction) → "
                "MobileNet-SSD forward pass → coordinate scaling → confidence gate.\n\n"
                "Recognizes 20 PASCAL VOC classes: person, car, dog, cat, bus, "
                "bicycle, chair, and more."
            )

        if run_det_btn:
            if uploaded_det is None:
                st.warning("Please upload an image first.")
            else:
                pil_img = Image.open(uploaded_det)
                input_image = pil_to_cv2(pil_img)

                with st.spinner("Running object detection..."):
                    t0 = time.time()
                    tmp_path = "output/_tmp_det_input.jpg"
                    os.makedirs("output", exist_ok=True)
                    cv2.imwrite(tmp_path, input_image)
                    result = detector.detect(tmp_path, confidence_threshold=conf_slider)
                    annotated = detector.annotate(result["original_image"], result["accepted_detections"])
                    elapsed = time.time() - t0

                c1, c2 = st.columns(2)
                with c1:
                    st.image(cv2_to_pil(result["original_image"]), caption="Original", use_container_width=True)
                with c2:
                    st.image(cv2_to_pil(annotated), caption=f"Detections ≥ {conf_slider*100:.0f}%", use_container_width=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("Accepted Detections", len(result["accepted_detections"]))
                m2.metric("Rejected (below threshold)", len(result["rejected_detections"]))
                m3.metric("Processing Time", f"{elapsed:.2f}s")

                if result["accepted_detections"]:
                    st.markdown("### Detected Objects")
                    for d in sorted(result["accepted_detections"], key=lambda x: -x["confidence"]):
                        st.write(f"🟢 **{d['label']}** — {d['confidence']:.1f}% at `{d['bbox']}`")
                else:
                    st.warning(
                        "No objects detected above this confidence threshold. "
                        "Try lowering the slider or using a photo with people/vehicles/animals."
                    )

# --------------------------------------------------------------------------- #
# TAB 3 — PRE-PROCESSING PIPELINE VISUALIZER
# --------------------------------------------------------------------------- #
with tab3:
    st.subheader("Systematic Image Pre-Processing — Step by Step")
    st.write(
        "Visualizes the exact 3-step pipeline: **Grayscale → Gaussian Blur → "
        "Deskew**, followed by **Otsu Adaptive Thresholding**."
    )

    uploaded_pp = st.file_uploader(
        "Upload an image to visualize preprocessing", type=["jpg", "jpeg", "png"], key="pp_upload"
    )
    use_sample_pp = st.checkbox("Use bundled sample image", value=uploaded_pp is None, key="pp_sample")

    if st.button("Run Pipeline", key="run_pp"):
        if use_sample_pp or uploaded_pp is None:
            input_image = cv2.imread(os.path.join("sample_images", "sample_text.jpg"))
        else:
            input_image = pil_to_cv2(Image.open(uploaded_pp))

        stages = full_pipeline(input_image)

        cols = st.columns(4)
        labels = ["1. Grayscale", "2. Gaussian Blur", "3. Deskewed", f"4. Binary (T={stages['otsu_threshold']:.0f})"]
        images = [stages["grayscale"], stages["blurred"], stages["deskewed"], stages["binary"]]

        for col, label, img in zip(cols, labels, images):
            with col:
                st.image(cv2_to_pil(img), caption=label, use_container_width=True)

# --------------------------------------------------------------------------- #
# TAB 4 — GATEKEEPER VALIDATION
# --------------------------------------------------------------------------- #
with tab4:
    st.subheader("The Gatekeeper Rule: Milestone Validation")
    st.write("The four uncompromising technical validations required to complete Project 4.")

    if st.button("Run Full Validation", key="run_validate"):
        results = []

        # 1. Library Integration
        try:
            import cv2 as _cv2  # noqa
            import pytesseract as _pt  # noqa
            lib_ok, lib_msg = True, "cv2.dnn and pytesseract import cleanly."
        except ImportError as e:
            lib_ok, lib_msg = False, str(e)
        results.append(("1. Library Integration", lib_ok, lib_msg))

        # 2. Pre-Processing Integrity
        sample = cv2.imread(os.path.join("sample_images", "sample_text.jpg"))
        stages = full_pipeline(sample)
        pp_ok = stages["binary"] is not None and stages["otsu_threshold"] > 0
        results.append(("2. Pre-Processing Integrity", pp_ok, f"Otsu threshold = {stages['otsu_threshold']:.1f}"))

        # 3. Accuracy Benchmarking
        engine = OCREngine(psm=3)
        ocr_result = engine.recognize(os.path.join("sample_images", "sample_text.jpg"))
        acc_ok = ocr_result["meets_gatekeeper_standard"]
        results.append(("3. Accuracy Benchmarking", acc_ok, f"Average confidence = {ocr_result['average_confidence']:.1f}% (need ≥80%)"))

        # 4. Visual Confirmation
        os.makedirs("output", exist_ok=True)
        annotated = engine.annotate(ocr_result["stages"]["original"], ocr_result["words"])
        out_path = "output/gatekeeper_check.jpg"
        cv2.imwrite(out_path, annotated)
        vis_ok = os.path.exists(out_path)
        results.append(("4. Visual Confirmation", vis_ok, f"Saved to {out_path}"))

        all_pass = all(r[1] for r in results)

        for name, ok, msg in results:
            icon = "✅" if ok else "❌"
            st.write(f"{icon} **{name}** — {msg}")

        st.divider()
        if all_pass:
            st.success("✅ ALL 4 GATEKEEPER RULES PASSED — Project 4 milestone complete.")
        else:
            st.error("❌ Some rules failed — review above.")

st.divider()
st.caption("DecodeLabs · Industrial Training Kit · Batch 2026 · decodelabs.tech@gmail.com")
