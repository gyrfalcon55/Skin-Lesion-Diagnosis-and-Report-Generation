import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

from utils.segmentation import segment_lesion
from utils.classification import classify, CLASS_NAMES
from utils.report_generator import generate_reports
from utils.gradcam import generate_gradcam, overlay_gradcam
from utils.pdf_generator import generate_pdf
from utils.preprocessing import preprocess_image


# --------------------------------------------------
# Streamlit page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Skin Lesion Diagnosis",
    layout="wide"
)

st.title("Skin Lesion Diagnosis and Report Generation")


# --------------------------------------------------
# Load classification model only once
# --------------------------------------------------
@st.cache_resource
def load_classifier():
    return tf.keras.models.load_model("models/efficientnetB1.h5")

classifier_model = load_classifier()


# --------------------------------------------------
# Image upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # --------------------------------------------------
    # Read uploaded image using OpenCV
    # --------------------------------------------------
    image_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    st.image(
        image,
        caption="Original Image",
        width=350
    )


    # --------------------------------------------------
    # Lesion segmentation (for visualization and metrics)
    # --------------------------------------------------
    segmented_image, mask = segment_lesion(image)

    st.image(
        segmented_image,
        caption="Segmented Lesion",
        width=350
    )


    # --------------------------------------------------
    # Classification (always use original image)
    # --------------------------------------------------
    disease, confidence = classify(image)

    st.success(
        f"Prediction: {disease} ({confidence * 100:.2f}%)"
    )

    lesion_info = f"Lesion area (pixels): {int(mask.sum())}"


    # --------------------------------------------------
    # Grad-CAM explainability
    # --------------------------------------------------
    st.subheader("Model Explanation (Grad-CAM)")

    preprocessed_image = preprocess_image(image)
    class_index = CLASS_NAMES.index(disease)

    heatmap = generate_gradcam(
        model=classifier_model,
        image=preprocessed_image,
        class_index=class_index
    )

    gradcam_image = overlay_gradcam(image, heatmap)

    st.image(
        gradcam_image,
        caption="Grad-CAM Heatmap",
        width=350
    )


    # --------------------------------------------------
    # Report generation
    # --------------------------------------------------
    if st.button("Generate Reports"):

        with st.spinner("Generating reports..."):
            patient_report, doctor_report = generate_reports(
                disease,
                confidence,
                lesion_info
            )

        # --------------------------------------------------
        # Display reports
        # --------------------------------------------------
        st.subheader("Patient Report")
        st.write(patient_report)

        st.subheader("Doctor Report")
        st.write(doctor_report)


        # --------------------------------------------------
        # Generate PDF files
        # --------------------------------------------------
        generate_pdf(
            "outputs/patient_report.pdf",
            "Patient Medical Report",
            patient_report
        )

        generate_pdf(
            "outputs/doctor_report.pdf",
            "Doctor Clinical Report",
            doctor_report
        )


        # --------------------------------------------------
        # Download buttons
        # --------------------------------------------------
        col1, col2 = st.columns(2)

        with col1:
            with open("outputs/patient_report.pdf", "rb") as file:
                st.download_button(
                    label="Download Patient Report (PDF)",
                    data=file,
                    file_name="patient_report.pdf",
                    mime="application/pdf"
                )

        with col2:
            with open("outputs/doctor_report.pdf", "rb") as file:
                st.download_button(
                    label="Download Doctor Report (PDF)",
                    data=file,
                    file_name="doctor_report.pdf",
                    mime="application/pdf"
                )
