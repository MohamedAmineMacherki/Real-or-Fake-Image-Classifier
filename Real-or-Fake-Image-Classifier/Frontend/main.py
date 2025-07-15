import streamlit as st
import requests
from PIL import Image

st.title("AI vs Real Image Classifier")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify Image"):
        st.write("Classifying...")

        # Send the image to the backend for prediction
        response = requests.post(
            "http://localhost:8000/predict/",
            files={"file": uploaded_file.getvalue()},
        )

        if response.status_code == 200:
            result = response.json()
            label = result.get("label", "Unknown")
            st.write(f"Prediction: {label}")
        else:
            st.write("Error: Unable to classify the image.")
