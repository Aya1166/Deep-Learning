import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json
import os

# -------------------------
# Load model
# -------------------------
MODEL_PATH = "models/best_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# -------------------------
# Load class names (FIXED)
# -------------------------
with open("classes.json", "r") as f:
    class_indices = json.load(f)

# flip dict: {"cane": 0, ...} -> {0: "cane", ...}
classes = {v: k for k, v in class_indices.items()}

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Animal Classifier", layout="centered")

st.title("🐾 Animal Image Classifier")
st.write("Upload an image and the model will predict the animal type.")

# -------------------------
# Upload image
# -------------------------
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # -------------------------
    # Preprocessing (FIXED - MobileNetV2)
    # -------------------------
    img = image.resize((224, 224))
    img_array = np.array(img)

    # MobileNetV2 preprocessing (replaces /255.0)
    img_array = preprocess_input(img_array)

    # add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # -------------------------
    # Prediction
    # -------------------------
    prediction = model.predict(img_array)

    predicted_index = int(np.argmax(prediction))
    predicted_class = classes[predicted_index]
    confidence = float(np.max(prediction))

    # -------------------------
    # Output
    # -------------------------
    st.subheader("Prediction Result")

    st.write(f"🐾 **Predicted Animal:** {predicted_class}")
    st.write(f"🎯 **Confidence:** {confidence:.2%}")

    st.progress(confidence)

    # optional debug
    with st.expander("Show raw probabilities"):
        for idx, prob in enumerate(prediction[0]):
            st.write(f"{classes[idx]}: {prob:.2%}")
