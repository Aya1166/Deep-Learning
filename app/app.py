import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# -------------------------
# Load model
# -------------------------
MODEL_PATH = "best_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# -------------------------
# Load class names (IMPORTANT FIX)
# -------------------------
with open("classes.json", "r") as f:
    class_indices = json.load(f)

# convert dict -> ordered list
classes = list(class_indices.keys())

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
    # Preprocessing (MATCH TRAINING)
    # -------------------------
    img = image.resize((224, 224))
    img_array = np.array(img)

    # normalize
    img_array = img_array / 255.0

    # add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # -------------------------
    # Prediction
    # -------------------------
    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)
    predicted_class = classes[predicted_index]
    confidence = float(np.max(prediction))

    # -------------------------
    # Output
    # -------------------------
    st.subheader("Prediction Result")

    st.write(f"🐶 **Predicted Animal:** {predicted_class}")
    st.write(f"🎯 **Confidence:** {confidence:.2f}")

    st.progress(confidence)

    # optional debug
    with st.expander("Show raw probabilities"):
        st.write(prediction)