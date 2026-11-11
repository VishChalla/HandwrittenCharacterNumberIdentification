"""Web app: upload a photo of a handwritten digit and see what the model reads.

    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from digits.preprocess import to_mnist_format

MODEL_PATH = Path(__file__).parent / "models" / "mnist_cnn.keras"

st.set_page_config(page_title="Handwritten Digit Recognition", layout="centered")


@st.cache_resource
def load_model():
    from tensorflow.keras.models import load_model as keras_load

    return keras_load(MODEL_PATH)


st.title("Handwritten Digit Recognition")
st.write("Upload a photo or scan of a single handwritten digit. Dark ink on light paper "
         "or light on dark both work.")

uploaded = st.file_uploader("Digit image", type=["png", "jpg", "jpeg"],
                            label_visibility="collapsed")

if uploaded is None:
    st.info("Waiting for an image.")
    st.stop()

if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()

image = Image.open(uploaded)
prepared = to_mnist_format(image)
probabilities = load_model().predict(prepared.reshape(1, 28, 28, 1), verbose=0)[0]
digit = int(np.argmax(probabilities))

left, right = st.columns(2)
left.image(image, caption="Your upload", use_container_width=True)
right.image(prepared, caption="What the model sees (28 x 28)", use_container_width=True, clamp=True)

st.metric("Predicted digit", digit, f"{probabilities[digit]:.1%} confident", delta_color="off")
st.bar_chart({"probability": {str(d): float(p) for d, p in enumerate(probabilities)}})

if probabilities[digit] < 0.7:
    st.warning("The model is not sure about this one. Try a clearer or more tightly cropped image.")

st.download_button("Download result", f"Predicted digit: {digit}\nConfidence: {probabilities[digit]:.4f}\n",
                   file_name="predicted_digit.txt")
