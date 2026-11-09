from __future__ import annotations

import streamlit as st
import pandas as pd
from PIL import Image

from src.inference import load_metadata, load_model, predict

st.set_page_config(page_title="Digit Recognition", page_icon="🔢", layout="wide")
st.title("Handwritten Digit Recognition")
st.caption("CNN-based recognition powered by the MNIST dataset")

@st.cache_resource
def get_model():
    return load_model()

metadata = load_metadata()
left, right = st.columns([1, 1.2], gap="large")

with left:
    st.subheader("Recognize a digit")
    uploaded = st.file_uploader("Upload a single handwritten digit", type=["png", "jpg", "jpeg"])
    st.caption("For best results, use one dark digit on a plain, light background.")
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
        try:
            result = predict(image, get_model())
            st.metric("Predicted digit", result["digit"], f"{result['confidence']:.1%} confidence")
            st.write("Top predictions")
            st.dataframe(pd.DataFrame(result["top_three"]).rename(columns={
                "digit": "Digit", "probability": "Probability"
            }), hide_index=True, use_container_width=True)
            st.bar_chart(pd.Series(result["probabilities"], name="Probability"))
        except FileNotFoundError as error:
            st.warning(str(error))

with right:
    st.subheader("Model information")
    if metadata:
        st.metric("Selected architecture", metadata.get("selected_model", "—"))
        candidates = metadata.get("candidates", [])
        if candidates:
            st.dataframe(pd.DataFrame(candidates), hide_index=True, use_container_width=True)
        st.caption(metadata.get("test_set_note", ""))
    else:
        st.info("Train the model once before using predictions.")
        st.code("python -m src.train", language="bash")
    st.markdown("**Dataset:** MNIST · 60,000 training images · 10,000 test images")
    st.markdown("**Classes:** digits 0–9")
