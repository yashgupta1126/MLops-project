import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="🌍 Location Classifier",
    page_icon="🌍",
    layout="centered"
)

# -------------------- CUSTOM THEME --------------------
st.markdown("""
<style>

/* Background gradient */
.stApp {
    background: linear-gradient(
        to bottom,
        #001c1b,
        #2f5e4a,
        #4a8f73,
        #9bc5ae
    );
    color: white;
}

/* Glass card container */
.block-container {
    background: rgba(0, 0, 0, 0.35);
    padding: 2rem;
    border-radius: 18px;
    backdrop-filter: blur(10px);
}

/* Title center */
h1 {
    text-align: center;
}

/* Button styling */
.stButton>button {
    background: linear-gradient(to right, #4a8f73, #9bc5ae);
    color: #001c1b;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: bold;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.03);
    background: linear-gradient(to right, #9bc5ae, #4a8f73);
}

/* Inputs */
.stTextInput input {
    border-radius: 10px;
    border: 1px solid #4a8f73;
}

/* File uploader */
.stFileUploader {
    border: 1px dashed #9bc5ae;
    border-radius: 10px;
    padding: 10px;
}

/* Progress bar */
.stProgress > div > div {
    background-color: #9bc5ae;
}

</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.markdown("<h1>🌍 Location Image Classifier</h1>", unsafe_allow_html=True)
st.caption("Upload an image or paste a URL to classify the location")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('newmodel1/')

model = load_model()

# -------------------- CLASSES + EMOJIS --------------------
classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

emoji_map = {
    "forest": "🌲",
    "sea": "🌊",
    "mountain": "⛰️",
    "street": "🛣️",
    "buildings": "🏢",
    "glacier": "❄️"
}

# -------------------- IMAGE PROCESSING --------------------
def decode_img(image):
    img = tf.io.decode_image(image, channels=3)
    img = tf.image.resize(img, [150, 150])
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# -------------------- INPUT SECTION --------------------
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

with col2:
    url = st.text_input("🔗 Image URL")

# -------------------- CLASSIFY BUTTON --------------------
if st.button("🚀 Classify Image"):

    content = None

    # Upload option
    if uploaded_file is not None:
        content = uploaded_file.read()

    # URL option
    elif url.strip() != "":
        if not url.startswith("http"):
            st.warning("⚠️ Please enter a valid URL")
            st.stop()
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            content = response.content
        except Exception:
            st.error("❌ Failed to fetch image from URL")
            st.stop()
    else:
        st.warning("⚠️ Please upload an image or provide a URL")
        st.stop()

    # -------------------- DISPLAY IMAGE --------------------
    image = Image.open(BytesIO(content))
    st.image(image, caption="📷 Input Image", use_container_width=True)

    # -------------------- PREDICTION --------------------
    with st.spinner("🧠 Classifying..."):
        preds = model.predict(decode_img(content))
        label = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))

    predicted_class = classes[label]

    # -------------------- RESULT --------------------
    st.success(f"{emoji_map[predicted_class]} {predicted_class.upper()}")

    st.progress(confidence)
    st.write(f"Confidence: **{confidence*100:.2f}%**")
    # trigger redeploy v2