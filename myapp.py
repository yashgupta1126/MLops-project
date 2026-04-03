import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import urllib.parse

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Location Classifier Model",
    page_icon="🌍",
    layout="centered"
)

# ─────────────────────────────────────────────
# GLOBAL STYLES (MATCHING NEW TARGET DESIGN)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg-deep:     #050505;
    --bg-card:     #0d1110;
    --border:      rgba(255, 255, 255, 0.08);
    --accent-mint: #4df0b5;
    --accent-lime: #b4f04d;
    --text-hi:     #ffffff;
    --text-lo:     #9ca3af;
}

/* ── Root Reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-deep) !important;
    font-family: 'Inter', sans-serif;
    color: var(--text-hi);
}

/* Simulate the dark earth curve background */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: absolute;
    top: -20%;
    left: 50%;
    transform: translateX(-50%);
    width: 150vw;
    height: 100vh;
    background: radial-gradient(ellipse at bottom, rgba(255,255,255,0.1) 0%, rgba(0,0,0,0) 60%);
    border-radius: 100%;
    pointer-events: none;
    z-index: 0;
    opacity: 0.5;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Block container ── */
.block-container {
    max-width: 800px !important;
    padding: 3rem 2rem 4rem !important;
    background: transparent !important;
    z-index: 1;
    position: relative;
}

/* ── Hero Section ── */
.hero-wrapper {
    text-align: center;
    margin-bottom: 4rem;
    padding-top: 2rem;
}
.badge-container {
    display: flex;
    justify-content: center;
    margin-bottom: 1.5rem;
}
.hero-badge {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255,255,255,0.1);
    color: var(--text-hi);
    font-size: 0.85rem;
    padding: 0.4rem 1rem;
    border-radius: 99px;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    backdrop-filter: blur(10px);
}
.hero-badge span {
    color: var(--accent-lime);
    font-size: 0.6rem;
}
.hero-title {
    font-size: 4rem;
    font-weight: 700;
    line-height: 1.1;
    letter-spacing: -0.04em;
    margin: 0;
    color: var(--text-hi);
}
.hero-subtitle {
    color: var(--text-lo);
    font-size: 1.05rem;
    margin-top: 1.5rem;
    font-weight: 400;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.6;
}

/* ── Cards — st.container blocks ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1.5rem 2rem !important;
    margin-bottom: 1.5rem !important;
}

/* Mint Green Section Headers */
.section-label {
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--accent-mint);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(255,255,255,0.15) !important;
    border-radius: 8px !important;
    background: rgba(255,255,255,0.02) !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"] section {
    color: var(--text-hi) !important;
}

/* ── Text input ── */
[data-testid="stTextInput"] input {
    background: rgba(0,0,0,0.3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-hi) !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent-mint) !important;
}

/* ── Buttons (Examples & Actions) ── */
.stButton > button {
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    border-radius: 8px !important;
    min-height: 3.5rem !important;
    height: auto !important;
    width: 100% !important;
    border: 1px solid var(--border) !important;
    background: rgba(255,255,255,0.03) !important;
    color: var(--text-hi) !important;
    transition: all 0.2s ease !important;
    white-space: nowrap !important; /* Prevents text from breaking into multiple lines */
}
.stButton > button p {
    white-space: nowrap !important;
}
.stButton > button:hover {
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(255,255,255,0.2) !important;
}

/* Primary classify button overrides */
div[data-testid="column"]:nth-child(1) .stButton > button {
    background: var(--accent-mint) !important;
    color: #000 !important;
    border: none !important;
    font-weight: 600 !important;
}
div[data-testid="column"]:nth-child(1) .stButton > button:hover {
    opacity: 0.9 !important;
}

/* ── Image ── */
[data-testid="stImage"] img {
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
}

/* ── Result card (Single High-Prob) ── */
.result-hero {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 0.5rem;
    padding: 1rem 0;
    text-align: center;
}
.result-emoji {
    font-size: 4rem;
    line-height: 1.2;
}
.result-label {
    font-size: 2rem;
    font-weight: 600;
    color: var(--text-hi);
    line-height: 1;
}
.result-conf {
    font-size: 1rem;
    color: var(--accent-mint);
    margin-top: 0.2rem;
    font-weight: 500;
}

/* ── OR separator ── */
.or-sep {
    text-align: center;
    color: var(--accent-mint);
    font-size: 0.85rem;
    margin: 1.5rem 0;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

EMOJI_MAP = {
    "buildings": "🏢 Building",
    "forest":    "🌲 Forest",
    "glacier":   "❄️ Glacier",
    "mountain":  "⛰️ Mountain",
    "sea":       "🌊 Sea",
    "street":    "🛣️ Street",
}

EXAMPLE_IMAGES = {
    "🌲 Forest":   "https://images.unsplash.com/photo-1448375240586-882707db888b?w=800",
    "🌊 Sea":      "https://images.unsplash.com/photo-1505118380757-91f5f5632de0?w=800",
    "⛰️ Mountain": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=800",
    "🏢 Building": "https://images.unsplash.com/photo-1486325212027-8081e485255e?w=800",
    "❄️ Glacier":  "https://images.unsplash.com/photo-1531366936337-7c912a4589a7?w=800",
    "🛣️ Street":   "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=800",
}

MAX_FILE_MB = 5


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model('newmodel1/')


# ─────────────────────────────────────────────
# IMAGE HELPERS
# ─────────────────────────────────────────────
def preprocess(image_bytes: bytes) -> np.ndarray:
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, [150, 150])
    img = tf.cast(img, tf.float32) / 255.0
    return np.expand_dims(img.numpy(), axis=0)

def predict(model, image_bytes: bytes):
    tensor = preprocess(image_bytes)
    preds  = model.predict(tensor, verbose=0)[0]
    order  = np.argsort(preds)[::-1]
    return [(CLASSES[i], float(preds[i])) for i in order]

def fetch_url(url: str) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL must start with http:// or https://")
    resp = requests.get(url, timeout=8)
    resp.raise_for_status()
    ct = resp.headers.get("Content-Type", "")
    if "image" not in ct:
        raise ValueError(f"URL does not point to an image (Content-Type: {ct})")
    return resp.content


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key, default in {
    "image_bytes": None,
    "image_source": None,
    "image_size": None,
    "results": None,
    "do_classify": False,
    "_last_upload_name": None,
    "_last_url": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

def reset():
    for k in ["image_bytes", "image_source", "image_size", "results",
              "do_classify", "_last_upload_name", "_last_url"]:
        st.session_state[k] = None
    st.session_state["do_classify"] = False


# ─────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="badge-container">
        <div class="hero-badge"><span>●</span> Introducing Classifier</div>
    </div>
    <h1 class="hero-title">Location Classifier Model</h1>
    <p class="hero-subtitle">Upload an image or paste a URL to automatically classify the scene into distinct geographic categories.</p>
</div>
""", unsafe_allow_html=True)

# Load model
model = load_model()

# ─────────────────────────────────────────────
# INPUT CARD
# ─────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="section-label">📁 IMAGE INPUT</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drag and drop file here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="or-sep">── or ──</div>', unsafe_allow_html=True)

    url_input = st.text_input(
        "Paste an image URL",
        placeholder="https://example.com/photo.jpg",
        label_visibility="collapsed"
    )

    # Example picker
    st.markdown('<div class="section-label" style="margin-top:2.5rem;">✨ TRY AN EXAMPLE</div>', unsafe_allow_html=True)
    ex_cols = st.columns(len(EXAMPLE_IMAGES))
    for col, (label, ex_url) in zip(ex_cols, EXAMPLE_IMAGES.items()):
        with col:
            if st.button(label, key=f"ex_{label}"):
                try:
                    data = fetch_url(ex_url)
                    img  = Image.open(BytesIO(data))
                    st.session_state.image_bytes       = data
                    st.session_state.image_source      = f"Example — {label}"
                    st.session_state.image_size        = img.size
                    st.session_state.results           = None
                    st.session_state._last_upload_name = None
                except Exception as e:
                    st.error(f"Could not load example: {e}")

# ─────────────────────────────────────────────
# STAGED INPUT → session state
# ─────────────────────────────────────────────
if uploaded_file is not None:
    if uploaded_file.name != st.session_state._last_upload_name:
        raw = uploaded_file.read()
        if len(raw) > MAX_FILE_MB * 1024 * 1024:
            st.error(f"⚠️ File too large — please upload an image under {MAX_FILE_MB} MB.")
        else:
            try:
                img = Image.open(BytesIO(raw))
                img.verify()
                img = Image.open(BytesIO(raw))
                st.session_state.image_bytes       = raw
                st.session_state.image_source      = f"Upload — {uploaded_file.name}"
                st.session_state.image_size        = img.size
                st.session_state.results           = None
                st.session_state._last_upload_name = uploaded_file.name
            except Exception:
                st.error("❌ The uploaded file appears to be corrupted or is not a valid image.")

elif url_input.strip():
    if url_input != st.session_state._last_url:
        st.session_state._last_url = url_input
        try:
            raw = fetch_url(url_input)
            if len(raw) > MAX_FILE_MB * 1024 * 1024:
                st.error(f"⚠️ Remote image exceeds {MAX_FILE_MB} MB limit.")
            else:
                img = Image.open(BytesIO(raw))
                img.verify()
                img = Image.open(BytesIO(raw))
                st.session_state.image_bytes  = raw
                st.session_state.image_source = "URL"
                st.session_state.image_size   = img.size
                st.session_state.results      = None
        except Exception:
            st.error("❌ Could not load image from URL — please verify it points directly to an image file.")


# ─────────────────────────────────────────────
# PREVIEW & RESULTS CARD
# ─────────────────────────────────────────────
if st.session_state.image_bytes:
    
    # 1. Show the Preview container first
    with st.container(border=True):
        st.markdown('<div class="section-label">👁️ PREVIEW</div>', unsafe_allow_html=True)

        st.image(
            Image.open(BytesIO(st.session_state.image_bytes)),
            use_container_width=True
        )

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        btn_col1, btn_col2 = st.columns([3, 1])
        with btn_col1:
            if st.button("Classify Image", use_container_width=True):
                st.session_state.do_classify = True
        with btn_col2:
            if st.button("Clear", use_container_width=True):
                reset()
                st.rerun()

        if st.session_state.do_classify:
            st.session_state.do_classify = False
            with st.spinner("Processing image..."):
                try:
                    st.session_state.results = predict(model, st.session_state.image_bytes)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")

    # 2. Show the Results container below the preview
    if st.session_state.results:
        results = st.session_state.results
        top_cls, top_conf = results[0]  # Show only the top prediction
        
        # Extract emoji and formatted label from our map
        formatted_label = EMOJI_MAP.get(top_cls, f"❓ {top_cls.title()}")
        emoji_only = formatted_label.split(" ")[0]
        text_only = formatted_label.split(" ")[1]

        with st.container(border=True):
            st.markdown(f"""
            <div class="result-hero">
                <div class="result-emoji">{emoji_only}</div>
                <div class="result-label">{text_only}</div>
                <div class="result-conf">{top_conf*100:.1f}% Confidence</div>
            </div>
            """, unsafe_allow_html=True)