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
    page_title="GeoLens · Location Classifier",
    page_icon="🌍",
    layout="centered"
)

# ─────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

:root {
    --bg-deep:     #050d0c;
    --bg-card:     rgba(10, 28, 26, 0.72);
    --border:      rgba(78, 195, 160, 0.18);
    --accent:      #3de8b0;
    --accent-dim:  #1db88a;
    --accent-glow: rgba(61, 232, 176, 0.25);
    --text-hi:     #e8f5f1;
    --text-lo:     #7aada0;
    --danger:      #ff6b6b;
    --warn:        #ffd166;
}

/* ── Root Reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-deep) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text-hi);
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 50% -10%, rgba(61,232,176,0.12) 0%, transparent 70%),
        radial-gradient(ellipse 50% 40% at 90% 80%, rgba(29,184,138,0.07) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Block container ── */
.block-container {
    max-width: 720px !important;
    padding: 3rem 2rem 4rem !important;
    background: transparent !important;
}

/* ── Typography ── */
h1, h2, h3 {
    font-family: 'Syne', sans-serif;
    letter-spacing: -0.02em;
}

/* ── Wordmark ── */
.wordmark {
    text-align: center;
    margin-bottom: 0.25rem;
}
.wordmark h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e8f5f1 30%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1;
}
.wordmark p {
    color: var(--text-lo);
    font-size: 0.95rem;
    font-weight: 300;
    margin: 0.4rem 0 0;
    letter-spacing: 0.04em;
}

/* ── Cards — st.container blocks ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    padding: 1.5rem !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    margin-bottom: 1.25rem !important;
    transition: border-color 0.3s !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(61,232,176,0.32) !important;
}

.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.9rem;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed var(--border) !important;
    border-radius: 14px !important;
    background: rgba(61,232,176,0.03) !important;
    transition: border-color 0.3s, background 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-dim) !important;
    background: rgba(61,232,176,0.06) !important;
}

/* ── Text input ── */
[data-testid="stTextInput"] input {
    background: rgba(5,13,12,0.7) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-hi) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    transition: border-color 0.25s;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em !important;
    border-radius: 12px !important;
    height: 3rem !important;
    width: 100% !important;
    border: none !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}

/* Primary classify button */
div[data-testid="column"]:nth-child(1) .stButton > button,
.classify-btn .stButton > button {
    background: linear-gradient(135deg, var(--accent-dim), var(--accent)) !important;
    color: #040d0c !important;
    box-shadow: 0 0 24px var(--accent-glow) !important;
}
div[data-testid="column"]:nth-child(1) .stButton > button:hover {
    transform: translateY(-1px) scale(1.01) !important;
    box-shadow: 0 0 36px rgba(61,232,176,0.4) !important;
}

/* Secondary / reset button */
div[data-testid="column"]:nth-child(2) .stButton > button {
    background: rgba(255,255,255,0.04) !important;
    color: var(--text-lo) !important;
    border: 1px solid var(--border) !important;
}
div[data-testid="column"]:nth-child(2) .stButton > button:hover {
    background: rgba(255,255,255,0.08) !important;
    color: var(--text-hi) !important;
}

/* ── Image ── */
[data-testid="stImage"] img {
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-size: 0.9rem !important;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--accent-dim), var(--accent)) !important;
    border-radius: 99px !important;
}
[data-testid="stProgress"] > div {
    background: rgba(255,255,255,0.07) !important;
    border-radius: 99px !important;
    height: 8px !important;
}

/* ── Result card ── */
.result-hero {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.2rem;
}
.result-emoji {
    font-size: 3.2rem;
    line-height: 1;
    filter: drop-shadow(0 0 12px var(--accent-glow));
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e8f5f1, var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.result-conf {
    font-size: 0.88rem;
    color: var(--text-lo);
    margin-top: 0.3rem;
}

/* ── Breakdown row ── */
.bar-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.65rem;
}
.bar-emoji { font-size: 1.1rem; width: 1.5rem; text-align: center; }
.bar-name {
    font-family: 'Syne', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-hi);
    width: 5.5rem;
    text-transform: capitalize;
}
.bar-track {
    flex: 1;
    height: 7px;
    background: rgba(255,255,255,0.07);
    border-radius: 99px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, var(--accent-dim), var(--accent));
    transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
}
.bar-fill.dim { background: rgba(61,232,176,0.25); }
.bar-pct {
    font-size: 0.8rem;
    color: var(--text-lo);
    width: 3.2rem;
    text-align: right;
    font-variant-numeric: tabular-nums;
}

/* ── Meta chips ── */
.meta-row {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-top: 0.5rem;
}
.chip {
    background: rgba(61,232,176,0.08);
    border: 1px solid var(--border);
    border-radius: 99px;
    padding: 0.25rem 0.75rem;
    font-size: 0.78rem;
    color: var(--text-lo);
    letter-spacing: 0.02em;
}
.chip span { color: var(--accent); font-weight: 500; }

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.2rem 0;
}

/* ── OR separator ── */
.or-sep {
    text-align: center;
    color: var(--text-lo);
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    margin: 0.6rem 0;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: var(--text-lo);
    font-size: 0.78rem;
    margin-top: 3rem;
    letter-spacing: 0.04em;
}
.footer a { color: var(--accent); text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

EMOJI_MAP = {
    "buildings": "🏢",
    "forest":    "🌲",
    "glacier":   "❄️",
    "mountain":  "⛰️",
    "sea":       "🌊",
    "street":    "🛣️",
}

EXAMPLE_IMAGES = {
    "🌲 Forest":   "https://images.unsplash.com/photo-1448375240586-882707db888b?w=800",
    "🌊 Sea":      "https://images.unsplash.com/photo-1505118380757-91f5f5632de0?w=800",
    "⛰️ Mountain": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=800",
    "🏢 Buildings":"https://images.unsplash.com/photo-1486325212027-8081e485255e?w=800",
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
# WORDMARK
# ─────────────────────────────────────────────
st.markdown("""
<div class="wordmark">
    <h1>GeoLens</h1>
    <p>AI-powered location scene classifier</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Load model (cached - spinner avoided to prevent empty box artifact)
model = load_model()

# ─────────────────────────────────────────────
# INPUT CARD
# ─────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="section-label">📂 Image Input</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop an image here or click to browse",
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
    st.markdown('<div class="section-label" style="margin-top:1rem;">✨ Try an example</div>', unsafe_allow_html=True)
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
# Only re-process when the source actually changes to avoid
# wiping results on every Streamlit rerun.
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
        except requests.exceptions.Timeout:
            st.error("❌ Request timed out — the server took too long to respond.")
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ HTTP error: {e}")
        except ValueError as e:
            st.error(f"⚠️ {e}")
        except Exception:
            st.error("❌ Could not load image from URL — please verify it points directly to an image file.")

# ─────────────────────────────────────────────
# PREVIEW CARD
# ─────────────────────────────────────────────
if st.session_state.image_bytes:
    with st.container(border=True):
        st.markdown('<div class="section-label">🖼 Preview</div>', unsafe_allow_html=True)

        st.image(
            Image.open(BytesIO(st.session_state.image_bytes)),
            use_container_width=True
        )

        w, h = st.session_state.image_size or (0, 0)
        st.markdown(f"""
        <div class="meta-row">
            <div class="chip">Resolution <span>{w} × {h} px</span></div>
            <div class="chip">Source <span>{st.session_state.image_source}</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    # ── Action buttons ──
    # Persist classify intent via session state so result survives the rerun
    # that Streamlit triggers immediately after a button click.
    btn_col1, btn_col2 = st.columns([3, 1])
    with btn_col1:
        if st.button("🔍 Classify Image", use_container_width=True):
            st.session_state.do_classify = True
    with btn_col2:
        if st.button("↩ Reset", use_container_width=True):
            reset()
            st.rerun()

    if st.session_state.do_classify:
        st.session_state.do_classify = False
        with st.spinner("Analysing scene…"):
            try:
                st.session_state.results = predict(model, st.session_state.image_bytes)
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

# ─────────────────────────────────────────────
# RESULTS CARD
# ─────────────────────────────────────────────
if st.session_state.results:
    results = st.session_state.results
    top_cls, top_conf = results[0]
    top_emoji = EMOJI_MAP[top_cls]

    with st.container(border=True):
        st.markdown('<div class="section-label">🎯 Classification Result</div>', unsafe_allow_html=True)

        # Hero
        st.markdown(f"""
        <div class="result-hero">
            <div class="result-emoji">{top_emoji}</div>
            <div>
                <div class="result-label">{top_cls.title()}</div>
                <div class="result-conf">Top prediction · {top_conf*100:.1f}% confidence</div>
            </div>
        </div>
        <hr class="divider">
        <div class="section-label">All Classes</div>
        """, unsafe_allow_html=True)

        # Per-class bars
        for i, (cls, conf) in enumerate(results):
            emoji = EMOJI_MAP[cls]
            pct   = conf * 100
            dim_class = "" if i == 0 else " dim"
            st.markdown(f"""
            <div class="bar-row">
                <div class="bar-emoji">{emoji}</div>
                <div class="bar-name">{cls}</div>
                <div class="bar-track">
                    <div class="bar-fill{dim_class}" style="width:{pct:.1f}%"></div>
                </div>
                <div class="bar-pct">{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with <a href="https://www.tensorflow.org">TensorFlow</a> &
    <a href="https://streamlit.io">Streamlit</a> &nbsp;·&nbsp;
    Classifies scenes into 6 categories: Buildings, Forest, Glacier, Mountain, Sea, Street
</div>
""", unsafe_allow_html=True)