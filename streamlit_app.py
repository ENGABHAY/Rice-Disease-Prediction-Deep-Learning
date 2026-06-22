import streamlit as st
import requests
from PIL import Image
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rice Leaf Disease Classifier",
    page_icon="🌾",
    layout="centered",
)

API_URL = "http://localhost:8000/predict"

CLASS_META = {
    "Bacterial leaf blight": {
        "emoji": "🦠",
        "color": "#E05C5C",
        "tip": "Caused by Xanthomonas oryzae. Remove infected tillers and avoid excessive nitrogen.",
    },
    "Brown spot": {
        "emoji": "🟤",
        "color": "#C47C2B",
        "tip": "Caused by Bipolaris oryzae. Improve soil nutrition (especially potassium).",
    },
    "Leaf smut": {
        "emoji": "⚫",
        "color": "#5A5A7A",
        "tip": "Caused by Entyloma oryzae. Use certified disease-free seeds and fungicide treatment.",
    },
}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.4rem;
        color: #2cde3b;
        line-height: 1.15;
        margin-bottom: 0.25rem;
    }
    .hero-sub {
        color: #556b55;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-card {
        background: #f7faf4;
        border: 1.5px solid #c4deba;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-top: 1.5rem;
    }
    .disease-label {
        font-size: 1.55rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .confidence-val {
        font-size: 0.95rem;
        color: #6b7c6b;
        margin-bottom: 1rem;
    }
    .prob-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.55rem;
        gap: 0.7rem;
    }
    .prob-label {
        width: 200px;
        font-size: 0.88rem;
        color: #2d3e2d;
        font-weight: 500;
    }
    .prob-bar-wrap {
        flex: 1;
        background: #dce8d8;
        border-radius: 99px;
        height: 10px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 99px;
        transition: width 0.5s ease;
    }
    .prob-pct {
        width: 46px;
        text-align: right;
        font-size: 0.85rem;
        font-weight: 600;
        color: #2d3e2d;
    }
    .tip-box {
        margin-top: 1.1rem;
        padding: 0.75rem 1rem;
        background: #eef5eb;
        border-left: 4px solid #5a9a5a;
        border-radius: 6px;
        font-size: 0.88rem;
        color: #2d4a2d;
    }
    .upload-hint {
        font-size: 0.82rem;
        color: #8a9e8a;
        margin-top: 0.4rem;
    }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #a8c9a0;
        border-radius: 12px;
        padding: 1rem;
        background: #f2f8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🌾 Rice Leaf Disease Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload a photo of a rice leaf — get an instant disease diagnosis.</div>', unsafe_allow_html=True)

# ── Upload widget ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Choose a rice leaf image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)
st.markdown('<p class="upload-hint">Supported formats: JPG, JPEG, PNG, WEBP</p>', unsafe_allow_html=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if uploaded:
    col_img, col_gap = st.columns([1, 0.05])
    with col_img:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded leaf", use_container_width=True)

    with st.spinner("Analysing leaf…"):
        try:
            # POST image bytes to FastAPI
            uploaded.seek(0)
            response = requests.post(
                API_URL,
                files={"file": (uploaded.name, uploaded.read(), uploaded.type)},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

        except requests.exceptions.ConnectionError:
            st.error("⚠️ Cannot reach the prediction server. Make sure FastAPI is running on port 8000.")
            st.stop()
        except requests.exceptions.HTTPError as e:
            detail = response.json().get("detail", str(e))
            st.error(f"Prediction failed: {detail}")
            st.stop()

    disease   = result["predicted_class"]
    confidence = result["confidence"]
    probs      = result["probabilities"]
    meta       = CLASS_META[disease]

    # ── Result card ───────────────────────────────────────────────────────────
    bars_html = ""
    for cls, pct in sorted(probs.items(), key=lambda x: -x[1]):
        color     = CLASS_META[cls]["color"]
        is_top    = cls == disease
        weight    = "font-weight:700;" if is_top else ""
        bars_html += f"""
        <div class="prob-row">
            <span class="prob-label" style="{weight}">{CLASS_META[cls]['emoji']} {cls}</span>
            <div class="prob-bar-wrap">
                <div class="prob-bar-fill" style="width:{pct}%; background:{color};"></div>
            </div>
            <span class="prob-pct">{pct}%</span>
        </div>
        """

    tip_html = f'<div class="tip-box">💡 <strong>Management tip:</strong> {meta["tip"]}</div>'

    st.markdown(
        f"""
        <div class="result-card">
            <div class="disease-label" style="color:{meta['color']}">
                {meta['emoji']} {disease}
            </div>
            <div class="confidence-val">Confidence: <strong>{confidence}%</strong></div>
            {bars_html}
            {tip_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── JSON expander for devs ────────────────────────────────────────────────
    with st.expander("Raw API response"):
        st.json(result)

else:
    st.info("Upload an image above to get started.", icon="👆")
