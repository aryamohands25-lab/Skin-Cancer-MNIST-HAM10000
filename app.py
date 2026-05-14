import os
import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skin Cancer Classifier",
    page_icon="🔬",
    layout="centered",
)

# ── Label order — verified against HAM10000 CSV sample counts ─────────────────
#
#   Index  Code    Count (train)   Confirmed match
#   0      akiec   262             akiec  327 × 0.8 = 262  ✓
#   1      bcc     411             bcc    514 × 0.8 = 411  ✓
#   2      bkl     879             bkl   1099 × 0.8 = 879  ✓
#   3      df       92             df     115 × 0.8 =  92  ✓
#   4      nv     5362             nv    6705 × 0.8 = 5364 ✓  ← was wrongly 'mel'
#   5      vasc    114             vasc   142 × 0.8 = 114  ✓  ← was wrongly 'nv'
#   6      mel     890             mel   1113 × 0.8 = 890  ✓  ← was wrongly 'vasc'
#
CLASS_INFO = {
    "akiec": {
        "full_name": "Actinic Keratoses",
        "nature": "Pre-malignant",
        "color": "#f59e0b",
        "description": (
            "Rough, scaly patches caused by years of sun exposure. "
            "Can progress to squamous cell carcinoma if untreated."
        ),
    },
    "bcc": {
        "full_name": "Basal Cell Carcinoma",
        "nature": "Malignant",
        "color": "#ef4444",
        "description": (
            "The most common form of skin cancer. Rarely spreads but "
            "can cause significant local tissue damage if left untreated."
        ),
    },
    "bkl": {
        "full_name": "Benign Keratosis",
        "nature": "Benign",
        "color": "#22c55e",
        "description": (
            "Non-cancerous skin growths including seborrheic keratoses "
            "and solar lentigines. Generally harmless."
        ),
    },
    "df": {
        "full_name": "Dermatofibroma",
        "nature": "Benign",
        "color": "#22c55e",
        "description": (
            "Firm, small benign skin nodules, commonly found on the legs. "
            "Usually harmless and require no treatment."
        ),
    },
    "nv": {
        "full_name": "Melanocytic Nevi",
        "nature": "Benign",
        "color": "#22c55e",
        "description": (
            "Common moles — benign proliferations of melanocytes. "
            "Most are harmless, but monitor for changes in size or color."
        ),
    },
    "vasc": {
        "full_name": "Vascular Lesions",
        "nature": "Benign",
        "color": "#3b82f6",
        "description": (
            "Includes angiomas, angiokeratomas, and pyogenic granulomas. "
            "Typically benign growths of blood vessels in the skin."
        ),
    },
    "mel": {
        "full_name": "Melanoma",
        "nature": "Malignant",
        "color": "#dc2626",
        "description": (
            "The most dangerous type of skin cancer. Develops in melanocytes "
            "and can spread to other organs. Early detection is critical."
        ),
    },
}

# Must match the integer label order in the training data (verified above)
CLASS_LABELS = ["akiec", "bcc", "bkl", "df", "nv", "vasc", "mel"]
IMG_SIZE     = (28, 28)
MODEL_PATH   = "skin_cancer_model.keras"


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    import tensorflow as tf

    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)

    try:
        from huggingface_hub import hf_hub_download
        hf_repo = st.secrets.get("HF_REPO_ID", "")
        if not hf_repo:
            st.error(
                "**Model not found.** Place `skin_cancer_model.keras` in the "
                "repo root, or set `HF_REPO_ID` in Streamlit secrets. "
                "See DEPLOYMENT.md for instructions."
            )
            st.stop()
        path = hf_hub_download(repo_id=hf_repo, filename=MODEL_PATH)
        return tf.keras.models.load_model(path)
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        st.stop()


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # (1, 28, 28, 3)


# ── Probability chart ──────────────────────────────────────────────────────────
def build_chart(probs: np.ndarray, predicted_idx: int) -> go.Figure:
    labels = [CLASS_INFO[c]["full_name"] for c in CLASS_LABELS]
    colors = [
        "#6366f1" if i == predicted_idx else "#cbd5e1"
        for i in range(len(CLASS_LABELS))
    ]
    fig = go.Figure(go.Bar(
        x=probs * 100,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        xaxis=dict(title="Confidence (%)", range=[0, 115]),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=60, t=10, b=40),
        height=320,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
    )
    return fig


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🔬 Skin Cancer Classifier")
st.caption("HAM10000 · 7-class CNN · For educational purposes only")

st.warning(
    "⚠️ **Medical disclaimer** — This tool is for research and educational use "
    "only. It is **not** a substitute for professional medical diagnosis. "
    "Consult a qualified dermatologist for any skin concerns.",
    icon="⚠️",
)

st.markdown("---")

uploaded = st.file_uploader(
    "Upload a dermatoscopic skin lesion image",
    type=["jpg", "jpeg", "png"],
    help="For best results use dermoscopy images similar to the HAM10000 dataset.",
)

if uploaded is not None:
    img = Image.open(uploaded)

    col_img, _ = st.columns([1, 1])
    with col_img:
        st.image(img, caption="Uploaded image", use_container_width=True)

    model  = load_model()
    tensor = preprocess(img)

    with st.spinner("Analysing…"):
        preds = model.predict(tensor, verbose=0)[0]

    pred_idx   = int(np.argmax(preds))
    pred_code  = CLASS_LABELS[pred_idx]
    pred_info  = CLASS_INFO[pred_code]
    confidence = float(preds[pred_idx])

    st.markdown("---")

    nature_color = pred_info["color"]
    bg = (
        "#fef2f2" if pred_info["nature"] == "Malignant"
        else "#f0fdf4" if pred_info["nature"] == "Benign"
        else "#fffbeb"
    )
    st.markdown(
        f"""
        <div style="
            border-left:5px solid {nature_color};padding:0.8rem 1.2rem;
            border-radius:6px;background:{bg};margin-bottom:1rem;">
          <p style="margin:0;font-size:0.85rem;color:#6b7280;">Predicted class</p>
          <h2 style="margin:0.2rem 0;color:#1f2937;">{pred_info['full_name']}</h2>
          <p style="margin:0;">
            <span style="background:{nature_color};color:white;padding:2px 10px;
              border-radius:99px;font-size:0.78rem;font-weight:600;">
              {pred_info['nature']}
            </span>
            &nbsp;
            <span style="font-size:0.95rem;color:#374151;">
              Confidence: <strong>{confidence*100:.1f}%</strong>
            </span>
          </p>
          <p style="margin:0.6rem 0 0;font-size:0.88rem;color:#4b5563;">
            {pred_info['description']}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Confidence across all classes")
    st.plotly_chart(build_chart(preds, pred_idx), use_container_width=True)

    with st.expander("📋 All class probabilities"):
        for i in np.argsort(preds)[::-1]:
            code = CLASS_LABELS[i]
            info = CLASS_INFO[code]
            st.markdown(
                f"**{info['full_name']}** (`{code}`)  "
                f"— {preds[i]*100:.2f}%  "
                f"— *{info['nature']}*"
            )

else:
    st.info("👆 Upload an image above to get a prediction.")

    with st.expander("📖 Lesion class reference"):
        for code, info in CLASS_INFO.items():
            st.markdown(
                f"**{info['full_name']}** (`{code}`) "
                f"<span style='background:{info['color']};color:white;"
                f"padding:1px 8px;border-radius:99px;font-size:0.75rem'>"
                f"{info['nature']}</span>",
                unsafe_allow_html=True,
            )
            st.caption(info["description"])
