import os
import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go

st.set_page_config(
    page_title="DermAI · Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #060d1a; color: #e2e8f0; }

[data-testid="stSidebar"] {
    background: #0b1628 !important;
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stHeader"]    { background: transparent; }
[data-testid="stToolbar"]   { display: none; }

[data-testid="stFileUploader"] {
    background: #0b1628;
    border: 2px dashed #1e3a5f;
    border-radius: 12px;
    padding: 1rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #3b82f6; }

.stButton > button {
    background: #1d4ed8; color: white; border: none;
    border-radius: 8px; font-family: 'DM Sans', sans-serif; font-weight: 500;
}
.stButton > button:hover { background: #2563eb; }

[data-testid="stExpander"] {
    background: #0b1628;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
}
[data-testid="stAlert"] {
    background: #0b1628 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 10px !important;
}
hr { border-color: #1e3a5f; }
[data-testid="metric-container"] {
    background: #0b1628;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CLASS_INFO = {
    "akiec": {
        "full_name": "Actinic Keratoses",
        "nature": "Pre-malignant",
        "icd": "L57.0",
        "color": "#f59e0b",
        "bg": "#1a1200",
        "description": "Rough, scaly patches from chronic UV exposure. Can progress to squamous cell carcinoma if untreated.",
    },
    "bcc": {
        "full_name": "Basal Cell Carcinoma",
        "nature": "Malignant",
        "icd": "C44",
        "color": "#ef4444",
        "bg": "#1a0000",
        "description": "Most common skin cancer. Rarely metastasises but causes significant local tissue damage.",
    },
    "bkl": {
        "full_name": "Benign Keratosis",
        "nature": "Benign",
        "icd": "L82",
        "color": "#22c55e",
        "bg": "#001a08",
        "description": "Non-cancerous growths including seborrheic keratoses and solar lentigines.",
    },
    "df": {
        "full_name": "Dermatofibroma",
        "nature": "Benign",
        "icd": "L98.0",
        "color": "#22c55e",
        "bg": "#001a08",
        "description": "Firm, small benign nodules typically found on the lower extremities.",
    },
    "nv": {
        "full_name": "Melanocytic Nevi",
        "nature": "Benign",
        "icd": "D22",
        "color": "#22c55e",
        "bg": "#001a08",
        "description": "Common moles — benign melanocyte proliferations. Monitor for ABCDE changes.",
    },
    "vasc": {
        "full_name": "Vascular Lesions",
        "nature": "Benign",
        "icd": "D18",
        "color": "#3b82f6",
        "bg": "#00081a",
        "description": "Angiomas, angiokeratomas, and pyogenic granulomas. Typically benign.",
    },
    "mel": {
        "full_name": "Melanoma",
        "nature": "Malignant",
        "icd": "C43",
        "color": "#dc2626",
        "bg": "#1a0000",
        "description": "Most dangerous skin cancer. Arises from melanocytes. Early detection is critical for survival.",
    },
}

CLASS_LABELS = ["akiec", "bcc", "bkl", "df", "nv", "vasc", "mel"]
IMG_SIZE     = (28, 28)
MODEL_PATH   = "skin_cancer_model.keras"


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising model…")
def load_model():
    import tensorflow as tf
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    try:
        from huggingface_hub import hf_hub_download
        hf_repo = st.secrets.get("HF_REPO_ID", "evans-1/skin-cancer-ham10000")
        path = hf_hub_download(repo_id=hf_repo, filename=MODEL_PATH)
        return tf.keras.models.load_model(path)
    except Exception as exc:
        st.error(f"Model load failed: {exc}")
        st.stop()


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Chart ──────────────────────────────────────────────────────────────────────
def build_chart(probs: np.ndarray, predicted_idx: int) -> go.Figure:
    labels = [CLASS_INFO[c]["full_name"] for c in CLASS_LABELS]
    colors = [CLASS_INFO[c]["color"] if i == predicted_idx else "#1e3a5f"
              for i, c in enumerate(CLASS_LABELS)]
    fig = go.Figure(go.Bar(
        x=probs * 100,
        y=labels,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=12),
        cliponaxis=False,
    ))
    fig.update_layout(
        xaxis=dict(
            title="Confidence (%)", range=[0, 118],
            gridcolor="#1e3a5f", tickfont=dict(color="#64748b"),
            title_font=dict(color="#64748b"), zeroline=False,
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color="#94a3b8"),
            gridcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=0, r=70, t=10, b=40),
        height=300,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", size=13),
    )
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0.5rem 0 1.5rem'>
        <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;
                    color:#e2e8f0;letter-spacing:-0.5px;'>🔬 DermAI</div>
        <div style='font-size:0.75rem;color:#475569;margin-top:2px;'>
            HAM10000 · CNN · v1.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Model")
    st.markdown("""
    <div style='font-size:0.82rem;color:#64748b;line-height:1.9'>
        Architecture &nbsp;·&nbsp; 3-block CNN<br>
        Input &nbsp;·&nbsp; 28 × 28 × 3<br>
        Classes &nbsp;·&nbsp; 7<br>
        Parameters &nbsp;·&nbsp; 471,207<br>
        Test accuracy &nbsp;·&nbsp; 55.9%<br>
        Weighted precision &nbsp;·&nbsp; 0.74
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Classes")
    for code, info in CLASS_INFO.items():
        icon = "🔴" if info["nature"] == "Malignant" else "🟡" if info["nature"] == "Pre-malignant" else "🟢"
        st.markdown(
            f"<div style='font-size:0.8rem;padding:3px 0;color:#94a3b8'>"
            f"{icon} <b style='color:#cbd5e1'>{info['full_name']}</b>"
            f" <span style='color:#475569'>({code})</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem;color:#334155;line-height:1.7;'>
        ⚠️ For educational and research use only.<br>
        Not a substitute for clinical diagnosis.<br>
        Consult a qualified dermatologist.
    </div>
    """, unsafe_allow_html=True)


# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:2rem 0 1.5rem'>
    <h1 style='font-family:Syne,sans-serif;font-size:2.4rem;font-weight:800;
               color:#e2e8f0;margin:0;letter-spacing:-1px;'>
        Skin Lesion Classifier
    </h1>
    <p style='color:#475569;margin:0.4rem 0 0;font-size:0.95rem;'>
        Upload a dermatoscopic image to classify into one of 7 lesion categories.
    </p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop an image here",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded is not None:
    img    = Image.open(uploaded)
    model  = load_model()
    tensor = preprocess(img)

    with st.spinner("Analysing…"):
        preds = model.predict(tensor, verbose=0)[0]

    pred_idx   = int(np.argmax(preds))
    pred_code  = CLASS_LABELS[pred_idx]
    pred_info  = CLASS_INFO[pred_code]
    confidence = float(preds[pred_idx])

    col_left, col_right = st.columns([1, 1.6], gap="large")

    with col_left:
        st.image(img, use_container_width=True)
        st.markdown(
            f"<div style='text-align:center;font-size:0.75rem;color:#334155;"
            f"margin-top:4px'>{uploaded.name} · {img.size[0]}×{img.size[1]}px</div>",
            unsafe_allow_html=True
        )

    with col_right:
        nature       = pred_info["nature"]
        nature_color = pred_info["color"]
        nature_bg    = pred_info["bg"]
        nature_icon  = "🔴" if nature == "Malignant" else "🟡" if nature == "Pre-malignant" else "🟢"

        st.markdown(f"""
        <div style="border:1px solid {nature_color}33;border-left:4px solid {nature_color};
                    background:{nature_bg};border-radius:12px;padding:1.4rem 1.6rem;
                    margin-bottom:1.2rem;">
            <div style="font-size:0.72rem;color:#475569;letter-spacing:1px;
                        text-transform:uppercase;margin-bottom:0.3rem;">
                Predicted Diagnosis
            </div>
            <div style="font-family:Syne,sans-serif;font-size:1.9rem;font-weight:800;
                        color:#e2e8f0;letter-spacing:-0.5px;line-height:1.1;
                        margin-bottom:0.7rem;">
                {pred_info['full_name']}
            </div>
            <div style="display:flex;gap:0.6rem;align-items:center;
                        flex-wrap:wrap;margin-bottom:0.9rem;">
                <span style="background:{nature_color}22;color:{nature_color};
                             border:1px solid {nature_color}44;padding:3px 12px;
                             border-radius:99px;font-size:0.75rem;font-weight:600;">
                    {nature_icon} {nature}
                </span>
                <span style="background:#1e3a5f;color:#94a3b8;padding:3px 12px;
                             border-radius:99px;font-size:0.75rem;">
                    ICD: {pred_info['icd']}
                </span>
                <span style="background:#1e3a5f;color:#94a3b8;padding:3px 12px;
                             border-radius:99px;font-size:0.75rem;font-weight:600;">
                    {confidence*100:.1f}% confidence
                </span>
            </div>
            <div style="font-size:0.85rem;color:#94a3b8;line-height:1.6;">
                {pred_info['description']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        runner_up_idx = np.argsort(preds)[-2]
        margin = confidence - preds[runner_up_idx]
        with m1:
            st.metric("Confidence", f"{confidence*100:.1f}%")
        with m2:
            st.metric("Runner-up", CLASS_LABELS[runner_up_idx].upper(),
                      f"{preds[runner_up_idx]*100:.1f}%")
        with m3:
            st.metric("Margin", f"{margin*100:.1f}pp")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;"
        "color:#cbd5e1;margin-bottom:0.5rem;'>Confidence Distribution</div>",
        unsafe_allow_html=True
    )
    st.plotly_chart(build_chart(preds, pred_idx), use_container_width=True)

    with st.expander("Full probability breakdown"):
        for i in np.argsort(preds)[::-1]:
            code = CLASS_LABELS[i]
            info = CLASS_INFO[code]
            bar_w = int(preds[i] * 200)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.8rem;
                        padding:0.4rem 0;border-bottom:1px solid #0f2236;">
                <div style="width:52px;font-size:0.75rem;color:#475569;font-weight:600;">{code}</div>
                <div style="flex:1;background:#0f2236;border-radius:4px;height:6px;">
                    <div style="width:{bar_w}px;max-width:100%;background:{info['color']};
                                height:6px;border-radius:4px;"></div>
                </div>
                <div style="width:90px;font-size:0.82rem;color:#94a3b8;">
                    {info['full_name'][:14]}
                </div>
                <div style="width:52px;text-align:right;font-size:0.82rem;
                            color:#cbd5e1;font-weight:600;">{preds[i]*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="border:2px dashed #1e3a5f;border-radius:16px;padding:3.5rem 2rem;
                text-align:center;margin:1rem 0;background:#0b1628;">
        <div style="font-size:2.5rem;margin-bottom:0.8rem;">🔬</div>
        <div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;
                    color:#475569;margin-bottom:0.4rem;">No image uploaded</div>
        <div style="font-size:0.85rem;color:#334155;">
            Upload a JPG or PNG dermoscopy image using the file picker above
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;"
        "color:#cbd5e1;margin:1.5rem 0 0.8rem;'>Lesion Reference</div>",
        unsafe_allow_html=True
    )
    cols = st.columns(2)
    for i, (code, info) in enumerate(CLASS_INFO.items()):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="border:1px solid #1e3a5f;border-left:3px solid {info['color']};
                        background:#0b1628;border-radius:8px;padding:0.8rem 1rem;
                        margin-bottom:0.6rem;">
                <div style="display:flex;justify-content:space-between;
                            align-items:center;margin-bottom:0.3rem;">
                    <span style="font-weight:600;color:#e2e8f0;font-size:0.88rem;">
                        {info['full_name']}
                    </span>
                    <span style="font-size:0.7rem;color:{info['color']};
                                 background:{info['color']}22;padding:1px 8px;
                                 border-radius:99px;">{info['nature']}</span>
                </div>
                <div style="font-size:0.75rem;color:#475569;">{info['description']}</div>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding-top:1rem;border-top:1px solid #1e3a5f;
            display:flex;justify-content:space-between;align-items:center;
            flex-wrap:wrap;gap:0.5rem;">
    <div style="font-size:0.75rem;color:#334155;">
        DermAI · HAM10000 Dataset · For educational use only
    </div>
    <div style="font-size:0.75rem;color:#334155;">
        Model on <a href="https://huggingface.co/evans-1/skin-cancer-ham10000"
        style="color:#3b82f6;text-decoration:none;">Hugging Face</a>
    </div>
</div>
""", unsafe_allow_html=True)