"""
Edge AI Image Classification — Streamlit Dashboard
Run: streamlit run app.py
"""

import os, json, time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
from tensorflow import keras

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Edge AI · Image Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f14;
    color: #e2e8f0;
}

/* Hero gradient */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 60% 40%, rgba(99,102,241,0.12) 0%, transparent 60%);
    pointer-events: none;
}
.hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #a5b4fc;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.5px;
}
.hero p {
    color: #94a3b8;
    font-size: 1.05rem;
    max-width: 700px;
    margin: 0;
    line-height: 1.7;
}
.badge {
    display: inline-block;
    background: rgba(99,102,241,0.18);
    border: 1px solid rgba(99,102,241,0.4);
    color: #a5b4fc;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 99px;
    margin: 0.5rem 0.3rem 0.5rem 0;
    letter-spacing: 0.5px;
}

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    color: #6366f1;
    border-left: 3px solid #6366f1;
    padding-left: 12px;
    margin: 2rem 0 1rem 0;
    letter-spacing: 0.3px;
}

/* Metric cards */
.metric-card {
    background: #161b2e;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #a5b4fc;
}
.metric-card .label {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* Explanation boxes */
.explain-box {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.explain-box h4 {
    color: #818cf8;
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    margin: 0 0 0.5rem 0;
}
.explain-box p {
    color: #94a3b8;
    font-size: 0.9rem;
    line-height: 1.6;
    margin: 0;
}

/* Inference box */
.pred-box {
    background: #0f1829;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
}
.pred-label {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    color: #a5b4fc;
    font-weight: 700;
}
.actual-label {
    font-size: 0.9rem;
    color: #64748b;
    margin-top: 6px;
}
.inf-time {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #34d399;
    margin-top: 10px;
}

/* Streamlit overrides */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
}
div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}
hr { border-color: #1e293b; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
CLASS_LABELS = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

MODELS_DIR = "models"

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADER  (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    (_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_test = x_test.astype("float32") / 255.0
    y_test = y_test.flatten()
    return x_test, y_test

@st.cache_data(show_spinner=False)
def load_metrics():
    path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_resource(show_spinner=False)
def load_keras_model(path):
    keras.mixed_precision.set_global_policy("float32")
    return keras.models.load_model(path)

@st.cache_resource(show_spinner=False)
def load_tflite_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def predict_keras(model, img):
    t0 = time.time()
    preds = model.predict(img[np.newaxis], verbose=0)
    elapsed = (time.time() - t0) * 1000
    return int(np.argmax(preds)), elapsed

def predict_tflite(interpreter, img):
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    interpreter.set_tensor(inp["index"], img[np.newaxis].astype("float32"))
    t0 = time.time()
    interpreter.invoke()
    elapsed = (time.time() - t0) * 1000
    result = interpreter.get_tensor(out["index"])
    return int(np.argmax(result)), elapsed

# ──────────────────────────────────────────────────────────────────────────────
# CHART HELPERS  (dark-themed matplotlib)
# ──────────────────────────────────────────────────────────────────────────────
DARK = "#0d0f14"
CARD = "#161b2e"
ACCENT = ["#6366f1", "#34d399", "#f59e0b"]

def dark_fig(figsize=(7, 3.5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=DARK)
    ax.set_facecolor(DARK)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e293b")
    ax.tick_params(colors="#64748b", labelsize=9)
    ax.xaxis.label.set_color("#64748b")
    ax.yaxis.label.set_color("#64748b")
    ax.title.set_color("#94a3b8")
    return fig, ax

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
x_test, y_test = load_data()
metrics = load_metrics()

models_ready = (
    os.path.exists(f"{MODELS_DIR}/baseline.keras") and
    os.path.exists(f"{MODELS_DIR}/mixed_precision.keras") and
    os.path.exists(f"{MODELS_DIR}/model.tflite")
)

# ══════════════════════════════════════════════════════════════════════════════
# ── A. HERO / OVERVIEW ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>🔬 Edge AI · Image Classification</h1>
  <p>
    A complete exploration of <strong>Mixed Precision Training (FP16)</strong> and
    <strong>Post-Training Quantization (INT8)</strong> applied to a CNN trained on CIFAR-10.
    Compare model size, accuracy, training speed, and inference latency across three model variants.
  </p>
  <div style="margin-top:1rem">
    <span class="badge">TensorFlow 2.x</span>
    <span class="badge">CIFAR-10</span>
    <span class="badge">Mixed Precision</span>
    <span class="badge">TFLite INT8</span>
    <span class="badge">Edge AI</span>
    <span class="badge">Streamlit</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Aim
st.markdown('<div class="section-header">🎯 Project Aim</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="explain-box">
    <h4>Baseline (FP32)</h4>
    <p>Train a standard CNN in full 32-bit floating point as the reference benchmark.</p>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="explain-box">
    <h4>Mixed Precision (FP16)</h4>
    <p>Re-train the same architecture using 16-bit floats to speed up training and reduce memory.</p>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="explain-box">
    <h4>Quantization (INT8)</h4>
    <p>Convert the FP32 model to 8-bit integers via TFLite for ultra-fast edge deployment.</p>
    </div>""", unsafe_allow_html=True)

if not models_ready:
    st.warning("⚠️  Models not found. Please run `python train_models.py` first, then refresh this page.")
    st.stop()

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ── B. DATASET PREVIEW ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📦 CIFAR-10 Dataset</div>', unsafe_allow_html=True)
st.markdown("**10 classes · 60,000 images · 32×32 pixels · 3 channels**")

fig, axes = plt.subplots(2, 5, figsize=(12, 5), facecolor=DARK)
fig.patch.set_facecolor(DARK)

# Show one example from each class
shown = {c: False for c in range(10)}
idx = 0
grid_images = []
for i in range(len(y_test)):
    c = int(y_test[i])
    if not shown[c]:
        grid_images.append((x_test[i], c))
        shown[c] = True
    if len(grid_images) == 10:
        break

for ax, (img, label) in zip(axes.flatten(), grid_images):
    ax.imshow(img)
    ax.set_title(CLASS_LABELS[label], color="#a5b4fc", fontsize=9,
                 fontfamily="monospace", pad=4)
    ax.axis("off")

plt.tight_layout(pad=0.5)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ── C. MODEL EXPLANATIONS ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🧠 Model Explanations</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔷 CNN Architecture", "⚡ Mixed Precision", "📦 Quantization"])

with tab1:
    st.markdown("""
    **Convolutional Neural Network (CNN)** — designed for image recognition.

    | Layer | Type | Output Shape | Purpose |
    |---|---|---|---|
    | 1 | Conv2D (32 filters, 3×3) + ReLU | 32×32×32 | Extract edges & textures |
    | 2 | MaxPooling2D (2×2) | 16×16×32 | Downsample spatially |
    | 3 | Conv2D (64 filters, 3×3) + ReLU | 16×16×64 | Detect shapes |
    | 4 | MaxPooling2D (2×2) | 8×8×64 | Downsample |
    | 5 | Conv2D (128 filters, 3×3) + ReLU | 8×8×128 | High-level features |
    | 6 | Flatten | 8192 | Convert to vector |
    | 7 | Dense (128) + ReLU + Dropout | 128 | Classification head |
    | 8 | Dense (10) + Softmax | 10 | Class probabilities |
    """)

with tab2:
    st.markdown("""
    **Mixed Precision Training** uses both FP16 and FP32 in the same training run.

    - **Compute (FP16):** Convolutions and matrix multiplications run in 16-bit → **2× faster** on modern GPUs/TPUs.
    - **Storage (FP32):** Weight updates and loss scaling remain in 32-bit to avoid underflow.
    - **Memory:** FP16 activations use **half the memory** → larger batch sizes possible.
    - **TensorFlow API:** `tf.keras.mixed_precision.set_global_policy('mixed_float16')`

    > ⚠️ Output layers should always be `dtype='float32'` to avoid numerical instability.
    """)

with tab3:
    st.markdown("""
    **Post-Training Quantization (PTQ)** converts a trained FP32 model to INT8.

    - **INT8:** Weights and activations are quantized to 8-bit integers (range: −128 to 127).
    - **Calibration:** A small representative dataset is used to determine quantization ranges.
    - **Size reduction:** ~**4× smaller** model file (32 bits → 8 bits per weight).
    - **Speed:** INT8 arithmetic is **2–4× faster** on edge CPUs and microcontrollers.
    - **TensorFlow Lite:** The standard format for deploying on mobile & IoT devices.

    ```python
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    ```
    """)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ── D. PERFORMANCE COMPARISON ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📊 Performance Comparison</div>', unsafe_allow_html=True)

m = metrics  # shorthand

# Summary table
import pandas as pd
df = pd.DataFrame({
    "Model":         ["FP32 Baseline", "Mixed Precision FP16", "TFLite INT8"],
    "Accuracy (%)":  [m["fp32_acc"],   m["fp16_acc"],          m["tflite_acc"]],
    "Train Time (s)":[m["fp32_time"],  m["fp16_time"],         "—"],
    "Size (KB)":     [m["fp32_size"],  m["fp16_size"],         m["tflite_size"]],
})
st.dataframe(df, use_container_width=True, hide_index=True)

# Charts
col_a, col_b, col_c = st.columns(3)

with col_a:
    fig, ax = dark_fig((5, 3.5))
    models_x = ["FP32", "FP16", "TFLite\nINT8"]
    accs = [m["fp32_acc"], m["fp16_acc"], m["tflite_acc"]]
    bars = ax.bar(models_x, accs, color=ACCENT, width=0.5, zorder=3)
    ax.set_ylim(min(accs) - 5, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Accuracy")
    ax.grid(axis="y", color="#1e293b", zorder=0)
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v:.1f}%", ha="center", va="bottom", color="#e2e8f0", fontsize=9)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col_b:
    fig, ax = dark_fig((5, 3.5))
    times_x = ["FP32", "FP16"]
    times_y = [m["fp32_time"], m["fp16_time"]]
    bars = ax.bar(times_x, times_y, color=[ACCENT[0], ACCENT[1]], width=0.4, zorder=3)
    ax.set_ylabel("Seconds")
    ax.set_title("Training Time")
    ax.grid(axis="y", color="#1e293b", zorder=0)
    for bar, v in zip(bars, times_y):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.0f}s", ha="center", va="bottom", color="#e2e8f0", fontsize=9)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col_c:
    fig, ax = dark_fig((5, 3.5))
    sz_x = ["FP32", "FP16", "TFLite\nINT8"]
    sz_y = [m["fp32_size"], m["fp16_size"], m["tflite_size"]]
    bars = ax.bar(sz_x, sz_y, color=ACCENT, width=0.5, zorder=3)
    ax.set_ylabel("Size (KB)")
    ax.set_title("Model Size")
    ax.grid(axis="y", color="#1e293b", zorder=0)
    for bar, v in zip(bars, sz_y):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f"{v:.0f}", ha="center", va="bottom", color="#e2e8f0", fontsize=9)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# Inference time chart
st.markdown("**Inference Time (ms / image)**")
fig, ax = dark_fig((8, 3))
inf_x = ["FP32 Keras", "TFLite INT8"]
inf_y = [m["fp32_inf"], m["tflite_inf"]]
bars = ax.barh(inf_x, inf_y, color=[ACCENT[0], ACCENT[2]], height=0.4, zorder=3)
ax.set_xlabel("ms per image")
ax.set_title("Inference Latency Comparison")
ax.grid(axis="x", color="#1e293b", zorder=0)
for bar, v in zip(bars, inf_y):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f"{v:.3f} ms", va="center", color="#e2e8f0", fontsize=9)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ── E. LIVE INFERENCE DEMO ────────────────────────────────────════════════════
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🚀 Live Inference Demo</div>', unsafe_allow_html=True)

col_ctrl, col_result = st.columns([1, 2])

with col_ctrl:
    model_choice = st.selectbox(
        "Select Model",
        ["Baseline FP32", "Mixed Precision FP16", "TFLite INT8"],
    )
    img_idx = st.slider("Test Image Index", 0, len(x_test) - 1, 42)
    run_btn = st.button("▶  Run Inference", use_container_width=True)

if run_btn:
    img = x_test[img_idx]
    actual = CLASS_LABELS[int(y_test[img_idx])]

    with st.spinner("Running inference..."):
        if model_choice == "Baseline FP32":
            fp32_m = load_keras_model(f"{MODELS_DIR}/baseline.keras")
            pred_idx, inf_ms = predict_keras(fp32_m, img)
        elif model_choice == "Mixed Precision FP16":
            fp16_m = load_keras_model(f"{MODELS_DIR}/mixed_precision.keras")
            pred_idx, inf_ms = predict_keras(fp16_m, img)
        else:
            tflite_interp = load_tflite_model(f"{MODELS_DIR}/model.tflite")
            pred_idx, inf_ms = predict_tflite(tflite_interp, img)

    predicted = CLASS_LABELS[pred_idx]
    correct = predicted == actual

    with col_result:
        r1, r2 = st.columns([1, 1])
        with r1:
            # Show image
            fig_img, ax_img = plt.subplots(figsize=(3, 3), facecolor=DARK)
            ax_img.imshow(img)
            ax_img.axis("off")
            ax_img.set_title(f"Image #{img_idx}", color="#64748b", fontsize=9)
            st.pyplot(fig_img, use_container_width=True)
            plt.close(fig_img)

        with r2:
            status_icon = "✅" if correct else "❌"
            st.markdown(f"""
            <div class="pred-box">
              <div style="font-size:2.5rem; margin-bottom:8px">{status_icon}</div>
              <div class="pred-label">{predicted}</div>
              <div class="actual-label">Actual: <strong>{actual}</strong></div>
              <div class="inf-time">⏱ {inf_ms:.3f} ms</div>
              <div style="color:#64748b; font-size:0.78rem; margin-top:8px">{model_choice}</div>
            </div>
            """, unsafe_allow_html=True)
else:
    with col_result:
        st.info("Select a model and image index, then click **▶ Run Inference**.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ── F. DEPLOYMENT SECTION ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📱 Edge Deployment</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="explain-box">
    <h4>TensorFlow Lite</h4>
    <p>TFLite is a lightweight runtime for running ML models on mobile, embedded, and IoT devices.
    It supports Android, iOS, Raspberry Pi, microcontrollers, and more.
    The INT8 quantized model is ideal for deployment — it runs with minimal CPU and RAM, and
    requires no internet connection for inference.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="explain-box">
    <h4>Supported Platforms</h4>
    <p>
    🤖 <strong>Android</strong> — via TFLite Android AAR<br>
    🍎 <strong>iOS</strong> — via TFLite Swift / Obj-C<br>
    🍓 <strong>Raspberry Pi</strong> — Python TFLite runtime<br>
    🔲 <strong>Microcontrollers</strong> — TensorFlow Lite Micro<br>
    ☁️ <strong>Web</strong> — TensorFlow.js (WASM backend)
    </p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ── G. LIMITATIONS & FUTURE WORK ──────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">⚠️ Limitations & Future Work</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Current Limitations**")
    st.markdown("""
    - **Small dataset:** CIFAR-10 has only 60K images at 32×32 pixels — far below real-world complexity.
    - **Simple CNN:** The 3-layer CNN has limited capacity; real-world tasks need deeper networks.
    - **No QAT:** True Quantization-Aware Training (QAT) would yield higher INT8 accuracy than PTQ.
    - **CPU inference only:** Benefits of FP16 are most visible on GPU/TPU hardware.
    """)

with col2:
    st.markdown("**Future Improvements**")
    st.markdown("""
    - 🏗️ Replace custom CNN with **EfficientNet** or **MobileNetV3** backbone.
    - 🎯 Apply **Quantization-Aware Training (QAT)** for higher quantized accuracy.
    - 📲 Deploy TFLite model to a **Flutter mobile app** for real-time camera inference.
    - 🔬 Experiment with **pruning** (weight sparsity) combined with quantization.
    - 📊 Add **Grad-CAM** visualization to explain CNN predictions.
    """)

st.divider()

# Footer
st.markdown("""
<div style="text-align:center; color:#334155; font-family:'Space Mono',monospace; font-size:0.75rem; padding: 1rem 0">
  Edge AI · Image Classification &nbsp;|&nbsp; TensorFlow · CIFAR-10 · TFLite
</div>
""", unsafe_allow_html=True)
