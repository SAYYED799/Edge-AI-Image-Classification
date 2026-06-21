# 🔬 Edge AI Image Classification
### Mixed Precision Training & Post-Training Quantization on CIFAR-10

---

## 📌 Project Description

This mini-project demonstrates **Edge AI optimization techniques** applied to image classification:

- **Baseline FP32** — Standard CNN trained in full 32-bit floating point
- **Mixed Precision FP16** — Same CNN trained with 16-bit floats for speed
- **TFLite INT8** — Post-training quantization for edge/mobile deployment

Dataset: **CIFAR-10** (10 classes, 60,000 images, 32×32 pixels)

---

## ✨ Features

| Feature | Details |
|---|---|
| Dataset | CIFAR-10 via `keras.datasets` |
| Model | Custom 3-block CNN (Conv→Pool×2 + Dense) |
| Mixed Precision | `tf.keras.mixed_precision` FP16 policy |
| Quantization | TFLite Post-Training INT8 quantization |
| Metrics | Accuracy, training time, model size, inference latency |
| Dashboard | Interactive Streamlit web app with live inference |

---

## 📁 Project Structure

```
edge_ai_project/
│
├── train_models.py        ← Train all 3 models & save metrics
├── app.py                 ← Streamlit dashboard
├── requirements.txt       ← Python dependencies
├── README.md              ← This file
│
└── models/                ← Created after training
    ├── baseline.keras
    ├── mixed_precision.keras
    ├── model.tflite
    └── metrics.json
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train all models
```bash
python train_models.py
```

This will:
- Download CIFAR-10 automatically
- Train FP32 baseline model (~5–10 min on CPU)
- Train Mixed Precision FP16 model
- Convert to TFLite INT8
- Save all models + `metrics.json` in the `models/` folder
- Print a results summary table

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📊 Expected Output

After training, you will see a summary like:

```
═══════════════════════════════════════════════════════
  RESULTS SUMMARY
═══════════════════════════════════════════════════════
  Model                    Accuracy   Train Time   Size (KB)
─────────────────────────────────────────────────────────
  FP32 Baseline              72.10%       340.2s    1820.5
  Mixed Precision FP16       71.85%       290.1s    1820.5
  TFLite INT8                71.40%            —      471.2
═══════════════════════════════════════════════════════

  Inference Time:
    FP32 Keras  : 1.842 ms/image
    TFLite INT8 : 0.612 ms/image
```

> Actual values will vary based on your hardware.

---

## 🧠 Concepts Covered

### Convolutional Neural Network (CNN)
A deep learning architecture optimized for images. Uses convolutional filters to detect spatial features like edges, shapes, and textures hierarchically.

### Mixed Precision Training (FP16)
Uses 16-bit floating point for compute-heavy operations (convolutions) and 32-bit for weight updates. Reduces memory usage and speeds up training on modern hardware.

### Post-Training Quantization (INT8)
Converts model weights from 32-bit floats to 8-bit integers after training. Results in ~4× smaller model size and faster inference, ideal for edge/mobile deployment.

### TensorFlow Lite
A runtime for running TensorFlow models on mobile phones, microcontrollers, and IoT devices with minimal resources.

---

## ⚠️ Limitations

- CIFAR-10 is a small, simple benchmark — not representative of real-world complexity
- The custom CNN has limited capacity compared to ResNet / EfficientNet
- FP16 speed benefits are most visible on GPU hardware (NVIDIA Tensor Cores)

## 🔮 Future Work

- Replace CNN with MobileNetV3 or EfficientNet backbone
- Apply Quantization-Aware Training (QAT) for better INT8 accuracy
- Deploy TFLite model in a Flutter or React Native mobile app
- Add model pruning for further compression

---

## 📦 Requirements

- Python 3.8+
- TensorFlow 2.13+
- Streamlit 1.32+
- NumPy, Matplotlib, Pandas
