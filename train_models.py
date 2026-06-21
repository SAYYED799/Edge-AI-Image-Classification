"""
Edge AI Image Classification
Mixed Precision and Quantization-Aware Training
Dataset: CIFAR-10
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras

# ─── Setup ────────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

CLASS_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

EPOCHS = 8
BATCH_SIZE = 64

# ─── 1. Load & Preprocess CIFAR-10 ────────────────────────────────────────────
print("\n[1] Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

y_train = y_train.flatten()
y_test  = y_test.flatten()

print(f"  Train: {x_train.shape}, Test: {x_test.shape}")

# ─── 2. CNN Builder ───────────────────────────────────────────────────────────
def build_cnn(output_dtype="float32"):
    """Simple CNN: Conv→Pool→Conv→Pool→Conv→Flatten→Dense→Output"""
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                            input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D(2, 2),

        keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2, 2),

        keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),

        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),

        # Output layer always float32 (required for mixed precision)
        keras.layers.Dense(10, activation="softmax", dtype=output_dtype),
    ])
    return model

def compile_and_train(model, label):
    """Compile, train and time the model."""
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    print(f"\n  Training {label} model...")
    start = time.time()
    model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )
    elapsed = time.time() - start
    return elapsed

# ─── 3. Baseline FP32 Model ───────────────────────────────────────────────────
print("\n[2] ── BASELINE FP32 ──")

# Reset policy to float32
keras.mixed_precision.set_global_policy("float32")

fp32_model = build_cnn(output_dtype="float32")
fp32_time  = compile_and_train(fp32_model, "FP32 Baseline")

fp32_loss, fp32_acc = fp32_model.evaluate(x_test, y_test, verbose=0)
print(f"  FP32 Accuracy : {fp32_acc*100:.2f}%")
print(f"  Training Time : {fp32_time:.1f}s")

fp32_model.save("models/baseline.keras")
print("  Saved → models/baseline.keras")

# ─── 4. Mixed Precision FP16 Model ────────────────────────────────────────────
print("\n[3] ── MIXED PRECISION FP16 ──")

keras.mixed_precision.set_global_policy("mixed_float16")

fp16_model = build_cnn(output_dtype="float32")   # output layer stays fp32
fp16_time  = compile_and_train(fp16_model, "Mixed Precision FP16")

# Evaluate (cast inputs to float32 for eval compatibility)
fp16_loss, fp16_acc = fp16_model.evaluate(x_test, y_test, verbose=0)
print(f"  FP16 Accuracy : {fp16_acc*100:.2f}%")
print(f"  Training Time : {fp16_time:.1f}s")

fp16_model.save("models/mixed_precision.keras")
print("  Saved → models/mixed_precision.keras")

# Reset policy back to float32
keras.mixed_precision.set_global_policy("float32")

# ─── 5. Post-Training Quantization (TFLite INT8) ──────────────────────────────
print("\n[4] ── POST-TRAINING QUANTIZATION (TFLite) ──")

# Representative dataset for INT8 calibration
def representative_dataset():
    for i in range(200):
        sample = x_train[i:i+1].astype("float32")
        yield [sample]

converter = tf.lite.TFLiteConverter.from_keras_model(fp32_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.float32   # keep float I/O for simplicity
converter.inference_output_type = tf.float32

tflite_model = converter.convert()

with open("models/model.tflite", "wb") as f:
    f.write(tflite_model)
print("  Saved → models/model.tflite")

# ─── 6. Model Size Comparison ─────────────────────────────────────────────────
def file_size_kb(path):
    return os.path.getsize(path) / 1024

fp32_size   = file_size_kb("models/baseline.keras")
fp16_size   = file_size_kb("models/mixed_precision.keras")
tflite_size = file_size_kb("models/model.tflite")

# ─── 7. Inference Time Comparison ─────────────────────────────────────────────
def measure_inference_keras(model, samples=100):
    batch = x_test[:samples]
    start = time.time()
    model.predict(batch, verbose=0)
    return (time.time() - start) / samples * 1000  # ms per image

def measure_inference_tflite(tflite_path, samples=100):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    start = time.time()
    for i in range(samples):
        interpreter.set_tensor(inp["index"], x_test[i:i+1])
        interpreter.invoke()
    return (time.time() - start) / samples * 1000

fp32_inf   = measure_inference_keras(fp32_model)
tflite_inf = measure_inference_tflite("models/model.tflite")

# ─── 8. TFLite Accuracy ───────────────────────────────────────────────────────
def tflite_accuracy(tflite_path, n=1000):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    correct = 0
    for i in range(n):
        interpreter.set_tensor(inp["index"], x_test[i:i+1])
        interpreter.invoke()
        pred = np.argmax(interpreter.get_tensor(out["index"]))
        if pred == y_test[i]:
            correct += 1
    return correct / n

print("\n[5] Measuring TFLite accuracy (1000 samples)...")
tflite_acc = tflite_accuracy("models/model.tflite")

# ─── 9. Save Metrics for Streamlit ───────────────────────────────────────────
import json
metrics = {
    "fp32_acc":    round(fp32_acc * 100, 2),
    "fp16_acc":    round(fp16_acc * 100, 2),
    "tflite_acc":  round(tflite_acc * 100, 2),
    "fp32_time":   round(fp32_time, 1),
    "fp16_time":   round(fp16_time, 1),
    "fp32_size":   round(fp32_size, 1),
    "fp16_size":   round(fp16_size, 1),
    "tflite_size": round(tflite_size, 1),
    "fp32_inf":    round(fp32_inf, 3),
    "tflite_inf":  round(tflite_inf, 3),
}
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ─── 10. Final Summary ────────────────────────────────────────────────────────
print("\n" + "═"*55)
print("  RESULTS SUMMARY")
print("═"*55)
print(f"  {'Model':<22} {'Accuracy':>10} {'Train Time':>12} {'Size (KB)':>10}")
print("─"*55)
print(f"  {'FP32 Baseline':<22} {fp32_acc*100:>9.2f}% {fp32_time:>10.1f}s {fp32_size:>10.1f}")
print(f"  {'Mixed Precision FP16':<22} {fp16_acc*100:>9.2f}% {fp16_time:>10.1f}s {fp16_size:>10.1f}")
print(f"  {'TFLite INT8':<22} {tflite_acc*100:>9.2f}% {'—':>11} {tflite_size:>10.1f}")
print("═"*55)
print(f"\n  Inference Time:")
print(f"    FP32 Keras  : {fp32_inf:.3f} ms/image")
print(f"    TFLite INT8 : {tflite_inf:.3f} ms/image")
print("\n  All models saved in models/")
print("  Run: streamlit run app.py\n")
