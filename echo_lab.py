# ============================================================
# 📣 EchoReport Lab — Trains CNN and Generates Unified HTML
# Calls echo_report() from report_dual_html module
# ============================================================
print("🔥 EchoLab script started")


import os
import sys
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical
from keras.datasets import mnist

# Unified HTML report generator with TSNE + history
from echo_report.report_dual_html import report_dual_html as echo_report

# ------------------------------------------------------------
# 1. Environment Setup
# ------------------------------------------------------------
print(f"Interpreter: {sys.executable}")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ------------------------------------------------------------
# 2. Load and Preprocess MNIST
# ------------------------------------------------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(f"🧪 Training samples: {X_train.shape[0]}")
print(f"🧪 Test samples: {X_test.shape[0]}")

X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]
print(f"✅ Preprocessing complete — {num_classes} classes")

# ------------------------------------------------------------
# 3. Define CNN Model
# ------------------------------------------------------------
def convolutional_model():
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(16, (5, 5), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(8, (2, 2), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

print("📦 Model constructed. Preparing to train...")
model = convolutional_model()

# ------------------------------------------------------------
# 4. Train Model
# ------------------------------------------------------------
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=10,
                    batch_size=200,
                    verbose=2)
print("✅ Training finished.")

# ------------------------------------------------------------
# 5. Evaluate Model
# ------------------------------------------------------------
scores = model.evaluate(X_test, y_test, verbose=0)
print(f"📊 Accuracy: {scores[1]*100:.2f}%")
print(f"❌ Error Rate: {100 - scores[1]*100:.2f}%")

# ------------------------------------------------------------
# 6. Generate Report via echo_report()
# ------------------------------------------------------------
echo_report(
    model=model,
    history=history,
    scores=scores,
    X_embed=X_test,
    y_embed=y_test,
    dataset_info="MNIST handwritten digits — civic benchmark",
    serial_no=1,
    notes=[
        "EchoReport Lab — symbolic archive of CNN training",
        "Reconstructed package version"
    ]
)
