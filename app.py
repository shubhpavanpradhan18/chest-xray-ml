"""Chest X-ray pneumonia screening demo — research prototype, not a diagnostic tool."""

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import matplotlib.cm as cm

MODEL_PATH = "pneumonia_model.keras"
IMG_SIZE = (160, 160)
CLASSES = ["NORMAL", "PNEUMONIA"]

st.set_page_config(page_title="Chest X-ray pneumonia screening", page_icon="🫁", layout="wide")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def prepare(img):
    """Same preprocessing the model was trained with: pixels scaled to -1..1."""
    img = img.convert("RGB").resize(IMG_SIZE)
    return preprocess_input(np.expand_dims(np.array(img).astype("float32"), axis=0))


def load_threshold(default=0.5):
    """Decision threshold chosen on validation data during training."""
    try:
        import json
        with open("metrics.json") as f:
            return float(json.load(f).get("threshold", default))
    except Exception:
        return default


def grad_cam(model, x, class_index, layer_name="Conv_1"):
    """Heatmap of the regions that drove the prediction.

    The last layer's softmax is switched off for this pass: at 99% confidence
    softmax output is flat at 1.0, its gradient is zero, and the heatmap comes
    out blank. The raw score before softmax still has a usable gradient.
    """
    last = model.layers[-1]
    saved_activation = last.activation
    last.activation = tf.keras.activations.linear
    try:
        grad_model = tf.keras.models.Model(
            model.inputs, [model.get_layer(layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_out, scores = grad_model(x, training=False)
            loss = scores[:, class_index]
        grads = tape.gradient(loss, conv_out)
        if grads is None:
            return None
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis]).numpy()
    finally:
        last.activation = saved_activation

    cam = np.maximum(cam, 0)
    if cam.max() <= 1e-8:          # nothing positive: fall back to magnitude
        cam = np.abs(cam)
    if cam.max() <= 1e-8:
        return None
    cam = cam / cam.max()
    return np.array(Image.fromarray((cam * 255).astype("uint8")).resize(IMG_SIZE))


def colourise(cam):
    """Grey heatmap values -> the same jet colours used in the overlay."""
    return (cm.jet(cam / 255.0)[..., :3] * 255).astype("uint8")


def overlay(img, cam, alpha=0.4):
    base = np.array(img.convert("RGB").resize(IMG_SIZE)) / 255.0  # display only
    heat = cm.jet(cam / 255.0)[..., :3]
    return np.clip(base * (1 - alpha) + heat * alpha, 0, 1)


def show_image(target, image, caption):
    """st.image renamed this argument between Streamlit versions, so try both."""
    try:
        target.image(image, caption=caption, use_container_width=True)
    except TypeError:
        target.image(image, caption=caption, use_column_width=True)


st.title("Chest X-ray pneumonia screening")
st.caption(
    "MobileNetV2 with transfer learning, trained on a public paediatric chest X-ray dataset. "
    "Research prototype for demonstration only — not a diagnostic tool, and not for real patient care."
)

uploaded = st.file_uploader("Upload a chest X-ray (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    x = prepare(img)
    model = load_model()
    probs = model.predict(x, verbose=0)[0]
    threshold = load_threshold()
    idx = 1 if probs[1] >= threshold else 0

    left, middle, right = st.columns(3)
    show_image(left, img, "Uploaded X-ray")

    try:
        cam = grad_cam(model, x, idx)
        if cam is None:
            raise RuntimeError("no usable gradient")
        show_image(middle, colourise(cam), "Grad-CAM heatmap")
        show_image(right, overlay(img, cam), "Overlay")
    except Exception:
        middle.info("Heatmap unavailable for this model structure.")

    st.subheader(f"Prediction: {CLASSES[idx]}")
    for name, p in zip(CLASSES, probs):
        st.write(f"{name}: {p:.1%}")
        st.progress(float(p))

    st.warning(
        "On the held-out test set this model catches almost every pneumonia case "
        "but flags roughly a third of healthy X-rays as pneumonia. Treat a PNEUMONIA "
        "result as 'worth a radiologist's look', never as a diagnosis."
    )
else:
    st.info("Upload an X-ray to see a prediction and the region the model focused on.")
