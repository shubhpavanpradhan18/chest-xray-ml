"""Chest X-ray pneumonia screening demo — research prototype, not a diagnostic tool."""

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import matplotlib.cm as cm

WEIGHTS_PATH = "pneumonia_weights.h5"
IMG_SIZE = (160, 160)
CLASSES = ["NORMAL", "PNEUMONIA"]
UNCERTAIN_LOW = 0.35      # below this: call it normal; above the threshold: pneumonia

st.set_page_config(page_title="Chest X-ray pneumonia screening", page_icon="🫁", layout="wide")


@st.cache_resource
def load_model():
    """Rebuild the training architecture, then load the trained weights.

    Rebuilding in code (instead of loading a .keras file) keeps the demo
    working across TensorFlow versions -- a saved model carries serialized
    layer config that often fails to reload on a different version.
    """
    base = tf.keras.applications.MobileNetV2(
        weights=None, include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)
    model = tf.keras.models.Model(inputs=base.input, outputs=out)
    model.load_weights(WEIGHTS_PATH)
    return model


def looks_like_xray(img):
    """Cheap sanity check on the upload.

    The model will answer confidently for ANY image -- a holiday photo included --
    because softmax always sums to 1. These checks catch the obvious cases:
    an X-ray is greyscale, roughly portrait or square, and reasonably large.
    Returns (ok, reason).
    """
    w, h = img.size
    if min(w, h) < 100:
        return False, f"This image is only {w}x{h} pixels. X-rays are much larger."

    ratio = w / h
    if ratio < 0.5 or ratio > 2.0:
        return False, f"This image is {ratio:.1f}:1. Chest X-rays are closer to square."

    small = np.array(img.convert("RGB").resize((64, 64))).astype("float32")
    colour_spread = float(np.abs(small - small.mean(axis=2, keepdims=True)).mean())
    if colour_spread > 12:
        return False, "This image is in colour. Chest X-rays are greyscale."

    grey = small.mean(axis=2)
    if grey.std() < 12:
        return False, "This image is nearly uniform, with none of the contrast an X-ray has."

    # Scans, screenshots and documents are mostly white paper; X-rays are not.
    if float(np.mean(grey > 230)) > 0.45:
        return False, "This looks like a document or screenshot rather than a radiograph."

    return True, ""


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

    ok, reason = looks_like_xray(img)
    if not ok:
        st.error(f"This does not look like a chest X-ray. {reason}")
        st.caption(
            "The model has only ever seen chest X-rays, and it will still produce a confident "
            "answer for any image you give it. That answer would be meaningless."
        )
        show_image(st, img, "Uploaded image")
        if not st.checkbox("Analyse it anyway (the result will not mean anything)"):
            st.stop()

    x = prepare(img)
    model = load_model()
    probs = model.predict(x, verbose=0)[0]
    threshold = load_threshold()
    p_pneumonia = float(probs[1])
    if p_pneumonia >= threshold:
        idx, verdict, tone = 1, "PNEUMONIA", "error"
    elif p_pneumonia >= UNCERTAIN_LOW:
        idx, verdict, tone = 1, "UNCERTAIN", "warning"
    else:
        idx, verdict, tone = 0, "NORMAL", "success"

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

    st.subheader(f"Result: {verdict}")
    if verdict == "UNCERTAIN":
        st.warning(
            f"The model puts pneumonia at {p_pneumonia:.0%}, between the two decision points "
            f"({UNCERTAIN_LOW:.0%} and {threshold:.0%}). It is not confident either way, so this "
            "needs a radiologist rather than a label."
        )
    elif verdict == "PNEUMONIA":
        st.error("Signs consistent with pneumonia. This is a flag for review, not a diagnosis.")
    else:
        st.success("No signs of pneumonia found by the model.")
    for name, p in zip(CLASSES, probs):
        st.write(f"{name}: {p:.1%}")
        st.progress(float(p))

    st.caption(
        "Measured on 624 held-out images from one public paediatric dataset: sensitivity 0.97, "
        "specificity 0.84. It missed 13 of 390 pneumonia cases and flagged 37 of 234 healthy ones. "
        "Performance on adult X-rays, other hospitals or other machines is untested and may be "
        "considerably worse."
    )
else:
    st.info("Upload an X-ray to see a prediction and the region the model focused on.")
