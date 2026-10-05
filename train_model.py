# ======================================================
# CHEST X-RAY TRAINING SCRIPT
# MobileNetV2 + transfer learning + Grad-CAM, CPU friendly
# ======================================================

# ======================================================
# 1. IMPORT LIBRARIES
# ======================================================
import os
import json
import tensorflow as tf
import numpy as np
import cv2

# Headless mode (important for Jenkins & no display systems)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


# ======================================================
# 2. DATASET & OUTPUT PATHS
# ------------------------------------------------------
# Anyone can run this without editing the file:
#   set XRAY_DATA=D:\datasets\chest_xray
#   python train_model.py
# Falls back to a "chest_xray" folder next to this script.
# ======================================================
HERE = os.path.dirname(os.path.abspath(__file__))

BASE_PATH = os.environ.get("XRAY_DATA", os.path.join(HERE, "chest_xray"))
TRAIN_PATH = os.path.join(BASE_PATH, "train")
VAL_PATH = os.path.join(BASE_PATH, "val")
TEST_PATH = os.path.join(BASE_PATH, "test")

OUTPUT_DIR = os.environ.get("XRAY_OUT", os.path.join(HERE, "outputs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

for path in (TRAIN_PATH, VAL_PATH, TEST_PATH):
    if not os.path.isdir(path):
        raise SystemExit(
            f"Missing folder: {path}\n"
            "Point XRAY_DATA at the chest_xray folder that holds train/ val/ test/."
        )

print("Dataset Path:", BASE_PATH)
print("Output Path :", OUTPUT_DIR)


# ======================================================
# 3. PERFORMANCE SETTINGS (OPTIMIZED FOR LAPTOP CPU)
# ======================================================
IMG_SIZE = (160, 160)   # Good balance of speed + accuracy
BATCH_SIZE = 16         # Faster than 4 and still laptop friendly
EPOCHS = int(os.environ.get("XRAY_EPOCHS", 12))


# ======================================================
# 4. DATA AUGMENTATION (IMPROVES ACCURACY)
# ======================================================
# MobileNetV2 expects pixels in -1..1, which preprocess_input does.
# (Plain rescale=1./255 gives 0..1 and quietly wastes the pretrained features.)
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.15      # a real validation set, carved out of train/
)

val_test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_gen.flow_from_directory(
    TRAIN_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", subset="training"
)
# The dataset's own val/ folder holds only 16 images, far too few to judge an
# epoch by, so validation comes from 15% of train/ instead.
val_data = train_gen.flow_from_directory(
    TRAIN_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", subset="validation", shuffle=False
)
test_data = val_test_gen.flow_from_directory(
    TEST_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False
)

NUM_CLASSES = train_data.num_classes
CLASS_LABELS = list(train_data.class_indices.keys())
print("Detected Classes:", CLASS_LABELS)


# ======================================================
# 4b. CLASS WEIGHTS  <-- fixes the low specificity
# ------------------------------------------------------
# The training set holds far more PNEUMONIA than NORMAL images, so an
# unweighted model learns that guessing "pneumonia" usually pays. These
# weights make a mistake on the rarer class cost proportionally more.
# ======================================================
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(weights))
print("Class weights:", {CLASS_LABELS[i]: round(w, 3) for i, w in class_weights.items()})


# ======================================================
# 5. LOAD PRETRAINED MOBILENETV2
# ------------------------------------------------------
# Downloads ImageNet weights once and caches them. For an offline or
# Jenkins box, set XRAY_WEIGHTS to a local .h5 file instead.
# ======================================================
WEIGHTS_PATH = os.environ.get("XRAY_WEIGHTS", "")

if WEIGHTS_PATH and os.path.isfile(WEIGHTS_PATH):
    print("Loading pretrained weights from:", WEIGHTS_PATH)
    base_model = MobileNetV2(
        weights=None, include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.load_weights(WEIGHTS_PATH)
else:
    print("Loading ImageNet weights (downloaded and cached by Keras)")
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

print("Pretrained weights loaded successfully!")


# ======================================================
# 6. FINE-TUNING STRATEGY
# ======================================================
for layer in base_model.layers[:-40]:
    layer.trainable = False
for layer in base_model.layers[-40:]:
    layer.trainable = True


# ======================================================
# 7. CUSTOM CLASSIFICATION HEAD
# ======================================================
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)


# ======================================================
# 8. COMPILE MODEL
# ======================================================
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()


# ======================================================
# 9. TRAIN MODEL
# ------------------------------------------------------
# This dataset's val/ folder holds only a handful of images, so val_loss
# is noisy. Patience of 3 stops the run reacting to that noise.
# ======================================================
early_stop = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)

print("\nStarting training...")
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    class_weight=class_weights,
    callbacks=[early_stop]
)


# ======================================================
# 10. FINAL EVALUATION
# ------------------------------------------------------
# Accuracy alone is misleading on an imbalanced test set, so sensitivity
# and specificity are reported too. For screening, sensitivity (catching
# real pneumonia) matters most, with specificity close behind.
# ======================================================
loss, accuracy = model.evaluate(test_data)
print("\nFinal Test Accuracy (threshold 0.5):", accuracy)

PNE = CLASS_LABELS.index("PNEUMONIA") if "PNEUMONIA" in CLASS_LABELS else 1

# Pick the decision threshold on VALIDATION data, never on test.
# Youden's J = sensitivity + specificity - 1, i.e. the best balance of the two.
val_true = val_data.classes
val_prob = model.predict(val_data)[:, PNE]
best_t, best_j = 0.5, -1.0
for t in np.arange(0.05, 0.96, 0.01):
    pred = (val_prob >= t).astype(int)
    tp = int(((pred == 1) & (val_true == PNE)).sum())
    fn = int(((pred == 0) & (val_true == PNE)).sum())
    tn = int(((pred == 0) & (val_true != PNE)).sum())
    fp = int(((pred == 1) & (val_true != PNE)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    if sens + spec - 1 > best_j:
        best_j, best_t = sens + spec - 1, float(t)
print(f"\nChosen decision threshold (from validation data): {best_t:.2f}")

y_true = test_data.classes
y_pred_prob = model.predict(test_data)
y_pred = (y_pred_prob[:, PNE] >= best_t).astype(int)
if PNE == 0:
    y_pred = 1 - y_pred

report = classification_report(y_true, y_pred, target_names=CLASS_LABELS, digits=3)
print("\nClassification Report:\n")
print(report)

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix (rows = true, columns = predicted):\n", cm)

metrics = {"test_accuracy": float((y_pred == y_true).mean()), "classes": CLASS_LABELS,
           "threshold": best_t, "confusion_matrix": cm.tolist()}

if NUM_CLASSES == 2:
    pos = CLASS_LABELS.index("PNEUMONIA") if "PNEUMONIA" in CLASS_LABELS else 1
    neg = 1 - pos
    tp = cm[pos][pos]
    fn = cm[pos][neg]
    tn = cm[neg][neg]
    fp = cm[neg][pos]
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    metrics.update(sensitivity=float(sensitivity), specificity=float(specificity),
                   true_positives=int(tp), false_negatives=int(fn),
                   true_negatives=int(tn), false_positives=int(fp))
    print(f"\nSensitivity (pneumonia caught) : {sensitivity:.3f}  "
          f"-> {fn} of {tp + fn} pneumonia cases missed")
    print(f"Specificity (normal correct)   : {specificity:.3f}  "
          f"-> {fp} of {tn + fp} healthy X-rays flagged")

# Save the numbers so they never have to be read off a graph again
with open(os.path.join(OUTPUT_DIR, "results.txt"), "w") as f:
    f.write(f"Decision threshold: {best_t:.2f}\n")
    f.write(f"Test accuracy: {(y_pred == y_true).mean():.4f}\n\n{report}\n")
    f.write(f"Confusion matrix (rows = true, cols = predicted):\n{cm}\n")
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)


# ======================================================
# 11. SAVE MODEL & OUTPUT GRAPHS (HEADLESS SAFE)
# ======================================================
model_path = os.path.join(OUTPUT_DIR, "pneumonia_model.keras")
model.save(model_path)
print("Model saved to:", model_path)

# Accuracy Graph
plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"), dpi=120, bbox_inches="tight")
plt.close()

# Loss Graph
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=120, bbox_inches="tight")
plt.close()

# Confusion Matrix -- with the counts printed inside each cell,
# so the numbers can be read instead of guessed from the colours.
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(range(NUM_CLASSES), CLASS_LABELS, rotation=45)
plt.yticks(range(NUM_CLASSES), CLASS_LABELS)
plt.xlabel("Predicted")
plt.ylabel("True")
threshold = cm.max() / 2.0
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        plt.text(j, i, str(cm[i][j]), ha="center", va="center",
                 color="white" if cm[i][j] > threshold else "black", fontsize=14)
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=120, bbox_inches="tight")
plt.close()


# ======================================================
# 12. GRAD-CAM VISUALIZATION (SAVED AS IMAGE)
# ======================================================
print("\nGenerating Grad-CAM Visualization...")

sample_image_path = os.path.join(TEST_PATH, test_data.filenames[0])

img = tf.keras.preprocessing.image.load_img(sample_image_path, target_size=IMG_SIZE)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])


def grad_cam(model, img_array, last_conv_layer_name, class_index):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap + 1e-8)
    return heatmap.numpy()


heatmap = grad_cam(model, img_array, "Conv_1", predicted_class)

heatmap = np.uint8(255 * heatmap)
heatmap = cv2.resize(heatmap, IMG_SIZE)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

img_cv = cv2.imread(sample_image_path)
img_cv = cv2.resize(img_cv, IMG_SIZE)
overlay = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

gradcam_path = os.path.join(OUTPUT_DIR, "gradcam_result.png")
cv2.imwrite(gradcam_path, overlay)

print("Grad-CAM image saved at:", gradcam_path)
print("\nTraining Completed Successfully!")
print("All outputs saved in:", OUTPUT_DIR)
