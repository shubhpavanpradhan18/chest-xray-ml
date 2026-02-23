# ======================================================
# FINAL OPTIMIZED CHEST X-RAY TRAINING SCRIPT
# FIXED FOR USERNAME = heman (Local + Jenkins Compatible)
# Includes: Grad-CAM + High Accuracy + Stable CPU Training
# ======================================================

# ======================================================
# 1. IMPORT LIBRARIES
# ======================================================
import os
import tensorflow as tf
import numpy as np
import cv2

# Headless mode (important for Jenkins & no display systems)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix


# ======================================================
# 2. DATASET & OUTPUT PATHS (LOCAL WINDOWS SETUP)
# ======================================================
BASE_PATH = r"C:\jenkins-ml\datasets\chest_xray"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
VAL_PATH   = os.path.join(BASE_PATH, "val")
TEST_PATH  = os.path.join(BASE_PATH, "test")

# Output folder (model + graphs + gradcam)
OUTPUT_DIR = r"C:\jenkins-ml\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Dataset Path:", BASE_PATH)


# ======================================================
# 3. PERFORMANCE SETTINGS (OPTIMIZED FOR LAPTOP CPU)
# ======================================================
IMG_SIZE = (160, 160)   # Good balance of speed + accuracy
BATCH_SIZE = 4          # Low RAM usage (safe for laptops)
EPOCHS = 6              # Faster training with good accuracy


# ======================================================
# 4. DATA AUGMENTATION (IMPROVES ACCURACY)
# ======================================================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

val_test_gen = ImageDataGenerator(rescale=1./255)

# Load training data
train_data = train_gen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Load validation data
val_data = val_test_gen.flow_from_directory(
    VAL_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Load test data
test_data = val_test_gen.flow_from_directory(
    TEST_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = train_data.num_classes
CLASS_LABELS = list(train_data.class_indices.keys())

print("Detected Classes:", CLASS_LABELS)


# ======================================================
# 5. LOAD PRETRAINED MOBILENETV2 WEIGHTS (USERNAME = heman)
# ======================================================
WEIGHTS_PATH = r"C:\Users\heman\.keras\models\mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5"

print("Loading pretrained weights from:", WEIGHTS_PATH)

# Initialize base model WITHOUT auto-download
base_model = MobileNetV2(
    weights=None,  # Prevents internet download & Jenkins cache issues
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Manually load local weights
base_model.load_weights(WEIGHTS_PATH)
print("Pretrained ImageNet weights loaded successfully!")


# ======================================================
# 6. FINE-TUNING STRATEGY (FOR 80%+ ACCURACY)
# ======================================================
# Freeze early layers (generic features)
for layer in base_model.layers[:-40]:
    layer.trainable = False

# Unfreeze last layers (learn X-ray patterns)
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
# 9. TRAIN MODEL (EARLY STOPPING = FASTER + STABLE)
# ======================================================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

print("\nStarting Optimized Training (Fast + High Accuracy)...")

history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    callbacks=[early_stop]
)


# ======================================================
# 10. FINAL EVALUATION
# ======================================================
loss, accuracy = model.evaluate(test_data)
print("\nFinal Test Accuracy:", accuracy)

y_true = test_data.classes
y_pred_prob = model.predict(test_data)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=CLASS_LABELS))

cm = confusion_matrix(y_true, y_pred)


# ======================================================
# 11. SAVE MODEL & OUTPUT GRAPHS (HEADLESS SAFE)
# ======================================================
model_path = os.path.join(OUTPUT_DIR, "optimized_model.keras")
model.save(model_path)

# Accuracy Graph
plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"))
plt.close()

# Loss Graph
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
plt.close()

# Confusion Matrix
plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(range(NUM_CLASSES), CLASS_LABELS, rotation=45)
plt.yticks(range(NUM_CLASSES), CLASS_LABELS)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()


# ======================================================
# 12. GRAD-CAM VISUALIZATION (SAVED AS IMAGE)
# ======================================================
print("\nGenerating Grad-CAM Visualization...")

sample_image_path = os.path.join(TEST_PATH, test_data.filenames[0])

img = tf.keras.preprocessing.image.load_img(
    sample_image_path, target_size=IMG_SIZE
)
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

heatmap = grad_cam(
    model,
    img_array,
    last_conv_layer_name="Conv_1",
    class_index=predicted_class
)

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