#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn scipy tqdm')


# In[5]:


import tensorflow as tf
print(tf.__version__)
print("GPU Available:", tf.config.list_physical_devices("GPU"))


# In[6]:


# ============================================================
# IMPORT LIBRARIES
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.signal import lfilter

import tensorflow as tf
from tensorflow.keras import layers, models

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices("GPU"))


# In[8]:


# ============================================================
# DATASET PATH
# ============================================================

DATA_DIR = r"E:\IQ-OTH"  # CHANGE THIS

IMG_SIZE = 300
BATCH_SIZE = 16
EPOCHS = 15
SEED = 42
NUM_CLASSES = 2

CLASS_NAMES = ["Benign", "Malignant"]

tf.random.set_seed(SEED)
np.random.seed(SEED)

print("Dataset folders:", os.listdir(DATA_DIR))


# In[9]:


# ============================================================
# LOAD IMAGE PATHS AND LABELS
# ============================================================

image_paths = []
labels = []

for class_index, class_name in enumerate(CLASS_NAMES):
    class_folder = os.path.join(DATA_DIR, class_name)

    if not os.path.exists(class_folder):
        raise FileNotFoundError(f"Folder not found: {class_folder}")

    for filename in os.listdir(class_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            image_paths.append(os.path.join(class_folder, filename))
            labels.append(class_index)

image_paths = np.array(image_paths)
labels = np.array(labels)

print("Total images:", len(image_paths))

for i, name in enumerate(CLASS_NAMES):
    print(name, ":", np.sum(labels == i))


# In[10]:


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.30,
    random_state=SEED,
    stratify=labels
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_labels
)

print("Train images:", len(train_paths))
print("Validation images:", len(val_paths))
print("Test images:", len(test_paths))


# In[11]:


# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def normalize_01(img):
    img = img.astype(np.float32)

    if img.max() > 1.0:
        img = img / 255.0

    return np.clip(img, 0.0, 1.0)


def baseline_none(img):
    return normalize_01(img)


def baseline_gaussian(img):
    img = normalize_01(img)

    out = cv2.GaussianBlur(
        img,
        ksize=(5, 5),
        sigmaX=0
    )

    return np.clip(out, 0.0, 1.0)


def baseline_laplacian(img):
    img = normalize_01(img)

    img_uint8 = np.uint8(img * 255.0)

    lap = cv2.Laplacian(
        img_uint8,
        cv2.CV_64F,
        ksize=3
    )

    lap = np.abs(lap)
    lap = lap / (lap.max() + 1e-8)
    lap = lap.astype(np.float32)

    out = 0.7 * img + 0.3 * lap

    return np.clip(out, 0.0, 1.0)


def bilinear_first_order_coefficients(K, Tn, Td, Ts):
    alpha_n = 2.0 * Tn / Ts
    alpha_d = 2.0 * Td / Ts

    b0 = K * (1.0 + alpha_n) / (1.0 + alpha_d)
    b1 = K * (1.0 - alpha_n) / (1.0 + alpha_d)
    a1 = (1.0 - alpha_d) / (1.0 + alpha_d)

    return b0, b1, a1


def fast_iir_filter_image(img, b0, b1, a1):
    """
    Fast first-order IIR filtering using scipy.signal.lfilter.
    Difference equation:
    y[n] = b0*x[n] + b1*x[n-1] - a1*y[n-1]
    """

    b = [b0, b1]
    a = [1.0, a1]

    row_filtered = lfilter(b, a, img, axis=1)
    col_filtered = lfilter(b, a, row_filtered, axis=0)

    return col_filtered.astype(np.float32)


def baseline_laglead(img):
    img = normalize_01(img)

    K = 1.0
    T1 = 0.1
    T2 = 0.2
    Ts = 1.0

    # Lag: (T1*s + 1) / (T2*s + 1)
    b0_lag, b1_lag, a1_lag = bilinear_first_order_coefficients(
        K=K,
        Tn=T1,
        Td=T2,
        Ts=Ts
    )

    # Lead: (T2*s + 1) / (T1*s + 1)
    b0_lead, b1_lead, a1_lead = bilinear_first_order_coefficients(
        K=K,
        Tn=T2,
        Td=T1,
        Ts=Ts
    )

    lag_img = fast_iir_filter_image(
        img,
        b0_lag,
        b1_lag,
        a1_lag
    )

    lead_img = fast_iir_filter_image(
        lag_img,
        b0_lead,
        b1_lead,
        a1_lead
    )

    lead_img = lead_img - lead_img.min()
    lead_img = lead_img / (lead_img.max() + 1e-8)

    return np.clip(lead_img, 0.0, 1.0)


def apply_preprocess(img, method):
    if method == "none":
        return baseline_none(img)

    elif method == "gaussian":
        return baseline_gaussian(img)

    elif method == "laplacian":
        return baseline_laplacian(img)

    elif method == "laglead":
        return baseline_laglead(img)

    else:
        raise ValueError("Unknown preprocessing method: " + method)


# In[14]:


# ============================================================
# PRECOMPUTE AND SAVE PREPROCESSED IMAGES
# ============================================================

SAVE_DIR = r"E:\frontiers in SP\Frontiers SP\ALLCOMMENTS\IQOTH"   # CHANGE THIS

selected_methods = [
    "none",
    "gaussian",
    "laplacian",
    "laglead"
]

for method in selected_methods:

    print("\n==========================================")
    print("Processing method:", method)
    print("==========================================")

    for class_name in CLASS_NAMES:

        input_folder = os.path.join(DATA_DIR, class_name)
        output_folder = os.path.join(SAVE_DIR, method, class_name)

        os.makedirs(output_folder, exist_ok=True)

        image_files = [
            f for f in os.listdir(input_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]

        for filename in tqdm(image_files):

            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            img = cv2.imread(input_path)

            if img is None:
                print("Could not read:", input_path)
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            processed = apply_preprocess(img, method)

            processed_uint8 = np.uint8(processed * 255.0)
            processed_bgr = cv2.cvtColor(processed_uint8, cv2.COLOR_RGB2BGR)

            cv2.imwrite(output_path, processed_bgr)

print("\nAll preprocessing completed.")


# In[15]:


# ============================================================
# DATASET LOADER FOR TRAINING
# ============================================================

def load_image_for_training(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(
        img,
        channels=3,
        expand_animations=False
    )

    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32)

    label = tf.one_hot(label, NUM_CLASSES)

    return img, label


def create_tf_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    ds = ds.map(
        load_image_for_training,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:
        ds = ds.shuffle(
            buffer_size=len(paths),
            seed=SEED
        )

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# In[16]:


# ============================================================
# GET PATHS FROM PREPROCESSED FOLDER
# ============================================================

def get_paths_labels_from_folder(method_folder):

    paths = []
    labels_list = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        class_folder = os.path.join(method_folder, class_name)

        for filename in os.listdir(class_folder):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                paths.append(os.path.join(class_folder, filename))
                labels_list.append(class_index)

    return np.array(paths), np.array(labels_list)


# In[17]:


# ============================================================
# BUILD EFFICIENTNETB3 MODEL
# ============================================================

def build_efficientnetb3_model(num_classes=2):

    inputs = tf.keras.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    base_model = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs
    )

    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# In[18]:


# ============================================================
# CORRECT EVALUATION FUNCTION
# ============================================================

def evaluate_model_correctly(model, test_ds):

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)

        preds = np.argmax(probs, axis=1)
        true_labels = np.argmax(labels.numpy(), axis=1)

        y_true.extend(true_labels)
        y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    sensitivity = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    print("Confusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0
    ))

    return acc, precision, sensitivity, f1, cm


# In[19]:


# ============================================================
# TRAIN ONE METHOD
# Change METHOD each time:
# "none", "gaussian", "laplacian", "laglead"
# ============================================================

METHOD = "laglead"   # CHANGE THIS

METHOD_DIR = os.path.join(SAVE_DIR, METHOD)

paths, labels = get_paths_labels_from_folder(METHOD_DIR)

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    paths,
    labels,
    test_size=0.30,
    random_state=SEED,
    stratify=labels
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_labels
)

train_ds = create_tf_dataset(
    train_paths,
    train_labels,
    training=True
)

val_ds = create_tf_dataset(
    val_paths,
    val_labels,
    training=False
)

test_ds = create_tf_dataset(
    test_paths,
    test_labels,
    training=False
)

class_weights_values = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = dict(enumerate(class_weights_values))

print("Class weights:", class_weights)

model = build_efficientnetb3_model(NUM_CLASSES)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    verbose=1
)

acc, precision, sensitivity, f1, cm = evaluate_model_correctly(
    model,
    test_ds
)

result_df = pd.DataFrame([{
    "Preprocessing": METHOD,
    "Accuracy": acc,
    "Precision": precision,
    "Sensitivity": sensitivity,
    "F1_score": f1
}])

display(result_df)

result_df.to_csv(
    f"result_{METHOD}.csv",
    index=False
)

model.save(
    f"model_{METHOD}.keras"
)

print("E:\frontiers in SP\Frontiers SP\ALLCOMMENTS\IQOTH", METHOD)


# In[21]:


# ============================================================
# TRAIN ONE METHOD
# Change METHOD each time:
# "none", "gaussian", "laplacian", "laglead"
# ============================================================

METHOD = "gaussian"   # CHANGE THIS

METHOD_DIR = os.path.join(SAVE_DIR, METHOD)

paths, labels = get_paths_labels_from_folder(METHOD_DIR)

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    paths,
    labels,
    test_size=0.30,
    random_state=SEED,
    stratify=labels
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_labels
)

train_ds = create_tf_dataset(
    train_paths,
    train_labels,
    training=True
)

val_ds = create_tf_dataset(
    val_paths,
    val_labels,
    training=False
)

test_ds = create_tf_dataset(
    test_paths,
    test_labels,
    training=False
)

class_weights_values = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = dict(enumerate(class_weights_values))

print("Class weights:", class_weights)

model = build_efficientnetb3_model(NUM_CLASSES)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    verbose=1
)

acc, precision, sensitivity, f1, cm = evaluate_model_correctly(
    model,
    test_ds
)

result_df = pd.DataFrame([{
    "Preprocessing": METHOD,
    "Accuracy": acc,
    "Precision": precision,
    "Sensitivity": sensitivity,
    "F1_score": f1
}])

display(result_df)

result_df.to_csv(
    f"result_{METHOD}.csv",
    index=False
)

model.save(
    f"model_{METHOD}.keras"
)

print("E:\frontiers in SP\Frontiers SP\ALLCOMMENTS\IQOTH", METHOD)


# In[23]:


# ============================================================
# TRAIN ONE METHOD
# Change METHOD each time:
# "none", "gaussian", "laplacian", "laglead"
# ============================================================

METHOD = "laplacian"   # CHANGE THIS

METHOD_DIR = os.path.join(SAVE_DIR, METHOD)

paths, labels = get_paths_labels_from_folder(METHOD_DIR)

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    paths,
    labels,
    test_size=0.30,
    random_state=SEED,
    stratify=labels
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_labels
)

train_ds = create_tf_dataset(
    train_paths,
    train_labels,
    training=True
)

val_ds = create_tf_dataset(
    val_paths,
    val_labels,
    training=False
)

test_ds = create_tf_dataset(
    test_paths,
    test_labels,
    training=False
)

class_weights_values = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = dict(enumerate(class_weights_values))

print("Class weights:", class_weights)

model = build_efficientnetb3_model(NUM_CLASSES)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    verbose=1
)

acc, precision, sensitivity, f1, cm = evaluate_model_correctly(
    model,
    test_ds
)

result_df = pd.DataFrame([{
    "Preprocessing": METHOD,
    "Accuracy": acc,
    "Precision": precision,
    "Sensitivity": sensitivity,
    "F1_score": f1
}])

display(result_df)

result_df.to_csv(
    f"result_{METHOD}.csv",
    index=False
)

model.save(
    f"model_{METHOD}.keras"
)

print("E:\frontiers in SP\Frontiers SP\ALLCOMMENTS\IQOTH", METHOD)


# In[39]:


# ============================================================
# TRAIN ONE METHOD
# Change METHOD each time:
# "none", "gaussian", "laplacian", "laglead"
# ============================================================

METHOD = "none"   # CHANGE THIS

METHOD_DIR = os.path.join(SAVE_DIR, METHOD)

paths, labels = get_paths_labels_from_folder(METHOD_DIR)

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    paths,
    labels,
    test_size=0.30,
    random_state=SEED,
    stratify=labels
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_labels
)

train_ds = create_tf_dataset(
    train_paths,
    train_labels,
    training=True
)

val_ds = create_tf_dataset(
    val_paths,
    val_labels,
    training=False
)

test_ds = create_tf_dataset(
    test_paths,
    test_labels,
    training=False
)

class_weights_values = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = dict(enumerate(class_weights_values))

print("Class weights:", class_weights)

model = build_efficientnetb3_model(NUM_CLASSES)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    verbose=1
)

acc, precision, sensitivity, f1, cm = evaluate_model_correctly(
    model,
    test_ds
)

result_df = pd.DataFrame([{
    "Preprocessing": METHOD,
    "Accuracy": acc,
    "Precision": precision,
    "Sensitivity": sensitivity,
    "F1_score": f1
}])

display(result_df)

result_df.to_csv(
    f"result_{METHOD}.csv",
    index=False
)

model.save(
    f"model_{METHOD}.keras"
)

print("E:\frontiers in SP\Frontiers SP\ALLCOMMENTS\IQOTH", METHOD)


# In[ ]:





# In[20]:


# ============================================================
# TRAINING CURVES
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], marker="o", label="Training Accuracy")
plt.plot(history.history["val_accuracy"], marker="o", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"Training and Validation Accuracy - {METHOD}")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"accuracy_curve_{METHOD}.png", dpi=300)
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], marker="o", label="Training Loss")
plt.plot(history.history["val_loss"], marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Training and Validation Loss - {METHOD}")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"loss_curve_{METHOD}.png", dpi=300)
plt.show()


# In[27]:


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

plt.figure(figsize=(5, 4))
plt.imshow(cm)

plt.title(f"Confusion Matrix - {METHOD}")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    np.arange(len(CLASS_NAMES)),
    CLASS_NAMES
)

plt.yticks(
    np.arange(len(CLASS_NAMES)),
    CLASS_NAMES
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=12
        )

plt.colorbar()
plt.tight_layout()
plt.savefig(f"confusion_matrix_{METHOD}.png", dpi=300)
plt.show()


# In[22]:


# ============================================================
# TRAINING CURVES
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], marker="o", label="Training Accuracy")
plt.plot(history.history["val_accuracy"], marker="o", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"Training and Validation Accuracy - {METHOD}")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"accuracy_curve_{METHOD}.png", dpi=300)
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], marker="o", label="Training Loss")
plt.plot(history.history["val_loss"], marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Training and Validation Loss - {METHOD}")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"loss_curve_{METHOD}.png", dpi=300)
plt.show()


# In[26]:


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

plt.figure(figsize=(5, 4))
plt.imshow(cm)

plt.title(f"Confusion Matrix - {METHOD}")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    np.arange(len(CLASS_NAMES)),
    CLASS_NAMES
)

plt.yticks(
    np.arange(len(CLASS_NAMES)),
    CLASS_NAMES
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=12
        )

plt.colorbar()
plt.tight_layout()
plt.savefig(f"confusion_matrix_{METHOD}.png", dpi=300)
plt.show()


# In[24]:


# ============================================================
# TRAINING CURVES
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], marker="o", label="Training Accuracy")
plt.plot(history.history["val_accuracy"], marker="o", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"Training and Validation Accuracy - {METHOD}")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"accuracy_curve_{METHOD}.png", dpi=300)
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], marker="o", label="Training Loss")
plt.plot(history.history["val_loss"], marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Training and Validation Loss - {METHOD}")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"loss_curve_{METHOD}.png", dpi=300)
plt.show()


# In[25]:


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

plt.figure(figsize=(5, 4))
plt.imshow(cm)

plt.title(f"Confusion Matrix - {METHOD}")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    np.arange(len(CLASS_NAMES)),
    CLASS_NAMES
)

plt.yticks(
    np.arange(len(CLASS_NAMES)),
    CLASS_NAMES
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=12
        )

plt.colorbar()
plt.tight_layout()
plt.savefig(f"confusion_matrix_{METHOD}.png", dpi=300)
plt.show()


# In[28]:


# ============================================================
# COMBINE ALL RESULTS
# ============================================================

all_results = []

for method in ["none", "gaussian", "laplacian", "laglead"]:
    file_path = f"result_{method}.csv"

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        all_results.append(df)

final_results_df = pd.concat(all_results, ignore_index=True)

display(final_results_df)

final_results_df.to_csv(
    "final_preprocessing_comparison_results.csv",
    index=False
)


# In[41]:


# ============================================================
# MODERN HEATMAP PERFORMANCE PLOT
# ============================================================

metrics = ["Accuracy", "Precision", "Sensitivity", "F1_score"]

heatmap_data = final_results_df[metrics].values
methods = final_results_df["Preprocessing"].values

plt.figure(figsize=(8, 5))
plt.imshow(heatmap_data, aspect="auto")

plt.colorbar(label="Score")

plt.xticks(
    np.arange(len(metrics)),
    metrics,
    rotation=30,
    ha="right"
)

plt.yticks(
    np.arange(len(methods)),
    methods
)

for i in range(len(methods)):
    for j in range(len(metrics)):
        plt.text(
            j,
            i,
            f"{heatmap_data[i, j]:.3f}",
            ha="center",
            va="center",
            fontsize=10
        )

plt.title("Performance Heatmap of Preprocessing Methods")
plt.xlabel("Evaluation Metrics")
plt.ylabel("Preprocessing Methods")

plt.tight_layout()
plt.savefig(
    "performance_heatmap_preprocessing_methods.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[49]:


import numpy as np
import matplotlib.pyplot as plt

metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

plot_data = {
    "None": {
        "scores": [0.75, 0.74, 0.73, 0.74],
        "lower":  [0.70, 0.68, 0.67, 0.68],
        "upper":  [0.80, 0.79, 0.78, 0.79],
        "color":  "#9467bd",
        "marker": "o"
    },
    "Gaussian": {
        "scores": [0.78, 0.77, 0.76, 0.77],
        "lower":  [0.73, 0.72, 0.71, 0.72],
        "upper":  [0.83, 0.82, 0.81, 0.82],
        "color":  "#1f77b4",
        "marker": "s"
    },
    "Laplacian": {
        "scores": [0.80, 0.79, 0.78, 0.79],
        "lower":  [0.75, 0.74, 0.73, 0.74],
        "upper":  [0.85, 0.84, 0.83, 0.84],
        "color":  "#54a24b",
        "marker": "^"
    },
    "LagLead": {
        "scores": [0.84, 0.83, 0.82, 0.83],
        "lower":  [0.79, 0.78, 0.77, 0.78],
        "upper":  [0.88, 0.87, 0.86, 0.87],
        "color":  "#003f5c",
        "marker": "D"
    }
}

x = np.arange(len(metrics))
offsets = [-0.27, -0.09, 0.09, 0.27]

plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.set_facecolor("#f8f8f8")

for i, (label, info) in enumerate(plot_data.items()):
    scores = np.array(info["scores"])
    lower = np.array(info["lower"])
    upper = np.array(info["upper"])

    yerr = np.vstack([scores - lower, upper - scores])
    xpos = x + offsets[i]

    plt.errorbar(
        xpos,
        scores,
        yerr=yerr,
        fmt=info["marker"],
        color=info["color"],
        ecolor=info["color"],
        elinewidth=2,
        capsize=6,
        markersize=10,
        linestyle="None",
        label=label
    )

    for xi, yi in zip(xpos, scores):
        plt.text(
            xi,
            yi + 0.008,
            f"{yi:.2f}",
            ha="center",
            fontsize=9,
            color=info["color"],
            fontweight="bold"
        )

plt.xticks(x, metrics, fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.ylim(0.60, 1.00)
#plt.title("Preprocessing Method Comparison", fontsize=14, fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.35)
plt.legend(loc="lower right", frameon=True)

plt.tight_layout()
plt.savefig("preprocessing_errorbar_plot.png", dpi=300, bbox_inches="tight")
plt.show()


# In[47]:


import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# DATA
# Replace these values with your final values
# scores = mean values
# lower  = lower CI / lower bound
# upper  = upper CI / upper bound
# ============================================================

metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

plot_data = {
    "None": {
        "scores": [0.75, 0.74, 0.73, 0.74],
        "lower":  [0.70, 0.68, 0.67, 0.68],
        "upper":  [0.80, 0.79, 0.78, 0.79],
        "color":  "#F28E2B",   # pastel orange
        "marker": "o"
    },
    "Gaussian": {
        "scores": [0.78, 0.77, 0.76, 0.77],
        "lower":  [0.73, 0.72, 0.71, 0.72],
        "upper":  [0.83, 0.82, 0.81, 0.82],
        "color":  "#4E79A7",   # pastel blue
        "marker": "s"
    },
    "Laplacian": {
        "scores": [0.80, 0.79, 0.78, 0.79],
        "lower":  [0.75, 0.74, 0.73, 0.74],
        "upper":  [0.85, 0.84, 0.83, 0.84],
        "color":  "#59A14F",   # pastel green
        "marker": "^"
    },
    "LagLead": {
        "scores": [0.84, 0.83, 0.82, 0.83],
        "lower":  [0.79, 0.78, 0.77, 0.78],
        "upper":  [0.88, 0.87, 0.86, 0.87],
        "color":  "#0B4F6C",   # dark teal-blue
        "marker": "D"
    }
}

# ============================================================
# STYLE SETTINGS
# ============================================================

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

x = np.arange(len(metrics))
offsets = [-0.24, -0.08, 0.08, 0.24]

fig, ax = plt.subplots(figsize=(11, 6.5))
ax.set_facecolor("#F7F7F7")

# ============================================================
# PLOT
# ============================================================

for i, (label, info) in enumerate(plot_data.items()):
    scores = np.array(info["scores"])
    lower = np.array(info["lower"])
    upper = np.array(info["upper"])
    color = info["color"]
    marker = info["marker"]

    xpos = x + offsets[i]

    # Shaded confidence interval
    ax.fill_between(
        xpos,
        lower,
        upper,
        color=color,
        alpha=0.12,
        zorder=1
    )

    # Mean line
    ax.plot(
        xpos,
        scores,
        color=color,
        linewidth=2.8,
        zorder=3
    )

    # Markers
    ax.scatter(
        xpos,
        scores,
        s=120,
        color=color,
        marker=marker,
        edgecolor="white",
        linewidth=0.9,
        zorder=4,
        label=label
    )

    # Optional small error bars on top of shading
    yerr = np.vstack([scores - lower, upper - scores])
    ax.errorbar(
        xpos,
        scores,
        yerr=yerr,
        fmt="none",
        ecolor=color,
        elinewidth=1.4,
        capsize=5,
        alpha=0.9,
        zorder=2
    )

    # Value labels
    for xi, yi in zip(xpos, scores):
        ax.text(
            xi,
            yi + 0.008,
            f"{yi:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=color
        )

# ============================================================
# AXES / LABELS / LEGEND
# ============================================================

ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel("Score", fontweight="bold")
#ax.set_title("Preprocessing Method Comparison", fontweight="bold")

ax.set_ylim(0.60, 1.00)
ax.grid(axis="y", linestyle="--", alpha=0.30)

# Cleaner border styling
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(True)

legend = ax.legend(
    loc="lower right",
    frameon=True,
    fancybox=True,
    framealpha=0.95
)

plt.tight_layout()
plt.savefig("preprocessing_plot.png", dpi=600, bbox_inches="tight")
plt.show()


# In[33]:


# ============================================================
# BUBBLE RANKING PLOT
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

metrics = ["Accuracy", "Precision", "Sensitivity", "F1_score"]

plot_df = final_results_df.copy()

x_labels = metrics
y_labels = plot_df["Preprocessing"].tolist()

plt.figure(figsize=(10, 6))

for i, method in enumerate(y_labels):
    for j, metric in enumerate(metrics):

        value = float(plot_df.loc[plot_df["Preprocessing"] == method, metric].values[0])

        plt.scatter(
            j,
            i,
            s=value * 1800,
            alpha=0.65
        )

        plt.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=9
        )

plt.xticks(np.arange(len(x_labels)), x_labels, fontsize=11)
plt.yticks(np.arange(len(y_labels)), y_labels, fontsize=11)

plt.xlabel("Evaluation Metrics")
plt.ylabel("Preprocessing Methods")
plt.title("Bubble Ranking Plot of Preprocessing Performance")

plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()

plt.savefig("bubble_ranking_preprocessing_plot.png", dpi=300, bbox_inches="tight")
plt.show()


# In[ ]:




