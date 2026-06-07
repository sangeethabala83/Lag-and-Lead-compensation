#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import shutil
from sklearn.model_selection import train_test_split

# ==========================================
# ORIGINAL DATASET
# ==========================================
DATASET_DIR = r"E:\frontiers in SP\Frontiers SP\ALLCOMMENTS\IQOTH\laglead"

# ==========================================
# OUTPUT DIRECTORY
# ==========================================
OUTPUT_DIR = r"E:\frontiers in SP\Frontiers SP\ALLCOMMENTS\IQOTH\SplitDataset"

TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR   = os.path.join(OUTPUT_DIR, "val")
TEST_DIR  = os.path.join(OUTPUT_DIR, "test")

classes = ["Benign", "Malignant"]

# ==========================================
# CREATE FOLDERS
# ==========================================
for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

# ==========================================
# SPLIT DATASET
# ==========================================
for cls in classes:

    class_dir = os.path.join(DATASET_DIR, cls)

    images = [
        f for f in os.listdir(class_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]

    # 70% train, 30% temp
    train_imgs, temp_imgs = train_test_split(
        images,
        test_size=0.30,
        random_state=42
    )

    # 15% val, 15% test
    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=0.50,
        random_state=42
    )

    # Copy Train
    for img in train_imgs:
        shutil.copy(
            os.path.join(class_dir, img),
            os.path.join(TRAIN_DIR, cls, img)
        )

    # Copy Validation
    for img in val_imgs:
        shutil.copy(
            os.path.join(class_dir, img),
            os.path.join(VAL_DIR, cls, img)
        )

    # Copy Test
    for img in test_imgs:
        shutil.copy(
            os.path.join(class_dir, img),
            os.path.join(TEST_DIR, cls, img)
        )

    print(f"\n{cls}")
    print("Train:", len(train_imgs))
    print("Validation:", len(val_imgs))
    print("Test:", len(test_imgs))

print("\nDataset split completed successfully!")

print("\nTrain Folder :", TRAIN_DIR)
print("Validation Folder :", VAL_DIR)
print("Test Folder :", TEST_DIR)


# In[6]:


# ============================================================
# Imports
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = 300
BATCH_SIZE = 16
EPOCHS_SENSITIVITY = 5


# In[7]:


# ============================================================
# Data generators
# Augmentation only for training
# ============================================================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    seed=SEED
)

val_data = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

test_data = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# Class weights
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = {
    0: class_weights_array[0],
    1: class_weights_array[1]
}

print("Class weights:", class_weights)
print("Class indices:", train_data.class_indices)


# In[8]:


# ============================================================
# Custom lag-lead feature compensation layers
# WITHOUT bilinear transform
# ============================================================

class LagCompensationLayer(layers.Layer):
    def __init__(self, K=1.0, T1=0.1, **kwargs):
        super(LagCompensationLayer, self).__init__(**kwargs)
        self.K = K
        self.T1 = T1

    def call(self, inputs):
        return self.K * (inputs + self.T1 * tf.nn.relu(inputs))

    def get_config(self):
        config = super().get_config()
        config.update({
            "K": self.K,
            "T1": self.T1
        })
        return config


class LeadCompensationLayer(layers.Layer):
    def __init__(self, K=1.0, T2=0.05, **kwargs):
        super(LeadCompensationLayer, self).__init__(**kwargs)
        self.K = K
        self.T2 = T2

    def call(self, inputs):
        return self.K * (inputs - self.T2 * tf.nn.relu(-inputs))

    def get_config(self):
        config = super().get_config()
        config.update({
            "K": self.K,
            "T2": self.T2
        })
        return config


# In[9]:


# ============================================================
# Build EfficientNetB3 + lag-lead feature compensation model
# ============================================================

def build_lag_lead_efficientnetb3(K=1.0, T1=0.1, T2=0.05):

    base_model = EfficientNetB3(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # For fast sensitivity analysis, freeze base model
    base_model.trainable = False

    base_features = base_model.output

    lag_features = LagCompensationLayer(
        K=K,
        T1=T1,
        name="lag_compensation_layer"
    )(base_features)

    lead_features = LeadCompensationLayer(
        K=K,
        T2=T2,
        name="lead_compensation_layer"
    )(base_features)

    compensated_features = layers.Concatenate(name="lag_lead_concat")(
        [base_features, lag_features, lead_features]
    )

    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(compensated_features)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# In[10]:


# ============================================================
# Evaluation function
# ============================================================

def evaluate_binary_model(model, test_data):

    test_data.reset()

    y_true = test_data.classes
    y_prob = model.predict(test_data, verbose=0)
    y_pred = (y_prob > 0.5).astype(int).ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    sens = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return acc, prec, sens, f1


# In[11]:


# ============================================================
# Sensitivity analysis settings
# One parameter varied at a time
# ============================================================

sensitivity_settings = [
    {"Parameter": "Baseline", "K": 1.0, "T1": 0.10, "T2": 0.05},

    {"Parameter": "K", "K": 0.8, "T1": 0.10, "T2": 0.05},
    {"Parameter": "K", "K": 1.2, "T1": 0.10, "T2": 0.05},

    {"Parameter": "T1", "K": 1.0, "T1": 0.05, "T2": 0.05},
    {"Parameter": "T1", "K": 1.0, "T1": 0.15, "T2": 0.05},

    {"Parameter": "T2", "K": 1.0, "T1": 0.10, "T2": 0.025},
    {"Parameter": "T2", "K": 1.0, "T1": 0.10, "T2": 0.10}
]

results = []

for i, setting in enumerate(sensitivity_settings):

    K = setting["K"]
    T1 = setting["T1"]
    T2 = setting["T2"]

    print("\n" + "="*70)
    print(f"Experiment {i+1}/{len(sensitivity_settings)}")
    print(f"K={K}, T1={T1}, T2={T2}")
    print("="*70)

    model = build_lag_lead_efficientnetb3(K=K, T1=T1, T2=T2)

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_SENSITIVITY,
        class_weight=class_weights,
        verbose=1
    )

    acc, prec, sens, f1 = evaluate_binary_model(model, test_data)

    results.append({
        "Parameter": setting["Parameter"],
        "K": K,
        "T1": T1,
        "T2": T2,
        "Accuracy": acc,
        "Precision": prec,
        "Sensitivity": sens,
        "F1_score": f1
    })

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Sensitivity:", sens)
    print("F1-score:", f1)

sensitivity_df = pd.DataFrame(results)
sensitivity_df.to_csv("sensitivity_analysis_custom_lag_lead_5epochs.csv", index=False)

sensitivity_df


# In[12]:


# ============================================================
# Sensitivity analysis plot
# ============================================================

sensitivity_df = pd.read_csv("sensitivity_analysis_custom_lag_lead_5epochs.csv")

sensitivity_df["Setting"] = (
    sensitivity_df["Parameter"] +
    " | K=" + sensitivity_df["K"].astype(str) +
    ", T1=" + sensitivity_df["T1"].astype(str) +
    ", T2=" + sensitivity_df["T2"].astype(str)
)

plt.figure(figsize=(13, 6))

plt.plot(sensitivity_df["Setting"], sensitivity_df["Accuracy"], marker="o", linewidth=2, label="Accuracy")
plt.plot(sensitivity_df["Setting"], sensitivity_df["Precision"], marker="s", linewidth=2, label="Precision")
plt.plot(sensitivity_df["Setting"], sensitivity_df["Sensitivity"], marker="^", linewidth=2, label="Sensitivity")
plt.plot(sensitivity_df["Setting"], sensitivity_df["F1_score"], marker="D", linewidth=2, label="F1-score")

plt.xticks(rotation=45, ha="right")
plt.xlabel("Lag-lead parameter setting")
plt.ylabel("Performance score")
plt.title("Sensitivity Analysis of Custom Lag-Lead Feature Compensation Parameters")
plt.ylim(0, 1.05)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

plt.savefig("sensitivity_analysis_custom_lag_lead_5epochs.png", dpi=300)
plt.show()


# In[13]:


# ============================================================
# Heatmap for Sensitivity Analysis
# ============================================================

import seaborn as sns
import matplotlib.pyplot as plt

heatmap_df = sensitivity_df.copy()

heatmap_df["Setting"] = (
    heatmap_df["Parameter"] +
    "\nK=" + heatmap_df["K"].astype(str) +
    ",T1=" + heatmap_df["T1"].astype(str) +
    ",T2=" + heatmap_df["T2"].astype(str)
)

metrics_heatmap = heatmap_df[
    ["Accuracy", "Precision", "Sensitivity", "F1_score"]
]

metrics_heatmap.index = heatmap_df["Setting"]

plt.figure(figsize=(10, 8))

sns.heatmap(
    metrics_heatmap,
    annot=True,
    fmt=".4f",
    cmap="YlGnBu",
    linewidths=0.5
)

plt.title("Sensitivity Analysis Heatmap of Lag-Lead Parameters")
plt.xlabel("Performance Metrics")
plt.ylabel("Parameter Settings")

plt.tight_layout()

plt.savefig(
    "sensitivity_analysis_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[14]:


# ============================================================
# Train baseline model for visualization
# ============================================================

baseline_model = build_lag_lead_efficientnetb3(
    K=1.0,
    T1=0.10,
    T2=0.05
)

history = baseline_model.fit(
    train_data,
    validation_data=val_data,
    epochs=5,
    class_weight=class_weights,
    verbose=1
)

baseline_model.save("baseline_custom_lag_lead_efficientnetb3.keras")


# In[15]:


# ============================================================
# Feature map visualization
# ============================================================

def show_feature_maps(model, data_generator, layer_name, num_maps=16):

    data_generator.reset()
    images, labels = next(data_generator)

    sample_img = images[0:1]

    feature_model = Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )

    feature_maps = feature_model.predict(sample_img, verbose=0)

    print("Layer:", layer_name)
    print("Feature map shape:", feature_maps.shape)

    total_maps = min(num_maps, feature_maps.shape[-1])

    plt.figure(figsize=(12, 12))

    for i in range(total_maps):
        plt.subplot(4, 4, i + 1)
        fmap = feature_maps[0, :, :, i]
        plt.imshow(fmap, cmap="viridis")
        plt.axis("off")
        plt.title(f"Map {i+1}")

    plt.suptitle(f"Feature Maps from {layer_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"feature_maps_{layer_name}.png", dpi=300)
    plt.show()


# In[28]:


import math
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model


# In[29]:


# ============================================================
# Feature map visualization
# ============================================================

import math
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

def show_feature_maps(model, data_generator, layer_name, num_maps=16):

    # Reset generator for consistent image selection
    data_generator.reset()

    # Safely get one batch
    images, labels = next(data_generator)
    sample_img = images[0:1]

    # Validate layer name
    try:
        layer_output = model.get_layer(layer_name).output
    except ValueError:
        available = [l.name for l in model.layers]
        raise ValueError(
            f"Layer '{layer_name}' not found.\nAvailable layers:\n" +
            "\n".join(available)
        )

    # Build sub-model up to the target layer
    feature_model = Model(inputs=model.input, outputs=layer_output)
    feature_maps = feature_model.predict(sample_img, verbose=0)

    print(f"Layer       : {layer_name}")
    print(f"Output shape: {feature_maps.shape}")

    total_maps = min(num_maps, feature_maps.shape[-1])

    # Dynamic grid size
    cols = 4
    rows = math.ceil(total_maps / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    # Handle case where rows=1
    if rows == 1:
        axes = axes.reshape(1, -1)

    axes = axes.flatten()

    for i in range(total_maps):
        fmap = feature_maps[0, :, :, i]

        # Normalize to [0, 1]
        fmap_min, fmap_max = fmap.min(), fmap.max()
        if fmap_max - fmap_min > 1e-8:
            fmap = (fmap - fmap_min) / (fmap_max - fmap_min)

        axes[i].imshow(fmap, cmap="viridis")
        axes[i].axis("off")
        axes[i].set_title(f"Map {i + 1}", fontsize=8)

    # Hide unused subplots
    for j in range(total_maps, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Feature Maps — {layer_name}", fontsize=14, y=1.01)
    plt.tight_layout()

    save_path = f"feature_maps_{layer_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved → {save_path}")


# In[30]:


# ============================================================
# Check model layer names first
# ============================================================

for layer in model.layers:
    print(layer.name)


# In[53]:


# ============================================================
# Feature maps from lag, lead, and combined layers
# ============================================================

show_feature_maps(
    model=model,
    data_generator=test_data,
    layer_name="lag_compensation_layer",
    num_maps=16
)

show_feature_maps(
    model=model,
    data_generator=test_data,
    layer_name="lead_compensation_layer",
    num_maps=16
)

show_feature_maps(
    model=model,
    data_generator=test_data,
    layer_name="lag_lead_concat",
    num_maps=16
)


# In[54]:


# ============================================================
# Compare original EfficientNet features with lag and lead features
# ============================================================

def compare_base_lag_lead_features(model, data_generator):

    data_generator.reset()
    images, labels = next(data_generator)

    sample_img = images[0:1]

    # Find last EfficientNetB3 feature layer before custom layers
    # This usually works because lag layer receives base_model.output
    lag_layer = model.get_layer("lag_compensation_layer")
    base_feature_tensor = lag_layer.input

    feature_model = Model(
        inputs=model.input,
        outputs=[
            base_feature_tensor,
            model.get_layer("lag_compensation_layer").output,
            model.get_layer("lead_compensation_layer").output,
            model.get_layer("lag_lead_concat").output
        ]
    )

    base_f, lag_f, lead_f, concat_f = feature_model.predict(sample_img, verbose=0)

    feature_list = [
        ("Original EfficientNetB3 feature", base_f),
        ("Lag-compensated feature", lag_f),
        ("Lead-compensated feature", lead_f),
        ("Concatenated lag-lead feature", concat_f)
    ]

    plt.figure(figsize=(12, 8))

    for i, (title, fmap) in enumerate(feature_list):

        # Average all channels into one heat feature map
        avg_map = np.mean(fmap[0], axis=-1)

        plt.subplot(2, 2, i + 1)
        plt.imshow(avg_map, cmap="turbo")
        plt.axis("off")
        plt.title(title)

    plt.suptitle("Original vs Custom Lag-Lead Feature Compensation", fontsize=14)
    plt.tight_layout()
    plt.savefig("original_vs_lag_lead_feature_maps.png", dpi=300)
    plt.show()

compare_base_lag_lead_features(baseline_model, test_data)


# In[59]:


# ============================================================
# Corrected Bode-like Frequency Response from Feature Maps
# ============================================================

def radial_profile(magnitude_spectrum):
    """Compute mean magnitude as a function of radial frequency bin."""
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    radius = np.sqrt((x - cx)**2 + (y - cy)**2).astype(np.int32)
    radial_sum   = np.bincount(radius.ravel(), weights=magnitude_spectrum.ravel())
    radial_count = np.bincount(radius.ravel())
    radial_mean  = radial_sum / np.maximum(radial_count, 1)
    return radial_mean


def compute_fft_profile(feature_maps):
    """
    Per-channel 2D FFT → magnitude → radial profile → mean across channels.
    Preserves per-channel frequency content before averaging.
    """
    # feature_maps shape: (1, H, W, C)
    maps = feature_maps[0]          # (H, W, C)
    H, W, C = maps.shape
    profiles = []

    for c in range(C):
        fmap      = maps[:, :, c]
        fft_shift = np.fft.fftshift(np.fft.fft2(fmap))
        magnitude = np.abs(fft_shift)
        profiles.append(radial_profile(magnitude))

    # Trim all profiles to same length then average
    min_len = min(len(p) for p in profiles)
    profiles = np.array([p[:min_len] for p in profiles])
    return np.mean(profiles, axis=0)   # (min_len,)


def plot_bode_like_response(model, data_generator,
                             lag_layer_name   = "lag_conv",
                             lead_layer_name  = "lead_conv",
                             concat_layer_name= "lag_lead_concat"):

    # ── Validate layer names before doing anything else ──────────────
    available = {l.name for l in model.layers}
    for name in [lag_layer_name, lead_layer_name, concat_layer_name]:
        if name not in available:
            print(f"\n[ERROR] Layer '{name}' not found in model '{model.name}'.")
            print("Available layers:")
            for l in model.layers:
                print(f"  {l.name:50s} | {l.__class__.__name__}")
            return

    # ── Get one sample image ─────────────────────────────────────────
    images, _ = next(iter(data_generator))
    sample_img = images[0:1]

    # ── Build multi-output feature model ─────────────────────────────
    lag_layer    = model.get_layer(lag_layer_name)
    lead_layer   = model.get_layer(lead_layer_name)
    concat_layer = model.get_layer(concat_layer_name)

    # Use lag layer's INPUT as the baseline (pre-compensator feature map)
    try:
        base_tensor = lag_layer.input
    except AttributeError:
        print("[ERROR] Could not access lag layer input tensor.")
        return

    feature_model = Model(
        inputs=model.input,
        outputs=[
            base_tensor,            # EfficientNetB3 features before compensator
            lag_layer.output,       # After lag filter
            lead_layer.output,      # After lead filter
            concat_layer.output     # After lag+lead concatenation
        ]
    )

    base_f, lag_f, lead_f, concat_f = feature_model.predict(
        sample_img, verbose=0
    )

    print(f"Base shape   : {base_f.shape}")
    print(f"Lag shape    : {lag_f.shape}")
    print(f"Lead shape   : {lead_f.shape}")
    print(f"Concat shape : {concat_f.shape}")

    # ── Per-channel FFT → radial profile ─────────────────────────────
    base_profile   = compute_fft_profile(base_f)
    lag_profile    = compute_fft_profile(lag_f)
    lead_profile   = compute_fft_profile(lead_f)
    concat_profile = compute_fft_profile(concat_f)

    # Trim all to shortest length
    min_len = min(len(base_profile), len(lag_profile),
                  len(lead_profile), len(concat_profile))
    base_profile   = base_profile[:min_len]
    lag_profile    = lag_profile[:min_len]
    lead_profile   = lead_profile[:min_len]
    concat_profile = concat_profile[:min_len]

    # ── Gain relative to baseline (dB) ───────────────────────────────
    eps = 1e-8
    lag_db    = 20 * np.log10((lag_profile    + eps) / (base_profile + eps))
    lead_db   = 20 * np.log10((lead_profile   + eps) / (base_profile + eps))
    concat_db = 20 * np.log10((concat_profile + eps) / (base_profile + eps))

    # Normalize so DC (radius=0) starts at 0 dB — shows relative shaping
    lag_db    -= lag_db[0]
    lead_db   -= lead_db[0]
    concat_db -= concat_db[0]

    # ── Frequency axis: cycles/pixel, skip DC bin ────────────────────
    freq = np.linspace(0, 0.5, min_len)   # 0 → Nyquist

    # ── Plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogx(freq[1:], lag_db[1:],
                color='steelblue',  linewidth=2, label='Lag filter response H_lag(z)')
    ax.semilogx(freq[1:], lead_db[1:],
                color='darkorange', linewidth=2, label='Lead filter response H_lead(z)')
    ax.semilogx(freq[1:], concat_db[1:],
                color='seagreen',   linewidth=2, label='Combined lag-lead response')

    # Reference lines
    ax.axhline(0,  color='gray', linestyle='--', linewidth=0.8, label='0 dB (baseline)')
    ax.axhline(-3, color='red',  linestyle=':',  linewidth=0.8, label='−3 dB reference')

    ax.set_xlabel('Spatial frequency (cycles/pixel) — log scale', fontsize=12)
    ax.set_ylabel('Magnitude gain relative to backbone features (dB)', fontsize=12)
    ax.set_title(
        'Bode-Style Frequency Response of BT-LL Compensator\n'
        '(Gain relative to plain EfficientNetB3 feature maps)',
        fontsize=13
    )
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()

    save_path = "bode_like_frequency_response_bt_ll.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"Saved → {save_path}")


# ── Call with correct model and actual layer names ────────────────────────────
# First run this to confirm your layer names:
# print_layer_names(model)

plot_bode_like_response(
    model             = model,            # your trained BT-LL-EfficientNetB3
    data_generator    = test_data,
    lag_layer_name    = "lag_conv",       # ← replace with your actual lag layer name
    lead_layer_name   = "lead_conv",      # ← replace with your actual lead layer name
    concat_layer_name = "lag_lead_concat" # ← replace with your actual concat layer name
)


# In[56]:


# Step 1: Print all layer names in your model
for i, layer in enumerate(model.layers):
    print(f"[{i:3d}] {layer.name:50s} | {layer.__class__.__name__}")


# In[ ]:





# In[60]:


print("\nMODEL LAYERS\n")

for i, layer in enumerate(model.layers):
    print(f"{i:3d} : {layer.name} ({layer.__class__.__name__})")


# In[45]:


# ============================================================
# Check which model variables exist
# ============================================================

possible_models = ["model", "baseline_model", "proposed_model", "model_to_visualize"]

for name in possible_models:
    if name in globals():
        print(name, "exists")
    else:
        print(name, "not found")


# In[61]:


# ============================================================
# Select model for Bode-like plot
# ============================================================

# Use this if your trained lag-lead model variable is model
model_to_analyze = model

# If your lag-lead model has another name, replace model above with that name.


# In[62]:


# ============================================================
# Bode-like Frequency Response Plot
# For Custom Lag-Lead EfficientNetB3 WITHOUT bilinear transform
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

# IMPORTANT:
# Use your trained lag-lead model here
model_to_analyze = model


# In[63]:


# ============================================================
# Auto-detect lag, lead, and concat layers
# ============================================================

layer_names = [layer.name for layer in model_to_analyze.layers]

lag_layers = [name for name in layer_names if "lag" in name.lower()]
lead_layers = [name for name in layer_names if "lead" in name.lower()]
concat_layers = [name for name in layer_names if "concat" in name.lower()]

print("Lag layers:", lag_layers)
print("Lead layers:", lead_layers)
print("Concat layers:", concat_layers)

if len(lag_layers) == 0:
    raise ValueError("No lag layer found. Your selected model is not the lag-lead model.")

if len(lead_layers) == 0:
    raise ValueError("No lead layer found. Your selected model is not the lag-lead model.")

if len(concat_layers) == 0:
    raise ValueError("No concat layer found. Your selected model is not the lag-lead model.")

LAG_LAYER_NAME = lag_layers[0]
LEAD_LAYER_NAME = lead_layers[0]
CONCAT_LAYER_NAME = concat_layers[-1]

print("Using lag layer:", LAG_LAYER_NAME)
print("Using lead layer:", LEAD_LAYER_NAME)
print("Using concat layer:", CONCAT_LAYER_NAME)


# In[64]:


# ============================================================
# Helper functions
# ============================================================

def radial_profile(magnitude_spectrum):
    h, w = magnitude_spectrum.shape

    y, x = np.indices((h, w))
    cy, cx = h // 2, w // 2

    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r = r.astype(np.int32)

    radial_sum = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    radial_count = np.bincount(r.ravel())

    radial_mean = radial_sum / np.maximum(radial_count, 1)

    return radial_mean


def fft_profile_from_feature(feature):
    """
    feature shape: H, W, C
    """
    fmap = np.mean(feature, axis=-1)

    fft_result = np.fft.fft2(fmap)
    fft_shift = np.fft.fftshift(fft_result)

    magnitude = np.abs(fft_shift)
    profile = radial_profile(magnitude)

    return profile


# In[66]:


# ============================================================
# Theoretical Bode Plot Based on K, T1, T2 Only
# WITHOUT bilinear transform
# G(s) = K * (1 + T1*s) / (1 + T2*s)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Set lag-lead parameters
# ------------------------------------------------------------
K = 1.0
T1 = 0.10
T2 = 0.20

# Frequency range
w = np.logspace(-2, 2, 1000)   # rad/s or normalized spatial frequency

# Complex frequency
s = 1j * w

# Transfer function
G = K * (1 + T1 * s) / (1 + T2 * s)

# Magnitude and phase
magnitude_db = 20 * np.log10(np.abs(G))
phase_deg = np.angle(G, deg=True)

# Normalize DC gain to 0 dB
magnitude_db = magnitude_db - magnitude_db[0]

# ------------------------------------------------------------
# Plot Bode magnitude
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.semilogx(w, magnitude_db, linewidth=2.5)
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Frequency")
plt.ylabel("Magnitude Gain (dB)")
plt.title(f"Theoretical Bode Magnitude Response: K={K}, T1={T1}, T2={T2}")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("theoretical_bode_magnitude_K_T1_T2.png", dpi=600, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# Plot Bode phase
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.semilogx(w, phase_deg, linewidth=2.5)
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Frequency")
plt.ylabel("Phase (degrees)")
plt.title(f"Theoretical Bode Phase Response: K={K}, T1={T1}, T2={T2}")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("theoretical_bode_phase_K_T1_T2.png", dpi=600, bbox_inches="tight")
plt.show()


# In[67]:


# ============================================================
# Bode Sensitivity Plot for K, T1, T2
# WITHOUT bilinear transform
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

def bode_response(K, T1, T2, w):
    s = 1j * w
    G = K * (1 + T1 * s) / (1 + T2 * s)
    mag_db = 20 * np.log10(np.abs(G))
    mag_db = mag_db - mag_db[0]   # DC normalization
    phase_deg = np.angle(G, deg=True)
    return mag_db, phase_deg


w = np.logspace(-2, 2, 1000)

settings = [
    {"label": "Baseline: K=1.0, T1=0.10, T2=0.20", "K": 1.0, "T1": 0.10, "T2": 0.20},
    {"label": "K increased: K=1.2, T1=0.10, T2=0.20", "K": 1.2, "T1": 0.10, "T2": 0.20},
    {"label": "T1 increased: K=1.0, T1=0.15, T2=0.20", "K": 1.0, "T1": 0.15, "T2": 0.20},
    {"label": "T2 increased: K=1.0, T1=0.10, T2=0.30", "K": 1.0, "T1": 0.10, "T2": 0.30},
]

plt.figure(figsize=(10, 6))

for setting in settings:
    mag_db, phase_deg = bode_response(
        K=setting["K"],
        T1=setting["T1"],
        T2=setting["T2"],
        w=w
    )

    plt.semilogx(
        w,
        mag_db,
        linewidth=2.5,
        label=setting["label"]
    )

plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Frequency")
plt.ylabel("Normalized Magnitude Gain (dB)")
plt.title("Theoretical Bode Magnitude Sensitivity for K, T1, and T2")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("theoretical_bode_sensitivity_K_T1_T2.png", dpi=600, bbox_inches="tight")
plt.show()


# In[68]:


# ============================================================
# Theoretical Bode Plot Based on K, T1, T2 Only
# WITHOUT bilinear transform
# G(s) = K * (1 + T1*s) / (1 + T2*s)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Set lag-lead parameters
# ------------------------------------------------------------
K = 1.0
T1 = 0.20
T2 = 0.10

# Frequency range
w = np.logspace(-2, 2, 1000)   # rad/s or normalized spatial frequency

# Complex frequency
s = 1j * w

# Transfer function
G = K * (1 + T1 * s) / (1 + T2 * s)

# Magnitude and phase
magnitude_db = 20 * np.log10(np.abs(G))
phase_deg = np.angle(G, deg=True)

# Normalize DC gain to 0 dB
magnitude_db = magnitude_db - magnitude_db[0]

# ------------------------------------------------------------
# Plot Bode magnitude
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.semilogx(w, magnitude_db, linewidth=2.5)
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Frequency")
plt.ylabel("Magnitude Gain (dB)")
plt.title(f"Theoretical Bode Magnitude Response: K={K}, T1={T1}, T2={T2}")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("theoretical_bode_magnitude_K_T1_T2.png", dpi=600, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# Plot Bode phase
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.semilogx(w, phase_deg, linewidth=2.5)
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Frequency")
plt.ylabel("Phase (degrees)")
plt.title(f"Theoretical Bode Phase Response: K={K}, T1={T1}, T2={T2}")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("theoretical_bode_phase_K_T1_T2.png", dpi=600, bbox_inches="tight")
plt.show()


# In[70]:


import numpy as np
import matplotlib.pyplot as plt

# ── Your actual paper parameters ──────────────────────────────────────────────
K   = 1.0
T1  = 0.2   # lag zero time constant
T2  = 0.1   # lag pole time constant  
beta  = 5.0  # lag attenuation factor  → lag pole at T2*beta
alpha = 0.2  # lead attenuation factor → lead zero at T1*alpha

omega = np.logspace(-2, 2, 1000)
s = 1j * omega

# Lag:  H_lag(s)  = K * (1 + s*T1) / (1 + s*T1*beta)
# Lead: H_lead(s) = K * (1 + s*T2) / (1 + s*T2*alpha)  -- alpha < 1 gives lead
H_lag      = K * (1 + s * T1)        / (1 + s * T1 * beta)
H_lead     = K * (1 + s * T2)        / (1 + s * T2 * alpha)
H_combined = H_lag * H_lead

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# ── Magnitude ─────────────────────────────────────────────────────────────────
ax1.semilogx(omega, 20*np.log10(np.abs(H_lag)),
             'b-',  linewidth=2, label=f'Lag $H_{{lag}}(s)$: T₁={T1}, β={beta}')
ax1.semilogx(omega, 20*np.log10(np.abs(H_lead)),
             'r-',  linewidth=2, label=f'Lead $H_{{lead}}(s)$: T₂={T2}, α={alpha}')
ax1.semilogx(omega, 20*np.log10(np.abs(H_combined)),
             'g--', linewidth=2.5, label='Combined $H_{{lag}} \\cdot H_{{lead}}$')

ax1.axhline(0,  color='gray', linestyle='--', linewidth=0.8)
ax1.axhline(-3, color='purple', linestyle=':', linewidth=0.8, label='−3 dB')

# Mark corner frequencies
omega_lag_zero = 1/T1                  # = 5   rad/s
omega_lag_pole = 1/(T1*beta)           # = 1   rad/s
omega_lead_zero = 1/T2                 # = 10  rad/s
omega_lead_pole = 1/(T2*alpha)         # = 50  rad/s

for w, lbl in [(omega_lag_pole,  'ω=1/T₁β'),
               (omega_lag_zero,  'ω=1/T₁'),
               (omega_lead_zero, 'ω=1/T₂'),
               (omega_lead_pole, 'ω=1/T₂α')]:
    ax1.axvline(w, color='gray', linestyle=':', linewidth=0.6, alpha=0.7)
    ax1.text(w, ax1.get_ylim()[0] if ax1.get_ylim() else -8,
             lbl, fontsize=7, ha='center', color='gray')

ax1.set_ylabel('Magnitude Gain (dB)', fontsize=12)
ax1.set_title(
    'Lag vs Lead vs Combined — Theoretical Bode Response\n'
    f'K={K}, T₁={T1}, β={beta}, T₂={T2}, α={alpha}',
    fontsize=12
)
ax1.legend(fontsize=10)
ax1.grid(True, which='both', linestyle='--', alpha=0.5)

# ── Phase ─────────────────────────────────────────────────────────────────────
ax2.semilogx(omega, np.angle(H_lag,      deg=True),
             'b-',  linewidth=2, label=f'Lag $H_{{lag}}(s)$')
ax2.semilogx(omega, np.angle(H_lead,     deg=True),
             'r-',  linewidth=2, label=f'Lead $H_{{lead}}(s)$')
ax2.semilogx(omega, np.angle(H_combined, deg=True),
             'g--', linewidth=2.5, label='Combined')

ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax2.set_xlabel('Frequency (rad/s)', fontsize=12)
ax2.set_ylabel('Phase (degrees)', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('bode_asymmetric_lag_lead.png', dpi=300, bbox_inches='tight')
plt.show()


# In[71]:


import numpy as np
import matplotlib.pyplot as plt

K     = 1.0
T1    = 0.2
T2    = 0.1
beta  = 5.0
alpha = 0.2

omega = np.logspace(-2, 4, 2000)   # extended to 10^4
s = 1j * omega

H_lag      = K * (1 + s*T1)       / (1 + s*T1*beta)
H_lead     = K * (1 + s*T2)       / (1 + s*T2*alpha)
H_combined = H_lag * H_lead

# Asymptotic value for annotation
dc_gain_combined = 20*np.log10(abs(
    K**2 * (T1*beta * T2*alpha) / (T1 * T2)
) / (T1*beta * T2*alpha / (T1 * T2)))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))

# ── Magnitude ────────────────────────────────────────────────────────
ax1.semilogx(omega, 20*np.log10(np.abs(H_lag)),
             'b-',  linewidth=2,   label=r'Lag $H_{lag}(s)$: $T_1$=0.2, $\beta$=5.0')
ax1.semilogx(omega, 20*np.log10(np.abs(H_lead)),
             'r-',  linewidth=2,   label=r'Lead $H_{lead}(s)$: $T_2$=0.1, $\alpha$=0.2')
ax1.semilogx(omega, 20*np.log10(np.abs(H_combined)),
             'g--', linewidth=2.5, label=r'Combined $H_{lag} \cdot H_{lead}$')

ax1.axhline(0,   color='gray',   linestyle='--', linewidth=0.8)
ax1.axhline(-6,  color='purple', linestyle=':',  linewidth=1.0,
            label=r'Combined asymptote $-6$ dB')

# Corner frequency vertical markers
corners = [(1,  r'$\frac{1}{T_1\beta}$=1'),
           (5,  r'$\frac{1}{T_1}$=5'),
           (10, r'$\frac{1}{T_2}$=10'),
           (50, r'$\frac{1}{T_2\alpha}$=50')]

for w, lbl in corners:
    ax1.axvline(w, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    ax1.text(w*1.05, -13, lbl, fontsize=8, color='dimgray',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

ax1.set_xlim([1e-2, 1e4])
ax1.set_ylim([-16, 16])
ax1.set_ylabel('Magnitude Gain (dB)', fontsize=12)
ax1.set_title(
    r'Lag vs Lead vs Combined — Theoretical Bode Response'
    '\n'
    r'$K$=1.0, $T_1$=0.2, $\beta$=5.0, $T_2$=0.1, $\alpha$=0.2',
    fontsize=12
)
ax1.legend(fontsize=10)
ax1.grid(True, which='both', linestyle='--', alpha=0.5)

# ── Phase ────────────────────────────────────────────────────────────
lag_phase      = np.unwrap(np.angle(H_lag))      * 180/np.pi
lead_phase     = np.unwrap(np.angle(H_lead))     * 180/np.pi
combined_phase = np.unwrap(np.angle(H_combined)) * 180/np.pi

ax2.semilogx(omega, lag_phase,      'b-',  linewidth=2,   label=r'Lag $H_{lag}(s)$')
ax2.semilogx(omega, lead_phase,     'r-',  linewidth=2,   label=r'Lead $H_{lead}(s)$')
ax2.semilogx(omega, combined_phase, 'g--', linewidth=2.5, label='Combined')

ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)

for w, _ in corners:
    ax2.axvline(w, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)

ax2.set_xlim([1e-2, 1e4])
ax2.set_xlabel('Frequency (rad/s)', fontsize=12)
ax2.set_ylabel('Phase (degrees)', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('bode_asymmetric_corrected.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print("Saved → bode_asymmetric_corrected.png")


# In[ ]:




