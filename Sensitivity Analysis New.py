#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================================
# CELL 1: IMPORT LIBRARIES
# ============================================================

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)


# In[2]:


# ============================================================
# CELL 2: DATASET PATH
# ============================================================

DATA_DIR = r"E:\IQ-OTH"   # CHANGE THIS PATH

CLASS_NAMES = ["Benign", "Malignant"]

IMG_SIZE = (300, 300)
BATCH_SIZE = 16

OUTPUT_DIR = Path("Lag-Lead_Preprocessing_Revision")
OUTPUT_DIR.mkdir(exist_ok=True)

print("Dataset path:", DATA_DIR)
print("Available folders:", os.listdir(DATA_DIR))


# In[4]:


# ============================================================
# CELL 2: ALREADY SPLIT DATASET PATH
# ============================================================

DATA_DIR = r"E:\frontiers in SP\Frontiers SP\ALLCOMMENTS\IQOTH\SplitDataset"  # CHANGE THIS

TRAIN_DIR = Path(DATA_DIR) / "train"
VAL_DIR   = Path(DATA_DIR) / "val"
TEST_DIR  = Path(DATA_DIR) / "test"

CLASS_NAMES = ["Benign", "Malignant"]

IMG_SIZE = (300, 300)
BATCH_SIZE = 16

OUTPUT_DIR = Path("LagLead_Preprocessing_SecondRevision")
OUTPUT_DIR.mkdir(exist_ok=True)

print("Train folder:", TRAIN_DIR)
print("Validation folder:", VAL_DIR)
print("Test folder:", TEST_DIR)

print("\nTrain exists:", TRAIN_DIR.exists())
print("Validation exists:", VAL_DIR.exists())
print("Test exists:", TEST_DIR.exists())


# In[6]:


# ============================================================
# CELL 3: CHECK SPLIT FOLDERS 
# ============================================================

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def count_images_unique(folder):
    folder = Path(folder)

    image_files = []

    for file in folder.rglob("*"):
        if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS:
            image_files.append(file)

    # remove duplicate paths safely
    image_files = list(set(image_files))

    return len(image_files)


def check_split_folder(split_dir, split_name):
    print("\n" + "=" * 60)
    print(split_name.upper())
    print("=" * 60)

    if not split_dir.exists():
        print("ERROR: folder not found:", split_dir)
        return

    print("Inside folder:")
    for item in split_dir.iterdir():
        print(" -", item.name)

    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name

        if class_dir.exists():
            count = count_images_unique(class_dir)
            print(class_name, "images:", count)
        else:
            print("Missing class folder:", class_dir)


check_split_folder(TRAIN_DIR, "train")
check_split_folder(VAL_DIR, "validation")
check_split_folder(TEST_DIR, "test")


# In[7]:


# ============================================================
# CELL 4: CREATE DATAFRAME FROM EXISTING SPLIT FOLDERS
# NO DOUBLE COUNTING
# ============================================================

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def collect_split_images(split_dir, split_name):
    rows = []

    for class_name in CLASS_NAMES:
        class_dir = Path(split_dir) / class_name

        if not class_dir.exists():
            print("WARNING: class folder not found:", class_dir)
            continue

        image_files = []

        for file in class_dir.rglob("*"):
            if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS:
                image_files.append(file)

        # remove duplicate paths
        image_files = sorted(list(set(image_files)))

        print(split_name, class_name, ":", len(image_files))

        for img_path in image_files:
            rows.append({
                "filepath": str(img_path),
                "label": class_name,
                "split": split_name
            })

    return rows


all_rows = []
all_rows.extend(collect_split_images(TRAIN_DIR, "train"))
all_rows.extend(collect_split_images(VAL_DIR, "val"))
all_rows.extend(collect_split_images(TEST_DIR, "test"))

split_df = pd.DataFrame(all_rows)

if split_df.empty:
    raise ValueError("No images found. Check DATA_DIR, split folder names, and CLASS_NAMES.")

print("\nTotal images:", len(split_df))
print(split_df.groupby(["split", "label"]).size())

split_csv_path = OUTPUT_DIR / "existing_split_file_list.csv"
split_df.to_csv(split_csv_path, index=False)

print("\nSaved file list:", split_csv_path)


# In[8]:


# ============================================================
# CELL 5: PREPARE TRAIN / VAL / TEST DATAFRAMES
# ============================================================

label_to_id = {
    CLASS_NAMES[0]: 0,
    CLASS_NAMES[1]: 1
}

split_df["label_id"] = split_df["label"].map(label_to_id)

train_df = split_df[split_df["split"] == "train"].copy()
val_df   = split_df[split_df["split"] == "val"].copy()
test_df  = split_df[split_df["split"] == "test"].copy()

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nTest class distribution:")
print(test_df["label"].value_counts())

print("\nLabel mapping:", label_to_id)


# In[9]:


# ============================================================
# CELL 6: CREATE TENSORFLOW DATASETS
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

def load_image(filepath, label):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    return image, tf.cast(label, tf.float32)


def make_dataset(df, training=False):
    filepaths = df["filepath"].values
    labels = df["label_id"].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)

    return ds


train_ds = make_dataset(train_df, training=True)
val_ds   = make_dataset(val_df, training=False)
test_ds  = make_dataset(test_df, training=False)


# In[10]:


# ============================================================
# CELL 7: LAG–LEAD PREPROCESSING LAYER
# ============================================================

@tf.keras.utils.register_keras_serializable()
class LagLeadPreprocessing(layers.Layer):
    """
    Lag–Lead Preprocessing.

    Applied before EfficientNetB3.
    No bilinear transform.
    No MBConv insertion.
    No alpha/beta.
    """

    def __init__(self, K=1.0, T1=0.10, T2=0.05, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.K = float(K)
        self.T1 = float(T1)
        self.T2 = float(T2)
        self.kernel_size = int(kernel_size)

    def get_config(self):
        config = super().get_config()
        config.update({
            "K": self.K,
            "T1": self.T1,
            "T2": self.T2,
            "kernel_size": self.kernel_size
        })
        return config

    def build(self, input_shape):
        sigma = max(0.30, self.T1 * 10.0)

        ax = tf.range(
            -(self.kernel_size // 2),
            self.kernel_size // 2 + 1,
            dtype=tf.float32
        )

        xx, yy = tf.meshgrid(ax, ax)

        kernel = tf.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / tf.reduce_sum(kernel)

        kernel = tf.reshape(kernel, [self.kernel_size, self.kernel_size, 1, 1])
        kernel = tf.tile(kernel, [1, 1, 3, 1])

        self.gaussian_kernel = tf.constant(kernel, dtype=tf.float32)

    def call(self, inputs):
        x = inputs / 255.0

        lag_output = tf.nn.depthwise_conv2d(
            x,
            self.gaussian_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME"
        )

        detail = x - lag_output

        lead_gain = 10.0 * self.T2

        y = lag_output + (1.0 + lead_gain) * detail
        y = self.K * y
        y = tf.clip_by_value(y, 0.0, 1.0)

        return y * 255.0


# In[11]:


# ============================================================
# CELL 8: BUILD EFFICIENTNETB3 WITH LAG–LEAD PREPROCESSING
# ============================================================

def build_laglead_model(K=1.0, T1=0.10, T2=0.05, learning_rate=1e-4):

    inputs = layers.Input(shape=(300, 300, 3), name="input_image")

    x = LagLeadPreprocessing(
        K=K,
        T1=T1,
        T2=T2,
        name="laglead_preprocessing"
    )(inputs)

    x = layers.RandomFlip("horizontal_and_vertical", seed=SEED)(x)
    x = layers.RandomRotation(0.05, seed=SEED)(x)
    x = layers.RandomZoom(0.10, seed=SEED)(x)

    backbone = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        pooling=None
    )

    backbone.trainable = False

    x = backbone.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    return model


# In[12]:


# ============================================================
# CELL 9: METRICS AND PREDICTION SAVING
# ============================================================

def calculate_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0)
    }


def save_predictions(model, test_ds, test_df, variant_name):
    y_prob = model.predict(test_ds).ravel()
    y_true = test_df["label_id"].values.astype(int)
    y_pred = (y_prob >= 0.5).astype(int)

    pred_df = test_df.copy()
    pred_df["y_true"] = y_true
    pred_df["y_probability"] = y_prob
    pred_df["y_prediction"] = y_pred

    pred_path = OUTPUT_DIR / f"predictions_{variant_name}.csv"
    pred_df.to_csv(pred_path, index=False)

    metrics = calculate_metrics(y_true, y_prob)

    result = {
        "variant": variant_name,
        "test_size": len(y_true),
        **metrics,
        "prediction_file": str(pred_path)
    }

    return result


# In[17]:


# ============================================================
# UPDATED SENSITIVITY SETTINGS: LAG–LEAD PREPROCESSING ONLY
# Code names do not use hyphen to avoid Python/file-loading confusion
# ============================================================

BASE_K = 1.0
BASE_T1 = 0.10
BASE_T2 = 0.05

sensitivity_settings = [
    {
        "variant": "LagLead_Preprocessing_K_0p8",
        "parameter": "K",
        "value": 0.8,
        "K": 0.8,
        "T1": BASE_T1,
        "T2": BASE_T2
    },
    {
        "variant": "LagLead_Preprocessing_K_1p0",
        "parameter": "K",
        "value": 1.0,
        "K": 1.0,
        "T1": BASE_T1,
        "T2": BASE_T2
    },
    {
        "variant": "LagLead_Preprocessing_K_1p2",
        "parameter": "K",
        "value": 1.2,
        "K": 1.2,
        "T1": BASE_T1,
        "T2": BASE_T2
    },

    {
        "variant": "LagLead_Preprocessing_T1_0p05",
        "parameter": "T1",
        "value": 0.05,
        "K": BASE_K,
        "T1": 0.05,
        "T2": BASE_T2
    },
    {
        "variant": "LagLead_Preprocessing_T1_0p10",
        "parameter": "T1",
        "value": 0.10,
        "K": BASE_K,
        "T1": 0.10,
        "T2": BASE_T2
    },
    {
        "variant": "LagLead_Preprocessing_T1_0p15",
        "parameter": "T1",
        "value": 0.15,
        "K": BASE_K,
        "T1": 0.15,
        "T2": BASE_T2
    },

    {
        "variant": "LagLead_Preprocessing_T2_0p025",
        "parameter": "T2",
        "value": 0.025,
        "K": BASE_K,
        "T1": BASE_T1,
        "T2": 0.025
    },
    {
        "variant": "LagLead_Preprocessing_T2_0p05",
        "parameter": "T2",
        "value": 0.05,
        "K": BASE_K,
        "T1": BASE_T1,
        "T2": 0.05
    },
    {
        "variant": "LagLead_Preprocessing_T2_0p10",
        "parameter": "T2",
        "value": 0.10,
        "K": BASE_K,
        "T1": BASE_T1,
        "T2": 0.10
    },
]

sensitivity_settings_df = pd.DataFrame(sensitivity_settings)
sensitivity_settings_df


# In[18]:


# ============================================================
# CELL 11: RUN SENSITIVITY ANALYSIS - FINAL CORRECTED VERSION
# ============================================================

EPOCHS = 15
LEARNING_RATE = 1e-4

all_results = []

for setting in sensitivity_settings:

    variant = setting["variant"]
    K = setting["K"]
    T1 = setting["T1"]
    T2 = setting["T2"]

    print("\n" + "=" * 80)
    print("Running:", variant)
    print("K =", K, "T1 =", T1, "T2 =", T2)
    print("=" * 80)

    tf.keras.backend.clear_session()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    model = build_laglead_model(
        K=K,
        T1=T1,
        T2=T2,
        learning_rate=LEARNING_RATE
    )

    checkpoint_path = OUTPUT_DIR / f"best_model_{variant}.keras"

    callbacks = [
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Correct custom layer loading
    best_model = tf.keras.models.load_model(
        checkpoint_path,
        custom_objects={"LagLeadPreprocessing": LagLeadPreprocessing}
    )

    result = save_predictions(
        model=best_model,
        test_ds=test_ds,
        test_df=test_df,
        variant_name=variant
    )

    result.update({
        "parameter": setting["parameter"],
        "value": setting["value"],
        "K": K,
        "T1": T1,
        "T2": T2
    })

    all_results.append(result)

    pd.DataFrame(all_results).to_csv(
        OUTPUT_DIR / "sensitivity_results_live.csv",
        index=False
    )

sensitivity_results = pd.DataFrame(all_results)

sensitivity_results.to_csv(
    OUTPUT_DIR / "final_sensitivity_results.csv",
    index=False
)

sensitivity_results


# In[19]:


# ============================================================
# CELL 12: SENSITIVITY PLOTS
# ============================================================

sensitivity_results = pd.read_csv(OUTPUT_DIR / "final_sensitivity_results.csv")

metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]

for parameter in ["K", "T1", "T2"]:

    temp_df = sensitivity_results[sensitivity_results["parameter"] == parameter].copy()
    temp_df = temp_df.sort_values("value")

    plt.figure(figsize=(8, 5))

    for metric in metrics_to_plot:
        plt.plot(
            temp_df["value"],
            temp_df[metric],
            marker="o",
            linewidth=2,
            label=metric.replace("_", " ").title()
        )

    plt.xlabel(parameter)
    plt.ylabel("Score")
    plt.title(f"Sensitivity Analysis of Lag–Lead Preprocessing Parameter {parameter}")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    save_path = OUTPUT_DIR / f"Sensitivity_LagLead_{parameter}.png"
    plt.savefig(save_path, dpi=300)
    plt.show()

    print("Saved:", save_path)


# In[22]:


# ============================================================
# CELL 14: CONFIDENCE INTERVAL FUNCTIONS
# ============================================================

def bootstrap_confidence_intervals(y_true, y_prob, threshold=0.5, n_bootstrap=5000, seed=42):
    rng = np.random.default_rng(seed)

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    n = len(y_true)

    boot_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": []
    }

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)

        if len(np.unique(y_true[idx])) < 2:
            continue

        result = calculate_metrics(y_true[idx], y_prob[idx], threshold)

        for metric in boot_metrics:
            boot_metrics[metric].append(result[metric])

    ci = {}

    for metric, values in boot_metrics.items():
        values = np.asarray(values)
        ci[metric + "_ci_lower"] = np.percentile(values, 2.5)
        ci[metric + "_ci_upper"] = np.percentile(values, 97.5)

    return ci


# In[25]:


# ============================================================
# CHECK AVAILABLE PREDICTION FILES
# ============================================================

from pathlib import Path

print("OUTPUT_DIR:", OUTPUT_DIR)

prediction_files = sorted(OUTPUT_DIR.glob("predictions_*.csv"))

print("\nAvailable prediction files:")
for file in prediction_files:
    print(file.name)

if len(prediction_files) == 0:
    print("\nNo prediction files found. First complete Cell 11 training/evaluation.")


# In[26]:


# ============================================================
# CELL 15: FINAL CONFIDENCE INTERVAL TABLE - CORRECTED
# ============================================================

FINAL_PREDICTION_FILE = OUTPUT_DIR / "predictions_LagLead_Preprocessing_K_1p0.csv"

print("Reading prediction file:", FINAL_PREDICTION_FILE)
print("File exists:", FINAL_PREDICTION_FILE.exists())

pred_df = pd.read_csv(FINAL_PREDICTION_FILE)

print(pred_df.head())
print(pred_df.columns)


# In[27]:


# ============================================================
# CONFIDENCE INTERVAL FUNCTIONS
# ============================================================

def bootstrap_confidence_intervals(y_true, y_prob, threshold=0.5, n_bootstrap=5000, seed=42):
    rng = np.random.default_rng(seed)

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    n = len(y_true)

    boot_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": []
    }

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)

        if len(np.unique(y_true[idx])) < 2:
            continue

        result = calculate_metrics(y_true[idx], y_prob[idx], threshold)

        for metric in boot_metrics:
            boot_metrics[metric].append(result[metric])

    ci = {}

    for metric, values in boot_metrics.items():
        values = np.asarray(values)
        ci[metric + "_ci_lower"] = np.percentile(values, 2.5)
        ci[metric + "_ci_upper"] = np.percentile(values, 97.5)

    return ci


# In[28]:


# ============================================================
# GENERATE FINAL CONFIDENCE INTERVAL TABLE
# ============================================================

pred_df = pd.read_csv(FINAL_PREDICTION_FILE)

y_true = pred_df["y_true"].values.astype(int)
y_prob = pred_df["y_probability"].values.astype(float)

final_metrics = calculate_metrics(y_true, y_prob)
final_ci = bootstrap_confidence_intervals(
    y_true,
    y_prob,
    n_bootstrap=5000
)

ci_table = pd.DataFrame([
    {
        "Metric": "Accuracy",
        "Value": final_metrics["accuracy"],
        "95% CI Lower": final_ci["accuracy_ci_lower"],
        "95% CI Upper": final_ci["accuracy_ci_upper"]
    },
    {
        "Metric": "Precision",
        "Value": final_metrics["precision"],
        "95% CI Lower": final_ci["precision_ci_lower"],
        "95% CI Upper": final_ci["precision_ci_upper"]
    },
    {
        "Metric": "Recall/Sensitivity",
        "Value": final_metrics["recall"],
        "95% CI Lower": final_ci["recall_ci_lower"],
        "95% CI Upper": final_ci["recall_ci_upper"]
    },
    {
        "Metric": "F1-score",
        "Value": final_metrics["f1_score"],
        "95% CI Lower": final_ci["f1_score_ci_lower"],
        "95% CI Upper": final_ci["f1_score_ci_upper"]
    }
])

ci_table["Report Format"] = ci_table.apply(
    lambda row: f"{row['Value']:.3f} [{row['95% CI Lower']:.3f}, {row['95% CI Upper']:.3f}]",
    axis=1
)

ci_table_path = OUTPUT_DIR / "LagLead_Preprocessing_Final_Confidence_Intervals.csv"
ci_table.to_csv(ci_table_path, index=False)

print("Saved CI table:", ci_table_path)
ci_table


# In[39]:


# ============================================================
# CONFIDENCE INTERVAL PLOT
# ============================================================

x = np.arange(len(ci_table))

values = ci_table["Value"].values
lower_error = values - ci_table["95% CI Lower"].values
upper_error = ci_table["95% CI Upper"].values - values

plt.figure(figsize=(8, 5))

plt.errorbar(
    x,
    values,
    yerr=[lower_error, upper_error],
    fmt="o",
    capsize=6,
    linewidth=3
)

plt.xticks(x, ci_table["Metric"], rotation=20)
plt.ylabel("Metric Value")
plt.ylim(0, 1.05)
plt.title("Lag–Lead Preprocessing Final Test Performance with 95% Confidence Intervals")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

ci_plot_path = OUTPUT_DIR / "LagLead_Preprocessing_Final_Confidence_Intervals_Plot.png"
plt.savefig(ci_plot_path, dpi=300)
plt.show()

print("Saved CI plot:", ci_plot_path)


# In[43]:


# ============================================================
# CONFIDENCE INTERVAL PLOT IN PERCENTAGE
# ============================================================

x = np.arange(len(ci_table))

# Convert values to percentage
values = ci_table["Value"].values * 100
lower_ci = ci_table["95% CI Lower"].values * 100
upper_ci = ci_table["95% CI Upper"].values * 100

lower_error = values - lower_ci
upper_error = upper_ci - values

plt.figure(figsize=(8, 5))

plt.errorbar(
    x,
    values,
    yerr=[lower_error, upper_error],
    fmt="o",
    capsize=6,
    linewidth=3
)

plt.xticks(x, ci_table["Metric"], rotation=20)
plt.ylabel("Performance Score (%)")
plt.ylim(0, 110)
plt.title("Lag–Lead Preprocessing Performance with 95% Confidence Intervals")
plt.grid(True, linestyle="--", alpha=1.0)
plt.tight_layout()

ci_plot_path = OUTPUT_DIR / "LagLead_Preprocessing_Final_Confidence_Intervals_Percentage_Plot.png"
plt.savefig(ci_plot_path, dpi=300)
plt.show()

print("Saved CI plot:", ci_plot_path)


# In[31]:


# ============================================================
# SHOW SENSITIVITY ANALYSIS PLOTS
# ============================================================

sensitivity_results = pd.read_csv(OUTPUT_DIR / "final_sensitivity_results.csv")

metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]

for parameter in ["K", "T1", "T2"]:

    temp_df = sensitivity_results[sensitivity_results["parameter"] == parameter].copy()
    temp_df = temp_df.sort_values("value")

    plt.figure(figsize=(8, 5))

    for metric in metrics_to_plot:
        plt.plot(
            temp_df["value"],
            temp_df[metric],
            marker="o",
            linewidth=2,
            label=metric.replace("_", " ").title()
        )

    plt.xlabel(parameter)
    plt.ylabel("Score")
    plt.title(f"Sensitivity Analysis of Lag–Lead Preprocessing Parameter {parameter}")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    save_path = OUTPUT_DIR / f"Sensitivity_LagLead_Preprocessing_{parameter}.png"
    plt.savefig(save_path, dpi=300)
    plt.show()

    print("Saved:", save_path)


# In[32]:


# ============================================================
# OVERALL SENSITIVITY ANALYSIS PLOT
# K, T1, and T2 in one figure
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

# Load final sensitivity results
sensitivity_results = pd.read_csv(
    OUTPUT_DIR / "final_sensitivity_results.csv"
)

metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
parameters = ["K", "T1", "T2"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, parameter in zip(axes, parameters):

    temp_df = sensitivity_results[
        sensitivity_results["parameter"] == parameter
    ].copy()

    temp_df = temp_df.sort_values("value")

    for metric in metrics_to_plot:
        ax.plot(
            temp_df["value"],
            temp_df[metric],
            marker="o",
            linewidth=2,
            label=metric.replace("_", " ").title()
        )

    ax.set_title(f"Parameter {parameter}")
    ax.set_xlabel(parameter)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)

axes[0].set_ylabel("Performance score")

# one common legend for all subplots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.08)
)

fig.suptitle(
    "Sensitivity Analysis of Lag–Lead Preprocessing Parameters",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0.08, 1, 0.92])

save_path = OUTPUT_DIR / "Overall_Sensitivity_LagLead_Preprocessing.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", save_path)


# In[44]:


# ============================================================
# OVERALL SENSITIVITY ANALYSIS PLOT IN PERCENTAGE
# K, T1, and T2 in one figure
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

# Load final sensitivity results
sensitivity_results = pd.read_csv(
    OUTPUT_DIR / "final_sensitivity_results.csv"
)

metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
parameters = ["K", "T1", "T2"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, parameter in zip(axes, parameters):

    temp_df = sensitivity_results[
        sensitivity_results["parameter"] == parameter
    ].copy()

    temp_df = temp_df.sort_values("value")

    for metric in metrics_to_plot:
        ax.plot(
            temp_df["value"],
            temp_df[metric] * 100,   # convert to percentage
            marker="o",
            linewidth=2,
            label=metric.replace("_", " ").title()
        )

    ax.set_title(f"Parameter {parameter}")
    ax.set_xlabel(parameter)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.5)

axes[0].set_ylabel("Performance Score (%)")

# one common legend for all subplots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.08)
)

fig.suptitle(
    "Sensitivity Analysis of Lag–Lead Preprocessing Parameters",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0.08, 1, 0.92])

save_path = OUTPUT_DIR / "Overall_Sensitivity_LagLead_Preprocessing_Percentage.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", save_path)


# In[34]:


# ============================================================
# THEORETICAL BODE PLOT FOR LAG–LEAD PREPROCESSING
# K = 1, T1 = 0.1, T2 = 0.05
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Output folder
OUTPUT_DIR = Path("LagLead_Preprocessing_SecondRevision")
OUTPUT_DIR.mkdir(exist_ok=True)

# Baseline parameters
K = 1.0
T1 = 0.10
T2 = 0.05

# Frequency range in rad/s
omega = np.logspace(-1, 3, 1000)
s = 1j * omega

# ------------------------------------------------------------
# Theoretical first-order responses
# Lag part: low-pass behavior
# Lead part: high-frequency detail enhancement behavior
# Lag–Lead Preprocessing: product response for theoretical visualization
# ------------------------------------------------------------

G_lag = K / (1 + T1 * s)

G_lead = K * (1 + T2 * s)

G_laglead = G_lag * G_lead

# Magnitude in dB
mag_lag = 20 * np.log10(np.abs(G_lag))
mag_lead = 20 * np.log10(np.abs(G_lead))
mag_laglead = 20 * np.log10(np.abs(G_laglead))

# Phase in degrees
phase_lag = np.angle(G_lag, deg=True)
phase_lead = np.angle(G_lead, deg=True)
phase_laglead = np.angle(G_laglead, deg=True)

# ============================================================
# Plot magnitude
# ============================================================

plt.figure(figsize=(9, 5))

plt.semilogx(omega, mag_lag, linewidth=2, label="Lag part")
plt.semilogx(omega, mag_lead, linewidth=2, label="Lead part")
plt.semilogx(omega, mag_laglead, linewidth=2, label="Lag–Lead Preprocessing")

plt.axvline(1 / T1, linestyle="--", linewidth=1, label="1/T1")
plt.axvline(1 / T2, linestyle="--", linewidth=1, label="1/T2")

plt.xlabel("Angular frequency ω (rad/s)")
plt.ylabel("Magnitude (dB)")
#plt.title("Theoretical Magnitude Response of Lag–Lead Preprocessing")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

mag_path = OUTPUT_DIR / "Theoretical_Bode_Magnitude_LagLead_Preprocessing.png"
plt.savefig(mag_path, dpi=300)
plt.show()

print("Saved magnitude plot:", mag_path)

# ============================================================
# Plot phase
# ============================================================

plt.figure(figsize=(9, 5))

plt.semilogx(omega, phase_lag, linewidth=2, label="Lag part")
plt.semilogx(omega, phase_lead, linewidth=2, label="Lead part")
plt.semilogx(omega, phase_laglead, linewidth=2, label="Lag–Lead Preprocessing")

plt.axvline(1 / T1, linestyle="--", linewidth=1, label="1/T1")
plt.axvline(1 / T2, linestyle="--", linewidth=1, label="1/T2")

plt.xlabel("Angular frequency ω (rad/s)")
plt.ylabel("Phase (degrees)")
#plt.title("Theoretical Phase Response of Lag–Lead Preprocessing")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

phase_path = OUTPUT_DIR / "Theoretical_Bode_Phase_LagLead_Preprocessing.png"
plt.savefig(phase_path, dpi=300)
plt.show()

print("Saved phase plot:", phase_path)


# In[35]:


# ============================================================
# THEORETICAL FREQUENCY-RESPONSE PLOT
# MAGNITUDE + PHASE IN ONE FIGURE
# K = 1, T1 = 0.10, T2 = 0.05
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("LagLead_Preprocessing_SecondRevision")
OUTPUT_DIR.mkdir(exist_ok=True)

# Baseline parameters
K = 1.0
T1 = 0.10
T2 = 0.05

# Frequency range
omega = np.logspace(-1, 3, 1000)
s = 1j * omega

# Theoretical transfer functions
G_lag = K / (1 + T1 * s)
G_lead = K * (1 + T2 * s)
G_laglead = G_lag * G_lead

# Magnitude
mag_lag = 20 * np.log10(np.abs(G_lag))
mag_lead = 20 * np.log10(np.abs(G_lead))
mag_laglead = 20 * np.log10(np.abs(G_laglead))

# Phase
phase_lag = np.angle(G_lag, deg=True)
phase_lead = np.angle(G_lead, deg=True)
phase_laglead = np.angle(G_laglead, deg=True)

# Corner frequencies
w1 = 1 / T1
w2 = 1 / T2

# ============================================================
# Plot magnitude and phase in one figure
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# ----------------------------
# Magnitude subplot
# ----------------------------
axes[0].semilogx(omega, mag_lag, linewidth=2, label="Lag part")
axes[0].semilogx(omega, mag_lead, linewidth=2, label="Lead part")
axes[0].semilogx(omega, mag_laglead, linewidth=2, label="Lag–Lead Preprocessing")

axes[0].axvline(w1, linestyle="--", linewidth=1, label="1/T1 = 10 rad/s")
axes[0].axvline(w2, linestyle="--", linewidth=1, label="1/T2 = 20 rad/s")

axes[0].set_ylabel("Magnitude (dB)")
#axes[0].set_title("Theoretical Frequency Response of Lag–Lead Preprocessing")
axes[0].grid(True, which="both", linestyle="--", alpha=0.5)
axes[0].legend(loc="best")

# ----------------------------
# Phase subplot
# ----------------------------
axes[1].semilogx(omega, phase_lag, linewidth=2, label="Lag part")
axes[1].semilogx(omega, phase_lead, linewidth=2, label="Lead part")
axes[1].semilogx(omega, phase_laglead, linewidth=2, label="Lag–Lead Preprocessing")

axes[1].axvline(w1, linestyle="--", linewidth=1, label="1/T1 = 10 rad/s")
axes[1].axvline(w2, linestyle="--", linewidth=1, label="1/T2 = 20 rad/s")

axes[1].set_xlabel("Angular frequency ω (rad/s)")
axes[1].set_ylabel("Phase (degrees)")
axes[1].grid(True, which="both", linestyle="--", alpha=0.5)
axes[1].legend(loc="best")

plt.tight_layout()

save_path = OUTPUT_DIR / "Theoretical_Frequency_Response_LagLead_Preprocessing.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", save_path)


# In[45]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data
data = {
    "Accuracy":   [0.80, 0.84, 0.96],
    "Precision":  [0.93, 0.92, 0.94],
    "Recall":     [0.79, 0.86, 0.97],
    "F1_score":   [0.85, 0.89, 0.95]
}

index = ["Gaussian", "Laplacian", "Lag-Lead"]

df = pd.DataFrame(data, index=index)

# Plot heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(
    df,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    linewidths=0.5,
    cbar_kws={"label": "Score"}
)

plt.title("Performance Heatmap of Preprocessing Methods")
plt.xlabel("Evaluation Metrics")
plt.ylabel("Preprocessing Methods")
plt.xticks(rotation=30)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# In[4]:


from pathlib import Path

OUTPUT_DIR = Path("LagLead_Preprocessing_SecondRevision")

FINAL_PREDICTION_FILE = OUTPUT_DIR / "predictions_LagLead_Preprocessing_K_1p0.csv"

print("Prediction file path:", FINAL_PREDICTION_FILE)
print("File exists:", FINAL_PREDICTION_FILE.exists())


# In[7]:


from pathlib import Path
import pandas as pd

FINAL_PREDICTION_FILE = Path(
    "LagLead_Preprocessing_SecondRevision/predictions_LagLead_Preprocessing_K_1p0.csv"
)

print("Prediction file path:", FINAL_PREDICTION_FILE)
print("File exists:", FINAL_PREDICTION_FILE.exists())

pred_df = pd.read_csv(FINAL_PREDICTION_FILE)
pred_df.head()


# In[9]:


# ============================================================
# WILSON SCORE CONFIDENCE INTERVALS
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np

from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------
# Define file path
# ------------------------------------------------------------

OUTPUT_DIR = Path("LagLead_Preprocessing_SecondRevision")

FINAL_PREDICTION_FILE = OUTPUT_DIR / "predictions_LagLead_Preprocessing_K_1p0.csv"

print("Reading:", FINAL_PREDICTION_FILE)
print("File exists:", FINAL_PREDICTION_FILE.exists())

# ------------------------------------------------------------
# Read prediction file
# ------------------------------------------------------------

pred_df = pd.read_csv(FINAL_PREDICTION_FILE)

y_true = pred_df["y_true"].values.astype(int)
y_pred = pred_df["y_prediction"].values.astype(int)

# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print("TN =", tn)
print("FP =", fp)
print("FN =", fn)
print("TP =", tp)

# ------------------------------------------------------------
# Wilson CI function
# ------------------------------------------------------------

def wilson_ci(successes, total, alpha=0.05):
    lower, upper = proportion_confint(
        count=successes,
        nobs=total,
        alpha=alpha,
        method="wilson"
    )
    return lower, upper

# ------------------------------------------------------------
# Metric values
# ------------------------------------------------------------

total = tp + tn + fp + fn

accuracy = (tp + tn) / total
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# ------------------------------------------------------------
# Wilson confidence intervals
# ------------------------------------------------------------

acc_ci = wilson_ci(tp + tn, total)
precision_ci = wilson_ci(tp, tp + fp)
recall_ci = wilson_ci(tp, tp + fn)
specificity_ci = wilson_ci(tn, tn + fp)

# ------------------------------------------------------------
# Create table
# ------------------------------------------------------------

wilson_table = pd.DataFrame([
    {
        "Metric": "Accuracy",
        "Value": accuracy,
        "95% CI Lower": acc_ci[0],
        "95% CI Upper": acc_ci[1]
    },
    {
        "Metric": "Precision",
        "Value": precision,
        "95% CI Lower": precision_ci[0],
        "95% CI Upper": precision_ci[1]
    },
    {
        "Metric": "Recall/Sensitivity",
        "Value": recall,
        "95% CI Lower": recall_ci[0],
        "95% CI Upper": recall_ci[1]
    },
    {
        "Metric": "Specificity",
        "Value": specificity,
        "95% CI Lower": specificity_ci[0],
        "95% CI Upper": specificity_ci[1]
    }
])

wilson_table["Report Format"] = wilson_table.apply(
    lambda row: f"{row['Value']:.3f} [{row['95% CI Lower']:.3f}, {row['95% CI Upper']:.3f}]",
    axis=1
)

wilson_table_path = OUTPUT_DIR / "Wilson_Score_Confidence_Intervals.csv"
wilson_table.to_csv(wilson_table_path, index=False)

print("Saved:", wilson_table_path)
wilson_table


# In[10]:


from pathlib import Path

OUTPUT_DIR = Path("LagLead_Preprocessing_SecondRevision")

for file in OUTPUT_DIR.glob("*.csv"):
    print(file.name)


# In[11]:


# ============================================================
# WILSON SCORE CONFIDENCE INTERVAL PLOT
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Output folder
OUTPUT_DIR = Path("LagLead_Preprocessing_SecondRevision")
OUTPUT_DIR.mkdir(exist_ok=True)

# Wilson score CI values
wilson_table = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall/Sensitivity",
        "Specificity"
    ],
    "Value": [
        0.774648,
        0.893617,
        0.792453,
        0.722222
    ],
    "95% CI Lower": [
        0.664848,
        0.774057,
        0.665423,
        0.491273
    ],
    "95% CI Upper": [
        0.856253,
        0.953695,
        0.879954,
        0.875002
    ]
})

# Convert to percentage
x = np.arange(len(wilson_table))

values = wilson_table["Value"].values * 100
lower_ci = wilson_table["95% CI Lower"].values * 100
upper_ci = wilson_table["95% CI Upper"].values * 100

lower_error = values - lower_ci
upper_error = upper_ci - values

# Plot
plt.figure(figsize=(8, 5))

plt.errorbar(
    x,
    values,
    yerr=[lower_error, upper_error],
    fmt="o",
    capsize=7,
    linewidth=2.5,
    markersize=8
)

plt.xticks(x, wilson_table["Metric"], rotation=20)
plt.ylabel("Performance Score (%)")
plt.ylim(0, 105)
plt.title("Wilson Score 95% Confidence Intervals for Final Test Performance")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

save_path = OUTPUT_DIR / "Wilson_Score_Confidence_Interval_Plot.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", save_path)


# In[ ]:




