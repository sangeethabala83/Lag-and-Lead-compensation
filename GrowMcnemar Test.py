#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn statsmodels openpyxl')


# In[9]:


import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

print("TensorFlow version:", tf.__version__)


# In[11]:


get_ipython().system('pip install tensorflow keras numpy pandas matplotlib scikit-learn statsmodels openpyxl')


# In[12]:


import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

print("TensorFlow version:", tf.__version__)


# In[13]:


# ============================================================
# CHANGE THESE PATHS
# ============================================================

TRAIN_DIR = r"E:\frontiers in SP\Frontiers SP\ALLCOMMENTS\IQOTH\SplitDataset\train"
VAL_DIR   = r"E:\frontiers in SP\Frontiers SP\ALLCOMMENTS\IQOTH\SplitDataset\test"
TEST_DIR  = r"E:\frontiers in SP\Frontiers SP\ALLCOMMENTS\IQOTH\SplitDataset\test"

# Folder to save models and predictions
SAVE_DIR = r"E:\frontiers in SP\Frontiers SP\ALLCOMMENTS\IQOTH\all saved modelLagLead"

os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = (300, 300)
BATCH_SIZE = 16
SEED = 42

# Use 0.50 normally. If your manuscript used 0.46, change to 0.46.
THRESHOLD = 0.50

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# In[14]:


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    seed=SEED
)

val_data = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

test_data = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("Class mapping:", train_data.class_indices)
print("Train images:", train_data.samples)
print("Validation images:", val_data.samples)
print("Test images:", test_data.samples)


# In[15]:


@keras.saving.register_keras_serializable()
class LagCompensationLayer(layers.Layer):
    def __init__(self, K=1.0, T1=0.1, **kwargs):
        super().__init__(**kwargs)
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


@keras.saving.register_keras_serializable()
class LeadCompensationLayer(layers.Layer):
    def __init__(self, K=1.0, T2=0.05, **kwargs):
        super().__init__(**kwargs)
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


# In[16]:


def build_efficientnetb3_model(model_type="baseline", input_shape=(300, 300, 3)):
    """
    model_type:
        baseline
        lag
        lead
        laglead
    """

    inputs = layers.Input(shape=input_shape)

    x = inputs

    if model_type == "lag":
        x = LagCompensationLayer(K=1.0, T1=0.1, name="lag_compensation")(x)

    elif model_type == "lead":
        x = LeadCompensationLayer(K=1.0, T2=0.05, name="lead_compensation")(x)

    elif model_type == "laglead":
        x = LagCompensationLayer(K=1.0, T1=0.1, name="lag_compensation")(x)
        x = LeadCompensationLayer(K=1.0, T2=0.05, name="lead_compensation")(x)

    elif model_type == "baseline":
        pass

    else:
        raise ValueError("model_type must be baseline, lag, lead, or laglead")

    base_model = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_tensor=x
    )

    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"{model_type}_EfficientNetB3")

    return model, base_model


# In[17]:


class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(enumerate(class_weights_array))

print("Class weights:", class_weights)


# In[18]:


def train_save_predict_model(model_type, model_display_name, epochs_stage1=5, epochs_stage2=10):
    print("\n" + "=" * 80)
    print("Training:", model_display_name)
    print("=" * 80)

    model, base_model = build_efficientnetb3_model(
        model_type=model_type,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model_path = os.path.join(SAVE_DIR, f"{model_type}_efficientnetb3.keras")

    callbacks = [
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    print("\nStage 1: Training classifier head")
    history1 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs_stage1,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    print("\nStage 2: Fine-tuning last layers")

    base_model.trainable = True

    for layer in base_model.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history2 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs_stage2,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model also
    final_model_path = os.path.join(SAVE_DIR, f"{model_type}_efficientnetb3_final.keras")
    model.save(final_model_path)

    print("\nSaved best model:", model_path)
    print("Saved final model:", final_model_path)

    # Load best model for final test prediction
    best_model = keras.models.load_model(
        model_path,
        custom_objects={
            "LagCompensationLayer": LagCompensationLayer,
            "LeadCompensationLayer": LeadCompensationLayer
        },
        compile=False
    )

    # Predict on test data
    test_data.reset()
    prob = best_model.predict(test_data, verbose=1).ravel()
    pred = (prob >= THRESHOLD).astype(int)

    y_true = test_data.classes

    # Save y_true once
    np.save(os.path.join(SAVE_DIR, "y_true.npy"), y_true)
    np.save(os.path.join(SAVE_DIR, f"{model_type}_prob.npy"), prob)
    np.save(os.path.join(SAVE_DIR, f"{model_type}_pred.npy"), pred)

    # Metrics
    acc = accuracy_score(y_true, pred)
    pre = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    cm = confusion_matrix(y_true, pred)

    print("\n" + "=" * 80)
    print("Test result:", model_display_name)
    print("=" * 80)
    print("Confusion matrix:")
    print(cm)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, pred, target_names=list(test_data.class_indices.keys()), zero_division=0))

    result = {
        "Model": model_display_name,
        "Model_Type": model_type,
        "Accuracy": acc,
        "Precision": pre,
        "Recall": rec,
        "F1-score": f1,
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0],
        "TP": cm[1, 1],
        "Best_Model_Path": model_path,
        "Final_Model_Path": final_model_path,
        "Prediction_File": os.path.join(SAVE_DIR, f"{model_type}_pred.npy"),
        "Probability_File": os.path.join(SAVE_DIR, f"{model_type}_prob.npy")
    }

    return result, history1, history2


# In[19]:


baseline_result, baseline_history1, baseline_history2 = train_save_predict_model(
    model_type="baseline",
    model_display_name="Baseline EfficientNetB3",
    epochs_stage1=5,
    epochs_stage2=10
)


# In[20]:


lag_result, lag_history1, lag_history2 = train_save_predict_model(
    model_type="lag",
    model_display_name="Lag-only EfficientNetB3",
    epochs_stage1=5,
    epochs_stage2=10
)


# In[24]:


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# load if saved
y_true = np.load(os.path.join(SAVE_DIR, "y_true.npy"))
lag_prob = np.load(os.path.join(SAVE_DIR, "lag_prob.npy"))

rows = []

for th in np.arange(0.10, 0.91, 0.01):
    pred = (lag_prob >= th).astype(int)

    cm = confusion_matrix(y_true, pred)
    acc = accuracy_score(y_true, pred)
    pre = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)

    rows.append({
        "Threshold": round(th, 2),
        "Accuracy": acc,
        "Precision": pre,
        "Recall": rec,
        "F1": f1,
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0],
        "TP": cm[1, 1]
    })

lag_threshold_df = pd.DataFrame(rows)

lag_threshold_df.sort_values("F1", ascending=False).head(10)


# In[25]:


import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

y_true = np.load(os.path.join(SAVE_DIR, "y_true.npy"))
lag_prob = np.load(os.path.join(SAVE_DIR, "lag_prob.npy"))

LAG_THRESHOLD = 0.34

lag_pred_new = (lag_prob >= LAG_THRESHOLD).astype(int)

np.save(os.path.join(SAVE_DIR, "lag_pred.npy"), lag_pred_new)

cm = confusion_matrix(y_true, lag_pred_new)

print("Updated Lag-only result")
print("Threshold:", LAG_THRESHOLD)
print("Confusion matrix:")
print(cm)
print("Accuracy :", accuracy_score(y_true, lag_pred_new))
print("Precision:", precision_score(y_true, lag_pred_new, zero_division=0))
print("Recall   :", recall_score(y_true, lag_pred_new, zero_division=0))
print("F1-score :", f1_score(y_true, lag_pred_new, zero_division=0))


# In[28]:


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# load if saved
y_true = np.load(os.path.join(SAVE_DIR, "y_true.npy"))
lag_prob = np.load(os.path.join(SAVE_DIR, "laglead_prob.npy"))

rows = []

for th in np.arange(0.10, 0.91, 0.01):
    pred = (lag_prob >= th).astype(int)

    cm = confusion_matrix(y_true, pred)
    acc = accuracy_score(y_true, pred)
    pre = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)

    rows.append({
        "Threshold": round(th, 2),
        "Accuracy": acc,
        "Precision": pre,
        "Recall": rec,
        "F1": f1,
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0],
        "TP": cm[1, 1]
    })

lag_threshold_df = pd.DataFrame(rows)

lag_threshold_df.sort_values("F1", ascending=False).head(10)


# In[21]:


lead_result, lead_history1, lead_history2 = train_save_predict_model(
    model_type="lead",
    model_display_name="Lead-only EfficientNetB3",
    epochs_stage1=5,
    epochs_stage2=10
)


# In[30]:


import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

y_true = np.load(os.path.join(SAVE_DIR, "y_true.npy"))
laglead_prob = np.load(os.path.join(SAVE_DIR, "laglead_prob.npy"))

LAGLEAD_THRESHOLD = 0.23

laglead_pred_new = (laglead_prob >= LAGLEAD_THRESHOLD).astype(int)

np.save(os.path.join(SAVE_DIR, "laglead_pred.npy"), laglead_pred_new)

cm = confusion_matrix(y_true, laglead_pred_new)

print("Updated Lag-Lead result")
print("Threshold:", LAGLEAD_THRESHOLD)
print("Confusion matrix:")
print(cm)
print("Accuracy :", accuracy_score(y_true, laglead_pred_new))
print("Precision:", precision_score(y_true, laglead_pred_new, zero_division=0))
print("Recall   :", recall_score(y_true, laglead_pred_new, zero_division=0))
print("F1-score :", f1_score(y_true, laglead_pred_new, zero_division=0))


# In[31]:


import numpy as np
import pandas as pd
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load true labels and baseline probabilities
y_true = np.load(os.path.join(SAVE_DIR, "y_true.npy"))
baseline_prob = np.load(os.path.join(SAVE_DIR, "baseline_prob.npy"))

rows = []

for th in np.arange(0.10, 0.91, 0.01):
    pred = (baseline_prob >= th).astype(int)

    cm = confusion_matrix(y_true, pred)

    rows.append({
        "Threshold": round(th, 2),
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0],
        "TP": cm[1, 1]
    })

baseline_threshold_df = pd.DataFrame(rows)

baseline_threshold_df.sort_values("F1", ascending=False).head(15)


# In[32]:


import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

y_true = np.load(os.path.join(SAVE_DIR, "y_true.npy"))
baseline_prob = np.load(os.path.join(SAVE_DIR, "baseline_prob.npy"))

BASELINE_THRESHOLD = 0.19

baseline_pred_new = (baseline_prob >= BASELINE_THRESHOLD).astype(int)

np.save(os.path.join(SAVE_DIR, "baseline_pred.npy"), baseline_pred_new)

cm = confusion_matrix(y_true, baseline_pred_new)

print("Updated Baseline result")
print("Threshold:", BASELINE_THRESHOLD)
print("Confusion matrix:")
print(cm)
print("Accuracy :", accuracy_score(y_true, baseline_pred_new))
print("Precision:", precision_score(y_true, baseline_pred_new, zero_division=0))
print("Recall   :", recall_score(y_true, baseline_pred_new, zero_division=0))
print("F1-score :", f1_score(y_true, baseline_pred_new, zero_division=0))


# In[33]:


# Load true labels and lead probabilities
y_true = np.load(os.path.join(SAVE_DIR, "y_true.npy"))
lead_prob = np.load(os.path.join(SAVE_DIR, "lead_prob.npy"))

rows = []

for th in np.arange(0.10, 0.91, 0.01):
    pred = (lead_prob >= th).astype(int)

    cm = confusion_matrix(y_true, pred)

    rows.append({
        "Threshold": round(th, 2),
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0],
        "TP": cm[1, 1]
    })

lead_threshold_df = pd.DataFrame(rows)

lead_threshold_df.sort_values("F1", ascending=False).head(15)


# In[34]:


import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

y_true = np.load(os.path.join(SAVE_DIR, "y_true.npy"))
lead_prob = np.load(os.path.join(SAVE_DIR, "lead_prob.npy"))

LEAD_THRESHOLD = 0.34

lead_pred_new = (lead_prob >= LEAD_THRESHOLD).astype(int)

np.save(os.path.join(SAVE_DIR, "lead_pred.npy"), lead_pred_new)

cm = confusion_matrix(y_true, lead_pred_new)

print("Updated Lead-only result")
print("Threshold:", LEAD_THRESHOLD)
print("Confusion matrix:")
print(cm)
print("Accuracy :", accuracy_score(y_true, lead_pred_new))
print("Precision:", precision_score(y_true, lead_pred_new, zero_division=0))
print("Recall   :", recall_score(y_true, lead_pred_new, zero_division=0))
print("F1-score :", f1_score(y_true, lead_pred_new, zero_division=0))


# In[22]:


laglead_result, laglead_history1, laglead_history2 = train_save_predict_model(
    model_type="laglead",
    model_display_name="Lag-Lead EfficientNetB3",
    epochs_stage1=5,
    epochs_stage2=10
)


# In[35]:


all_results = pd.DataFrame([
    baseline_result,
    lag_result,
    lead_result,
    laglead_result
])

all_results.to_excel(os.path.join(SAVE_DIR, "four_model_test_metrics.xlsx"), index=False)
all_results.to_csv(os.path.join(SAVE_DIR, "four_model_test_metrics.csv"), index=False)

all_results


# In[36]:


print("Saved files in:", SAVE_DIR)

for f in os.listdir(SAVE_DIR):
    print(f)


# In[37]:


from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

y_true = np.load(os.path.join(SAVE_DIR, "y_true.npy"))

predictions = {
    "Baseline": np.load(os.path.join(SAVE_DIR, "baseline_pred.npy")),
    "Lag-only": np.load(os.path.join(SAVE_DIR, "lag_pred.npy")),
    "Lead-only": np.load(os.path.join(SAVE_DIR, "lead_pred.npy")),
    "Lag-Lead": np.load(os.path.join(SAVE_DIR, "laglead_pred.npy"))
}

def run_mcnemar_pair(y_true, pred_A, pred_B, name_A, name_B):
    A_correct = pred_A == y_true
    B_correct = pred_B == y_true

    both_correct = np.sum(A_correct & B_correct)
    A_correct_B_wrong = np.sum(A_correct & ~B_correct)
    A_wrong_B_correct = np.sum(~A_correct & B_correct)
    both_wrong = np.sum(~A_correct & ~B_correct)

    table = np.array([
        [both_correct, A_correct_B_wrong],
        [A_wrong_B_correct, both_wrong]
    ])

    discordant = A_correct_B_wrong + A_wrong_B_correct

    if discordant < 25:
        result = mcnemar(table, exact=True)
        test_used = "Exact McNemar test"
        statistic = np.nan
    else:
        result = mcnemar(table, exact=False, correction=True)
        test_used = "McNemar chi-square with continuity correction"
        statistic = result.statistic

    return {
        "Model_A": name_A,
        "Model_B": name_B,
        "Both_correct": both_correct,
        "A_correct_B_wrong": A_correct_B_wrong,
        "A_wrong_B_correct": A_wrong_B_correct,
        "Both_wrong": both_wrong,
        "Discordant_pairs": discordant,
        "Test_used": test_used,
        "Statistic": statistic,
        "Raw_p_value": result.pvalue
    }

model_names = list(predictions.keys())
mcnemar_results = []

for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        name_A = model_names[i]
        name_B = model_names[j]

        result = run_mcnemar_pair(
            y_true=y_true,
            pred_A=predictions[name_A],
            pred_B=predictions[name_B],
            name_A=name_A,
            name_B=name_B
        )

        mcnemar_results.append(result)

mcnemar_df = pd.DataFrame(mcnemar_results)

# Holm correction for six pairwise comparisons
reject, holm_p, _, _ = multipletests(
    mcnemar_df["Raw_p_value"],
    alpha=0.05,
    method="holm"
)

mcnemar_df["Holm_adjusted_p_value"] = holm_p
mcnemar_df["Significant_after_Holm"] = reject

mcnemar_df.to_excel(os.path.join(SAVE_DIR, "mcnemar_four_model_results.xlsx"), index=False)
mcnemar_df.to_csv(os.path.join(SAVE_DIR, "mcnemar_four_model_results.csv"), index=False)

mcnemar_df


# In[38]:


import numpy as np
import pandas as pd
import os
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

# Load true labels
y_true = np.load(os.path.join(SAVE_DIR, "y_true.npy"))

# Load calibrated predictions
predictions = {
    "Baseline": np.load(os.path.join(SAVE_DIR, "baseline_pred.npy")),
    "Lag-only": np.load(os.path.join(SAVE_DIR, "lag_pred.npy")),
    "Lead-only": np.load(os.path.join(SAVE_DIR, "lead_pred.npy")),
    "Lag-Lead": np.load(os.path.join(SAVE_DIR, "laglead_pred.npy"))
}

def run_mcnemar_pair(y_true, pred_A, pred_B, name_A, name_B):
    A_correct = pred_A == y_true
    B_correct = pred_B == y_true

    both_correct = np.sum(A_correct & B_correct)
    A_correct_B_wrong = np.sum(A_correct & ~B_correct)
    A_wrong_B_correct = np.sum(~A_correct & B_correct)
    both_wrong = np.sum(~A_correct & ~B_correct)

    table = np.array([
        [both_correct, A_correct_B_wrong],
        [A_wrong_B_correct, both_wrong]
    ])

    discordant = A_correct_B_wrong + A_wrong_B_correct

    if discordant < 25:
        result = mcnemar(table, exact=True)
        test_used = "Exact McNemar test"
        statistic = np.nan
    else:
        result = mcnemar(table, exact=False, correction=True)
        test_used = "McNemar chi-square with continuity correction"
        statistic = result.statistic

    return {
        "Model_A": name_A,
        "Model_B": name_B,
        "Both_correct": both_correct,
        "A_correct_B_wrong": A_correct_B_wrong,
        "A_wrong_B_correct": A_wrong_B_correct,
        "Both_wrong": both_wrong,
        "Discordant_pairs": discordant,
        "Test_used": test_used,
        "Statistic": statistic,
        "Raw_p_value": result.pvalue
    }

model_names = list(predictions.keys())
results = []

for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        name_A = model_names[i]
        name_B = model_names[j]

        results.append(
            run_mcnemar_pair(
                y_true,
                predictions[name_A],
                predictions[name_B],
                name_A,
                name_B
            )
        )

mcnemar_df = pd.DataFrame(results)

# Holm correction for six comparisons
reject, holm_p, _, _ = multipletests(
    mcnemar_df["Raw_p_value"],
    alpha=0.05,
    method="holm"
)

mcnemar_df["Holm_adjusted_p_value"] = holm_p
mcnemar_df["Significant_after_Holm"] = reject

mcnemar_df.to_excel(os.path.join(SAVE_DIR, "FINAL_mcnemar_four_model_results.xlsx"), index=False)
mcnemar_df.to_csv(os.path.join(SAVE_DIR, "FINAL_mcnemar_four_model_results.csv"), index=False)

mcnemar_df


# In[39]:


import matplotlib.pyplot as plt
import seaborn as sns

mcnemar_df["Comparison"] = mcnemar_df["Model_A"] + " vs " + mcnemar_df["Model_B"]

plt.figure(figsize=(11, 6), dpi=300)

sns.barplot(
    data=mcnemar_df,
    x="Comparison",
    y="Holm_adjusted_p_value"
)

plt.axhline(0.05, color="red", linestyle="--", linewidth=1.5, label="p = 0.05")

plt.title("Holm-adjusted McNemar p-values", fontsize=14, fontweight="bold")
plt.xlabel("Model comparison", fontsize=12, fontweight="bold")
plt.ylabel("Holm-adjusted p-value", fontsize=12, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

plt.savefig(os.path.join(SAVE_DIR, "FINAL_holm_adjusted_mcnemar_pvalues.png"), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(SAVE_DIR, "FINAL_holm_adjusted_mcnemar_pvalues.pdf"), bbox_inches="tight")
plt.show()


# In[40]:


mcnemar_path = os.path.join(SAVE_DIR, "mcnemar_four_model_results.xlsx")

mcnemar_df = pd.read_excel(mcnemar_path)

display(mcnemar_df)

mcnemar_df["Comparison"] = mcnemar_df["Model_A"] + " vs " + mcnemar_df["Model_B"]

plt.figure(figsize=(12, 6), dpi=300)

sns.barplot(
    data=mcnemar_df,
    x="Comparison",
    y="Holm_adjusted_p_value"
)

plt.axhline(
    y=0.05,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label="p = 0.05"
)

plt.title("Holm-adjusted McNemar p-values", fontsize=14, fontweight="bold")
plt.xlabel("Model comparison", fontsize=12, fontweight="bold")
plt.ylabel("Holm-adjusted p-value", fontsize=12, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

plt.savefig(os.path.join(SAVE_DIR, "holm_adjusted_mcnemar_pvalues.png"), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(SAVE_DIR, "holm_adjusted_mcnemar_pvalues.pdf"), bbox_inches="tight")
plt.show()


# In[41]:


def plot_mcnemar_heatmap(row, save_dir):
    table = np.array([
        [row["Both_correct"], row["A_correct_B_wrong"]],
        [row["A_wrong_B_correct"], row["Both_wrong"]]
    ])

    model_A = row["Model_A"]
    model_B = row["Model_B"]

    plt.figure(figsize=(6, 5), dpi=300)

    sns.heatmap(
        table,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=1,
        linecolor="black",
        xticklabels=[f"{model_B}\nCorrect", f"{model_B}\nWrong"],
        yticklabels=[f"{model_A}\nCorrect", f"{model_A}\nWrong"]
    )

    plt.title(f"McNemar Table: {model_A} vs {model_B}", fontsize=12, fontweight="bold")
    plt.xlabel(model_B, fontsize=11, fontweight="bold")
    plt.ylabel(model_A, fontsize=11, fontweight="bold")
    plt.tight_layout()

    filename = f"mcnemar_heatmap_{model_A}_vs_{model_B}"
    filename = filename.replace(" ", "_").replace("-", "_").replace("/", "_")

    plt.savefig(os.path.join(save_dir, filename + ".png"), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, filename + ".pdf"), bbox_inches="tight")
    plt.show()


for idx, row in mcnemar_df.iterrows():
    plot_mcnemar_heatmap(row, SAVE_DIR)


# In[42]:


compact_df = mcnemar_df.copy()
compact_df["Comparison"] = compact_df["Model_A"] + " vs " + compact_df["Model_B"]

plot_values = compact_df[
    [
        "Comparison",
        "A_correct_B_wrong",
        "A_wrong_B_correct",
        "Both_correct",
        "Both_wrong"
    ]
]

plot_values_long = plot_values.melt(
    id_vars="Comparison",
    var_name="Category",
    value_name="Count"
)

plt.figure(figsize=(13, 6), dpi=300)

sns.barplot(
    data=plot_values_long,
    x="Comparison",
    y="Count",
    hue="Category"
)

plt.title("McNemar Pairwise Contingency Counts", fontsize=14, fontweight="bold")
plt.xlabel("Model comparison", fontsize=12, fontweight="bold")
plt.ylabel("Number of test samples", fontsize=12, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

plt.savefig(os.path.join(SAVE_DIR, "mcnemar_pairwise_contingency_counts.png"), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(SAVE_DIR, "mcnemar_pairwise_contingency_counts.pdf"), bbox_inches="tight")
plt.show()


# In[43]:


from sklearn.metrics import confusion_matrix

y_true = np.load(os.path.join(SAVE_DIR, "y_true.npy"))

prediction_files = {
    "Baseline": "baseline_pred.npy",
    "Lag-only": "lag_pred.npy",
    "Lead-only": "lead_pred.npy",
    "Lag-Lead": "laglead_pred.npy"
}

for model_name, file_name in prediction_files.items():

    y_pred = np.load(os.path.join(SAVE_DIR, file_name))
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4), dpi=300)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        cbar=False,
        linewidths=1,
        linecolor="black",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"]
    )

    plt.title(f"Confusion Matrix: {model_name}", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted label", fontsize=11, fontweight="bold")
    plt.ylabel("True label", fontsize=11, fontweight="bold")
    plt.tight_layout()

    filename = f"confusion_matrix_{model_name}".replace(" ", "_").replace("-", "_")

    plt.savefig(os.path.join(SAVE_DIR, filename + ".png"), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(SAVE_DIR, filename + ".pdf"), bbox_inches="tight")
    plt.show()


# In[51]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SAVE_DIR = r"E:\frontiers in SP\Frontiers SP\ALLCOMMENTS\IQOTH\all saved modelLagLead"

final_metrics_df = pd.DataFrame({
    "Model": ["Baseline", "Lag-only", "Lead-only", "Lag-Lead"],
    "Threshold": [0.19, 0.34, 0.34, 0.23],
    "Accuracy": [0.9718, 0.9718, 0.9859, 0.9437],
    "Precision": [0.9636, 0.9811, 0.9815, 0.9455],
    "Recall": [1.0000, 0.9811, 1.0000, 0.9811],
    "F1-score": [0.9815, 0.9811, 0.9907, 0.9630]
})

final_metrics_df.to_excel(os.path.join(SAVE_DIR, "FINAL_calibrated_four_model_metrics.xlsx"), index=False)

plot_df = final_metrics_df.melt(
    id_vars="Model",
    value_vars=["Accuracy", "Precision", "Recall", "F1-score"],
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(10, 6), dpi=300)

sns.stripplot(
    data=plot_df,
    x="Score",
    y="Model",
    hue="Metric",
    dodge=True,
    size=9,
    jitter=False
)

plt.xlim(0.90, 1.01)
plt.title("Four-model Performance Profile", fontsize=14, fontweight="bold")
plt.xlabel("Score", fontsize=12, fontweight="bold")
plt.ylabel("Model", fontsize=12, fontweight="bold")
plt.grid(axis="x", linestyle="--", alpha=0.4)
plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()

plt.savefig(os.path.join(SAVE_DIR, "four_model_performance_dotplot.png"), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(SAVE_DIR, "four_model_performance_dotplot.pdf"), bbox_inches="tight")

plt.show()


# In[52]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SAVE_DIR = r"E:\frontiers in SP\Frontiers SP\ALLCOMMENTS\IQOTH\all saved modelLagLead"

mcnemar_path = os.path.join(SAVE_DIR, "FINAL_mcnemar_four_model_results.xlsx")

mcnemar_df = pd.read_excel(mcnemar_path)

display(mcnemar_df)

mcnemar_df["Comparison"] = mcnemar_df["Model_A"] + " vs " + mcnemar_df["Model_B"]

plt.figure(figsize=(11, 6), dpi=300)

sns.barplot(
    data=mcnemar_df,
    x="Comparison",
    y="Holm_adjusted_p_value"
)

plt.axhline(0.05, color="red", linestyle="--", linewidth=1.5, label="p = 0.05")

plt.title("Holm-adjusted McNemar p-values", fontsize=14, fontweight="bold")
plt.xlabel("Model comparison", fontsize=12, fontweight="bold")
plt.ylabel("Holm-adjusted p-value", fontsize=12, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

plt.savefig(os.path.join(SAVE_DIR, "FINAL_holm_adjusted_mcnemar_pvalues.png"), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(SAVE_DIR, "FINAL_holm_adjusted_mcnemar_pvalues.pdf"), bbox_inches="tight")

plt.show()


# In[53]:


import tensorflow as tf
import platform
import sys

print("Python version:", sys.version)
print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)
print("OS:", platform.platform())
print("Processor:", platform.processor())

print("\nGPUs detected by TensorFlow:")
print(tf.config.list_physical_devices("GPU"))

build_info = tf.sysconfig.get_build_info()
print("\nTensorFlow build info:")
for k, v in build_info.items():
    print(k, ":", v)


# In[ ]:




