from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# -------------------------------------------------
# PATHS
# -------------------------------------------------
FILE_DIR = Path(__file__).resolve().parent
ROOT_DIR = FILE_DIR.parent

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

DATASET = DATA_DIR / "behavior_training_data.csv"

LSTM_MODEL_FILE = MODELS_DIR / "lstm_behavior_model.keras"
SCALER_FILE = MODELS_DIR / "lstm_scaler.pkl"
META_FILE = MODELS_DIR / "lstm_model_meta.json"

MODELS_DIR.mkdir(exist_ok=True)


# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
SEQUENCE_LENGTH = 5
THRESHOLD = 0.5

FEATURES = [
    "avg_typing_speed",
    "location_encoded",
    "login_hour",
    "failed_attempts",
    "location_mismatch",
    "password_length",
    "day_of_week",
]

TARGET = "label"


# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------
if not DATASET.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET}")

df = pd.read_csv(DATASET)

required_cols = FEATURES + [TARGET]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

df = df.dropna(subset=required_cols).reset_index(drop=True)

for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)


# -------------------------------------------------
# SCALE FEATURES
# -------------------------------------------------
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[FEATURES])


# -------------------------------------------------
# CREATE SEQUENCES
# -------------------------------------------------
X_sequences = []
y_sequences = []

for i in range(SEQUENCE_LENGTH, len(scaled_features) + 1):
    sequence = scaled_features[i - SEQUENCE_LENGTH:i]
    label = df[TARGET].iloc[i - 1]

    X_sequences.append(sequence)
    y_sequences.append(label)

X = np.array(X_sequences)
y = np.array(y_sequences)

if len(X) == 0:
    raise ValueError("Not enough rows to create LSTM sequences.")

print("\nSequence dataset created")
print("X shape:", X.shape)
print("y shape:", y.shape)


# -------------------------------------------------
# TRAIN / TEST SPLIT
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# -------------------------------------------------
# BUILD LSTM MODEL
# -------------------------------------------------
model = Sequential(
    [
        LSTM(
            64,
            input_shape=(SEQUENCE_LENGTH, len(FEATURES)),
            return_sequences=False,
        ),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=80,
    batch_size=8,
    callbacks=[early_stop],
    verbose=1,
)


# -------------------------------------------------
# EVALUATION
# -------------------------------------------------
y_prob = model.predict(X_test).ravel()
y_pred = (y_prob >= THRESHOLD).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)


# -------------------------------------------------
# SAVE MODEL, SCALER, METADATA
# -------------------------------------------------
model.save(LSTM_MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)

meta = {
    "model_type": "LSTM",
    "purpose": "Sequential behavioural login analysis",
    "features": FEATURES,
    "target": TARGET,
    "sequence_length": SEQUENCE_LENGTH,
    "threshold": THRESHOLD,
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "roc_auc": float(roc_auc),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "dataset_rows": int(len(df)),
    "sequence_rows": int(len(X)),
    "confusion_matrix": cm.tolist(),
    "label_distribution": {
        str(label): int(count)
        for label, count in df[TARGET].value_counts().sort_index().items()
    },
    "note": (
        "This LSTM model is used as a secondary sequential model. "
        "It requires enough login history to create a sequence. "
        "Random Forest remains the primary deployed model for single-login prediction."
    ),
}

with open(META_FILE, "w") as f:
    json.dump(meta, f, indent=4)


# -------------------------------------------------
# OUTPUT
# -------------------------------------------------
print("\nLSTM Training Complete")
print("-" * 40)
print("Dataset rows:", len(df))
print("Sequence rows:", len(X))
print("Training sequences:", len(X_train))
print("Testing sequences:", len(X_test))

print("\nMetrics:")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nSaved files:")
print("LSTM model saved to:", LSTM_MODEL_FILE)
print("Scaler saved to:", SCALER_FILE)
print("Metadata saved to:", META_FILE)