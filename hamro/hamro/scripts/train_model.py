from pathlib import Path
import json

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
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

# -------------------------------------------------
# PATHS
# -------------------------------------------------
FILE_DIR = Path(__file__).resolve().parent
ROOT_DIR = FILE_DIR.parent

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

DATASET = DATA_DIR / "behavior_training_data.csv"

MODEL_FILE = MODELS_DIR / "behavior_model.pkl"
META_FILE = MODELS_DIR / "model_meta.json"

MODELS_DIR.mkdir(exist_ok=True)

# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------
if not DATASET.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET}")

df = pd.read_csv(DATASET)

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

required_cols = FEATURES + [TARGET]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

df = df.dropna(subset=required_cols)

X = df[FEATURES]
y = df[TARGET]

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
# RANDOM FOREST MODEL
# -------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
)

model.fit(X_train, y_train)

# -------------------------------------------------
# EVALUATION
# -------------------------------------------------
threshold = 0.5

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

cm = confusion_matrix(y_test, y_pred)

feature_importance = dict(
    sorted(
        zip(FEATURES, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
)

# -------------------------------------------------
# SAVE MODEL
# -------------------------------------------------
joblib.dump(model, MODEL_FILE)

meta = {
    "model_type": "RandomForestClassifier",
    "features": FEATURES,
    "target": TARGET,
    "threshold": threshold,
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "roc_auc": float(roc_auc),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "dataset_rows": int(len(df)),
    "label_distribution": {
        str(label): int(count)
        for label, count in y.value_counts().sort_index().items()
    },
    "confusion_matrix": cm.tolist(),
    "feature_importance": {
        key: float(value) for key, value in feature_importance.items()
    },
}

with open(META_FILE, "w") as f:
    json.dump(meta, f, indent=4)

# -------------------------------------------------
# OUTPUT
# -------------------------------------------------
print("\nTraining Complete")
print("-" * 40)
print("Model:", "RandomForestClassifier")
print("Dataset rows:", len(df))
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("\nLabel distribution:")
print(y.value_counts().sort_index())

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

print("\nFeature Importance:")
for feature, importance in feature_importance.items():
    print(f"{feature}: {importance:.4f}")

print("\nSaved files:")
print("Model saved to:", MODEL_FILE)
print("Metadata saved to:", META_FILE)