from pathlib import Path
import numpy as np
import pandas as pd

# Paths
FILE_DIR = Path(__file__).resolve().parent
ROOT_DIR = FILE_DIR.parent

DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = DATA_DIR / "behavior_training_data.csv"

np.random.seed(42)

rows = []

locations = {
    "Australia": 0,
    "Nepal": 1,
    "USA": 2,
    "India": 3,
    "Russia": 4,
    "Other": 5
}

# -------------------------------
# Legitimate login behaviour
# label = 0
# -------------------------------
for i in range(100):
    home_location = np.random.choice(["Australia", "Nepal", "India"])

    rows.append({
        "avg_typing_speed": round(np.random.normal(2.8, 0.45), 2),
        "location_encoded": locations[home_location],
        "login_hour": int(np.random.choice([8, 9, 10, 18, 19, 20, 21, 22])),
        "failed_attempts": int(np.random.choice([0, 0, 0, 1])),
        "location_mismatch": 0,
        "password_length": int(np.random.randint(10, 18)),
        "day_of_week": int(np.random.randint(0, 7)),
        "label": 0
    })

# -------------------------------
# Suspicious login behaviour
# label = 1
# -------------------------------
for i in range(100):
    suspicious_location = np.random.choice(["Russia", "USA", "Other"])

    rows.append({
        "avg_typing_speed": round(np.random.choice([
            np.random.normal(0.5, 0.2),
            np.random.normal(6.5, 1.0)
        ]), 2),
        "location_encoded": locations[suspicious_location],
        "login_hour": int(np.random.choice([0, 1, 2, 3, 4, 5])),
        "failed_attempts": int(np.random.choice([2, 3, 4, 5])),
        "location_mismatch": 1,
        "password_length": int(np.random.randint(6, 20)),
        "day_of_week": int(np.random.randint(0, 7)),
        "label": 1
    })

df = pd.DataFrame(rows)

# Clean negative typing speed if generated
df["avg_typing_speed"] = df["avg_typing_speed"].clip(lower=0.1)

# Shuffle data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv(OUTPUT_FILE, index=False)

print("Balanced dataset created successfully.")
print("Saved to:", OUTPUT_FILE)
print("\nDataset shape:", df.shape)
print("\nLabel distribution:")
print(df["label"].value_counts())
print("\nPreview:")
print(df.head())