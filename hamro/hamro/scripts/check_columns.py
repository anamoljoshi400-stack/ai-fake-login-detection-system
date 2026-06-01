# check_columns.py
import pandas as pd

df = pd.read_csv("../data/behavior_training_data.csv")
print("📋 Columns in dataset:", df.columns.tolist())
print("🧾 Sample data:\n", df.head())
