import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# Ensure all numeric columns are numeric
numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Group by 'label' (your crop column) and calculate mean and std
crop_stats = df.groupby("label")[numeric_cols].agg(["mean", "std"])

print(crop_stats)

# Optional: function to build a rotation matrix (example)
def build_rotation_matrix():
    # For example, create a simple identity matrix for each crop
    crops = df['label'].unique()
    rotation_matrices = {crop: np.identity(len(numeric_cols)) for crop in crops}
    return rotation_matrices

rotation_matrices = build_rotation_matrix()
print(rotation_matrices)
