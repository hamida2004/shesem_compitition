import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# Ensure all numeric columns are numeric
numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Group by 'label' (your crop column) and calculate mean and std
crop_stats = df.groupby("label")[numeric_cols].agg(["mean", "std"])

# Build optimal_ranges as {crop: {feature: (min, max)}} using mean ± std
optimal_ranges = {}
for crop in crop_stats.index:
    crop_dict = {}
    for feature in numeric_cols:
        mean = crop_stats.loc[crop, (feature, 'mean')]
        std = crop_stats.loc[crop, (feature, 'std')]
        min_val = mean - std
        max_val = mean + std
        crop_dict[feature] = (min_val, max_val)
    optimal_ranges[crop] = crop_dict

# Optional: Save optimal_ranges to pickle if not already done
# import joblib
# joblib.dump(optimal_ranges, "recommendations/models/optimal_ranges.pkl")

def build_rotation_matrix():
    crops = df['label'].unique().tolist()

    # Define crop groups based on common rotation principles
    legumes = [c for c in crops if any(leg in c.lower() for leg in ['chickpea', 'kidneybeans', 'blackgram', 'mungbean', 'mothbeans', 'pigeonpeas', 'lentil'])]  # Restorative (nitrogen-fixing)
    cereals_fibers = [c for c in crops if any(cf in c.lower() for cf in ['rice', 'maize', 'jute', 'cotton'])]  # Exhaustive
    fruits = [c for c in crops if any(fr in c.lower() for fr in ['banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya', 'pomegranate'])]
    others = [c for c in crops if any(ot in c.lower() for ot in ['coconut', 'coffee'])]  # Perennials/trees

    exhaustive = cereals_fibers + fruits + others

    # Build rotation dict: previous_crop -> [recommended_next_crops]
    rotation = {}
    for crop in exhaustive:
        rotation[crop] = legumes[:]  # Primarily follow with legumes; add some variety if desired
    for crop in legumes:
        rotation[crop] = exhaustive[:]  # Follow with exhaustive crops
    for crop in others:
        rotation[crop] = fruits + others  # Keep similar for perennials

    # Avoid self-rotation where possible
    for crop in rotation:
        if crop in rotation[crop]:
            rotation[crop].remove(crop)

    return rotation

rotation_matrix = build_rotation_matrix()
print(rotation_matrix)  # For debugging