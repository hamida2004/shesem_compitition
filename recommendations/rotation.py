# rotation.py
import pandas as pd

# Load dataset (ideally, this could be from DB instead of CSV)
df = pd.read_csv("Crop_recommendation.csv", names=["N","P","K","temperature","humidity","ph","rainfall","crop"])
crop_stats = df.groupby("crop").agg(["mean","std"])

def rotation_score(current_crop_stats, next_crop_stats, land_conditions):
    score = 0
    # Nutrient balance
    if current_crop_stats["N"]["mean"] > 50:
        score += 2 if next_crop_stats["N"]["mean"] < 40 else -1
    if current_crop_stats["P"]["mean"] > 50:
        if next_crop_stats["P"]["mean"] < 40:
            score += 1
    if current_crop_stats["K"]["mean"] > 50:
        if next_crop_stats["K"]["mean"] < 40:
            score += 1

    # Environmental compatibility
    for feature in ["temperature","humidity","ph","rainfall"]:
        mean = next_crop_stats[feature]["mean"]
        std = next_crop_stats[feature]["std"]
        if land_conditions.get(feature) is not None:
            if mean - std <= land_conditions[feature] <= mean + std:
                score += 1

    # Diversity
    if current_crop_stats.name != next_crop_stats.name:
        score += 1
    return score

def build_rotation_matrix(land_conditions=None, top_n=2):
    if land_conditions is None:
        land_conditions = {feat: df[feat].mean() for feat in ["temperature","humidity","ph","rainfall"]}

    rotation_matrix = {}
    for current_crop in crop_stats.index:
        scores = {}
        for next_crop in crop_stats.index:
            if next_crop == current_crop:
                continue
            scores[next_crop] = rotation_score(crop_stats.loc[current_crop], crop_stats.loc[next_crop], land_conditions)
        rotation_matrix[current_crop] = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return rotation_matrix
