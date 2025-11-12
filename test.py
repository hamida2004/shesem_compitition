# import pandas as pd

# # Load your dataset
# df = pd.read_csv("Crop_recommendation.csv")

# # Get unique values in the 'label' column
# unique_labels = df['label'].unique()

# # Print them
# for label in unique_labels:
#     print(label)

import pandas as pd

df = pd.read_csv("Crop_recommendation.csv")

# استبعاد القيم الفارغة
unique_labels = df['label'].dropna().unique()

for label in unique_labels:
    print(label)
