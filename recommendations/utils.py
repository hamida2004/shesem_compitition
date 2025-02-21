import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os

# Step 1: Load and preprocess the data
def load_and_preprocess_data(filepath):
    # Load the dataset
    data = pd.read_csv(filepath)
    # Separate features and target
    X = data.drop("label", axis=1).values  # Features (N, P, K, temperature, humidity, ph, rainfall)
    y = data["label"].values  # Target (crop names)
    distinct_labels = data["label"].nunique()
    print(f"Number of distinct crop labels: {distinct_labels}")

    # Encode the target labels (crop names) into integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoader for training and testing
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, label_encoder, scaler, data

# Step 2: Define the neural network
class CropRecommendationModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CropRecommendationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Step 3: Train the model
def train_model(train_loader, input_size, hidden_size, output_size, num_epochs=20):
    # Define the model
    model = CropRecommendationModel(input_size, hidden_size, output_size)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print loss for each epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    return model

# Step 4: Evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# Step 5: Calculate optimal ranges
def calculate_optimal_ranges(data):
    optimal_ranges = {}
    for crop, group in data.groupby("label"):
        ranges = {}
        for feature in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]:
            mean = group[feature].mean()
            std_dev = group[feature].std()
            ranges[feature] = (mean - std_dev, mean + std_dev)  # Define range as mean ± std_dev
        optimal_ranges[crop] = ranges
    return optimal_ranges

# Step 6: Save the model and preprocessing objects
def save_model_and_objects(model, label_encoder, scaler, optimal_ranges, save_dir="recommendations/models"):
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(save_dir, "crop_recommendation_model.pth")
    torch.save(model.state_dict(), model_path)

    # Save the label encoder
    label_encoder_path = os.path.join(save_dir, "label_encoder.pkl")
    joblib.dump(label_encoder, label_encoder_path)

    # Save the scaler
    scaler_path = os.path.join(save_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    # Save the optimal ranges
    optimal_ranges_path = os.path.join(save_dir, "optimal_ranges.pkl")
    joblib.dump(optimal_ranges, optimal_ranges_path)

    print(f"Model, preprocessing objects, and optimal ranges saved in {save_dir}")

# Step 7: Train and save the model
def train_and_save_model(filepath, save_dir="recommendations/models"):
    # Step 1: Load and preprocess the data
    train_loader, test_loader, label_encoder, scaler, data = load_and_preprocess_data(filepath)

    # Step 2: Define model parameters
    input_size = 7  # Number of features (N, P, K, temperature, humidity, ph, rainfall)
    hidden_size = 64  # Number of neurons in the hidden layer
    output_size = len(label_encoder.classes_)  # Number of unique crops

    # Step 3: Train the model
    model = train_model(train_loader, input_size, hidden_size, output_size)

    # Step 4: Evaluate the model
    evaluate_model(model, test_loader)

    # Step 5: Calculate optimal ranges
    optimal_ranges = calculate_optimal_ranges(data)

    # Step 6: Save the model and preprocessing objects
    save_model_and_objects(model, label_encoder, scaler, optimal_ranges, save_dir)

# Run the function to train and save the model
if __name__ == "__main__":
    filepath = "Crop_recommendation.csv"  # Replace with your dataset path
    train_and_save_model(filepath)