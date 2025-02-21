import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import torch
import joblib
import pandas as pd
import os
import logging
from .utils import CropRecommendationModel  # Import the model class from utils.py

logger = logging.getLogger(__name__)

# Paths to the saved model and preprocessing objects
model_path = "recommendations/models/crop_recommendation_model.pth"
label_encoder_path = "recommendations/models/label_encoder.pkl"
scaler_path = "recommendations/models/scaler.pkl"
optimal_ranges_path = "recommendations/models/optimal_ranges.pkl"

# Load the label encoder, scaler, and optimal ranges
label_encoder = joblib.load(label_encoder_path)
scaler = joblib.load(scaler_path)
optimal_ranges = joblib.load(optimal_ranges_path)

# Define the model architecture
input_size = 7  # Number of features (N, P, K, temperature, humidity, ph, rainfall)
hidden_size = 64  # Number of neurons in the hidden layer
output_size = len(label_encoder.classes_)  # Number of unique crops

# Initialize the model
model = CropRecommendationModel(input_size, hidden_size, output_size)

# Load the trained model weights
model.load_state_dict(torch.load(model_path))

# Set the model to evaluation mode
model.eval()

@csrf_exempt
def crop_recommendation(request):
    if request.method == "POST":
        try:
            logger.info("Received POST request for crop recommendation.")

            # Parse JSON data from the request body
            data = json.loads(request.body)
            logger.info(f"Request data: {data}")

            # List of required fields
            required_fields = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

            # Check if all required fields are present
            for field in required_fields:
                if field not in data:
                    logger.error(f"Missing field: {field}")
                    return JsonResponse({"error": f"Missing field: {field}"}, status=400)

            # Get sensor data from the request
            new_data = {
                "N": float(data.get("N")),
                "P": float(data.get("P")),
                "K": float(data.get("K")),
                "temperature": float(data.get("temperature")),
                "humidity": float(data.get("humidity")),
                "ph": float(data.get("ph")),
                "rainfall": float(data.get("rainfall"))
            }

            logger.info(f"Processed data: {new_data}")

            # Preprocess the data
            new_df = pd.DataFrame([new_data])
            new_data_scaled = scaler.transform(new_df)
            new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

            # Make a prediction
            with torch.no_grad():
                output = model(new_data_tensor)
                _, predicted = torch.max(output, 1)

            # Decode the predicted label
            predicted_crop = label_encoder.inverse_transform(predicted.numpy())

            logger.info(f"Predicted crop: {predicted_crop[0]}")

            # Return the recommended crop
            return JsonResponse({"recommended_crop": predicted_crop[0]})

        except json.JSONDecodeError:
            logger.error("Invalid JSON data.")
            return JsonResponse({"error": "Invalid JSON data"}, status=400)
        except Exception as e:
            logger.error(f"Error in crop_recommendation: {str(e)}")
            return JsonResponse({"error": f"Invalid data: {str(e)}"}, status=400)

    logger.error("Invalid request method for crop_recommendation.")
    return JsonResponse({"error": "Invalid request method"}, status=400)

@csrf_exempt
def general_recommendation(request):
    if request.method == "POST":
        try:
            logger.info("Received POST request for general recommendation.")

            # Parse JSON data from the request body
            data = json.loads(request.body)
            logger.info(f"Request data: {data}")

            # List of required fields
            required_fields = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "selected_crop"]

            # Check if all required fields are present
            for field in required_fields:
                if field not in data:
                    logger.error(f"Missing field: {field}")
                    return JsonResponse({"error": f"Missing field: {field}"}, status=400)

            # Check if optimal ranges are available
            if not optimal_ranges:
                logger.error("Optimal ranges not available.")
                return JsonResponse({"error": "Optimal ranges not available. Please run the training script first."}, status=500)

            # Get sensor data and selected crop from the request
            new_data = {
                "N": float(data.get("N")),
                "P": float(data.get("P")),
                "K": float(data.get("K")),
                "temperature": float(data.get("temperature")),
                "humidity": float(data.get("humidity")),
                "ph": float(data.get("ph")),
                "rainfall": float(data.get("rainfall"))
            }
            selected_crop = data.get("selected_crop")  # Crop selected by the user

            logger.info(f"Processed data: {new_data}")
            logger.info(f"Selected crop: {selected_crop}")

            # Get the optimal range for the selected crop
            if selected_crop not in optimal_ranges:
                logger.error(f"Crop not found: {selected_crop}")
                return JsonResponse({"error": "Crop not found in database"}, status=400)

            crop_ranges = optimal_ranges[selected_crop]

            # Compare sensor data with optimal ranges and provide recommendations
            recommendations = []
            for feature, (min_val, max_val) in crop_ranges.items():
                value = new_data[feature]
                if value < min_val:
                    recommendations.append(f"{feature} is low. Optimal range: {min_val:.2f} - {max_val:.2f}.")
                elif value > max_val:
                    recommendations.append(f"{feature} is high. Optimal range: {min_val:.2f} - {max_val:.2f}.")

            if not recommendations:
                recommendations.append("All parameters are within the optimal range for the selected crop.")

            logger.info(f"Recommendations: {recommendations}")

            return JsonResponse({"recommendations": recommendations})

        except json.JSONDecodeError:
            logger.error("Invalid JSON data.")
            return JsonResponse({"error": "Invalid JSON data"}, status=400)
        except Exception as e:
            logger.error(f"Error in general_recommendation: {str(e)}")
            return JsonResponse({"error": f"Invalid data: {str(e)}"}, status=400)

    logger.error("Invalid request method for general_recommendation.")
    return JsonResponse({"error": "Invalid request method"}, status=400)