import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import torch
import joblib
import pandas as pd
import logging
from .utils import CropRecommendationModel  # Import the model class from utils.py
from .rotation import build_rotation_matrix  # Import the rotation matrix function

logger = logging.getLogger(__name__)

# Paths to saved model and preprocessing objects
model_path = "recommendations/models/crop_recommendation_model.pth"
label_encoder_path = "recommendations/models/label_encoder.pkl"
scaler_path = "recommendations/models/scaler.pkl"
optimal_ranges_path = "recommendations/models/optimal_ranges.pkl"

# Load the label encoder, scaler, and optimal ranges
label_encoder = joblib.load(label_encoder_path)
scaler = joblib.load(scaler_path)
optimal_ranges = joblib.load(optimal_ranges_path)

# Initialize the model
input_size = 7  # Number of features (N, P, K, temperature, humidity, ph, rainfall)
hidden_size = 64  # Number of neurons in the hidden layer
output_size = len(label_encoder.classes_)  # Number of unique crops
model = CropRecommendationModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load(model_path))
model.eval()

# Build the rotation matrix once at startup
rotation_matrix = build_rotation_matrix()


@csrf_exempt
def crop_recommendation(request):
    if request.method == "POST":
        try:
            logger.info("Received POST request for crop recommendation.")
            data = json.loads(request.body)
            required_fields = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

            for field in required_fields:
                if field not in data:
                    return JsonResponse({"error": f"Missing field: {field}"}, status=400)

            new_data = {f: float(data[f]) for f in required_fields}
            new_df = pd.DataFrame([new_data])
            new_data_scaled = scaler.transform(new_df)
            new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

            with torch.no_grad():
                output = model(new_data_tensor)
                _, predicted = torch.max(output, 1)

            predicted_crop = label_encoder.inverse_transform(predicted.numpy())
            return JsonResponse({"recommended_crop": predicted_crop[0]})

        except Exception as e:
            logger.error(f"Error in crop_recommendation: {str(e)}")
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=400)


@csrf_exempt
def general_recommendation(request):
    if request.method == "POST":
        try:
            logger.info("Received POST request for general recommendation.")
            data = json.loads(request.body)
            required_fields = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "selected_crop"]

            for field in required_fields:
                if field not in data:
                    return JsonResponse({"error": f"Missing field: {field}"}, status=400)

            new_data = {f: float(data[f]) for f in ["N","P","K","temperature","humidity","ph","rainfall"]}
            selected_crop = data["selected_crop"]

            if selected_crop not in optimal_ranges:
                return JsonResponse({"error": "Crop not found in database"}, status=400)

            crop_ranges = optimal_ranges[selected_crop]
            recommendations = []
            for feature, (min_val, max_val) in crop_ranges.items():
                value = new_data[feature]
                if value < min_val:
                    recommendations.append(f"{feature} is low. Optimal range: {min_val:.2f} - {max_val:.2f}.")
                elif value > max_val:
                    recommendations.append(f"{feature} is high. Optimal range: {min_val:.2f} - {max_val:.2f}.")

            if not recommendations:
                recommendations.append("All parameters are within the optimal range for the selected crop.")

            return JsonResponse({"recommendations": recommendations})

        except Exception as e:
            logger.error(f"Error in general_recommendation: {str(e)}")
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=400)

@csrf_exempt
def next_crop(request):
    """
    Endpoint to get recommended next crops based on crop rotation matrix and sensor data.
    Expects POST JSON:
    {
        "N": float,
        "P": float,
        "K": float,
        "temperature": float,
        "humidity": float,
        "ph": float,
        "rainfall": float,
        "previous_crop": str
    }
    """
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            required_fields = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "previous_crop"]

            # Check all required fields
            for field in required_fields:
                if field not in data:
                    return JsonResponse({"error": f"Missing field: {field}"}, status=400)

            # Extract sensor data
            sensor_data = {f: float(data[f]) for f in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]}
            previous_crop = data["previous_crop"]

            # Get possible next crops from rotation matrix
            possible_next_crops = rotation_matrix.get(previous_crop)
            if not possible_next_crops:
                return JsonResponse({"error": "Previous crop not found in rotation matrix"}, status=404)

            # Filter possible crops based on sensor data and optimal ranges
            suitable_crops = []
            for crop in possible_next_crops:
                if crop not in optimal_ranges:
                    continue  # Skip crops without range info

                crop_ranges = optimal_ranges[crop]
                suitable = True
                for feature, (min_val, max_val) in crop_ranges.items():
                    value = sensor_data[feature]
                    if value < min_val or value > max_val:
                        suitable = False
                        break
                if suitable:
                    suitable_crops.append(crop)

            # If none are perfectly suitable, return all possible next crops
            recommended_crops = suitable_crops if suitable_crops else possible_next_crops

            return JsonResponse({
                "previous_crop": previous_crop,
                "recommended_next_crops": recommended_crops
            })

        except Exception as e:
            logger.error(f"Error in next_crop: {str(e)}")
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=400)
