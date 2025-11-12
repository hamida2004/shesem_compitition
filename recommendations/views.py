import json
import logging
import random
import torch
import joblib
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .utils import CropRecommendationModel
from .rotation import build_rotation_matrix

logger = logging.getLogger(__name__)

# Paths to saved model and preprocessing objects
MODEL_PATH = "recommendations/models/crop_recommendation_model.pth"
LABEL_ENCODER_PATH = "recommendations/models/label_encoder.pkl"
SCALER_PATH = "recommendations/models/scaler.pkl"
OPTIMAL_RANGES_PATH = "recommendations/models/optimal_ranges.pkl"

# Load preprocessing and model objects
label_encoder = joblib.load(LABEL_ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)
optimal_ranges = joblib.load(OPTIMAL_RANGES_PATH)

input_size = 7  # N, P, K, temperature, humidity, ph, rainfall
hidden_size = 64
output_size = len(label_encoder.classes_)
model = CropRecommendationModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

rotation_matrix = build_rotation_matrix()


def validate_fields(data, fields):
    """Ensure all required fields exist in the input JSON."""
    missing = [f for f in fields if f not in data]
    if missing:
        return False, f"Missing fields: {', '.join(missing)}"
    return True, None


@csrf_exempt
def next_crop(request):
    """
    Endpoint to get a recommended next crop (single) based on crop rotation and soil parameters.
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
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=400)

    try:
        data = json.loads(request.body)
        required_fields = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "previous_crop"]
        valid, error_msg = validate_fields(data, required_fields)
        if not valid:
            return JsonResponse({"error": error_msg}, status=400)

        sensor_data = {f: float(data[f]) for f in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]}
        previous_crop = data["previous_crop"]

        # المحاصيل الممكنة من مصفوفة الدوران
        possible_next_crops = rotation_matrix.get(previous_crop)
        if not possible_next_crops:
            return JsonResponse({"error": "Previous crop not found in rotation matrix"}, status=404)

        # فلترة المحاصيل حسب القيم المثالية للتربة
        suitable_crops = []
        for crop in possible_next_crops:
            if crop not in optimal_ranges:
                continue
            crop_ranges = optimal_ranges[crop]
            suitable = all(crop_ranges[feature][0] <= sensor_data[feature] <= crop_ranges[feature][1]
                           for feature in sensor_data)
            if suitable:
                suitable_crops.append(crop)

        # اختيار محصول واحد من المحاصيل المناسبة أو من الممكنة إذا لم توجد مناسبة
        recommended_crop = random.choice(suitable_crops) if suitable_crops else random.choice(possible_next_crops)

        return JsonResponse({
            "previous_crop": previous_crop,
            "recommended_next_crop": recommended_crop
        })

    except Exception as e:
        logger.error(f"Error in next_crop: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)
