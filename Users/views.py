from django.shortcuts import render
from django.utils import timezone
from .models import PredictionLog
import pandas as pd
import numpy as np
import joblib
import os
from django.db.models import Count

SCALER_PATH = 'model/standard_scaler.pkl'
MODEL_PATH = 'model/voting_classifier_model.pkl'
DATA_PATH = 'model/SIEM_data.csv'

ENCODINGS = {
    1: "Suspicious",
    0: "Incriminated"
}

FEATURE_NAMES = [
    'AlertTitle', 'Sha256', 'IpAddress', 'Url', 'AccountUpn', 
    'AccountName', 'DeviceName', 'RegistryKey', 'RegistryValueName', 
    'RegistryValueData', 'ApplicationName', 'FileName', 'FolderPath', 
    'ResourceIdName', 'OSFamily', 'OSVersion'
]

# Load model and scaler once globally
def load_model_and_scaler():
    scaler, model = None, None

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    return scaler, model

SCALER, MODEL = load_model_and_scaler()

# Helper to preprocess input
def preprocess_input(raw_data, feature_names, scaler):
    try:
        data_list = [float(x.strip()) for x in raw_data.split(',')]
        if len(data_list) != len(feature_names):
            return None, f"Expected {len(feature_names)} features but got {len(data_list)}"
        input_array = np.array(data_list).reshape(1, -1)
        input_df = pd.DataFrame(input_array, columns=feature_names)
        input_scaled = scaler.transform(input_df)
        return input_scaled, None
    except Exception as e:
        return None, str(e)

# Create your views here.
def userhome(request):
    user = request.user
    return render(request, 'User/userhome.html', {'user':user})

def prediction(request):
    result = None
    error_message = None
    user_input = ""

    if request.method == "POST":
        user_input = request.POST.get("user_input", "").strip()
        if not user_input:
            error_message = "Input cannot be empty."
        elif SCALER is None or MODEL is None:
            error_message = "Prediction model or scaler is not loaded."
        else:
            input_scaled, error = preprocess_input(user_input, FEATURE_NAMES, SCALER)
            if error:
                error_message = error
            else:
                pred_label = MODEL.predict(input_scaled)[0]
                result = ENCODINGS.get(pred_label, "Unknown")

                # Save to database
                session_key = request.session.session_key
                if not session_key:
                    request.session.create()
                    session_key = request.session.session_key

                PredictionLog.objects.create(
                    user=request.user if request.user.is_authenticated else None,
                    session_key=session_key,
                    user_input=user_input,
                    predicted_result=result
                )

    return render(request, 'User/prediction.html', {
        "result": result,
        "user_input": user_input,
        "error_message": error_message
    })

def datavisulization(request):
    return render(request, 'User/datavisulization.html')

def exsisting(request):
    return render(request, 'User/exsisting.html')

def proposed(request):
    return render(request, 'User/proposed.html')

def history(request):
    if request.user.is_authenticated:
        logs = PredictionLog.objects.filter(user=request.user).order_by('-created_at')
    else:
        session_key = request.session.session_key
        if not session_key:
            request.session.create()
            session_key = request.session.session_key
        logs = PredictionLog.objects.filter(session_key=session_key).order_by('-created_at')

    return render(request, 'User/history.html', {
        "logs": logs
    })

def analytics(request):
    if request.user.is_authenticated:
        logs = PredictionLog.objects.filter(user=request.user)
    else:
        # Use session_key for anonymous users
        session_key = request.session.session_key
        if not session_key:
            request.session.create()
            session_key = request.session.session_key
        logs = PredictionLog.objects.filter(session_key=session_key)

    # Aggregate counts of predictions
    counts = logs.values('predicted_result').annotate(total=Count('predicted_result'))
    
    # Prepare data for Chart.js
    labels = [item['predicted_result'] for item in counts]
    data = [item['total'] for item in counts]

    return render(request, 'User/analytics.html', {
        'labels': labels,
        'data': data
    })