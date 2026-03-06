import pandas as pd
import numpy as np
import joblib
import os

SCALER_PATH = 'standard_scaler.pkl'
MODEL_PATH = 'voting_classifier_model.pkl'
DATA_PATH = 'SIEM_data.csv' 

ENCODINGS = {
    1: "Suspicious",
    0: "Incriminated"
}

raw_input_data = "1168,138268,360606,160396,69,53,153085,1631,635,860,3421,289573,117668,3586,5,66"

FEATURE_NAMES = [
    'AlertTitle', 'Sha256', 'IpAddress', 'Url', 'AccountUpn', 
    'AccountName', 'DeviceName', 'RegistryKey', 'RegistryValueName', 
    'RegistryValueData', 'ApplicationName', 'FileName', 'FolderPath', 
    'ResourceIdName', 'OSFamily', 'OSVersion'
]

def load_data(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            print(f"Successfully loaded dataset from: {path}")
            return df
        except Exception as e:
            print(f"Error loading actual data from {path}: {e}")

    print(f"Warning: Data file not found at {path}. Creating dummy DataFrame for feature alignment.")
    N_SAMPLES = 1 
    np.random.seed(42)

    dummy_data = {
        'Unnamed: 0': [0],
        'SuspicionLevel': ['Low'],
        'SuspicionLevel_label': [0],
    }

    for name in FEATURE_NAMES:
        dummy_data[name] = np.random.randint(0, 10, N_SAMPLES)

    all_cols = ['Unnamed: 0'] + FEATURE_NAMES + ['SuspicionLevel', 'SuspicionLevel_label']
    df = pd.DataFrame(dummy_data, columns=all_cols)
    print("Dummy dataset loaded successfully.")
    return df


def load_model_and_scaler(data_path):
    df = load_data(data_path)

    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"Successfully loaded StandardScaler from: {SCALER_PATH}")
    except FileNotFoundError:
        print(f"Error: StandardScaler file not found at {SCALER_PATH}. Please ensure the training script ran correctly.")
        scaler = None

    try:
        model = joblib.load(MODEL_PATH)
        print(f"Successfully loaded VotingClassifier model from: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}. Please ensure the training script ran correctly.")
        model = None

    return df, scaler, model


def preprocess_input(raw_data, feature_names, scaler):
    try:
        data_list = [float(x.strip()) for x in raw_data.split(',')]
        
        if len(data_list) != len(feature_names):
            print(f"Error: Expected {len(feature_names)} features but received {len(data_list)}.")
            return None

        input_array = np.array(data_list).reshape(1, -1) 
        input_df = pd.DataFrame(input_array, columns=feature_names)
        input_scaled = scaler.transform(input_df)
        return input_scaled
    except ValueError:
        print("Error: Input data contains non-numeric values.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during input preprocessing: {e}")
        return None


def predict_and_print(input_scaled, model):
    prediction_label = model.predict(input_scaled)[0]
    result = ENCODINGS.get(prediction_label, "Unknown Label (Check ENCODINGS dict)")
    
    print("\n--- Prediction Result ---")
    print(f"Predicted Class Label (Encoded): {prediction_label}")
    print(f"Predicted Suspicion Level: {result}")
    print("-------------------------")


if __name__ == "__main__":
    print(f"Attempting to load data and models...")
    data_df, scaler, voting_clf = load_model_and_scaler(DATA_PATH)

    if scaler is None or voting_clf is None or data_df is None:
        print("Prediction aborted due to missing file(s).")
    else:
        data_features = data_df.drop(columns=['Unnamed: 0', 'SuspicionLevel', 'SuspicionLevel_label'], errors='ignore').shape[1]
        if len(FEATURE_NAMES) != data_features:
            print(f"WARNING: Feature count mismatch. Expected {len(FEATURE_NAMES)} features, found {data_features} in loaded data.")
        input_scaled_data = preprocess_input(raw_input_data, FEATURE_NAMES, scaler)
        if input_scaled_data is not None:
            predict_and_print(input_scaled_data, voting_clf)
        else:
            print("Prediction aborted due to invalid input data.")
