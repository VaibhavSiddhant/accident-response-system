"""
Inference — FT-Transformer Risk Predictor
Matches data/preprocess.py (accident_prediction_india.csv schema)
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.ft_transformer import build_model
from data.preprocess import engineer_features, CATEGORICAL_FEATURES, NUMERICAL_FEATURES


RISK_LEVELS = {
    (0.0, 0.3):  ('LOW',      '🟢', 'Normal conditions. Standard monitoring.'),
    (0.3, 0.6):  ('MODERATE', '🟡', 'Elevated risk. Increase patrol frequency.'),
    (0.6, 0.8):  ('HIGH',     '🟠', 'High risk. Deploy traffic marshal.'),
    (0.8, 1.0):  ('CRITICAL', '🔴', 'Critical risk. Immediate intervention needed.'),
}


def get_risk_level(score: float):
    for (low, high), (level, emoji, action) in RISK_LEVELS.items():
        if low <= score < high:
            return level, emoji, action
    return 'CRITICAL', '🔴', 'Critical risk. Immediate intervention needed.'


def load_model_and_encoders(
    checkpoint_path: str = 'checkpoints/best_model.pt',
    encoders_path: str  = 'checkpoints/encoders.pkl',
    device: str = 'cuda',
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model not found at {checkpoint_path}. Run train.py first.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(
        n_features=checkpoint['n_features'],
        size=checkpoint['model_size'],
        device=device,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)

    print(f"Model loaded | Val F1: {checkpoint['val_f1']:.4f} | Val AUC: {checkpoint['val_auc']:.4f}")
    return model, encoders


def preprocess_input(sample: dict, encoders: dict) -> torch.Tensor:
    """
    sample must use RAW column names matching accident_prediction_india.csv:
    {
        'State Name': 'Karnataka',
        'Month': 'August',
        'Day of Week': 'Friday',
        'Time of Day': '23:15',
        'Weather Conditions': 'Rainy',
        'Road Type': 'National Highway',
        'Road Condition': 'Poor',
        'Lighting Conditions': 'Night - Street Lights Off',
        'Vehicle Type Involved': 'Two-Wheeler',
        'Traffic Control Presence': 'No',
        'Accident Location Details': 'Junction',
        'Alcohol Involvement': 'Yes',
        'Driver License Status': 'Invalid',
        'Driver Gender': 'Male',
        'Speed Limit (km/h)': 100,
        'Driver Age': 28,
        'Number of Vehicles Involved': 2,
    }
    """
    df = pd.DataFrame([sample])

    # Rename to internal schema (same mapping as load_and_map in preprocess.py)
    df = df.rename(columns={
        'State Name':                  'state',
        'Month':                       'month_str',
        'Day of Week':                 'day_str',
        'Time of Day':                 'time_str',
        'Number of Vehicles Involved': 'num_vehicles',
        'Vehicle Type Involved':       'vehicle_type',
        'Weather Conditions':          'weather',
        'Road Type':                   'road_type',
        'Road Condition':              'road_condition',
        'Lighting Conditions':         'light_condition',
        'Traffic Control Presence':    'traffic_control',
        'Speed Limit (km/h)':          'speed_limit',
        'Driver Age':                  'driver_age',
        'Driver Gender':               'driver_gender',
        'Driver License Status':       'license_status',
        'Alcohol Involvement':         'alcohol_involved',
        'Accident Location Details':   'accident_location',
    })

    df = engineer_features(df)

    cat_cols = encoders['cat_cols']
    num_cols = encoders['num_cols']

    # Encode categoricals
    for col in cat_cols:
        le = encoders[f'le_{col}']
        val = str(df[col].iloc[0]) if col in df.columns else 'Unknown'
        if val not in le.classes_:
            val = le.classes_[0]
        df[col] = le.transform([val])

    # Scale numericals — build full vector in correct order
    full_num = np.zeros((1, len(num_cols)))
    for i, col in enumerate(num_cols):
        if col in df.columns:
            full_num[0, i] = df[col].iloc[0]
    full_num = encoders['scaler'].transform(full_num)

    cat_data = df[cat_cols].values.astype(np.float32)
    X = np.concatenate([cat_data, full_num], axis=1).astype(np.float32)
    return torch.tensor(X, dtype=torch.float32)


@torch.no_grad()
def predict(sample: dict, model, encoders: dict, device: str = 'cuda') -> dict:
    x = preprocess_input(sample, encoders).to(device)
    logits = model(x)
    proba = F.softmax(logits, dim=1)

    risk_score = proba[0, 1].item()
    confidence = proba.max().item()
    level, emoji, action = get_risk_level(risk_score)

    return {
        'risk_score': round(risk_score, 4),
        'risk_level': level,
        'emoji':      emoji,
        'action':     action,
        'confidence': round(confidence, 4),
    }


def get_manual_input() -> dict:
    """Interactive terminal prompt for live demo."""
    print("\n=== Enter Accident Scenario ===")
    print("(Press Enter to accept the [default] shown)\n")

    def ask(prompt, default, choices=None):
        if choices:
            print(f"  Options: {', '.join(choices)}")
        val = input(f"{prompt} [{default}]: ").strip()
        return val if val else default

    sample = {
        'State Name':  ask("State", "Karnataka"),
        'Month':       ask("Month", "August",
                            ["January","February","March","April","May","June",
                             "July","August","September","October","November","December"]),
        'Day of Week': ask("Day of Week", "Friday",
                            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]),
        'Time of Day': ask("Time (24hr, e.g. 23:15)", "23:15"),
        'Weather Conditions': ask("Weather", "Rainy",
                            ["Clear","Rainy","Foggy","Mist","Hailstorm","Dust Storm"]),
        'Road Type':   ask("Road Type", "National Highway",
                            ["National Highway","State Highway","District Road","Village Road","City Road"]),
        'Road Condition': ask("Road Condition", "Poor", ["Good","Fair","Poor","Under Repair"]),
        'Lighting Conditions': ask("Lighting", "Night - Street Lights Off",
                            ["Daylight","Dawn/Dusk","Night - Street Lights On","Night - Street Lights Off"]),
        'Vehicle Type Involved': ask("Vehicle Type", "Two-Wheeler",
                            ["Two-Wheeler","Car","Auto-Rickshaw","Truck","Bus","LCV","Bicycle"]),
        'Traffic Control Presence': ask("Traffic Control Present?", "No", ["Yes","No"]),
        'Accident Location Details': ask("Location Type", "Junction",
                            ["Straight Road","Curve","Junction","Bridge","Toll Plaza","School Zone"]),
        'Alcohol Involvement': ask("Alcohol Involved?", "Yes", ["Yes","No"]),
        'Driver License Status': ask("License Status", "Invalid", ["Valid","Invalid","Not Available"]),
        'Driver Gender': ask("Driver Gender", "Male", ["Male","Female"]),
        'Speed Limit (km/h)': int(ask("Speed Limit (km/h)", "100")),
        'Driver Age': int(ask("Driver Age", "24")),
        'Number of Vehicles Involved': int(ask("Number of Vehicles Involved", "2")),
    }
    return sample


# ─── CLI Demo ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, encoders = load_model_and_encoders(device=device)

    print("\n1. Run preset scenarios")
    print("2. Enter my own scenario manually")
    mode = input("Choose (1/2): ").strip()

    if mode == '2':
        sample = get_manual_input()
        result = predict(sample, model, encoders, device=device)
        print(f"\n{result['emoji']} Risk: {result['risk_level']} ({result['risk_score']:.3f})")
        print(f"   Action: {result['action']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        sys.exit(0)

    high_risk = {
        'State Name':                  'Karnataka',
        'Month':                       'August',
        'Day of Week':                 'Friday',
        'Time of Day':                 '23:15',
        'Weather Conditions':          'Rainy',
        'Road Type':                   'National Highway',
        'Road Condition':              'Poor',
        'Lighting Conditions':         'Night - Street Lights Off',
        'Vehicle Type Involved':       'Two-Wheeler',
        'Traffic Control Presence':    'No',
        'Accident Location Details':   'Junction',
        'Alcohol Involvement':         'Yes',
        'Driver License Status':       'Invalid',
        'Driver Gender':               'Male',
        'Speed Limit (km/h)':          100,
        'Driver Age':                  24,
        'Number of Vehicles Involved': 2,
    }

    low_risk = {
        'State Name':                  'Karnataka',
        'Month':                       'February',
        'Day of Week':                 'Tuesday',
        'Time of Day':                 '10:30',
        'Weather Conditions':          'Clear',
        'Road Type':                   'City Road',
        'Road Condition':              'Good',
        'Lighting Conditions':         'Daylight',
        'Vehicle Type Involved':       'Car',
        'Traffic Control Presence':    'Yes',
        'Accident Location Details':   'Straight Road',
        'Alcohol Involvement':         'No',
        'Driver License Status':       'Valid',
        'Driver Gender':               'Female',
        'Speed Limit (km/h)':          40,
        'Driver Age':                  35,
        'Number of Vehicles Involved': 1,
    }

    print("\n─── Scenario 1: High Risk (Night + NH + Monsoon + Alcohol) ───")
    result = predict(high_risk, model, encoders, device=device)
    print(f"{result['emoji']} Risk: {result['risk_level']} ({result['risk_score']:.3f})")
    print(f"   Action: {result['action']}")
    print(f"   Confidence: {result['confidence']:.3f}")

    print("\n─── Scenario 2: Low Risk (Day + City Road + Clear) ───")
    result = predict(low_risk, model, encoders, device=device)
    print(f"{result['emoji']} Risk: {result['risk_level']} ({result['risk_score']:.3f})")
    print(f"   Action: {result['action']}")
    print(f"   Confidence: {result['confidence']:.3f}")

    print("\n─── Try your own scenario ───")
    print("Edit the `high_risk` / `low_risk` dicts at the bottom of this file")
    print("and re-run: python predict.py")