"""
Data Preprocessing Pipeline — India Road Accidents
Dataset: accident_prediction_india.csv (3000 records)
Columns: State Name, City Name, Year, Month, Day of Week, Time of Day,
         Accident Severity, Number of Vehicles Involved, Vehicle Type Involved,
         Number of Casualties, Number of Fatalities, Weather Conditions,
         Road Type, Road Condition, Lighting Conditions, Traffic Control Presence,
         Speed Limit (km/h), Driver Age, Driver Gender, Driver License Status,
         Alcohol Involvement, Accident Location Details
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os

# ── Feature lists ──────────────────────────────────────────────────────────────

CATEGORICAL_FEATURES = [
    'state',
    'weather',
    'road_type',
    'road_condition',
    'light_condition',
    'vehicle_type',
    'traffic_control',
    'accident_location',
    'alcohol_involved',
    'license_status',
    'driver_gender',
]

NUMERICAL_FEATURES = [
    'hour',
    'month_num',
    'day_of_week_num',
    'is_weekend',
    'is_monsoon',
    'is_festival_season',
    'is_peak_hour',
    'is_night',
    'speed_limit',
    'driver_age',
    'num_vehicles',
    'historical_risk',
    'road_lighting_score',
]

TARGET = 'target'  # 1 = Serious or Fatal, 0 = Minor

# ── Month helpers ──────────────────────────────────────────────────────────────

MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

DAY_MAP = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
    'Friday': 4, 'Saturday': 5, 'Sunday': 6
}

STATE_RISK = {
    'Tamil Nadu': 0.85, 'Uttar Pradesh': 0.82, 'Maharashtra': 0.78,
    'Madhya Pradesh': 0.80, 'Karnataka': 0.74, 'Rajasthan': 0.76,
    'Andhra Pradesh': 0.73, 'Gujarat': 0.70, 'Telangana': 0.71,
    'West Bengal': 0.68, 'Kerala': 0.66, 'Bihar': 0.79,
    'Haryana': 0.75, 'Punjab': 0.72, 'Delhi': 0.65,
    'Jammu and Kashmir': 0.77, 'Himachal Pradesh': 0.73,
    'Uttarakhand': 0.74, 'Jharkhand': 0.76, 'Odisha': 0.78,
    'Chhattisgarh': 0.75, 'Assam': 0.72,
}

LIGHTING_SCORE = {
    'Daylight': 1.0,
    'Dawn/Dusk': 0.6,
    'Dark (street lit)': 0.4,
    'Dark (unlit)': 0.1,
    'Night - Street Lights On': 0.4,
    'Night - Street Lights Off': 0.1,
    'Night': 0.2,
}


def parse_hour(time_str):
    """Parse '1:46', '21:30', '08:00' → integer hour."""
    try:
        return int(str(time_str).split(':')[0])
    except Exception:
        return 12


def load_and_map(csv_path: str) -> pd.DataFrame:
    """Load CSV and rename columns to our internal schema."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows | Columns: {df.columns.tolist()}")

    df = df.rename(columns={
        'State Name':                  'state',
        'City Name':                   'city',
        'Year':                        'year',
        'Month':                       'month_str',
        'Day of Week':                 'day_str',
        'Time of Day':                 'time_str',
        'Accident Severity':           'severity',
        'Number of Vehicles Involved': 'num_vehicles',
        'Vehicle Type Involved':       'vehicle_type',
        'Number of Casualties':        'num_casualties',
        'Number of Fatalities':        'num_fatalities',
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
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Time features ──────────────────────────────────────────────────────────
    df['hour'] = df['time_str'].apply(parse_hour)
    df['month_num'] = df['month_str'].map(MONTH_MAP).fillna(6).astype(int)
    df['day_of_week_num'] = df['day_str'].map(DAY_MAP).fillna(0).astype(int)

    df['is_weekend']        = (df['day_of_week_num'] >= 5).astype(int)
    df['is_monsoon']        = df['month_num'].apply(lambda m: 1 if 6 <= m <= 9 else 0)
    df['is_festival_season']= df['month_num'].apply(lambda m: 1 if m in [10, 11] else 0)
    df['is_peak_hour']      = df['hour'].apply(lambda h: 1 if (8<=h<=10) or (17<=h<=20) else 0)
    df['is_night']          = df['hour'].apply(lambda h: 1 if (h>=22 or h<=5) else 0)

    # ── Road lighting score ────────────────────────────────────────────────────
    df['road_lighting_score'] = df['light_condition'].map(LIGHTING_SCORE).fillna(0.5)

    # ── Historical state risk ──────────────────────────────────────────────────
    df['historical_risk'] = df['state'].map(STATE_RISK).fillna(0.70)

    # ── Numerical fillna ───────────────────────────────────────────────────────
    df['speed_limit']  = pd.to_numeric(df['speed_limit'],  errors='coerce').fillna(50)
    df['driver_age']   = pd.to_numeric(df['driver_age'],   errors='coerce').fillna(35)
    df['num_vehicles'] = pd.to_numeric(df['num_vehicles'], errors='coerce').fillna(2)

    # ── Categorical fillna ─────────────────────────────────────────────────────
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)

    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary target from Accident Severity:
      Fatal / Serious → 1 (high risk)
      Minor           → 0 (low risk)
    """
    df = df.copy()
    severity_col = df['severity'].str.strip().str.lower()
    df['target'] = severity_col.apply(
        lambda s: 1 if s in ['fatal', 'serious', 'severe', 'grievous'] else 0
    )
    print(f"\nSeverity value counts:\n{df['severity'].value_counts()}")
    print(f"\nTarget distribution:\n{df['target'].value_counts(normalize=True).round(3)}")
    return df


def encode_and_scale(
    df: pd.DataFrame,
    fit: bool = True,
    encoders_path: str = 'checkpoints/encoders.pkl'
) -> tuple:

    df = df.copy()
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    num_cols = [c for c in NUMERICAL_FEATURES  if c in df.columns]

    if fit:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str).fillna('Unknown'))
            encoders[f'le_{col}'] = le

        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols].fillna(0))
        encoders['scaler']   = scaler
        encoders['cat_cols'] = cat_cols
        encoders['num_cols'] = num_cols

        os.makedirs(os.path.dirname(encoders_path), exist_ok=True)
        with open(encoders_path, 'wb') as f:
            pickle.dump(encoders, f)
        print(f"Encoders saved → {encoders_path}")

    else:
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
        cat_cols = encoders['cat_cols']
        num_cols = encoders['num_cols']
        for col in cat_cols:
            le = encoders[f'le_{col}']
            df[col] = df[col].astype(str).fillna('Unknown')
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
        df[num_cols] = encoders['scaler'].transform(df[num_cols].fillna(0))

    feature_cols = cat_cols + num_cols
    X = df[feature_cols].values.astype(np.float32)
    y = df['target'].values.astype(np.int64) if 'target' in df.columns else None

    print(f"Feature matrix: {X.shape}  |  {len(cat_cols)} cat + {len(num_cols)} num features")
    return X, y


def prepare_pipeline(csv_path: str) -> tuple:
    """Full pipeline: load → map → engineer → target → encode → split."""
    df = load_and_map(csv_path)
    df = engineer_features(df)
    df = build_target(df)
    X, y = encode_and_scale(df, fit=True)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    print(f"Split → Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Synthetic fallback ─────────────────────────────────────────────────────────

def generate_synthetic_data(n_samples: int = 10000) -> pd.DataFrame:
    np.random.seed(42)
    rng = np.random.default_rng(42)
    states = list(STATE_RISK.keys())

    df = pd.DataFrame({
        'state':            rng.choice(states, n_samples),
        'weather':          rng.choice(['Clear', 'Rainy', 'Foggy', 'Hailstorm'], n_samples, p=[0.6,0.25,0.1,0.05]),
        'road_type':        rng.choice(['NH', 'SH', 'MDR', 'ODR', 'Village Road'], n_samples, p=[0.3,0.25,0.2,0.15,0.1]),
        'road_condition':   rng.choice(['Good', 'Fair', 'Poor'], n_samples, p=[0.5,0.3,0.2]),
        'light_condition':  rng.choice(['Daylight','Dawn/Dusk','Night - Street Lights On','Night - Street Lights Off'], n_samples, p=[0.5,0.1,0.25,0.15]),
        'vehicle_type':     rng.choice(['Two-wheeler','Car','Auto','Truck','Bus','LCV'], n_samples, p=[0.4,0.25,0.1,0.1,0.08,0.07]),
        'traffic_control':  rng.choice(['Yes','No'], n_samples, p=[0.4,0.6]),
        'accident_location':rng.choice(['Straight Road','Curve','Junction','Bridge','Other'], n_samples),
        'alcohol_involved': rng.choice(['Yes','No'], n_samples, p=[0.3,0.7]),
        'license_status':   rng.choice(['Valid','Invalid','Unknown'], n_samples, p=[0.6,0.25,0.15]),
        'driver_gender':    rng.choice(['Male','Female'], n_samples, p=[0.85,0.15]),
        'hour':             rng.integers(0, 24, n_samples),
        'month_num':        rng.integers(1, 13, n_samples),
        'day_of_week_num':  rng.integers(0, 7, n_samples),
        'speed_limit':      rng.choice([30,40,50,60,80,100], n_samples),
        'driver_age':       rng.integers(18, 70, n_samples),
        'num_vehicles':     rng.integers(1, 5, n_samples),
        'severity':         rng.choice(['Fatal','Serious','Minor'], n_samples, p=[0.15,0.35,0.50]),
    })

    # Derived features
    df['is_weekend']         = (df['day_of_week_num'] >= 5).astype(int)
    df['is_monsoon']         = df['month_num'].apply(lambda m: 1 if 6<=m<=9 else 0)
    df['is_festival_season'] = df['month_num'].apply(lambda m: 1 if m in [10,11] else 0)
    df['is_peak_hour']       = df['hour'].apply(lambda h: 1 if (8<=h<=10) or (17<=h<=20) else 0)
    df['is_night']           = df['hour'].apply(lambda h: 1 if (h>=22 or h<=5) else 0)
    df['road_lighting_score']= df['light_condition'].map(LIGHTING_SCORE).fillna(0.5)
    df['historical_risk']    = df['state'].map(STATE_RISK).fillna(0.70)

    print(f"Generated {n_samples:,} synthetic samples")
    return df


if __name__ == '__main__':
    # Test with real data
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_pipeline(path)
    else:
        df = generate_synthetic_data(5000)
        df['target'] = (df['severity'].isin(['Fatal','Serious'])).astype(int)
        X, y = encode_and_scale(df, fit=True)
        print(f"Synthetic ready. X: {X.shape}")