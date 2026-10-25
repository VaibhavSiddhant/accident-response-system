"""
Integration — Module 2 (Risk Prediction) → Module 3 (Emergency Routing)

Pipeline:
  accident scenario (raw fields) + GPS coordinates
      ↓
  FT-Transformer computes risk_score          [Module 2]
      ↓
  if risk_score is high enough → auto-dispatch
      ↓
  GAT-style dynamic weighting + A* routing    [Module 3]
      ↓
  unified result: risk + route + map

Run from the accident-system/ root directory:
    python integrate.py
"""

import os
import sys
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "prediction"))
sys.path.append(os.path.dirname(__file__))

import torch
from prediction.predict import load_model_and_encoders, predict as predict_risk
from routing.route import route_emergency, generate_map, load_or_build_graph


# Confidence threshold from earlier decision: routing triggers only on detected accident.
# Here we treat "risk_score > ACCIDENT_THRESHOLD" as the detection-confidence stand-in
# until Module 1 (RT-DETR) is wired in — same trigger logic, different upstream source.
ACCIDENT_THRESHOLD = 0.70


def analyze_and_dispatch(scenario: dict, lat: float, lng: float,
                          model, encoders, road_graph, device: str) -> dict:
    """
    Full pipeline: risk score -> conditional routing -> unified result.

    scenario: raw accident fields (same schema as predict.py's high_risk/low_risk dicts)
    lat, lng: accident GPS coordinates (Bengaluru)
    """
    # ── Step 1: Risk Prediction (Module 2) ──────────────────────────────────
    risk_result = predict_risk(scenario, model, encoders, device=device)
    risk_score  = risk_result["risk_score"]

    # Parse hour from the scenario's "Time of Day" field for routing's peak-hour logic
    hour = int(scenario.get("Time of Day", "12:00").split(":")[0])

    print(f"\n{'='*55}")
    print(f"  Risk Assessment")
    print(f"{'='*55}")
    print(f"{risk_result['emoji']} Risk: {risk_result['risk_level']} ({risk_score:.3f})")
    print(f"   Action: {risk_result['action']}")

    result = {
        "risk": risk_result,
        "location": {"lat": lat, "lng": lng},
        "routing": None,
        "dispatched": False,
    }

    # ── Step 2: Conditional Routing (Module 3) ──────────────────────────────
    if risk_score > ACCIDENT_THRESHOLD:
        print(f"\n   Risk exceeds threshold ({ACCIDENT_THRESHOLD}) — dispatching emergency routing...")
        routing_result = route_emergency(
            accident_lat=lat, accident_lng=lng,
            risk_score=risk_score, hour=hour,
            G=road_graph,
        )
        result["routing"] = routing_result
        result["dispatched"] = True

        map_path = generate_map(routing_result)
        result["map_path"] = map_path
    else:
        print(f"\n   Risk below threshold ({ACCIDENT_THRESHOLD}) — no dispatch triggered.")
        print(f"   Logged for risk heatmap only.")

    return result


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Module 2 (FT-Transformer)...")
    model, encoders = load_model_and_encoders(
        checkpoint_path="prediction/checkpoints/best_model.pt",
        encoders_path="prediction/checkpoints/encoders.pkl",
        device=device,
    )

    print("\nLoading Module 3 (Bengaluru road graph)...")
    road_graph = load_or_build_graph()

    # ── Test Scenario: High-risk accident near Silk Board Junction ─────────
    scenario = {
        "State Name":                  "Karnataka",
        "Month":                       "August",
        "Day of Week":                 "Friday",
        "Time of Day":                 "23:15",
        "Weather Conditions":          "Rainy",
        "Road Type":                   "National Highway",
        "Road Condition":              "Poor",
        "Lighting Conditions":         "Night - Street Lights Off",
        "Vehicle Type Involved":       "Two-Wheeler",
        "Traffic Control Presence":    "No",
        "Accident Location Details":   "Junction",
        "Alcohol Involvement":         "Yes",
        "Driver License Status":       "Invalid",
        "Driver Gender":               "Male",
        "Speed Limit (km/h)":          100,
        "Driver Age":                  24,
        "Number of Vehicles Involved": 2,
    }

    # Silk Board Junction coordinates
    accident_lat, accident_lng = 12.9175, 77.6231

    result = analyze_and_dispatch(scenario, accident_lat, accident_lng,
                                    model, encoders, road_graph, device)

    if result["dispatched"]:
        r = result["routing"]["primary_route"]
        print(f"\n{'='*55}")
        print(f"  Final Dispatch Summary")
        print(f"{'='*55}")
        print(f"  Hospital:  {r['hospital']}")
        print(f"  ETA:       {r['eta_minutes']} min")
        print(f"  Distance:  {r['distance_km']} km")
        print(f"  Map:       {result['map_path']}")

    # ── Test Scenario 2: Low-risk, should NOT trigger routing ──────────────
    print(f"\n\n{'#'*55}")
    print("  Second test — low risk scenario")
    print(f"{'#'*55}")

    safe_scenario = {
        "State Name":                  "Karnataka",
        "Month":                       "February",
        "Day of Week":                 "Tuesday",
        "Time of Day":                 "10:30",
        "Weather Conditions":          "Clear",
        "Road Type":                   "City Road",
        "Road Condition":              "Good",
        "Lighting Conditions":         "Daylight",
        "Vehicle Type Involved":       "Car",
        "Traffic Control Presence":    "Yes",
        "Accident Location Details":   "Straight Road",
        "Alcohol Involvement":         "No",
        "Driver License Status":       "Valid",
        "Driver Gender":               "Female",
        "Speed Limit (km/h)":          40,
        "Driver Age":                  35,
        "Number of Vehicles Involved": 1,
    }

    result2 = analyze_and_dispatch(safe_scenario, 12.9352, 77.6245,  # Koramangala
                                     model, encoders, road_graph, device)
    print(f"\nDispatched: {result2['dispatched']}")