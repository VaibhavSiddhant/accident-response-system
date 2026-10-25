"""
FastAPI Backend — Accident Risk Prediction + Emergency Routing

Run from the accident-system/ root:
    uvicorn api:app --reload --port 8000

Then open:
    http://127.0.0.1:8000/docs   ← interactive Swagger UI, test everything from the browser
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "prediction"))
sys.path.append(os.path.dirname(__file__))

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from prediction.predict import load_model_and_encoders, predict as predict_risk
from routing.route import route_emergency, generate_map, load_or_build_graph, load_hospitals


ACCIDENT_THRESHOLD = 0.70

# ── Global model state (loaded once at startup, not per-request) ──────────────

class ModelState:
    model = None
    encoders = None
    road_graph = None
    device = None

state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up — loading models...")
    state.device = "cuda" if torch.cuda.is_available() else "cpu"

    state.model, state.encoders = load_model_and_encoders(
        checkpoint_path="prediction/checkpoints/best_model.pt",
        encoders_path="prediction/checkpoints/encoders.pkl",
        device=state.device,
    )
    state.road_graph = load_or_build_graph()
    load_hospitals()

    print(f"Ready. Device: {state.device}")
    yield
    print("Shutting down.")


app = FastAPI(
    title="India Accident Detection, Prediction & Emergency Routing API",
    description="FT-Transformer risk prediction + GAT-weighted A* emergency routing for Bengaluru",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # relax for dev; restrict to your frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ─────────────────────────────────────────────────

class AccidentScenario(BaseModel):
    state_name: str                = Field(..., alias="State Name", examples=["Karnataka"])
    month: str                     = Field(..., alias="Month", examples=["August"])
    day_of_week: str               = Field(..., alias="Day of Week", examples=["Friday"])
    time_of_day: str               = Field(..., alias="Time of Day", examples=["23:15"])
    weather: str                   = Field(..., alias="Weather Conditions", examples=["Rainy"])
    road_type: str                 = Field(..., alias="Road Type", examples=["National Highway"])
    road_condition: str            = Field(..., alias="Road Condition", examples=["Poor"])
    lighting: str                  = Field(..., alias="Lighting Conditions", examples=["Night - Street Lights Off"])
    vehicle_type: str              = Field(..., alias="Vehicle Type Involved", examples=["Two-Wheeler"])
    traffic_control: str           = Field(..., alias="Traffic Control Presence", examples=["No"])
    location_type: str             = Field(..., alias="Accident Location Details", examples=["Junction"])
    alcohol: str                   = Field(..., alias="Alcohol Involvement", examples=["Yes"])
    license_status: str            = Field(..., alias="Driver License Status", examples=["Invalid"])
    driver_gender: str             = Field(..., alias="Driver Gender", examples=["Male"])
    speed_limit: int               = Field(..., alias="Speed Limit (km/h)", examples=[100])
    driver_age: int                = Field(..., alias="Driver Age", examples=[24])
    num_vehicles: int              = Field(..., alias="Number of Vehicles Involved", examples=[2])

    model_config = {"populate_by_name": True}


class AnalyzeRequest(BaseModel):
    scenario: AccidentScenario
    lat: float = Field(..., examples=[12.9175], description="Accident latitude (Bengaluru)")
    lng: float = Field(..., examples=[77.6231], description="Accident longitude (Bengaluru)")


class RiskResponse(BaseModel):
    risk_score: float
    risk_level: str
    action: str
    confidence: float


class RouteResponse(BaseModel):
    hospital: str
    eta_minutes: float
    distance_km: float
    coordinates: list
    ambulances: int


class AnalyzeResponse(BaseModel):
    risk: RiskResponse
    dispatched: bool
    routing: Optional[RouteResponse] = None
    black_spot_risk: Optional[float] = None
    message: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "India Accident Detection, Prediction & Emergency Routing API",
        "status": "running",
        "device": state.device,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "graph_loaded": state.road_graph is not None,
        "device": state.device,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    """
    Full pipeline: risk prediction -> conditional emergency routing.

    Send an accident scenario + GPS coordinates. Returns the risk score always;
    returns a dispatch route only if risk exceeds the accident-detection threshold.
    """
    if state.model is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    scenario_dict = request.scenario.model_dump(by_alias=True)

    # ── Module 2: Risk Prediction ────────────────────────────────────────────
    try:
        risk_result = predict_risk(scenario_dict, state.model, state.encoders, device=state.device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Risk prediction failed: {e}")

    risk_score = risk_result["risk_score"]
    hour = int(scenario_dict.get("Time of Day", "12:00").split(":")[0])

    response = {
        "risk": {
            "risk_score": risk_result["risk_score"],
            "risk_level": risk_result["risk_level"],
            "action":     risk_result["action"],
            "confidence": risk_result["confidence"],
        },
        "dispatched": False,
        "routing": None,
        "black_spot_risk": None,
        "message": f"Risk assessed: {risk_result['risk_level']} ({risk_score:.3f})",
    }

    # ── Module 3: Conditional Routing ────────────────────────────────────────
    if risk_score > ACCIDENT_THRESHOLD:
        try:
            routing_result = route_emergency(
                accident_lat=request.lat, accident_lng=request.lng,
                risk_score=risk_score, hour=hour,
                G=state.road_graph,
            )
            generate_map(routing_result)  # refresh emergency_map.html

            primary = routing_result["primary_route"]
            response["dispatched"] = True
            response["black_spot_risk"] = routing_result["black_spot_risk"]
            response["routing"] = {
                "hospital":     primary["hospital"],
                "eta_minutes":  primary["eta_minutes"],
                "distance_km":  primary["distance_km"],
                "coordinates":  primary["coordinates"],
                "ambulances":   primary["ambulances"],
            }
            response["message"] = routing_result["dispatch_message"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Routing failed: {e}")

    return response


@app.get("/blackspots")
def get_black_spots():
    """Returns all known Bengaluru accident black spots for map display."""
    from routing.route import BENGALURU_BLACK_SPOTS
    return BENGALURU_BLACK_SPOTS


@app.get("/hospitals")
def get_hospitals():
    """Returns all Bengaluru hospitals with emergency services."""
    from routing.route import HOSPITALS
    return HOSPITALS


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)