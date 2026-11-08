from pathlib import Path
import joblib
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'model.joblib'
FEATURES = ["trip_distance_km", "congestion_index", "avg_speed_kmh", "stop_count", "fuel_efficiency", "vehicle_age_years", "rain_intensity", "route_grade_pct", "driver_hours", "passenger_count"]
def get_bundle():
    if not MODEL_PATH.exists(): raise RuntimeError('Train first: python src/train.py --limit 20000')
    return joblib.load(MODEL_PATH)
def predict(values):
    bundle = get_bundle()
    row = {k: float(values.get(k, 0.0)) for k in FEATURES}
    return float(bundle['model'].predict(pd.DataFrame([row], columns=FEATURES))[0])

from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI(title='AI/ML Starter API')
class Request(BaseModel):
    features: dict[str, float]
@app.get('/health')
def health(): return {'status':'ok','model_ready':MODEL_PATH.exists()}
@app.post('/predict')
def predict_route(request: Request):
    return {'prediction': predict(request.features)}
