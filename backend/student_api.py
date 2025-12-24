import os
import yaml
import json
import uuid
import time
import joblib
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException, Header, BackgroundTasks, Body
from pydantic import BaseModel, Field, validator
from loguru import logger

# --- 1. VALIDATION SCHEMES (Pydantic) ---

class StudentInput(BaseModel):
    school: str = Field(..., example="GP")
    sex: str = Field(..., example="F")
    age: int = Field(..., ge=15, le=22)
    studytime: int = Field(..., ge=1, le=4)
    failures: int = Field(..., ge=0)
    absences: int = Field(..., ge=0)
    G1: int = Field(..., ge=0, le=20)
    G2: int = Field(..., ge=0, le=20)

class PipelineStep(BaseModel):
    step: str
    strategy: Optional[str] = None
    params: Dict[str, Any] = {}

class FullConfig(BaseModel):
    pipeline: Dict[str, List[PipelineStep]]
    learning: Dict[str, Any]

# --- 2. SETUP AND UTILS ---

app = FastAPI(title="EduPredict MLOps API")

CONFIG_BASE = "pipeline_config.yaml"
EXP_DIR = "outputs/experiments/"
LOG_FILE = "outputs/logs/inference_history.json"
MODELS_DIR = "models/"

os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs("outputs/logs/", exist_ok=True)

def log_inference(user_id: str, inputs: dict, outputs: dict):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "input": inputs,
        "output": outputs
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

# --- 3. SETUP ROUTES (Labo) ---

@app.get("/configuration")
def get_base_config():
    with open(CONFIG_BASE, 'r') as f:
        return yaml.safe_load(f)

@app.get("/configuration/history")
def get_config_history():
    return {"history": sorted([f for f in os.listdir(EXP_DIR) if f.endswith('.yaml')], reverse=True)}

@app.get("/configuration/{filename}")
def get_specific_config(filename: str):
    path = os.path.join(EXP_DIR, filename)
    if not os.path.exists(path): raise HTTPException(status_code=404)
    with open(path, 'r') as f: return yaml.safe_load(f)

@app.post("/configuration/experiment")
def save_experiment(config: FullConfig):
    exp_id = f"config_{int(time.time())}.yaml"
    with open(os.path.join(EXP_DIR, exp_id), 'w') as f:
        yaml.dump(config.model_dump(), f)
    return {"experiment_id": exp_id}

# --- 4. CORE ROUTES (Train & Predict) ---

@app.post("/train")
async def trigger_training(background_tasks: BackgroundTasks, config_id: str | None = None):
    def run_training():
        # Here we would instantiate the PipelineBuilder with config_id if present
        logger.info(f"🚀 Training started with config: {config_id or 'BASE'}")
        time.sleep(5) # Simulation
        logger.success("✅ Training finished")

    background_tasks.add_task(run_training)
    return {"status": "started"}

@app.post("/predict")
async def predict(data: StudentInput, strategy: str = "accuracy", x_user_id: str = Header("anonymous")):
    model_path = os.path.join(MODELS_DIR, f"best_{strategy}.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=503, detail="Unavailable model for the requested strategy.")
    
    model = joblib.load(model_path)
    df = pd.DataFrame([data.model_dump()])
    # Encode if necessary according to training_strategies.py
    df_encoded = pd.get_dummies(df).reindex(columns=model.feature_names_in_, fill_value=0)
    
    pred = model.predict(df_encoded)[0]
    prob = model.predict_proba(df_encoded)[0][1]
    
    res = {"is_failure": bool(pred), "probability": float(prob), "id": str(uuid.uuid4())}
    log_inference(x_user_id, data.model_dump(), res)
    return res

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now()}