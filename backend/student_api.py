from contextlib import asynccontextmanager
import os
from pathlib import Path
import sys
from fastapi.encoders import jsonable_encoder
import yaml
import json
import uuid
import time

import joblib
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException, Header, BackgroundTasks, Body
from backend.utils.validators.validators import PredictionResponse, StudentInput, FullPipelineConfig as FullConfig
from loguru import logger

# PROMETHEUS INSTRUMENTATOR
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
from core.pipeline_core.pipeline_builder import PipelineBuilder
from core.pipeline_core.pipeline_core import PipelineContext, PipelineOrchestrator


# Sets up Prometheus monitoring
PREDICTION_COUNT = Counter('edupredict_predictions_total', 'Total number of predictions made', ['is_failure', 'strategy'])
AVG_PROBABILITY = Gauge('edupredict_average_failure_probability', 'Average predicted failure probability')

# --- 1. SETUP AND UTILS ---

app = FastAPI(title="EduPredict MLOps API")

Instrumentator().instrument(app).expose(app)

BASE_DIR = Path(__file__).resolve().parent

CONFIG_BASE = "pipeline_config.yaml"
EXP_DIR = "outputs/experiments/"
LOG_FILE = "outputs/logs/inference_history.json"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs("outputs/logs/", exist_ok=True)

def log_inference(user_id: str, inputs: dict, outputs: dict):
    """Structured storage of queries for audit"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "input": inputs,
        "output": outputs
    }
    # Logging in JSON
    with open("logs/inference_history.json", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    logger.info(f"📝 Inference logged for user {user_id}")

# PRE-PROCESSING FUNCTION ---
def apply_pipeline_rules(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Apply compliance rules and encoding for inference"""
    
    # A. Minimization (Mjob, Fjob) - Must be identical to pipeline
    mapping = {"health": "other", "at_home": "other", "teacher": "other"}
    df["Mjob"] = df["Mjob"].replace(mapping)
    df["Fjob"] = df["Fjob"].replace(mapping)

    # B. Anonymization (Age limit)
    df["age"] = df["age"].apply(lambda x: x if x < 19 else 19)

    # C. Encodage One-Hot
    # Columns generation and alignment to feature_names
    df_encoded = pd.get_dummies(df)
    
    # Critical alignment : Add missing columns with 0
    # remove extra columns
    final_df = pd.DataFrame(columns=feature_names)
    for col in feature_names:
        if col in df_encoded.columns:
            final_df[col] = df_encoded[col]
        else:
            final_df[col] = 0 # Non present feature
            
    return final_df.fillna(0)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Logique au démarrage
    logger.info("🚀 Starting EduPredict API...")
    
    # Technicals checks at startup
    if not any(MODELS_DIR.iterdir()):
        logger.warning(f"⚠️ No models was found in {MODELS_DIR}. L'API may failed.")
    
    yield # L'application tourne ici
    
    # Logique à la fermeture
    logger.info("🛑 Stopping API...")

# --- 2. SETUP ROUTES (Labo) ---

@app.get("/configuration")
async def get_base_config():
    """Returns the current pipeline configuration"""
    try:
        with open(CONFIG_BASE, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lecture config: {e}")

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
    try:
        # 1. Unique ID generation (timestamp)
        exp_id = f"config_{int(time.time())}.yaml"
        file_path = os.path.join(EXP_DIR, exp_id)

        # 2. Save the validated config
        with open(file_path, 'w') as f:
            yaml.dump(config.model_dump(), f)
            
        logger.info(f"🧪 New experimental configuration created: {exp_id}")
        return {"experiment_id": exp_id, "path": file_path}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- 4. CORE ROUTES (Train & Predict) ---

@app.post("/train")
async def trigger_training(background_tasks: BackgroundTasks, config_id: str | None = None):
    def run_training():
        # Here we would instantiate the PipelineBuilder with config_id if present
        logger.info(f"🚀 Training started with config: {config_id or 'BASE'}")
        orchestrator = PipelineOrchestrator()
        PipelineBuilder.build_from_yaml(
            config_path=config_id if config_id else CONFIG_BASE,
            orchestrator=orchestrator
        )
        context = PipelineContext()
        orchestrator.run(context)
        logger.success("✅ Background training completed.")

    background_tasks.add_task(run_training)
    return {"message": "Training pipeline started in background", "monitor_url": "/mlflow"}

@app.post("/predict/{strategy}", response_model=PredictionResponse)
async def predict(strategy: str, data: StudentInput, x_user_id: str = Header(default="anonymous")):
    try:
        # 1. Model and features loading
        model_path = MODELS_DIR / f"student_model_{strategy}_latest.joblib"
        feat_path = MODELS_DIR / f"feature_names_{strategy}_latest.pkl"

        if not os.path.exists(model_path) or not os.path.exists(feat_path):
            raise HTTPException(status_code=404, detail=f"Model or features not found : {model_path} {feat_path}")
        
        model = joblib.load(model_path)
        expected_features = joblib.load(feat_path)

        # 2. Data transform
        input_df = pd.DataFrame([data.model_dump()])
        prepared_df = apply_pipeline_rules(input_df, expected_features)

        # 3. Inference
        prediction = model.predict(prepared_df)[0]
        prob = model.predict_proba(prepared_df)[0][1]
        
        # 3. Préparation de la réponse
        res = {
            "prediction_id": str(uuid.uuid4()),
            "model_type": strategy,
            "is_failure": bool(prediction),
            "probability": float(prob),
            "timestamp": datetime.now()
        }
        
        # 4. Journalisation
        log_inference(x_user_id, data.model_dump(), jsonable_encoder(res))
        
        # Mise à jour des métriques Prometheus
        PREDICTION_COUNT.labels(is_failure=str(prediction), strategy=strategy).inc()
        AVG_PROBABILITY.set(prob)

        return jsonable_encoder(res)
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during inference")

@app.get("/health")
async def health():
    """Evaluates the health of the API and models"""
    files_exist = {
        "acc": joblib.os.path.exists("models/student_model_accuracy_latest.joblib"),
        "auc": joblib.os.path.exists("models/student_model_auc_latest.joblib")
    }
    status = "ok" if all(files_exist.values()) else "degraded"
    return {"status": status, "models_loaded": files_exist, "timestamp": datetime.now()}