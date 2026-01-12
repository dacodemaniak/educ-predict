import mlflow
import mlflow.tracking
import os
import sys
import yaml
import requests
from loguru import logger
from core.pipeline_core.pipeline_builder import PipelineBuilder
from core.pipeline_core.pipeline_core import PipelineContext, PipelineOrchestrator
from dotenv import load_dotenv

load_dotenv()

# Settings
MIN_ACCURACY = float(os.getenv("MIN_ACCURACY_THRESHOLD", 0.70))

def run_and_audit():
    # MLFlow settings
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("Educ_Predict_Experiments")

    # Configure and run pipeline
    logger.info("🚀 Training pipeline starting (CI/CD Mode)...")
    orchestrator = PipelineOrchestrator()
    context = PipelineContext()

    try:
        PipelineBuilder.build_from_yaml("pipeline_config.yaml", orchestrator=orchestrator)
        orchestrator.run(context=context)
        logger.info("🏁 Pipeline ended successfuly")
    except Exception as e:
        logger.error(f"❌ **PIPELINE CRITICAL FAIL** : {str(e)}")
        sys.exit(1)

    logger.info(f"🔍 Performance audit (Threshold: {MIN_ACCURACY})...")

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Educ_Predict_Experiments")
    experiment_id = experiment.experiment_id if experiment is not None else 0

    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.scenario != 'Full_Features'",
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )

    if runs.empty:
        logger.error("❌ No run was found during audit")
        sys.exit(1)

    best_run = runs.iloc[0]
    accuracy = best_run["metrics.accuracy"]
    run_id = best_run["run_id"]
    scenario = best_run["tag.scenario"]

    if accuracy >= MIN_ACCURACY:
        success_msg = (f"✅ **DEPLOYMENT VALIDATED**\n"
                       f"- Model: {scenario}\n"
                       f"- Accuracy: {accuracy:.4f} (Threshold: {MIN_ACCURACY})\n"
                       f"- ID: {run_id}")
        logger.info(success_msg)
        sys.exit(0)
    else:
        fail_msg = (f"⚠️ **REJECTED DEPLOYMENT**\n"
                    f"- Unsufficient performance: {accuracy:.4f} < {MIN_ACCURACY}\n"
                    f"- Le modèle ne sera pas poussé en production.")
        logger.warning(fail_msg)
        sys.exit(1) # Échec pour stopper la CI/CD        

if __name__ == "__main__":
    run_and_audit()