import sys
import optuna
import yaml
import shutil
from datetime import datetime
from loguru import logger
from core.pipeline_core.pipeline_builder import PipelineBuilder
from core.pipeline_core.pipeline_core import PipelineContext, PipelineOrchestrator
import prefect
from prefect import flow, task


@task(name="Optuna_Search")
def optimize_hyperparams(algo: str = "RF"):
    def objective(trial):
        # ... Training (optimization mode) ...
        logger.info("🚀 Training pipeline starting (CI/CD Mode)...")
        orchestrator = PipelineOrchestrator()
        context = PipelineContext()
        
        if algo == "RF":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)
            }
            context.add_variable("temp_rf_params", params)
        else:
            params = {
                "C": trial.suggest_float("C", 0.001, 10.0, log=True),
                "max_iter": trial.suggest_int("max_iter", 100, 2000)
            }
            context.add_variable("temp_lr_params", params)
        

        try:
            PipelineBuilder.build_from_yaml("pipeline_config.yaml", orchestrator=orchestrator)
            orchestrator.run(context=context)
            logger.info("🏁 Pipeline ended successfuly")
            return context.get_variable("last_accuracy", 0)
        except Exception as e:
            logger.error(f"❌ **PIPELINE CRITICAL FAIL** : {str(e)}")
            sys.exit(1)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    return study.best_params

@task(name="Version_And_Update_Config")
def update_and_version_config(rf_best, lr_best):
    # Versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_name = f"pipeline_config_{timestamp}.yaml"
    shutil.copy("pipeline_config.yaml", f"backend/models/history/{version_name}")

    # Main file update
    with open("pipeline_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    if "learning" in config["pipeline"]:
        config["pipeline"]["learning"]["params"]["rf_params"].update(rf_best)
        config["pipeline"]["learning"]["params"]["lr_params"].update(lr_best)
            
    with open("pipeline_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"✅ pipeline_config.yaml was updated with Optuna best runs")

@flow(name="Smart_Retraining")
def training_flow():
    best_rf = optimize_hyperparams(algo="RF")
    best_lr = optimize_hyperparams(algo="LR")

    update_and_version_config(rf_best=best_rf, lr_best=best_lr)

    logger.info("🚀 Final retrain with optimized parameters")
    final_orch = PipelineOrchestrator()
    PipelineBuilder.build_from_yaml("pipeline_config.yaml", final_orch)
    final_orch.run(PipelineContext())