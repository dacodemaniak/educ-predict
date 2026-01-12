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
def optimize_hyperparams():
    def objective(trial):
        # Sets search space
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)
        }

        # ... Training (optimization mode) ...
        logger.info("🚀 Training pipeline starting (CI/CD Mode)...")
        orchestrator = PipelineOrchestrator()
        context = PipelineContext()
        # Add params in the context
        context.add_variable("temp_rf_params", params)

        try:
            PipelineBuilder.build_from_yaml("pipeline_config.yaml", orchestrator=orchestrator)
            orchestrator.run(context=context)
            logger.info("🏁 Pipeline ended successfuly")
            return context.get_variable("last_accuracy")
        except Exception as e:
            logger.error(f"❌ **PIPELINE CRITICAL FAIL** : {str(e)}")
            raise

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    return study.best_params

@task(name="Version_And_Update_Config")
def update_and_version_config(best_params):
    # Versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_name = f"pipeline_config_{timestamp}.yaml"
    shutil.copy("pipeline_config.yaml", f"backend/models/history/{version_name}")

    # Main file update
    with open("pipeline_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Dynamic RandomForest params update
    for strategy in config['pipeline']['learning']['strategies']:
        if "RandomForest" in strategy:
            # Inject new params
            config['pipeline']['params']['rf_params'] = best_params
            
    with open("pipeline_config.yaml", "w") as f:
        yaml.dump(config, f)

    return version_name

@flow(name="Smart_Retraining")
def training_flow():
    best_params = optimize_hyperparams()

    new_version = update_and_version_config(best_params=best_params)

    print(f"✅ Versioned and updated configuration: {new_version}")