from abc import ABC, abstractmethod
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, auc, roc_curve, root_mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix
import os
import joblib
from loguru import logger

from core.pipeline_core.pipeline_core import PipelineContext

class TrainingStrategy(ABC):
    """
    Abstraction layer for concrete strategies
    """
    def __init__(self, yaml_params: dict = None):
        self.pipeline_context = None
        self.yaml_params = yaml_params or {}
    
    def set_context(self, context: PipelineContext) -> None:
        self.pipeline_context = context
    
    @abstractmethod
    def execute(self, df: pd.DataFrame, scenario_name: str):
        """
        The core execution that must be implemented in children classes
        
        :param df: dataframe to train
        :type df: pd.DataFrame
        :param scenario_name: MLFlow scenario name
        :type scenario_name: str
        """
        pass

    def log_metrics(self, y_true, y_pred, metrics_type: str = "regression"):
        if metrics_type == "regression":
            mlflow.log_metric("rmse", root_mean_squared_error(y_true=y_true, y_pred=y_pred))
            mlflow.log_metric("r2", r2_score(y_true=y_true, y_pred=y_pred))
        else:
            mlflow.log_metric("accuracy", accuracy_score(y_true=y_true, y_pred=y_pred))
            mlflow.log_metric("f1_score", f1_score(y_true=y_true, y_pred=y_pred))

    def save_artifacts(self, model, X, y, y_pred, scenario_name):
        # Confusion matrix
        plt.figure(figsize=(8,6))
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Success", "Failed"],
                    yticklabels=["Success", "Failed"]
        )
        plt.title(f"Confusion Matrix {scenario_name}")
        plt.ylabel('Real')
        plt.xlabel('Predicted')
        cm_path = f"outputs/plots/confusion_matrix_{scenario_name}.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()

        # Correlation matrix
        plt.figure(figsize=(12,10))
        corr = X.corr()
        sns.heatmap(corr, annot=False, cmap="coolwarm")
        plt.title(f"Features correlation - {scenario_name}")
        corr_path = f"outputs/plots/correlation_{scenario_name}.png"
        plt.savefig(corr_path)
        mlflow.log_artifact(corr_path)
        plt.close()

        # ROC curve : Receiver Operating Characteristic
        # Use AUC metric (Area Under the Curve)
        plt.figure(figsize=(8,6))
        # Class 1 (failed) probality
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred

        RocCurveDisplay.from_predictions(y, y_prob, name=scenario_name)
        plt.plot([0,1], [0,1], "k--", label="Chance (AUC=0.5)")
        plt.title(f"ROC curve - {scenario_name}")
        roc_path = f"outputs/plots/roc_curve_{scenario_name}.png"
        plt.savefig(roc_path)
        mlflow.log_artifact(roc_path)

        # Log AUC in metrics
        fpr, tpr, _ = roc_curve(y, y_prob)
        mlflow.log_metric("auc_score", auc(fpr, tpr))

## ==========================================================================================
# Concrete strategies
## ==========================================================================================
class LogisticRegressionStrategy(TrainingStrategy):
    def __init__(self, scenario_id: str, exclusions = [], yaml_params: dict = {}):
        super().__init__(yaml_params=yaml_params)

        self.scenario_id = scenario_id
        # Exclusions are feed from handler
        self.exclusions = exclusions

        self.yaml_parameters = yaml_params or {"max_iter": 1000}

    def execute(self, df: pd.DataFrame, scenario_name: str):
        algo_name = "LR"
        logger.info(f"🚀 Scenario {scenario_name} ({algo_name}) running...")

        with mlflow.start_run(run_name=f"{algo_name}_{scenario_name}", nested=True):
            # Filter according scenario
            cols_to_drop = self.exclusions

            # Prepare model target
            df_model = df.copy() # Working on a copy
            target_col = 'G3'
            if target_col not in df_model.columns:
                target_col = "G3_x" if "G3_x" in df_model.columns else 'G3_y'

            if not target_col in df_model.columns:
                raise KeyError(f"Target column 'G3' was not found. Available columns: {df_model.columns.tolist()}")
            
            y = (df_model['G3'] < 10).astype(int)

            # Remove target and technical cols
            X = df_model.drop(columns=cols_to_drop + ['G3', 'source_origin'], errors='ignore')
            X = pd.get_dummies(df_model.drop(columns=self.exclusions + ['G3', 'source_origin'], errors='ignore'), drop_first=True) # Encode text variables
            feature_names = X.columns.tolist()

            # Save feature names
            temp_feat_file = "backend/models/feature_names.pkl"
            joblib.dump(feature_names, temp_feat_file)
            mlflow.log_artifact(temp_feat_file)
            mlflow.set_tag("algorithm",algo_name)
            mlflow.set_tag("scenario", scenario_name)
            
            # Clean up temp file
            if os.path.exists(temp_feat_file):
                os.remove(temp_feat_file)

            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            y_pred = model.predict(X)

            # Logging MLFlow
            mlflow.log_params({"scenario": self.scenario_id, "model": "LogisticRegression"})
            self.log_metrics(y, y_pred, metrics_type="classification")
            mlflow.sklearn.log_model(model, "model")
            
            # Specific artifact Recall
            pr_display = PrecisionRecallDisplay.from_estimator(model, X, y)
            pr_display.figure_.suptitle(f"PR Curve - {scenario_name}")
            plt.savefig("outputs/plots/pr_curve.png")
            mlflow.log_artifact("outputs/plots/pr_curve.png")
            plt.close()

            # Common artifacts
            self.save_artifacts(model, X, y, y_pred, scenario_name)

            logger.info(f"🏆 Scenario {self.scenario_id} ({algo_name}) done.")

class RandomForestStrategy(TrainingStrategy):
    def __init__(self, scenario_id: str, exclusions = [], yaml_params: dict = {}):
        super().__init__(yaml_params=yaml_params)

        self.scenario_id = scenario_id
        self.exclusions = exclusions
        
        self.yaml_params = yaml_params or {"n_estimators": 100, "random_state": 42}

    def execute(self, df: pd.DataFrame, scenario_name: str):
        algo_name = "RF"

        # Get params : Optuna priority
        params = self.yaml_params
        if self.pipeline_context:
            if self.pipeline_context.has_variable("temp_rf_params"):
                optuna_params = self.pipeline_context.get_variable("temp_rf_params")
                params.update(optuna_params)
        
        logger.info(f"🚀 Scenario {scenario_name} ({algo_name}) running...")
        with mlflow.start_run(run_name=f"{algo_name}_{scenario_name}", nested=True):
            df_model = df.copy()

            target_col = 'G3'
            if target_col not in df_model.columns:
                target_col = "G3_x" if "G3_x" in df_model.columns else 'G3_y'

            if not target_col in df_model.columns:
                raise KeyError(f"Target column 'G3' was not found. Available columns: {df_model.columns.tolist()}")
            
            y = (df_model['G3'] < 10).astype(int)
            X = pd.get_dummies(df_model.drop(columns=self.exclusions + ['G3', 'source_origin'], errors='ignore'), drop_first=True)

            feature_names = X.columns.tolist()

            # Save feature names
            temp_feat_file = "backend/models/feature_names.pkl"
            joblib.dump(feature_names, temp_feat_file)
            mlflow.log_artifact(temp_feat_file)
            mlflow.set_tag("algorithm",algo_name)
            mlflow.set_tag("scenario", scenario_name)

            # Clean up temp file
            if os.path.exists(temp_feat_file):
                os.remove(temp_feat_file)

            
            model = RandomForestClassifier(**params)
            model.fit(X, y)

            # Build graphs and artifacts
            import numpy as np
            # 1. Feature Importance Plot
            importances = model.feature_importances_
            indices = np.argsort(importances[-10:]) # 10 bests
            plt.figure(figsize=(10, 6))
            plt.title("RF Importances top ten")
            plt.barh(range(len(indices)), importances[indices], align="center")
            plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
            plt.xlabel("Relative Importance")
            plt.tight_layout()

            # Save and log
            feat_imp_path = "outputs/plots/rf_feature_importance.png"
            plt.savefig(feat_imp_path)
            mlflow.log_artifact(feat_imp_path)
            plt.close()

            # Precision-Recall Curve
            from sklearn.metrics import PrecisionRecallDisplay
            fig, ax = plt.subplots(figsize=(8, 6))
            PrecisionRecallDisplay.from_estimator(model, X, y, ax=ax)
            ax.set_title(f"Precision-Recall Curve - {scenario_name}")
            pr_path = "outputs/plots/rf_precision_recall.png"
            plt.savefig(pr_path)
            # Log
            mlflow.log_artifact(pr_path)
            plt.close()

            # One Tree
            from sklearn.tree import plot_tree
            plt.figure(figsize=(20, 10))
            plot_tree(model.estimators_[0], max_depth=3, feature_names=X.columns.tolist(), filled=True, rounded=True)
            plt.savefig("outputs/plots/rf_individual_tree.png")
            mlflow.log_artifact("outputs/plots/rf_individual_tree.png")
            plt.close()
            
            y_pred = model.predict(X)

            if self.pipeline_context:
                self.pipeline_context.add_variable("last_accuracy", accuracy_score(y, y_pred))
            
            mlflow.log_params({"scenario": self.scenario_id, "model": "RandomForest"})
            self.log_metrics(y, y_pred, metrics_type="classification")
            mlflow.sklearn.log_model(model, "model")