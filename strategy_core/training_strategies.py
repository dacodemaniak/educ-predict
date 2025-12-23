from abc import ABC, abstractmethod
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, auc, roc_curve, root_mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix

from loguru import logger

class TrainingStrategy(ABC):
    """
    Abstraction layer for concrete strategies
    """

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
    def __init__(self, scenario_id: str, exclusions = []):
        self.scenario_id = scenario_id
        # Exclusions are feed from handler
        self.exclusions = exclusions

    def execute(self, df: pd.DataFrame, scenario_name: str):
        logger.info(f"🚀 Scenario {scenario_name} (LR) running...")
        with mlflow.start_run(run_name=f"LR_{scenario_name}", nested=True):
            # Filter according scenario
            cols_to_drop = self.exclusions

            # Prepare model target
            df_model = df.copy() # Working on a copy
            target_col = 'G3'
            if target_col not in df_model.columns:
                target_col = "G3_x" if "G3_x" in df_model.columns else 'G3_y'

            if not target_col in df_model.columns:
                raise KeyError(f"La colonne cible 'G3' est introuvable. Colonnes disponibles : {df_model.columns.tolist()}")
            
            y = (df_model['G3'] < 10).astype(int)

            # Remove target and technical cols
            X = df_model.drop(columns=cols_to_drop + ['G3', 'source_origin'], errors='ignore')
            X = pd.get_dummies(X, drop_first=True) # Encode text variables

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

            logger.info(f"🏆 Scenario {self.scenario_id} (LR) done.")



class RandomForestStrategy(TrainingStrategy):
    def __init__(self, scenario_id: str, exclusions = []):
        self.scenario_id = scenario_id
        self.exclusions = exclusions

    def execute(self, df: pd.DataFrame, scenario_name: str):
        logger.info(f"🚀 Scenario {scenario_name} (RF) running...")
        with mlflow.start_run(run_name=f"RF_{scenario_name}", nested=True):
            df_model = df.copy()

            target_col = 'G3'
            if target_col not in df_model.columns:
                target_col = "G3_x" if "G3_x" in df_model.columns else 'G3_y'

            if not target_col in df_model.columns:
                raise KeyError(f"La colonne cible 'G3' est introuvable. Colonnes disponibles : {df_model.columns.tolist()}")
            
            y = (df_model['G3'] < 10).astype(int)
            X = pd.get_dummies(df_model.drop(columns=self.exclusions + ['G3', 'source_origin'], errors='ignore'), drop_first=True)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            y_pred = model.predict(X)

            mlflow.log_params({"scenario": self.scenario_id, "model": "RandomForest"})
            self.log_metrics(y, y_pred, metrics_type="classification")
            mlflow.sklearn.log_model(model, "model")