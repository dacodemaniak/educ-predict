from abc import ABC, abstractmethod
import pandas as pd
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import root_mean_squared_error, r2_score, accuracy_score, f1_score

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

## ==========================================================================================
# Concrete strategies
## ==========================================================================================
class LogisticRegressionStrategy(TrainingStrategy):
    def __init__(self, scenario_id: str, exclusions = []):
        self.scenario_id = scenario_id
        # Exclusions are feed from handler
        self.exclusions = exclusions

    def execute(self, df: pd.DataFrame, scenario_name: str):
        with mlflow.start_run(run_name=f"LR_{scenario_name}", nested=True):
            # Filter according scenario
            cols_to_drop = self.exclusions

            # Prepare model target
            df_model = df.copy() # Working on a copy
            y = (df_model['G3'] < 10).astype(int)

            # Remove target and technical cols
            X = df.drop(columns=cols_to_drop + ['G3', 'source_origin'], errors='ignore')
            X = pd.get_dummies(X, drop_first=True) # Encode text variables

            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            y_pred = model.predict(X)

            # Logging MLFlow
            mlflow.log_params({"scenario": self.scenario_id, "model": "LogisticRegression"})
            self.log_metrics(y, y_pred, metrics_type="classification")
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"🏆 Scenario {self.scenario_id} (LR) done.")



class RandomForestStrategy(TrainingStrategy):
    def __init__(self, scenario_id: str, exclusions = []):
        self.scenario_id = scenario_id
        self.exclusions = exclusions

    def execute(self, df: pd.DataFrame, scenario_name: str):
        with mlflow.start_run(run_name=f"RF_{scenario_name}", nested=True):
            df_model = df.copy()
            y = (df_model['G3'] < 10).astype(int)
            X = pd.get_dummies(df_model.drop(columns=self.exclusions + ['G3', 'source_origin'], errors='ignore'), drop_first=True)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            y_pred = model.predict(X)

            mlflow.log_params({"scenario": self.scenario_id, "model": "RandomForest"})
            self.log_metrics(y, y_pred, metrics_type="classification")
            mlflow.sklearn.log_model(model, "model")