from typing import cast
from core.pipeline_core.pipeline_core import DataHandler, PipelineContext
from core.strategy_core.training_strategies import TrainingStrategy
from sklearn.utils import shuffle
from pandas import DataFrame
from loguru import logger

class ModelHandler(DataHandler):
    def __init__(self, strategy: TrainingStrategy, scenario_label:str):
        """
        Initiate Training Model
        Params:
            strategy: TrainingStrategy one of the strategy to use
            scenario_label: str - Scenario to store into MLFlow tracking
        """
        super().__init__()
        self.strategy = strategy
        self.scenario_label = scenario_label

    def process(self, context: PipelineContext) -> PipelineContext:
        logger.info(f"🚀 Training launching: {self.scenario_label}")
        if context.final_df is None:
            logger.error("❌ dataframe is none. {self.scenario_label} cannot be completed!")
            raise ValueError("Dataframe is none or empty. Training was interrupted")
        
        working_df = context.final_df.copy()
        shuffled_df = cast(DataFrame, shuffle(working_df, random_state=42))

        try:
            self.strategy.set_context(context=context)
            self.strategy.execute(shuffled_df, self.scenario_label)
            context.logs[f"model_{self.scenario_label}"] = "Success"
            
        except Exception as e:
            logger.error(f"❌ Training failed for {self.scenario_label}: {e}")
            context.logs[f"model_{self.scenario_label}"] = f"Failed: {e}"
            raise  # Re-raise the exception after logging
        
        return context