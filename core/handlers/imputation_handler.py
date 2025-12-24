from sklearn.experimental import enable_iterative_imputer
from core.pipeline_core.pipeline_core import DataHandler, PipelineContext
from core.strategy_core.imputation_strategies import ImputationStrategy
from loguru import logger
import pandas as pd

class ImputationHandler(DataHandler):
    """
    Identify and impute missing values
    """
    def __init__(self, strategy: ImputationStrategy):
        self.strategy = strategy
        super().__init__()

    def process(self, context: PipelineContext) -> PipelineContext:

        if context.final_df is None:
            logger.error("❌ SmartImputationHandler : final_df is empty. This handler must be place AFTER MergerHandler.")
            return context
        
        df = context.final_df

        # 1. Automatic NaN columns detection
        nan_report = df.isna().sum()
        cols_with_nan = nan_report[nan_report > 0].index.tolist()

        # Only numercial columns are kept
        target_cols = [c for c in cols_with_nan if pd.api.types.is_numeric_dtype(df[c])]

        if not target_cols:
            logger.info("✅ No missing datas detected in the dataframe")
            return context
        

        logger.info("🛠️ Smart NaN imputation running...")

        # Applying strategy
        initial_nan_count = df[target_cols].isna().sum().sum()
        context.final_df = self.strategy.apply(df, target_cols)

        # Logging and metadatas
        context.logs["imputation_report"] = {
            "fixed_columns": target_cols,
            "total_values_filled": int(initial_nan_count)
        }

        logger.success(f"✨ {initial_nan_count} successfuly missing datas processed")
        
        return context