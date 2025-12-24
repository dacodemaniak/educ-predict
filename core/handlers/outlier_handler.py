from core.pipeline_core.pipeline_core import DataHandler, PipelineContext
from core.strategy_core.outliers_strategies import OutlierStrategy

from loguru import logger

class OutlierHandler(DataHandler):
    def __init__(self, strategy: OutlierStrategy, target_columns: list):
        super().__init__()
        self.strategy = strategy
        self.target_columns = target_columns

    def process(self, context: PipelineContext) -> PipelineContext:
        """
        Identify ludicrous data and remove all row if found
        """
        logger.info("🛠️ Outliers detection running...")
        
        final_df = context.final_df
        if final_df is not None:
            initial_count = len(final_df)
            df_inlier = self.strategy.detect_and_clean(df=final_df, columns=self.target_columns)
            removed = initial_count - len(df_inlier)

            context.final_df = df_inlier

            # Store metadatas
            if "outlier_reports" not in context.metadata:
                context.metadata['outlier_reports'] = {}
            # Get the concrete strategy name
            s_name = self.strategy.__class__.__name__
            context.metadata["outlier_reports"][s_name] = removed

            logger.debug(f"🗑️ {removed} outliers removed using {s_name}.")

        return context