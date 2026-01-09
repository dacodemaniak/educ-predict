from core.pipeline_core.pipeline_core import DataHandler, PipelineContext
from loguru import logger
import pandas as pd
from typing import Dict

class MinimizerHandler(DataHandler):
    """
    Gather socials and economical categories to conform to RGPD
    """
    def __init__(self, mapping: Dict[str, Dict[str, str]]):
        super().__init__()
        self.mapping = mapping

    def process(self, context: PipelineContext) -> PipelineContext:
        if context.final_df is None:
            logger.error("❌ MinimizerHandler : Empty dataframe")
            return context

        logger.info("📉 Minimize socials and economicals categories (Post-Merge)...")
        
        df = context.final_df
        for column, replace_map in self.mapping.items():
            if column in df.columns:
                before_counts = df[column].nunique()
                df[column] = df[column].replace(replace_map)
                after_counts = df[column].nunique()
                logger.debug(f"✅ Column '{column}' reduced from {before_counts} to {after_counts} categories.")
            else:
                logger.warning(f"⚠️ Columns '{column}' not found in final dataframe")
            
        context.final_df = df
        return context