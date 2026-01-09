from core.pipeline_core.pipeline_core import DataHandler, PipelineContext
from loguru import logger
from typing import Dict, Any
import pandas as pd

class EncodingHandler(DataHandler):
    """
    Categorial variables transformation to numerical cols (One-Hot Encoding)
    """
    def process(self, context: PipelineContext) -> PipelineContext:
        if context.final_df is None:
            return context

        logger.info("🔢 One-Hot Encoding on categorial columns...")
        
        # Only 'object' object columns are processed (string)
        categorical_cols = context.final_df.select_dtypes(include=['object']).columns.tolist()
        
        # Exclude source_origin from encoding if exists
        if 'source_origin' in categorical_cols:
            categorical_cols.remove('source_origin')

        context.final_df = pd.get_dummies(
            context.final_df, 
            columns=categorical_cols, 
            drop_first=True
        )
        
        logger.success(f"✅ Encoding done. Final dims : {context.final_df.shape}")
        return context