from core.pipeline_core.pipeline_core import DataHandler, PipelineContext
from loguru import logger
from typing import Dict, Any

class AnonymizerHandler(DataHandler):
    """
    Anonymize indirect ids
    """
    def __init__(self, binning_rules: Dict[str, Any]):
        super().__init__()
        self.binning_rules = binning_rules

    def process(self, context: PipelineContext) -> PipelineContext:
        if context.final_df is None:
            logger.error("❌ AnonymizerHandler : Dataframe is empty.")
            return context

        logger.info("🛡️ Indirect ids anonymization...")
        df = context.final_df

        # Age (Top-coding)
        if 'age' in df.columns and (age_limit := self.binning_rules.get('age_limit')):
            df['age'] = df['age'].apply(lambda x: x if x < age_limit else age_limit)
            logger.debug(f"✅ Age limited to {age_limit} years.")

        # School (hidden if decided)
        if 'school' in df.columns and self.binning_rules.get('mask_school', False):
            df['school'] = 'HIDDEN'
            logger.debug("✅ School id was hide.")

        context.final_df = df
        return context