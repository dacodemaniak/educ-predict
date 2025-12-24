from core.pipeline_core.pipeline_core import DataHandler, PipelineContext
from loguru import logger

class SensitiveDataHandler(DataHandler):
    """
    Remove all sensible columns from sources before merge the two dataframes
    See:
        DataHandler abstract class
    """
    def __init__(self, sensitive_columns: list):
        super().__init__()
        self.sensitive_columns = sensitive_columns

    def process(self, context: PipelineContext) -> PipelineContext:
        logger.info(f"🔎 Check sensitives columns in: {len(context.data_map)} sources")

        for name, df in context.data_map.items():
            to_drop = [col for col in self.sensitive_columns if col in df.columns]

            if to_drop:
                context.data_map[name] = df.drop(columns=to_drop)
                logger.debug(f"❌ Remove {to_drop} from source: {name}")
            else:
                logger.debug(f"🌱 No sensitive datas in source: {name}")
        return context