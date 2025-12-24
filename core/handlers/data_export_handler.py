import os
from datetime import datetime
from core.pipeline_core.pipeline_core import DataHandler, PipelineContext
from core.file_handling_core.file_manager import FileManager

from loguru import logger

class DataExportHandler(DataHandler):
    """
    Save dataframe using FileHandler
    Comes after cleaning and merging
    """
    def __init__(self, output_dir: str = "outputs/data_processed"):
        super().__init__()
        self.output_dir = output_dir
        self.file_manager = FileManager()
        
        # Create folder if not exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"📁 Folder created: {self.output_dir}")

    def process(self, context: PipelineContext) -> PipelineContext:
        if context.final_df is None or context.final_df.empty:
            logger.warning("⚠️ DataExportHandler: No data to save (df is empty).")
            return context

        # Filename generation: student_JJMMAAAA_HHMMss_processed.csv
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        file_name = f"student_{timestamp}_processed.csv"
        full_path = os.path.join(self.output_dir, file_name)

        try:
            logger.info(f"💾 Try to save to {full_path}")
            self.file_manager.save_processed_data(context.final_df, full_path)
            
            # Add log to context
            context.metadata['export_path'] = full_path
            
        except Exception as e:
            logger.error(f"❌ Data export failed: {e}")
            raise # Stop here if failed

        return context