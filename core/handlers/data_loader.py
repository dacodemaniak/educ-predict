from core.pipeline_core.pipeline_core import DataHandler, PipelineContext


class DataLoader(DataHandler):
    """
    Handle data loading from csv
    """
    def __init__(self, files_to_load, sep=","):
        self.files_to_load = files_to_load
        self.sep = sep

        super().__init__()

    def process(self, context: PipelineContext) -> PipelineContext:
        """
        """
        from file_handling_core.file_manager import FileManager
       

        file_manager = FileManager()

        for name, file in self.files_to_load.items():
            df = file_manager.load_data(file_path=file, sep=self.sep)
            if df is not None and len(df) > 0:
                context.data_map[name] = df

        return context
    