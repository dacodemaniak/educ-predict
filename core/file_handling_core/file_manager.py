import pandas as pd
from loguru import logger

class FileManager:
    """
    Manage standard operations on csv files
    - Open,
    - Load in Dataframe,
    - Save to other location
    """

    def __init__(self):
        self.original_data = None

    def load_data(self, file_path: str, sep: str = ",") -> pd.DataFrame:
        """
        Load a csv file and returns DataFrame
        
        :param file_path: Path to the csv file
        :type file_path: str
        :return: Pandas Dataframe from csv
        :rtype: DataFrame
        """

        try:
            self.original_data = pd.read_csv(file_path, sep=sep)
            logger.info(f"📄 Successfuly loaded data from: {file_path}")

            print(f"\nData shape: {self.original_data.shape}")
            print(f"\nColumns: {self.original_data.columns}")
            print(f"\nCRows: {len(self.original_data)}")
            print(f"\nTypes:\n {self.original_data.dtypes}")

            return self.original_data
        except Exception as e:
            logger.error(f"❌ Data loading error; {e}")
            return pd.DataFrame({})

    def save_processed_data(self, df, file_path: str):
        """Save processed data to CSV"""
        try:
            if df is not None:
               df.to_csv(file_path, index=False)
               logger.success(f" ✅  Processed data saved to {file_path}")
               logger.debug(f"Final data shape: {df.shape}")
            else:
                logger.warning("No dataframe to save")
        except Exception as e:
            logger.error(f"❌ Error saving data to {file_path}: {str(e)}")
            raise

       