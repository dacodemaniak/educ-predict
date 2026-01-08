from abc import ABC, abstractmethod
import pandas as pd
from sklearn.ensemble import IsolationForest

class OutlierStrategy(ABC):
    @abstractmethod
    def detect_and_clean(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Detect outliers and clean the input DataFrame according algorithm
        
        :param df: Dataframe to clean
        :type df: pd.DataFrame
        :param columns: Columns to include in strategy
        :type columns: list
        :return: Cleaned DataFrame
        :rtype: pd.DataFrame
        """
        pass

class IQRStrategy(OutlierStrategy):
    """
    IQR split datas into quartiles
    Datas out of the bounds are considered as abnormal, so... line is removed
    """
    def __init__(self, factor: float = 1.5) -> None:
        self.factor = factor
        super().__init__()

    def detect_and_clean(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        df_cleaned = df.copy()

        for col in columns:
            if col in df_cleaned and pd.api.types.is_numeric_dtype(df_cleaned[col]):
                # Sets quartiles
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - self.factor * IQR
                upper_bound = Q3 + self.factor * IQR

                # Process filtering
                df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]

        return df_cleaned
    
class IsolationForestStrategy(OutlierStrategy):
    """
    IsolationForest infer multidimensional datas instead of focus on a set of columns
    """
    def __init__(self, contamination: float = 0.05) -> None:
        try:
            self.contamination = float(contamination) if contamination is not None else 0.05
        except (ValueError, TypeError):
            self.contamination = 0.05
        
        super().__init__()

    def detect_and_clean(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        df_cleaned = df.copy()

        # IF works only on numerical columns
        numeric_cols = [c for c in columns if c in df_cleaned and pd.api.types.is_numeric_dtype(df_cleaned[c])]

        if not numeric_cols:
            return df_cleaned
        
        iso_forest = IsolationForest(contamination=self.contamination, random_state=42)
        # Forcast -1 for outliers, 1 for inliers
        outliers = iso_forest.fit_predict(df_cleaned[numeric_cols])

        return df_cleaned[outliers == 1].reset_index(drop=True)

