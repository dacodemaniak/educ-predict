from abc import ABC, abstractmethod
import pandas as pd
from sklearn.impute import IterativeImputer, SimpleImputer

class ImputationStrategy(ABC):
    @abstractmethod
    def apply(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Abstraction for data imputation
        
        :param df: The dataframe to impute
        :type df: pd.DataFrame
        :param columns: Columns to impute
        :type columns: list
        :return: Dataframe after imputation was applied
        :rtype: DataFrame
        """
        pass

class AIImputationStrategy(ImputationStrategy):
    """
    Use Regression to estimate missing datas
    Useful when the dataset is really big, hard to visually estimate
    """
    def __init__(self, max_iter=10) -> None:
        try:
            clean_max_iter = int(max_iter) if max_iter is not None else 10
        except (ValueError, TypeError):
            clean_max_iter = 10
        
        self.imputer = IterativeImputer(max_iter=clean_max_iter, random_state=42)
        super().__init__()

    def apply(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        if not columns: return df

        df_to_impute = df.copy()

        cols_present = [c for c in columns if c in df_to_impute.columns]

        if cols_present:
            # Process only numerical datas
            df_to_impute[columns] = self.imputer.fit_transform(df_to_impute[columns])

        return df_to_impute
    
class SimpleImputerStrategy(ImputationStrategy):
    """Classcial statistical approach (mean, median)"""
    def __init__(self, method="median"):
        self.imputer = SimpleImputer(strategy=method)

    def apply(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy[columns] = self.imputer.fit_transform(df_copy[columns])
        return df_copy