import pandas as pd
from abc import ABC, abstractmethod

class ValidationProcess(ABC):
    
    @abstractmethod
    def split_data(self, data: pd.DataFrame, labels: pd.Series, k:int) -> list[tuple[list[int], list[int]]]:

        pass
