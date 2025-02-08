import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class FeatureTransformerInterface(ABC):
    """
    Interfaccia per strategie di trasformazione di feature (e.g. scaling).
    """

    @abstractmethod
    def transform(self, data: pd.DataFrame, skip_columns: list = None) -> pd.DataFrame:
        pass


class Normalizer(FeatureTransformerInterface):
    """
    Applica una normalizzazione [0,1] alle colonne numeriche.
    """

    def transform(self, data: pd.DataFrame, skip_columns: list = None) -> pd.DataFrame:
        if skip_columns is None:
            skip_columns = []

        df = data.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns_to_scale = [col for col in numeric_cols if col not in skip_columns]

        for col in columns_to_scale:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:  # Evita divisione per zero
                df[col] = (df[col] - min_val) / (max_val - min_val)
        return df


class Standardizer(FeatureTransformerInterface):
    """
    Applica una standardizzazione (mean=0, std=1) alle colonne numeriche.
    """

    def transform(self, data: pd.DataFrame, skip_columns: list = None) -> pd.DataFrame:
        if skip_columns is None:
            skip_columns = []

        df = data.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns_to_scale = [col for col in numeric_cols if col not in skip_columns]

        for col in columns_to_scale:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val != 0:  # Evita divisione per zero
                df[col] = (df[col] - mean_val) / std_val
        return df


class FeatureTransformationManager:
    """
    Gestore delle strategie di trasformazione feature. 
    Consente di scegliere dinamicamente la strategia di scaling.
    """

    @staticmethod
    def apply_transformation(strategy: str, data: pd.DataFrame, skip_columns: list = None) -> pd.DataFrame:
        if strategy.lower() == 'normalize':
            transformer = Normalizer()
        elif strategy.lower() == 'standardize':
            transformer = Standardizer()
        else:
            raise ValueError("Strategia non supportata. Usa 'normalize' o 'standardize'.")

        return transformer.transform(data, skip_columns)
