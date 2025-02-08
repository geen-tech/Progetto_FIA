import pandas as pd
import numpy as np

class MissingDataHandler:
    """
    Classe che racchiude diversi metodi per la gestione dei valori mancanti.
    """

    @staticmethod
    def convert_numerical_values(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converte tutte le colonne che possono essere interpretate come numeriche in float,
        sostituendo i valori non validi con NaN.
        """
        converted_df = df.copy()
        for col in converted_df.columns:
            # Convertiamo forzatamente in float (coerce -> NaN se non convertibile)
            converted_series = pd.to_numeric(converted_df[col], errors='coerce')
            converted_df[col] = converted_series.astype(float)
        return converted_df

    @staticmethod
    def drop_rows_with_missing_target(df: pd.DataFrame, target_col: str = 'target_class') -> pd.DataFrame:
        """
        Elimina le righe prive di un valore di target.
        Di default, la colonna è 'target_class'.
        """
        if target_col in df.columns:
            return df.dropna(subset=[target_col])
        else:
            return df

    @staticmethod
    def remove_any_missing(df: pd.DataFrame) -> pd.DataFrame:
        """
        Rimuove le righe che hanno almeno un valore mancante.
        """
        return df.dropna(how='any')

    @staticmethod
    def fill_missing_with_mean(df: pd.DataFrame) -> pd.DataFrame:
        """
        Riempie i valori mancanti con la media colonna per colonna.
        """
        return df.fillna(df.mean(numeric_only=True))

    @staticmethod
    def fill_missing_with_median(df: pd.DataFrame) -> pd.DataFrame:
        """
        Riempie i valori mancanti con la mediana colonna per colonna.
        """
        return df.fillna(df.median(numeric_only=True))

    @staticmethod
    def fill_missing_with_mode(df: pd.DataFrame) -> pd.DataFrame:
        """
        Riempie i valori mancanti con la moda colonna per colonna.
        """
        mode_values = df.mode(dropna=True).iloc[0]
        return df.fillna(mode_values)

    @staticmethod
    def fill_missing_ffill(df: pd.DataFrame) -> pd.DataFrame:
        """
        Riempie i valori mancanti con il valore precedente (forward fill).
        """
        return df.ffill()


class MissingDataStrategyManager:
    """
    Classe che permette di applicare diverse strategie di gestione dei valori mancanti
    in maniera dinamica.
    """

    @staticmethod
    def handle_missing_data(strategy: str, data: pd.DataFrame, target_col: str = 'target_class') -> pd.DataFrame:
        """
        Parametri:
            - strategy: stringa che indica la strategia ('remove', 'mean', 'median', 'mode', 'ffill').
            - data: DataFrame Pandas da processare.
            - target_col: eventuale colonna target che NON deve presentare valori mancanti.
        """
        # Prima convertiamo tutte le colonne in numeriche (se possibile)
        df = MissingDataHandler.convert_numerical_values(data)

        # Eliminiamo righe con target mancante (se c'è la colonna)
        df = MissingDataHandler.drop_rows_with_missing_target(df, target_col)

        # Applichiamo la strategia di cleaning specificata
        strategy = strategy.lower()
        if strategy == 'remove':
            df = MissingDataHandler.remove_any_missing(df)
        elif strategy == 'mean':
            df = MissingDataHandler.fill_missing_with_mean(df)
        elif strategy == 'median':
            df = MissingDataHandler.fill_missing_with_median(df)
        elif strategy == 'mode':
            df = MissingDataHandler.fill_missing_with_mode(df)
        elif strategy == 'ffill':
            df = MissingDataHandler.fill_missing_ffill(df)
        else:
            raise ValueError("Strategia non valida. Scegli tra: 'remove', 'mean', 'median', 'mode', 'ffill'.")

        return df


# Esempio di esecuzione (solo se esegui direttamente questo file)
if __name__ == "__main__":
    from .file_parser import ParserDispatcher

    # Esempio: parsing di un file CSV
    file_path = "data/example_data.csv"
    parser = ParserDispatcher.get_parser(file_path)
    raw_data = parser.parse_file(file_path)

    print("\n[DEBUG] Dati originari:")
    print(raw_data.head())

    # Applichiamo la strategia "median" sui valori mancanti
    cleaned_data = MissingDataStrategyManager.handle_missing_data(strategy='median', data=raw_data, target_col='target_class')
    print("\n[DEBUG] Dati dopo gestione missing (median):")
    print(cleaned_data.head())

    # Applichiamo una normalizzazione alle feature numeric-only
    from .feature_transformer import FeatureTransformationManager
    normalized_data = FeatureTransformationManager.apply_transformation('normalize', cleaned_data, skip_columns=['target_class'])
    print("\n[DEBUG] Dati dopo normalizzazione:")
    print(normalized_data.head())
