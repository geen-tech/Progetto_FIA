import pandas as pd
from abc import ABC, abstractmethod


class AbstractFileParser(ABC):
    """
    Interfaccia astratta per il parsing di file in formato tabellare.
    """

    @abstractmethod
    def parse_file(self, file_path: str) -> pd.DataFrame:
        """
        Legge un file e lo converte in un DataFrame Pandas.
        """
        pass


class CSVFileParser(AbstractFileParser):
    """
    Parser per file CSV.
    """
    def parse_file(self, file_path: str) -> pd.DataFrame:
        print(f"[INFO] Parsing CSV: {file_path}")
        df = pd.read_csv(file_path)
        # Rimuove i duplicati in base alla colonna 'ID' (se esiste)
        if 'ID' in df.columns:
            df = df.drop_duplicates(subset='ID').set_index('ID')
        return df


class ExcelFileParser(AbstractFileParser):
    """
    Parser per file Excel (XLSX).
    """
    def parse_file(self, file_path: str) -> pd.DataFrame:
        print(f"[INFO] Parsing Excel: {file_path}")
        df = pd.read_excel(file_path)
        if 'ID' in df.columns:
            df = df.drop_duplicates(subset='ID').set_index('ID')
        return df


class JSONFileParser(AbstractFileParser):
    """
    Parser per file JSON.
    """
    def parse_file(self, file_path: str) -> pd.DataFrame:
        print(f"[INFO] Parsing JSON: {file_path}")
        df = pd.read_json(file_path)
        if 'ID' in df.columns:
            df = df.drop_duplicates(subset='ID').set_index('ID')
        return df


class TXTFileParser(AbstractFileParser):
    """
    Parser per file TXT (delimitato da virgole).
    """
    def parse_file(self, file_path: str) -> pd.DataFrame:
        print(f"[INFO] Parsing TXT: {file_path}")
        df = pd.read_csv(file_path, delimiter=',')
        if 'ID' in df.columns:
            df = df.drop_duplicates(subset='ID').set_index('ID')
        return df


class TSVFileParser(AbstractFileParser):
    """
    Parser per file TSV (delimitato da tab).
    """
    def parse_file(self, file_path: str) -> pd.DataFrame:
        print(f"[INFO] Parsing TSV: {file_path}")
        df = pd.read_csv(file_path, delimiter='\t')
        if 'ID' in df.columns:
            df = df.drop_duplicates(subset='ID').set_index('ID')
        return df


class ParserDispatcher:
    """
    Factory/Dispatcher per restituire il parser adeguato a seconda del formato del file.
    """

    @staticmethod
    def get_parser(file_path: str) -> AbstractFileParser:
        """
        Restituisce un oggetto parser specifico basato sull'estensione del file.
        """
        file_path_lower = file_path.lower()
        if file_path_lower.endswith(".csv"):
            return CSVFileParser()
        elif file_path_lower.endswith(".xlsx"):
            return ExcelFileParser()
        elif file_path_lower.endswith(".json"):
            return JSONFileParser()
        elif file_path_lower.endswith(".txt"):
            return TXTFileParser()
        elif file_path_lower.endswith(".tsv"):
            return TSVFileParser()
        else:
            raise ValueError(f"Formato file non riconosciuto: {file_path}")
