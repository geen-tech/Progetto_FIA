import unittest
import pandas as pd
import numpy as np

from preprocesso.missing_data_manager import (
    MissingDataHandler,
    MissingDataStrategyManager
)

class TestMissingDataManager(unittest.TestCase):

    def setUp(self):
        """
        Inizializza un DataFrame di esempio con alcuni valori mancanti,
        colonne numeriche/strings e una colonna target con NaN.
        
        Struttura:
          - Riga 0: A=1,   B=2,    classtype_v1=1,   C="10"    (completa, no NaN)
          - Riga 1: A=2,   B=2,    classtype_v1=NaN, C="20"    (target mancante)
          - Riga 2: A=NaN, B=3,    classtype_v1=3,   C="30"    (A mancante)
          - Riga 3: A=4,   B=4,    classtype_v1=4,   C="40"    (completa, no NaN)
        
        In questo modo, ci sono 2 righe senza alcun valore NaN (riga 0 e riga 3).
        La riga 1 ha target mancante, la riga 2 ha A mancante.
        """
        self.sample_data = pd.DataFrame({
            "A": [1, 2, np.nan, 4],
            "B": [2, 2, 3, 4],
            "classtype_v1": [1, np.nan, 3, 4],
            "C": ["10", "20", "30", "40"]
        })

    def test_convert_numerical_values(self):
        """
        Verifica che convert_numerical_values converta correttamente
        tutte le colonne in float (ove possibile), 
        sostituendo i valori non interpretabili con NaN.
        """
        converted_df = MissingDataHandler.convert_numerical_values(self.sample_data)

        # A, B, classtype_v1 e C devono diventare float
        for col in ["A", "B", "classtype_v1", "C"]:
            self.assertTrue(
                pd.api.types.is_float_dtype(converted_df[col]),
                f"La colonna {col} non è float come atteso."
            )

        # Riga 2, colonna A: era np.nan, rimane NaN
        self.assertTrue(pd.isna(converted_df.loc[2, "A"]))

        # Riga 1, colonna classtype_v1: era NaN, rimane NaN
        self.assertTrue(pd.isna(converted_df.loc[1, "classtype_v1"]))

        # Nessun valore 'non interpretabile': la colonna C ("10","20","30","40") diventa [10.0,20.0,30.0,40.0]
        # Quindi nessun NaN in col C
        self.assertFalse(converted_df["C"].isna().any())

    def test_drop_rows_with_missing_target(self):
        """
        Verifica che le righe con classtype_v1 = NaN vengano rimosse.
        """
        df_no_missing_target = MissingDataHandler.drop_rows_with_missing_target(
            self.sample_data, 
            target_col='classtype_v1'
        )
        # Riga 1 ha classtype_v1 = NaN, quindi viene eliminata
        # Restano righe 0, 2, 3 => tot 3 righe
        self.assertEqual(len(df_no_missing_target), 3)
        self.assertFalse(df_no_missing_target['classtype_v1'].isna().any())

    def test_remove_any_missing(self):
        """
        Verifica che remove_any_missing rimuova le righe con almeno un valore NaN.
        """
        df_no_missing = MissingDataHandler.remove_any_missing(self.sample_data)
        # Riga 0 è completa, riga 1 ha target mancante, riga 2 ha A mancante, riga 3 è completa.
        # Quindi rimangono le righe 0 e 3 => 2 righe
        self.assertEqual(len(df_no_missing), 2)
        self.assertFalse(df_no_missing.isna().any().any())

    def test_fill_missing_with_mean(self):
        """
        Verifica che fill_missing_with_mean riempia i NaN con la media delle colonne numeriche.
        """
        # Convertiamo tutto a numerico
        df_converted = MissingDataHandler.convert_numerical_values(self.sample_data)
        df_filled = MissingDataHandler.fill_missing_with_mean(df_converted)

        # Calcoliamo le medie manualmente (sulle righe NON NaN).
        # A => [1,2,4] => media = (1+2+4)/3 = 7/3 ≈ 2.3333
        # B => [2,2,3,4] => media = (2+2+3+4)/4 = 11/4 = 2.75 (ma nessuna riga è NaN su B tranne la row0? No, row0 è 2 => non manca)
        #   In realtà B non ha NaN => non viene modificato
        # classtype_v1 => [1,3,4] => media = (1+3+4)/3=8/3≈2.6667
        # C => [10,20,30,40] => nessun NaN => non viene modificato
        self.assertAlmostEqual(df_filled.loc[2, "A"], 7/3, places=4)   # riga 2 A era NaN, adesso 2.3333...
        self.assertAlmostEqual(df_filled.loc[1, "classtype_v1"], 8/3, places=4)  # riga 1 target era NaN, adesso 2.6667

    def test_fill_missing_with_median(self):
        """
        Verifica che fill_missing_with_median riempia i NaN con la mediana delle colonne numeriche.
        """
        df_converted = MissingDataHandler.convert_numerical_values(self.sample_data)
        df_filled = MissingDataHandler.fill_missing_with_median(df_converted)

        # A => [1,2,4] => mediana = 2
        # B => [2,2,3,4] => mediana = 2.5? (Nel caso di un numero pari di valori, Pandas fa la media dei 2 centrali => 2 e 3 => 2.5)
        #   Ma in colonna B non ci sono NaN, quindi non cambia
        # classtype_v1 => [1,3,4] => mediana=3
        # C => [10,20,30,40] => no NaN, rimane invariato
        self.assertEqual(df_filled.loc[2, "A"], 2.0)   # era NaN, adesso 2.0
        self.assertEqual(df_filled.loc[1, "classtype_v1"], 3.0)  # era NaN, adesso 3.0

    def test_fill_missing_with_mode(self):
        """
        Verifica che fill_missing_with_mode riempia i NaN con la moda delle colonne numeriche.
        """
        df_converted = MissingDataHandler.convert_numerical_values(self.sample_data)
        df_filled = MissingDataHandler.fill_missing_with_mode(df_converted)

        # A => [1,2,4], tutti diversi => Pandas prende la più piccola come mode => 1
        # B => [2,2,3,4], la moda è 2
        # classtype_v1 => [1,3,4], tutti diversi => la moda è 1 (valore più piccolo)
        # C => [10,20,30,40], tutti diversi => la moda=10 (valore più piccolo)
        # Ma B non è NaN in nessuna riga, C non è NaN in nessuna riga => nessun riempimento lì
        # => row2, col A => 1
        # => row1, col classtype_v1 => 1
        self.assertEqual(df_filled.loc[2, "A"], 1.0)
        self.assertEqual(df_filled.loc[1, "classtype_v1"], 1.0)

    def test_fill_missing_ffill(self):
        """
        Verifica il forward fill (ffill).
        """
        df_converted = MissingDataHandler.convert_numerical_values(self.sample_data)
        df_filled = MissingDataHandler.fill_missing_ffill(df_converted)

        # ffill sostituisce NaN con il valore precedente sulla stessa colonna.
        # A => [1, 2, NaN, 4] => riga 2 => prende valore di riga 1 (2)
        self.assertEqual(df_filled.loc[2, "A"], 2.0)
        # classtype_v1 => [1, NaN, 3, 4] => riga 1 => non c'è riga 0? Sì, riga 0 => 1 => quindi riga 1 prende 1
        self.assertEqual(df_filled.loc[1, "classtype_v1"], 1.0)

        # Se una colonna era NaN in riga 0, rimane NaN (non c'è "riga -1"). 
        # In questo dataset B[0] non è NaN => è 2, quindi non c'è problema.

    def test_handle_missing_data_remove(self):
        """
        Verifica la strategia 'remove' tramite MissingDataStrategyManager.
        """
        df_cleaned = MissingDataStrategyManager.handle_missing_data(
            strategy='remove',
            data=self.sample_data,
            target_col='classtype_v1'
        )
        # Passi interni:
        # 1) convert_numerical_values => 'A', 'B', 'classtype_v1', 'C' in float
        # 2) drop_rows_with_missing_target => rimuove riga 1 (target=NaN) => rimangono righe 0,2,3
        # 3) remove_any_missing => riga 2 ha A=NaN => via, riga 0 è ok, riga 3 è ok => rimangono righe 0 e 3
        self.assertEqual(len(df_cleaned), 2)
        self.assertFalse(df_cleaned.isna().any().any())

    def test_handle_missing_data_mean(self):
        """
        Verifica la strategia 'mean' tramite MissingDataStrategyManager.
        """
        df_cleaned = MissingDataStrategyManager.handle_missing_data(
            strategy='mean',
            data=self.sample_data,
            target_col='classtype_v1'
        )
        # Rimuoviamo riga 1 (target NaN), restano righe 0,2,3
        # Poi colonna A e classtype_v1 vengono riempite con la media
        # => nessun NaN residuo su A, B, classtype_v1, C.
        self.assertFalse(df_cleaned["A"].isna().any())
        self.assertFalse(df_cleaned["B"].isna().any())
        self.assertFalse(df_cleaned["classtype_v1"].isna().any())

    def test_handle_missing_data_invalid_strategy(self):
        """
        Verifica che venga sollevata ValueError se la strategia non è valida.
        """
        with self.assertRaises(ValueError):
            MissingDataStrategyManager.handle_missing_data(
                strategy='invalid',
                data=self.sample_data,
                target_col='classtype_v1'
            )

if __name__ == '__main__':
    unittest.main()
