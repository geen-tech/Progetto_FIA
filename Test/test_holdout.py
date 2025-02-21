import unittest
import pandas as pd
import numpy as np

from preprocesso.feature_transformer import (
    Normalizer,
    Standardizer,
    FeatureTransformationManager
)


class TestFeatureTransformer(unittest.TestCase):

    def setUp(self):
        """
        Questo metodo viene eseguito prima di ogni test.
        Inizializziamo qui un DataFrame di esempio.
        """
        self.sample_data = pd.DataFrame({
            'col_numeric_1': [10, 20, 30, 40, 50],
            'col_numeric_2': [100, 100, 100, 300, 500],
            'col_categorical': ['A', 'B', 'A', 'B', 'C'],
            'col_constant': [1, 1, 1, 1, 1],
        })

    def test_normalizer(self):
        """
        Verifica che il Normalizer applichi correttamente la normalizzazione [0,1]
        alle colonne numeriche non saltate.
        """
        normalizer = Normalizer()
        transformed_df = normalizer.transform(self.sample_data)

        # Verifica che la forma (shape) non sia cambiata
        self.assertEqual(transformed_df.shape, self.sample_data.shape)

        # Colonne da controllare
        numeric_cols = ['col_numeric_1', 'col_numeric_2', 'col_constant']

        for col in numeric_cols:
            col_min = transformed_df[col].min()
            col_max = transformed_df[col].max()

            # Se la colonna è costante, non dovrebbe essere trasformata 
            # (o rimane invariata a causa del check "Evita divisione per zero")
            if self.sample_data[col].nunique() == 1:
                # Confrontiamo se i valori sono invariati
                self.assertTrue((transformed_df[col] == self.sample_data[col]).all())
            else:
                # min ~ 0 e max ~ 1
                self.assertAlmostEqual(col_min, 0.0, places=3)
                self.assertAlmostEqual(col_max, 1.0, places=3)

        # Verifica che la colonna categorica sia invariata
        self.assertTrue((transformed_df['col_categorical'] == self.sample_data['col_categorical']).all())

    def test_normalizer_skip_columns(self):
        """
        Verifica che, se si specificano colonne da saltare, queste non vengano trasformate.
        """
        normalizer = Normalizer()
        skip_cols = ['col_numeric_2']
        transformed_df = normalizer.transform(self.sample_data, skip_columns=skip_cols)

        # La col_numeric_2 non deve essere cambiata
        self.assertTrue((transformed_df['col_numeric_2'] == self.sample_data['col_numeric_2']).all())

        # col_numeric_1 è normalizzata (non costante) => min=0, max=1
        col_min = transformed_df['col_numeric_1'].min()
        col_max = transformed_df['col_numeric_1'].max()
        self.assertAlmostEqual(col_min, 0.0, places=3)
        self.assertAlmostEqual(col_max, 1.0, places=3)

    def test_standardizer(self):
        """
        Verifica che lo Standardizer applichi correttamente la standardizzazione
        (mean=0, std=1) alle colonne numeriche non saltate.
        """
        standardizer = Standardizer()
        transformed_df = standardizer.transform(self.sample_data)

        self.assertEqual(transformed_df.shape, self.sample_data.shape)

        numeric_cols = ['col_numeric_1', 'col_numeric_2', 'col_constant']

        for col in numeric_cols:
            mean_val = transformed_df[col].mean()
            std_val = transformed_df[col].std()

            # Se la colonna è costante, rimane invariata
            if self.sample_data[col].nunique() == 1:
                self.assertTrue((transformed_df[col] == self.sample_data[col]).all())
            else:
                # mean ~ 0 e std ~ 1
                self.assertAlmostEqual(mean_val, 0.0, places=3)
                self.assertAlmostEqual(std_val, 1.0, places=3)

        self.assertTrue((transformed_df['col_categorical'] == self.sample_data['col_categorical']).all())

    def test_standardizer_skip_columns(self):
        """
        Verifica che, se si specificano colonne da saltare, queste non vengano trasformate.
        """
        standardizer = Standardizer()
        skip_cols = ['col_numeric_1']
        transformed_df = standardizer.transform(self.sample_data, skip_columns=skip_cols)

        # col_numeric_1 resta invariata
        self.assertTrue((transformed_df['col_numeric_1'] == self.sample_data['col_numeric_1']).all())

        # col_numeric_2 dev'essere standardizzata => mean ~ 0, std ~ 1
        col_mean = transformed_df['col_numeric_2'].mean()
        col_std = transformed_df['col_numeric_2'].std()
        self.assertAlmostEqual(col_mean, 0.0, places=3)
        self.assertAlmostEqual(col_std, 1.0, places=3)

        # La colonna costante rimane invariata
        self.assertTrue((transformed_df['col_constant'] == self.sample_data['col_constant']).all())

    def test_feature_transformation_manager_normalize(self):
        """
        Verifica che il FeatureTransformationManager applichi la strategia di normalizzazione.
        """
        transformed_df = FeatureTransformationManager.apply_transformation(
            strategy='normalize',
            data=self.sample_data
        )
        # col_numeric_1 e col_numeric_2 non costanti => min=0, max=1
        for col in ['col_numeric_1', 'col_numeric_2']:
            self.assertAlmostEqual(transformed_df[col].min(), 0.0, places=3)
            self.assertAlmostEqual(transformed_df[col].max(), 1.0, places=3)

        # col_constant rimane invariata
        self.assertTrue((transformed_df['col_constant'] == self.sample_data['col_constant']).all())
        # col_categorical invariata
        self.assertTrue((transformed_df['col_categorical'] == self.sample_data['col_categorical']).all())

    def test_feature_transformation_manager_standardize(self):
        """
        Verifica che il FeatureTransformationManager applichi la strategia di standardizzazione.
        """
        transformed_df = FeatureTransformationManager.apply_transformation(
            strategy='standardize',
            data=self.sample_data
        )
        # col_numeric_1 e col_numeric_2 => mean=0, std=1
        for col in ['col_numeric_1', 'col_numeric_2']:
            self.assertAlmostEqual(transformed_df[col].mean(), 0.0, places=3)
            self.assertAlmostEqual(transformed_df[col].std(), 1.0, places=3)

        # col_constant invariato
        self.assertTrue((transformed_df['col_constant'] == self.sample_data['col_constant']).all())
        # col_categorical invariata
        self.assertTrue((transformed_df['col_categorical'] == self.sample_data['col_categorical']).all())

    def test_feature_transformation_manager_invalid_strategy(self):
        """
        Verifica che venga sollevata ValueError per strategia inesistente.
        """
        with self.assertRaises(ValueError):
            FeatureTransformationManager.apply_transformation(
                strategy='invalid_strategy',
                data=self.sample_data
            )


if __name__ == '__main__':
    unittest.main()
