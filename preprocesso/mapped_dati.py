class ValidationMapper:
    def __init__(self, mapping={2: 0, 4: 1}):
        """
        Inizializza il mapper con un dizionario di mappatura predefinito.

        Args:
            mapping (dict, optional): Dizionario che definisce la mappatura delle classi.
                                      Default: {2: 0, 4: 1}
        """
        self.mapping = mapping

    def map(self, validation_data):
        """
        Trasforma validation_data mappando i valori secondo il dizionario fornito.

        Args:
            validation_data (list of tuples): Lista di tuple ([y_real], [y_pred], [y_proba])

        Returns:
            list of tuples: Nuova lista con i valori mappati.
        """
        return self._apply_mapping(validation_data, self.mapping)

    @staticmethod
    def map_static(validation_data, mapping={2: 0, 4: 1}):
        """
        Metodo statico per eseguire la mappatura senza istanziare la classe.

        Args:
            validation_data (list of tuples): Lista di tuple ([y_real], [y_pred], [y_proba])
            mapping (dict, optional): Dizionario che definisce la mappatura delle classi.

        Returns:
            list of tuples: Nuova lista con i valori mappati.
        """
        return ValidationMapper._apply_mapping(validation_data, mapping)

    @staticmethod
    def _apply_mapping(validation_data, mapping):
        """Applica la mappatura ai dati forniti, inclusa la probabilità se presente."""
        mapped_data = []
        for entry in validation_data:
            if len(entry) == 3:  # Controlla se c'è anche y_proba
                y_real, y_pred, probabilities = entry
                mapped_entry = (
                    [mapping.get(x, x) for x in y_real],  # Mappa y_real
                    [mapping.get(x, x) for x in y_pred],  # Mappa y_pred
                    probabilities  # Mantiene le probabilità inalterate
                )
            else:  # Se y_proba non è presente, ignora il terzo elemento
                y_real, y_pred = entry
                mapped_entry = (
                    [mapping.get(x, x) for x in y_real],
                    [mapping.get(x, x) for x in y_pred]
                )
            mapped_data.append(mapped_entry)
        return mapped_data