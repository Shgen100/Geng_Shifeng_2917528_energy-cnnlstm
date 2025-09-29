"""
Konfigurationsmodul - zentrale Verwaltung aller Projektparameter
"""


class Config:
    """Projektkonfiguration"""

    # ========== Pfadkonfiguration ==========
    CSV_PATH = "owid-energy_consumption_2000_2024.csv"
    OUTPUT_DIR = "exports"
    FIGURES_DIR = "exports/figures"
    MODELS_DIR = "exports/models"

    # ========== Spaltennamen im Datensatz ==========
    COUNTRY_COL = "country"
    YEAR_COL = "year"
    TARGET_COL = "primary_energy_consumption"
    FEATURE_COLS = [
        "coal_consumption",
        "gas_consumption",
        "oil_consumption",
        "renewables_consumption",
        "nuclear_consumption"
    ]

    # ========== Zeitreihenparameter ==========
    WINDOW_SIZE = 3  # Eingabefenster in Jahren
    PREDICTION_HORIZON = 1  # Vorhersageschritt in Jahren

    # ========== Datensplitting nach Jahren ==========
    TRAIN_START_YEAR = 2000  # Untere Grenze Training
    TRAIN_END_YEAR = 2021    # Ende Trainingsjahre
    VAL_END_YEAR = 2022      # Ende Validierungsjahre
    TEST_END_YEAR = 2024     # Obere Grenze Testjahre

    YEAR_MIN = 2000
    YEAR_MAX = 2024

    # ========== Trainingsparameter ==========
    BATCH_SIZE = 32
    EPOCHS = 150
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    RANDOM_SEED = 42

    # ========== Modellarchitektur ==========
    # LSTM
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT_RATE = 0.3

    # CNN (für CNN-LSTM)
    CNN_FILTERS = [16, 32, 64]
    KERNEL_SIZE = 3

    # ========== Early Stopping ==========
    EARLY_STOPPING_PATIENCE = 20
    MIN_DELTA = 1e-4

    # ========== Cross-Validation ==========
    CV_FOLDS = 3
    CV_REPEATS = 5  # Wiederholungen für Permutation Importance

    # ========== Datenvorverarbeitung ==========
    MISSING_THRESHOLD = 0.3  # maximal zulässiger Fehlwertanteil
    MIN_COUNTRY_SAMPLES = 3  # minimale Stichproben pro Land

    # ========== Länderspezifisches Training ==========
    MIN_SAMPLES_PER_COUNTRY = 10  # Mindestanzahl pro Land für länderspezifisches Training

    # ========== Visualisierung ==========
    FIGURE_DPI = 300
    FIGURE_FORMAT = 'png'
    PLOT_STYLE = 'seaborn-v0_8'

    @classmethod
    def get_device(cls):
        """Bestimme Rechengerät"""
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def create_directories(cls):
        """Erzeuge erforderliche Verzeichnisse"""
        import os
        directories = [cls.OUTPUT_DIR, cls.FIGURES_DIR, cls.MODELS_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        print(f"Verzeichnisse erstellt: {', '.join(directories)}")

    @classmethod
    def print_config(cls):
        """Gib Konfiguration aus"""
        print("=" * 50)
        print("Projektkonfiguration")
        print("=" * 50)
        print(f"Datendatei: {cls.CSV_PATH}")
        print(f"Zeitfenster: {cls.WINDOW_SIZE} Jahr(e)")
        print(f"Vorhersagehorizont: {cls.PREDICTION_HORIZON} Jahr(e)")
        print(f"Trainingsjahre: <= {cls.TRAIN_END_YEAR}")
        print(f"Validierungsjahre: {cls.TRAIN_END_YEAR + 1} - {cls.VAL_END_YEAR}")
        print(f"Testjahre: >= {cls.VAL_END_YEAR + 1}")
        print(f"Batchgröße: {cls.BATCH_SIZE}")
        print(f"Epochen: {cls.EPOCHS}")
        print(f"Lernrate: {cls.LEARNING_RATE}")
        print(f"Gerät: {cls.get_device()}")
        print("=" * 50)


# Globale Konfigurationsinstanz
config = Config()

# Direkter Aufruf: Konfiguration anzeigen und Verzeichnisse anlegen
if __name__ == "__main__":
    config.print_config()
    config.create_directories()
