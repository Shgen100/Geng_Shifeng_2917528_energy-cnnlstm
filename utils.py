"""
Hilfsfunktionenmodul: allgemeine Utilities
"""

import random
import numpy as np
import pandas as pd
import torch
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config import config


class Utils:
    """Sammlung allgemeiner Hilfsfunktionen"""

    @staticmethod
    def set_seed(seed=None):
        """Setzt den Zufallssamen für reproduzierbare Ergebnisse"""
        if seed is None:
            seed = config.RANDOM_SEED

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"Zufallssamen wurde gesetzt: {seed}")

    @staticmethod
    def _to_python(o):
        """Wandelt numpy/pandas Objekte rekursiv in Python-Basistypen um, damit JSON-serialisierbar"""
        if isinstance(o, dict):
            return {k: Utils._to_python(v) for k, v in o.items()}
        if isinstance(o, (list, tuple, set)):
            return [Utils._to_python(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, pd.Series):
            return o.tolist()
        if isinstance(o, pd.DataFrame):
            return o.to_dict(orient="records")
        return o

    @staticmethod
    def setup_plotting_style():
        """Setzt den Standardstil für Matplotlib-Grafiken"""
        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
        sns.set_palette("husl")
        print("Darstellungsstil wurde konfiguriert")

    @staticmethod
    def save_figure(path, fig=None, dpi=300):
        """Einheitliches Speichern von Abbildungen als PNG und PDF"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if fig is None:
            fig = plt.gcf()

        png = p.with_suffix('.png')
        pdf = p.with_suffix('.pdf')
        fig.savefig(png, dpi=dpi, bbox_inches='tight')
        fig.savefig(pdf, dpi=dpi, bbox_inches='tight')
        print(f"Grafik gespeichert: {png} und {pdf}")
        return str(png)

    @staticmethod
    def smart_read_csv(file_path, encoding='utf-8'):
        """Liest CSV intelligent ein und erkennt das Trennzeichen automatisch"""
        try:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Datei existiert nicht: {file_path}")

            # Erste Zeile lesen, um möglichem Trennzeichen nachzuspüren
            with open(file_path, 'r', encoding=encoding) as f:
                first_line = f.readline()

            separators = [',', ';', '\t', '|']
            separator_counts = {sep: first_line.count(sep) for sep in separators}
            separator = max(separator_counts, key=separator_counts.get)

            if separator_counts[separator] == 0:
                separator = ','

            df = pd.read_csv(file_path, sep=None, engine="python", encoding=encoding)
            print(f"Daten erfolgreich geladen: {df.shape}, vermutetes Trennzeichen: '{separator}'")
            return df

        except Exception as e:
            print(f"Datei konnte nicht gelesen werden: {e}")
            return None

    @staticmethod
    def find_real_column_name(dataframe_columns, expected_name):
        """Findet den tatsächlichen Spaltennamen im DataFrame, inklusive unscharfer Suche"""
        expected_clean = expected_name.strip().lower()

        # Exakte Übereinstimmung
        for col in dataframe_columns:
            if col.strip().lower() == expected_clean:
                return col

        # Unscharfe Übereinstimmung (Teilstrings)
        for col in dataframe_columns:
            col_clean = col.strip().lower()
            if expected_clean in col_clean or col_clean in expected_clean:
                print(f"Unscharfe Zuordnung: '{expected_name}' -> '{col}'")
                return col

        # Teilweise Übereinstimmung über Schlüsselwörter
        expected_keywords = expected_clean.split('_')
        for col in dataframe_columns:
            col_clean = col.strip().lower()
            if any(keyword in col_clean for keyword in expected_keywords):
                print(f"Schlüsselwort-Zuordnung: '{expected_name}' -> '{col}'")
                return col

        available_cols = list(dataframe_columns)[:10]
        raise KeyError(f"Spalte '{expected_name}' nicht gefunden. Verfügbare Spalten (Ausschnitt): {available_cols}")

    @staticmethod
    def validate_data_shape(X, y, expected_features=None):
        """Prüft die Konsistenz der Datenformen"""
        if len(X) != len(y):
            raise ValueError(f"Anzahl der Stichproben passt nicht: X={len(X)}, y={len(y)}")

        if X.ndim not in [2, 3]:
            raise ValueError(f"Falsche Dimensionalität von X: {X.ndim}, erwartet 2D oder 3D")

        if expected_features is not None:
            actual_features = X.shape[-1]
            if actual_features != expected_features:
                raise ValueError(f"Anzahl der Merkmale passt nicht: Ist={actual_features}, Erwartet={expected_features}")

        print(f"Formprüfung bestanden: X={X.shape}, y={y.shape}")

    @staticmethod
    def check_for_nan_inf(data, name="Daten"):
        """Prüft das Vorkommen von NaN- und Inf-Werten"""
        if isinstance(data, torch.Tensor):
            nan_count = torch.isnan(data).sum().item()
            inf_count = torch.isinf(data).sum().item()
        elif isinstance(data, np.ndarray):
            nan_count = np.isnan(data).sum()
            inf_count = np.isinf(data).sum()
        elif isinstance(data, pd.DataFrame):
            nan_count = data.isna().sum().sum()
            inf_count = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
        elif isinstance(data, pd.Series):
            nan_count = data.isna().sum()
            inf_count = np.isinf(data).sum() if pd.api.types.is_numeric_dtype(data) else 0
        else:
            raise TypeError(f"Nicht unterstützter Datentyp: {type(data)}")

        result = {
            'nan_count': nan_count,
            'inf_count': inf_count,
            'is_clean': nan_count == 0 and inf_count == 0
        }

        if result['is_clean']:
            print(f"{name}: keine NaN oder Inf gefunden")
        else:
            print(f"{name}: {nan_count} NaN und {inf_count} Inf gefunden")

        return result

    @staticmethod
    def save_results(data, filename, output_dir="exports", float_fmt="%.6g"):
        """Speichert Ergebnisse in Datei und wählt anhand des Typs das passende Format"""
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        filepath = outdir / filename

        if isinstance(data, pd.DataFrame):
            if filepath.suffix.lower() != ".csv":
                filepath = filepath.with_suffix(".csv")
            data.to_csv(filepath, index=False)

        elif isinstance(data, dict):
            if filepath.suffix.lower() != ".json":
                filepath = filepath.with_suffix(".json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(Utils._to_python(data), f, indent=2, ensure_ascii=False)

        elif isinstance(data, (list, np.ndarray)):
            if filepath.suffix.lower() != ".csv":
                filepath = filepath.with_suffix(".csv")
            np.savetxt(filepath, np.asarray(data), delimiter=",", fmt=float_fmt)

        else:
            raise TypeError(f"Nicht unterstützter Datentyp: {type(data)}")

        print(f"Ergebnisse gespeichert unter: {filepath}")
        return str(filepath)

    @staticmethod
    def format_number(num, decimals=2):
        """Formatiert Zahlen kompakt"""
        if num >= 1e6:
            return f"{num / 1e6:.{decimals}f}M"
        elif num >= 1e3:
            return f"{num / 1e3:.{decimals}f}K"
        else:
            return f"{num:.{decimals}f}"

    @staticmethod
    def print_data_summary(df, title="Datenübersicht"):
        """Gibt eine kompakte Übersicht über einen DataFrame aus"""
        print(f"\n{'=' * 50}")
        print(f"{title}")
        print(f"{'=' * 50}")
        print(f"Form: {df.shape}")
        print(f"Speichernutzung: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
        print(f"Gesamtzahl fehlender Werte: {df.isna().sum().sum()}")

        # Numerische Spalten
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"Anzahl numerischer Spalten: {len(numeric_cols)}")
            print("\nStatistische Kenngrößen:")
            print(df[numeric_cols].describe())

        # Kategorische Spalten
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            print(f"\nAnzahl kategorischer Spalten: {len(cat_cols)}")
            for col in cat_cols[:5]:
                unique_count = df[col].nunique()
                print(f"{col}: {unique_count} eindeutige Werte")

        print("=" * 50)

    @staticmethod
    def calculate_mae(y_true, y_pred):
        """Berechnet den mittleren absoluten Fehler (MAE)"""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def calculate_rmse(y_true, y_pred):
        """Berechnet die Wurzel des mittleren quadratischen Fehlers (RMSE)"""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def calculate_mape(y_true, y_pred, epsilon=1e-8):
        """Berechnet den mittleren absoluten prozentualen Fehler (MAPE)"""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        # Division durch 0 vermeiden
        y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
        return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

    @staticmethod
    def calculate_all_metrics(y_true, y_pred, prefix=""):
        """Berechnet MSE, MAE, RMSE und MAPE und gibt sie als Dictionary zurück"""
        mse = np.mean((y_true - y_pred) ** 2) if isinstance(y_true, np.ndarray) else torch.mean((y_true - y_pred) ** 2).item()
        mae = Utils.calculate_mae(y_true, y_pred)
        rmse = Utils.calculate_rmse(y_true, y_pred)
        mape = Utils.calculate_mape(y_true, y_pred)

        return {
            f'{prefix}loss': mse,
            f'{prefix}mae': mae,
            f'{prefix}rmse': rmse,
            f'{prefix}mape': mape
        }


# Bequeme Wrapper-Funktionen für den Direktimport
def set_seed(seed=42):
    """Bequemer Wrapper zum Setzen des Zufallssamens"""
    return Utils.set_seed(seed)

def smart_read_csv(file_path, encoding='utf-8'):
    """Bequemer Wrapper zum CSV-Einlesen"""
    return Utils.smart_read_csv(file_path, encoding)

def setup_plotting():
    """Bequemer Wrapper zum Setzen des Plot-Stils"""
    return Utils.setup_plotting_style()

def calculate_mae(y_true, y_pred):
    """Bequemer Wrapper für MAE"""
    return Utils.calculate_mae(y_true, y_pred)

def calculate_rmse(y_true, y_pred):
    """Bequemer Wrapper für RMSE"""
    return Utils.calculate_rmse(y_true, y_pred)

def calculate_mape(y_true, y_pred, epsilon=1e-8):
    """Bequemer Wrapper für MAPE"""
    return Utils.calculate_mape(y_true, y_pred, epsilon)

def calculate_all_metrics(y_true, y_pred, prefix=""):
    """Bequemer Wrapper für alle Kennzahlen"""
    return Utils.calculate_all_metrics(y_true, y_pred, prefix)


# Selbsttest
if __name__ == "__main__":
    print("Starte Selbsttest des Hilfsfunktionenmoduls...")

    # Basis
    set_seed(42)
    setup_plotting()

    # Testdaten
    test_data = pd.DataFrame({
        'col1': np.random.randn(100),
        'col2': np.random.choice(['A', 'B', 'C'], 100),
        'col3': np.random.uniform(0, 1, 100)
    })

    Utils.print_data_summary(test_data, "Testdaten")

    # Formprüfung
    X_test = np.random.randn(100, 5, 7)
    y_test = np.random.randn(100)
    Utils.validate_data_shape(X_test, y_test, expected_features=7)

    # Kennzahlen-Test
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.1

    print("\nKennzahlen-Test:")
    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}%")

    all_metrics = calculate_all_metrics(y_true, y_pred, prefix="test_")
    print(f"Alle Kennzahlen: {all_metrics}")

    print("\nSelbsttest abgeschlossen")
