"""
artifacts_utils.py: Verantwortlich für das Speichern und Laden von Artefakten.
"""

import json
import joblib
from pathlib import Path


def save_artifacts(scaler, feature_cols, country_list, save_dir="exports/artifacts"):
    """
    Speichert modellbezogene Artefakte.

    Parameter:
        scaler: Standardisiererobjekt (z. B. ein Dictionary mit Scaler).
        feature_cols: Liste der Merkmalsnamen.
        country_list: Liste der Länder.
        save_dir: Zielverzeichnis.

    Erzeugt:
        - scaler.joblib
        - artifacts_meta.json
    """
    try:
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)

        # Scaler speichern
        joblib.dump(scaler, d / "scaler.joblib")

        # Metadaten speichern
        meta = {
            "feature_cols": list(feature_cols) if feature_cols is not None else None,
            "countries": list(country_list) if country_list is not None else None,
        }

        with open(d / "artifacts_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"Artefakte gespeichert in {d.resolve()}.")

    except Exception as e:
        print(f"Speichern der Artefakte fehlgeschlagen: {e}")


def load_artifacts(save_dir="exports/artifacts"):
    """
    Lädt modellbezogene Artefakte.

    Parameter:
        save_dir: Verzeichnis, in dem die Artefakte liegen.

    Rückgabe:
        tuple: (scaler, feature_cols, countries)

    Ausnahmen:
        FileNotFoundError: Wenn erforderliche Dateien fehlen.
        Exception: Bei Fehlern während des Ladevorgangs.
    """
    try:
        d = Path(save_dir)
        scaler_path = d / "scaler.joblib"
        meta_path = d / "artifacts_meta.json"

        # Existenz prüfen
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler nicht gefunden: {scaler_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadatei nicht gefunden: {meta_path}")

        # Scaler laden
        scaler = joblib.load(scaler_path)

        # Metadaten laden
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        print(f"Artefakte geladen aus {d.resolve()}.")

        return scaler, meta.get("feature_cols"), meta.get("countries")

    except FileNotFoundError:
        raise
    except json.JSONDecodeError as e:
        raise Exception(f"Ungültiges JSON Format: {e}")
    except Exception as e:
        raise Exception(f"Laden der Artefakte fehlgeschlagen: {e}")


# Hinweis: Die Funktion save_figure befindet sich in utils.Utils.save_figure, um Dopplungen zu vermeiden.
