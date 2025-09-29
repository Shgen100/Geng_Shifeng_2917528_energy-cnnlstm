import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from config import config
from utils import Utils


class EnergyDataProcessor:
    """Vorverarbeitung von Energiedaten für Cross-Validation und klassische Splits"""

    def __init__(self):
        self.feature_scalers = {}
        self.target_scalers = {}
        self.feature_cols_real = None
        self.target_col_real = None
        self.country_col_real = None
        self.year_col_real = None
        self.cv_ready_data = None
        self.year_range = None

    def load_data(self, csv_path=None):
        """Daten einlesen und reale Spaltennamen ermitteln"""
        csv_path = csv_path or config.CSV_PATH
        print("\n=== 1. Datenimport ===")

        df_raw = Utils.smart_read_csv(csv_path)
        if df_raw is None:
            raise FileNotFoundError(f"Datei kann nicht gelesen werden: {csv_path}")
        print(f"Form des Rohdatensatzes: {df_raw.shape}")

        # Reale Spaltennamen auflösen
        self.country_col_real = Utils.find_real_column_name(df_raw.columns, config.COUNTRY_COL)
        self.year_col_real = Utils.find_real_column_name(df_raw.columns, config.YEAR_COL)
        self.target_col_real = Utils.find_real_column_name(df_raw.columns, config.TARGET_COL)

        # Verfügbare Feature-Spalten abgleichen
        self.feature_cols_real = []
        for col in config.FEATURE_COLS:
            try:
                real_col = Utils.find_real_column_name(df_raw.columns, col)
                if real_col not in self.feature_cols_real:
                    self.feature_cols_real.append(real_col)
            except KeyError:
                print(f"Spalte nicht gefunden und übersprungen: {col}")

        print(f"Spaltenzuordnung erfolgreich. Verwendete Feature-Spalten: {len(self.feature_cols_real)}")
        return df_raw

    def clean_data(self, df_raw, max_feature_nan_ratio=0.30, impute_target=False,
                   year_min_target=2000, year_max_target=2024):
        """Datenbereinigung je Land, Vorwärtsauffüllen der Features und Zielvariable nach Bedarf"""
        print("\n=== 2. Datenbereinigung ===")
        ccol, ycol, tcol = self.country_col_real, self.year_col_real, self.target_col_real
        fc = list(self.feature_cols_real)

        df = df_raw.copy()
        df[ycol] = pd.to_numeric(df[ycol], errors="coerce").astype("Int64")

        kept, dropped = [], []
        for country, g in df.groupby(ccol, sort=False):
            g = g.sort_values(ycol).copy()
            g[fc] = g[fc].ffill()
            g = g.dropna(subset=fc, how="any")

            if impute_target:
                g[tcol] = g[tcol].ffill().bfill()
            g = g.dropna(subset=[tcol])

            after_ratio = g[fc].isna().mean().mean() if len(g) else 1.0
            if len(g) >= 2 and after_ratio <= max_feature_nan_ratio:
                kept.append(g)
            else:
                dropped.append(country)

        if not kept:
            raise ValueError("Nach der Bereinigung sind keine Länder mehr verfügbar")

        df_clean = pd.concat(kept, ignore_index=True).sort_values([ccol, ycol]).reset_index(drop=True)
        print(f"Rohdaten: {len(df)}/{df[ccol].nunique()} | Bereinigt: {len(df_clean)}/{df_clean[ccol].nunique()}")

        # Jahresgrenzen anwenden
        need_hist = config.WINDOW_SIZE + config.PREDICTION_HORIZON - 1
        raw_min = year_min_target - need_hist
        before = len(df_clean)
        df_clean = df_clean[(df_clean[ycol] >= raw_min) & (df_clean[ycol] <= year_max_target)].copy()
        print(f"Jahreszuschnitt: {raw_min}-{year_max_target} → {len(df_clean)}/{before}")

        if df_clean[fc].isna().any().any():
            raise ValueError("FEATURE_COLS enthalten nach der Bereinigung weiterhin fehlende Werte")
        return df_clean

    def standardize_data_for_cv(self, df_clean):
        """Standardisierung je Land. Skalierer werden ausschließlich auf dem Trainingszeitraum (≤ TRAIN_END_YEAR) gefittet, um Leckagen zu vermeiden."""
        print("\n=== 3. Standardisierung ===")

        # Jahresbereich protokollieren
        self.year_range = {
            'min_year': df_clean[self.year_col_real].min(),
            'max_year': df_clean[self.year_col_real].max(),
            'unique_years': sorted(df_clean[self.year_col_real].unique())
        }
        print(f"Jahresbereich: {self.year_range['min_year']} bis {self.year_range['max_year']}")

        standardized_parts = []
        skipped_countries = []

        for country_name, country_data in df_clean.groupby(self.country_col_real):
            country_data = country_data.sort_values(self.year_col_real).copy()

            # Skalierer nur mit Trainingsjahren fitten (≤ TRAIN_END_YEAR)
            train_mask_year = country_data[self.year_col_real] <= config.TRAIN_END_YEAR
            train_data_year = country_data.loc[train_mask_year].copy()

            # Mindestlänge für robustes Fitten
            min_required_years = max(2, config.WINDOW_SIZE + config.PREDICTION_HORIZON + 1)

            # Fallback: so früh wie möglich abschneiden, jedoch nicht über TRAIN_END_YEAR hinaus
            if len(train_data_year) < 2:
                earliest_k = min(len(country_data), min_required_years)
                fallback_chunk = country_data.iloc[:earliest_k]
                fallback_chunk = fallback_chunk[fallback_chunk[self.year_col_real] <= config.TRAIN_END_YEAR]
                train_data_year = fallback_chunk

            if len(train_data_year) < 2:
                skipped_countries.append(f"{country_name}(zu wenige Trainingsjahre ≤ {config.TRAIN_END_YEAR})")
                continue

            try:
                # Fitten auf Trainingszeitraum
                feat_scaler = StandardScaler()
                targ_scaler = StandardScaler()

                feat_scaler.fit(train_data_year[self.feature_cols_real])
                targ_scaler.fit(train_data_year[[self.target_col_real]])

                # Numerische Stabilität
                if hasattr(feat_scaler, "scale_"):
                    feat_scaler.scale_ = np.maximum(feat_scaler.scale_, 1e-12)
                if hasattr(targ_scaler, "scale_"):
                    targ_scaler.scale_ = np.maximum(targ_scaler.scale_, 1e-12)

                # Transformation über den gesamten Zeitraum des Landes
                c_scaled = country_data.copy()
                c_scaled[self.feature_cols_real] = feat_scaler.transform(country_data[self.feature_cols_real])
                c_scaled["target_scaled"] = targ_scaler.transform(country_data[[self.target_col_real]])[:, 0]

                self.feature_scalers[country_name] = feat_scaler
                self.target_scalers[country_name] = targ_scaler
                standardized_parts.append(c_scaled)

            except Exception as e:
                print(f"Standardisierung fehlgeschlagen für {country_name}: {e}")
                skipped_countries.append(country_name)

        if not standardized_parts:
            raise ValueError("Keine Länder konnten erfolgreich standardisiert werden")

        df_standardized = (
            pd.concat(standardized_parts, ignore_index=True)
            .sort_values([self.country_col_real, self.year_col_real])
            .reset_index(drop=True)
        )
        print(f"Standardisierung abgeschlossen. Länder mit Skalierer: {len(self.feature_scalers)} | Übersprungen: {len(skipped_countries)}")
        return df_standardized

    def create_sliding_windows_for_cv(self, df_standardized):
        """Zeitfenster für Sequenzmodelle erzeugen, inklusive strenger Jahreskonsistenzprüfungen"""
        print("\n=== 4. Erzeuge Gleitfenster ===")

        X_list, y_list, year_list, country_list = [], [], [], []
        country_stats = {}

        for country, group in df_standardized.groupby(self.country_col_real):
            group = group.sort_values(self.year_col_real).reset_index(drop=True)
            min_length = config.WINDOW_SIZE + config.PREDICTION_HORIZON

            if len(group) < min_length:
                continue

            country_windows = 0
            for start_idx in range(len(group) - min_length + 1):
                window_end = start_idx + config.WINDOW_SIZE
                target_index = start_idx + config.WINDOW_SIZE + config.PREDICTION_HORIZON - 1

                # Jahresfolge muss lückenlos sein
                years_window = group.loc[start_idx:window_end - 1, self.year_col_real].to_numpy()
                if not np.all(np.diff(years_window) == 1):
                    continue

                # Zieljahr muss exakt dem Horizont entsprechen
                last_year = years_window[-1]
                target_year = group.loc[target_index, self.year_col_real]
                if target_year != last_year + config.PREDICTION_HORIZON:
                    continue

                # Daten extrahieren
                X_window = group.loc[start_idx:window_end - 1, self.feature_cols_real].to_numpy(dtype=np.float32)
                y_value = group.loc[target_index, "target_scaled"]

                # Vollständigkeit sicherstellen
                if np.isnan(X_window).any() or pd.isna(y_value) or pd.isna(target_year):
                    continue

                X_list.append(X_window)
                y_list.append(np.float32(y_value))
                year_list.append(np.int32(target_year))
                country_list.append(country)
                country_windows += 1

            if country_windows > 0:
                country_stats[country] = country_windows

        # In Arrays umwandeln
        X_windows = np.array(X_list, dtype=np.float32)
        y_windows = np.array(y_list, dtype=np.float32)
        target_years = np.array(year_list, dtype=np.int32)
        countries = np.array(country_list, dtype=object)

        print(f"Fenstererzeugung abgeschlossen. Gesamt: {len(X_windows)} | Formen: X={X_windows.shape}, y={y_windows.shape}")
        if len(X_windows) > 0:
            print(f"Jahre: {target_years.min()} bis {target_years.max()} | Länder: {len(np.unique(countries))}")

        print(f"\nBeispielhafte Jahresstatistik:")
        for year in sorted(np.unique(target_years)):
            count = np.sum(target_years == year)
            countries_in_year = len(np.unique(countries[target_years == year]))
            print(f"  {year}: {count} Stichproben, {countries_in_year} Länder")

        # Ergebnisse ablegen
        self.cv_ready_data = {
            'X_windows': X_windows, 'y_windows': y_windows,
            'target_years': target_years, 'countries': countries,
            'country_stats': country_stats
        }
        return X_windows, y_windows, target_years, countries

    def get_cv_data(self):
        """CV-Daten abrufen, nachdem create_sliding_windows_for_cv ausgeführt wurde"""
        if self.cv_ready_data is None:
            raise ValueError("CV-Daten sind nicht vorbereitet")
        return (self.cv_ready_data['X_windows'], self.cv_ready_data['y_windows'],
                self.cv_ready_data['target_years'], self.cv_ready_data['countries'])

    def split_dataset_by_years(self, X_windows, y_windows, target_years, countries,
                               train_end_year=None, val_end_year=None):
        """Datensatz nach Jahren in Train, Val und Test aufteilen"""
        train_end_year = train_end_year or config.TRAIN_END_YEAR
        val_end_year = val_end_year or config.VAL_END_YEAR

        print(f"\n=== Datensplitting (Train ≤ {train_end_year}, Val = {val_end_year}, Test > {val_end_year}) ===")

        train_mask = (target_years <= train_end_year)
        val_mask = (target_years == val_end_year)
        test_mask = (target_years > val_end_year)

        if not train_mask.any():
            raise ValueError(f"Trainingsmenge leer. TRAIN_END_YEAR prüfen ({train_end_year})")

        # Teilmengen extrahieren
        train_data = (X_windows[train_mask], y_windows[train_mask], countries[train_mask])
        val_data = (X_windows[val_mask], y_windows[val_mask], countries[val_mask])
        test_data = (X_windows[test_mask], y_windows[test_mask], countries[test_mask])

        print(f"Train: {train_data[0].shape[0]} | Val: {val_data[0].shape[0]} | Test: {test_data[0].shape[0]}")
        return train_data, val_data, test_data

    def analyze_temporal_distribution(self, target_years, countries):
        """Zeitliche Verteilung der Zieljahre analysieren"""
        print("\n=== Analyse der Zeitverteilung ===")
        year_counts = pd.Series(target_years).value_counts().sort_index()
        print(f"Jahresverteilung (Top 5): {year_counts.head().to_dict()}")

        country_year_info = {}
        for country in np.unique(countries):
            country_years = target_years[countries == country]
            country_year_info[country] = {
                'min_year': country_years.min(), 'max_year': country_years.max(),
                'span': country_years.max() - country_years.min() + 1,
                'sample_count': len(country_years)
            }

        spans = [info['span'] for info in country_year_info.values()]
        print(f"Zeitspanne: Mittel {np.mean(spans):.1f} Jahre, Bereich {min(spans)} bis {max(spans)} Jahre")
        return country_year_info

    def inverse_transform_by_country(self, scaled_values, countries_array):
        """Rücktransformation der Zielwerte je Land anhand der gelernten Skalierer"""
        original_values = np.empty_like(scaled_values, dtype=np.float32)
        for i, country in enumerate(countries_array):
            if country in self.target_scalers:
                original_values[i] = self.target_scalers[country].inverse_transform([[scaled_values[i]]])[0, 0]
            else:
                original_values[i] = scaled_values[i]
                print(f"Kein Ziel-Skalierer für Land {country} gefunden")
        return original_values


class EnergyDataset(Dataset):
    """PyTorch-Dataset für gleitende Zeitfenster mit Länder- und optionalen Jahresangaben"""

    def __init__(self, X, y, countries, years=None):
        if len(X) != len(y) or len(X) != len(countries):
            raise ValueError(f"Längen passen nicht zusammen: X={len(X)}, y={len(y)}, countries={len(countries)}")

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        self.countries = np.array(countries, dtype=object)
        self.years = np.array(years, dtype=np.int32) if years is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.years is not None:
            return self.X[idx], self.y[idx], self.countries[idx], self.years[idx]
        else:
            return self.X[idx], self.y[idx], self.countries[idx]


def create_cv_dataset(X_windows, y_windows, countries, years):
    """Convenience-Funktion: CV-Dataset aus Fenstern erstellen"""
    return EnergyDataset(X_windows, y_windows, countries, years)


def create_cv_only_dataset(processor, X_windows, y_windows, target_years, countries, max_year=None):
    """
    Datensatz nur für Cross-Validation erstellen. Testperiode (> max_year) wird entfernt, um Datenleckagen zu vermeiden.
    Standardwert für max_year ist config.TRAIN_END_YEAR.
    """
    if max_year is None:
        max_year = config.TRAIN_END_YEAR

    cv_mask = target_years <= max_year
    cv_X = X_windows[cv_mask]
    cv_y = y_windows[cv_mask]
    cv_countries = countries[cv_mask]
    cv_years = target_years[cv_mask]

    print("\n=== Erzeuge CV-exklusiven Datensatz (Leckagevermeidung) ===")
    print(f"Ausgangsdaten: {len(X_windows)} Stichproben (Jahre: {target_years.min()} bis {target_years.max()})")
    print(f"CV-Daten:      {len(cv_X)} Stichproben (Jahre: {cv_years.min()} bis {cv_years.max()})")
    print(f"Entfernt: {(~cv_mask).sum()} Teststichproben (> {max_year})")

    return EnergyDataset(cv_X, cv_y, cv_countries, cv_years)


def create_dataloaders(train_data, val_data, test_data, batch_size=None):
    """DataLoader für Train, Val und Test erzeugen"""
    batch_size = batch_size or config.BATCH_SIZE
    loader_kwargs = {'batch_size': batch_size, 'num_workers': 0, 'pin_memory': False}

    loaders = []
    for data in [train_data, val_data, test_data]:
        if data[0] is not None and len(data[0]) > 0:
            shuffle = data is train_data  # Nur Training wird gemischt
            loader = DataLoader(EnergyDataset(data[0], data[1], data[2]),
                                shuffle=shuffle, drop_last=False, **loader_kwargs)
            loaders.append(loader)
        else:
            loaders.append(None)

    return tuple(loaders)


def complete_data_processing_for_cv(csv_path=None):
    """Komplette Pipeline zur CV-Vorbereitung"""
    Utils.set_seed()
    print("=== Starte CV-Datenverarbeitung ===")

    processor = EnergyDataProcessor()
    df_raw = processor.load_data(csv_path)
    df_clean = processor.clean_data(df_raw)
    df_standardized = processor.standardize_data_for_cv(df_clean)
    X_windows, y_windows, target_years, countries = processor.create_sliding_windows_for_cv(df_standardized)
    processor.analyze_temporal_distribution(target_years, countries)
    cv_dataset = create_cv_dataset(X_windows, y_windows, countries, target_years)

    print("=== CV-Datenverarbeitung abgeschlossen ===")
    return processor, cv_dataset, (X_windows, y_windows, target_years, countries)


def complete_data_processing(csv_path=None):
    """Klassische Pipeline mit Train-, Val- und Test-Split"""
    Utils.set_seed()

    processor = EnergyDataProcessor()
    df_raw = processor.load_data(csv_path)
    df_clean = processor.clean_data(df_raw)
    df_standardized = processor.standardize_data_for_cv(df_clean)
    X_windows, y_windows, target_years, countries = processor.create_sliding_windows_for_cv(df_standardized)
    train_data, val_data, test_data = processor.split_dataset_by_years(X_windows, y_windows, target_years, countries)
    train_loader, val_loader, test_loader = create_dataloaders(train_data, val_data, test_data)

    return processor, train_loader, val_loader, test_loader, (X_windows, y_windows, target_years, countries)


def evaluate_model_with_proper_denormalization(model, test_loader, criterion, processor, device=None):
    """Modellbewertung mit landesspezifischer Rücktransformation der Zielwerte"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_predictions, all_targets, all_countries = [], [], []
    total_loss = 0.0

    print("=== Starte Modellbewertung ===")
    with torch.no_grad():
        for batch_data in test_loader:
            # Unterschiedliche Batch-Formate unterstützen
            if len(batch_data) == 4:
                X_batch, y_batch, countries_batch, _ = batch_data
            else:
                X_batch, y_batch, countries_batch = batch_data

            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            total_loss += criterion(outputs, y_batch).item()

            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            all_countries.extend([str(c).strip() for c in np.ravel(countries_batch)])

    # In Arrays konvertieren
    pred_np = np.array(all_predictions, dtype=np.float32).ravel()
    true_np = np.array(all_targets, dtype=np.float32).ravel()
    countries_np = np.array(all_countries, dtype=object)

    print(f"Skalierte Werte – Vorhersage: [{pred_np.min():.3f}, {pred_np.max():.3f}]  Wahrheit: [{true_np.min():.3f}, {true_np.max():.3f}]")

    # Rücktransformation
    if hasattr(processor, 'target_scalers') and processor.target_scalers:
        pred_original = processor.inverse_transform_by_country(pred_np, countries_np)
        true_original = processor.inverse_transform_by_country(true_np, countries_np)
        denorm_method = "Rücktransformation je Land"
        print(f"Rücktransformation je Land verwendet. Anzahl der Skalierer: {len(processor.target_scalers)}")
    else:
        pred_original, true_original = pred_np, true_np
        denorm_method = "Keine Rücktransformation"
        print("Keine target_scalers gefunden. Skalierte Werte werden beibehalten")

    print(f"Werte nach Rücktransformation – Vorhersage: [{pred_original.min():.3f}, {pred_original.max():.3f}]  Wahrheit: [{true_original.min():.3f}, {true_original.max():.3f}]")

    # Kennzahlen berechnen
    def calc_metrics(pred, true):
        mse = np.mean((pred - true) ** 2)
        mae = np.mean(np.abs(pred - true))
        return mse, mae, np.sqrt(mse)

    mse_scaled, mae_scaled, rmse_scaled = calc_metrics(pred_np, true_np)
    mse_original, mae_original, rmse_original = calc_metrics(pred_original, true_original)

    # Prozentfehler
    mask = np.abs(true_original) > 1e-8
    if mask.sum() > 0:
        true_safe, pred_safe = true_original[mask], pred_original[mask]
        mape = np.mean(np.abs((true_safe - pred_safe) / true_safe)) * 100
        smape = np.mean(2 * np.abs(pred_safe - true_safe) / (np.abs(pred_safe) + np.abs(true_safe))) * 100
        wape = np.sum(np.abs(pred_safe - true_safe)) / np.sum(np.abs(true_safe)) * 100
        ss_res = np.sum((true_safe - pred_safe) ** 2)
        ss_tot = np.sum((true_safe - np.mean(true_safe)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    else:
        mape = smape = wape = r2 = np.nan

    # Bericht ausgeben
    avg_loss = total_loss / len(test_loader)
    print(f"\n=== Bewertungsergebnisse ===")
    print(f"Anzahl Stichproben: {len(pred_original)} | Durchschnittlicher Loss: {avg_loss:.6f}")
    print(f"Skalierte Metriken   – MSE: {mse_scaled:.6f}  MAE: {mae_scaled:.6f}  RMSE: {rmse_scaled:.6f}")
    print(f"Rücktransformierte   – MSE: {mse_original:.6f}  MAE: {mae_original:.6f}  RMSE: {rmse_original:.6f}")
    print(f"Prozentfehler        – MAPE: {mape:.2f}%  sMAPE: {smape:.2f}%  WAPE: {wape:.2f}%  R²: {r2:.4f}")

    return {
        'predictions': pred_original, 'targets': true_original, 'countries': countries_np,
        'predictions_scaled': pred_np, 'targets_scaled': true_np, 'avg_loss': avg_loss,
        'mse_scaled': mse_scaled, 'mae_scaled': mae_scaled, 'rmse_scaled': rmse_scaled,
        'mse_original': mse_original, 'mae_original': mae_original, 'rmse_original': rmse_original,
        'mape': mape, 'smape': smape, 'wape': wape, 'r2': r2, 'denormalization_method': denorm_method
    }


if __name__ == "__main__":
    print("Starte Datentest für Vorverarbeitung...")
    try:
        processor, cv_dataset, raw_data = complete_data_processing_for_cv()
        print(f"CV-Verarbeitung abgeschlossen. Datensatzgröße: {len(cv_dataset)}")

        processor2, train_loader, val_loader, test_loader, raw_data2 = complete_data_processing()
        print("Klassische Verarbeitung abgeschlossen.")

        for name, loader in [("Training", train_loader), ("Validierung", val_loader), ("Test", test_loader)]:
            if loader:
                print(f"Anzahl Batches in {name}: {len(loader)}")

    except Exception as e:
        print(f"Datenverarbeitung fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
