"""
Länderspezifischer Trainer – trainiert für jedes Land ein eigenes Modell
Zur Gegenüberstellung mit gemeinsamem Training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import os
from pathlib import Path

from config import config
from models import SimpleLSTM, SimpleCNNLSTM
from utils import Utils


class CountrySpecificTrainer:
    """Trainer für länderspezifische Modelle"""

    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.country_models = {}
        self.country_metrics = {}

    def split_data_by_country(self, dataset, countries):
        """Teilt den Datensatz nach Ländern auf"""
        country_data = defaultdict(list)

        # Indizes pro Land sammeln
        for idx, country in enumerate(countries):
            country_data[country].append(idx)

        country_datasets = {}
        for country, indices in country_data.items():
            if len(indices) >= config.MIN_SAMPLES_PER_COUNTRY:  # Mindestanzahl an Stichproben prüfen
                country_datasets[country] = Subset(dataset, indices)
            else:
                print(f"Land {country} wird übersprungen: zu wenige Stichproben ({len(indices)} < {config.MIN_SAMPLES_PER_COUNTRY})")

        return country_datasets

    def train_country_model(self, country, country_dataset, model_type="LSTM", input_size=5):
        """Trainiert ein Modell für ein einzelnes Land"""
        print(f"\nTrainiere {model_type}-Modell für {country} ...")
        print(f"Anzahl Stichproben: {len(country_dataset)}")

        # Modell erzeugen
        if model_type == "LSTM":
            model = SimpleLSTM(
                input_size=input_size,
                hidden_size=config.HIDDEN_SIZE,
                num_layers=config.NUM_LAYERS,
                dropout=config.DROPOUT_RATE
            )
        else:  # CNN-LSTM
            model = SimpleCNNLSTM(
                input_size=input_size,
                hidden_size=config.HIDDEN_SIZE,
                num_layers=config.NUM_LAYERS,
                dropout=config.DROPOUT_RATE,
                cnn_filters=config.CNN_FILTERS,
                kernel_size=config.KERNEL_SIZE
            )

        model = model.to(self.device)

        # DataLoader
        train_loader = DataLoader(
            country_dataset,
            batch_size=min(config.BATCH_SIZE, len(country_dataset)),
            shuffle=True
        )

        # Optimierer und Verlustfunktion
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        criterion = nn.MSELoss()

        # Training mit Early Stopping
        model.train()
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(config.EPOCHS):
            epoch_loss = 0.0
            num_batches = 0

            for batch_data in train_loader:
                X, y = batch_data[:2]
                X, y = X.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                pred = model(X)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches

            # Early-Stopping-Kontrolle
            if avg_loss < best_loss - config.MIN_DELTA:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"  Frühzeitiges Stoppen bei Epoche {epoch + 1}, Loss: {best_loss:.6f}")
                break

            if (epoch + 1) % 20 == 0:
                print(f"  Epoche {epoch + 1}: Loss={avg_loss:.6f}")

        return model, best_loss

    def evaluate_country_model(self, model, country_dataset):
        """Bewertet ein länderspezifisches Modell"""
        model.eval()
        criterion = nn.MSELoss()

        test_loader = DataLoader(
            country_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False
        )

        all_predictions = []
        all_targets = []
        total_loss = 0.0

        with torch.no_grad():
            for batch_data in test_loader:
                X, y = batch_data[:2]
                X, y = X.to(self.device), y.to(self.device)

                pred = model(X)
                loss = criterion(pred, y)

                total_loss += loss.item()
                all_predictions.append(pred.cpu())
                all_targets.append(y.cpu())

        # Vorhersagen zusammenführen
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Kennzahlen berechnen
        metrics = Utils.calculate_all_metrics(all_targets, all_predictions)
        metrics['loss'] = total_loss / len(test_loader)

        return metrics, all_predictions.numpy(), all_targets.numpy()

    def train_all_countries(self, train_dataset, test_dataset, train_countries, test_countries,
                            model_type="LSTM", input_size=5):
        """Trainiert Modelle für alle Länder"""
        print(f"\n=== Starte länderspezifisches {model_type}-Training ===")

        # Trainings- und Testdaten nach Land aufteilen
        country_train_datasets = self.split_data_by_country(train_dataset, train_countries)
        country_test_datasets = self.split_data_by_country(test_dataset, test_countries)

        # Nur Länder trainieren, die auch im Test enthalten sind
        valid_countries = set(country_train_datasets.keys()) & set(country_test_datasets.keys())
        print(f"Gültige Länderanzahl: {len(valid_countries)}")

        results = {}

        for country in sorted(valid_countries):
            try:
                # Modell trainieren
                model, train_loss = self.train_country_model(
                    country, country_train_datasets[country], model_type, input_size
                )

                # Modell testen
                test_metrics, predictions, targets = self.evaluate_country_model(
                    model, country_test_datasets[country]
                )

                results[country] = {
                    'model': model,
                    'train_loss': train_loss,
                    'test_metrics': test_metrics,
                    'predictions': predictions,
                    'targets': targets,
                    'test_samples': len(country_test_datasets[country])
                }

                print(
                    f"{country}: RMSE={test_metrics['rmse']:.4f}, MAE={test_metrics['mae']:.4f}, MAPE={test_metrics['mape']:.2f}%")

            except Exception as e:
                print(f"Training für {country} fehlgeschlagen: {e}")

        return results

    def save_country_models(self, results, model_type, save_dir):
        """Speichert die Modelle je Land"""
        models_dir = Path(save_dir) / "country_models" / model_type.lower()
        models_dir.mkdir(parents=True, exist_ok=True)

        for country, result in results.items():
            model_path = models_dir / f"{country.replace(' ', '_')}_model.pt"
            torch.save({
                'model_state_dict': result['model'].state_dict(),
                'train_loss': result['train_loss'],
                'test_metrics': result['test_metrics']
            }, model_path)

        print(f"Länderspezifische {model_type}-Modelle gespeichert unter: {models_dir}")

    def analyze_country_results(self, results, model_type):
        """Analysiert die Ergebnisse des länderspezifischen Trainings"""
        if not results:
            return {}

        # Alle Kennzahlen sammeln
        metrics_list = []
        for country, result in results.items():
            metrics = result['test_metrics'].copy()
            metrics['country'] = country
            metrics['test_samples'] = result['test_samples']
            metrics_list.append(metrics)

        metrics_df = pd.DataFrame(metrics_list)

        # Statistische Zusammenfassung berechnen
        summary = {
            'total_countries': len(results),
            'mean_rmse': metrics_df['rmse'].mean(),
            'std_rmse': metrics_df['rmse'].std(),
            'mean_mae': metrics_df['mae'].mean(),
            'std_mae': metrics_df['mae'].std(),
            'mean_mape': metrics_df['mape'].mean(),
            'std_mape': metrics_df['mape'].std(),
            'best_country': metrics_df.loc[metrics_df['mape'].idxmin(), 'country'],
            'worst_country': metrics_df.loc[metrics_df['mape'].idxmax(), 'country'],
            'best_mape': metrics_df['mape'].min(),
            'worst_mape': metrics_df['mape'].max()
        }

        print(f"\n=== Zusammenfassung des {model_type}-Trainings nach Ländern ===")
        print(f"Anzahl trainierter Länder: {summary['total_countries']}")
        print(f"Durchschnittlicher RMSE: {summary['mean_rmse']:.4f} ± {summary['std_rmse']:.4f}")
        print(f"Durchschnittlicher MAE: {summary['mean_mae']:.4f} ± {summary['std_mae']:.4f}")
        print(f"Durchschnittliche MAPE: {summary['mean_mape']:.2f}% ± {summary['std_mape']:.2f}%")
        print(f"Bestes Land: {summary['best_country']} (MAPE: {summary['best_mape']:.2f}%)")
        print(f"Schlechtestes Land: {summary['worst_country']} (MAPE: {summary['worst_mape']:.2f}%)")

        return {
            'summary': summary,
            'details': metrics_df,
            'results': results
        }


def compare_training_approaches(joint_test_metrics, country_lstm_analysis, country_cnn_analysis):
    """Vergleicht gemeinsames Training mit länderspezifischem Training"""
    print("\n" + "=" * 60)
    print("Vergleich: gemeinsames Training vs. länderspezifisches Training")
    print("=" * 60)

    # Ergebnisse des gemeinsamen Trainings
    joint_mape = joint_test_metrics.get('mape', 0)
    joint_rmse = joint_test_metrics.get('rmse', 0)
    joint_mae = joint_test_metrics.get('mae', 0)

    print(f"\nErgebnisse gemeinsames Training:")
    print(f"  RMSE: {joint_rmse:.4f}")
    print(f"  MAE: {joint_mae:.4f}")
    print(f"  MAPE: {joint_mape:.2f}%")

    # Ergebnisse des länderspezifischen Trainings
    if country_lstm_analysis and 'summary' in country_lstm_analysis:
        lstm_summary = country_lstm_analysis['summary']
        print(f"\nErgebnisse länderspezifisches LSTM:")
        print(f"  Durchschnittlicher RMSE: {lstm_summary['mean_rmse']:.4f} ± {lstm_summary['std_rmse']:.4f}")
        print(f"  Durchschnittlicher MAE: {lstm_summary['mean_mae']:.4f} ± {lstm_summary['std_mae']:.4f}")
        print(f"  Durchschnittliche MAPE: {lstm_summary['mean_mape']:.2f}% ± {lstm_summary['std_mape']:.2f}%")

    if country_cnn_analysis and 'summary' in country_cnn_analysis:
        cnn_summary = country_cnn_analysis['summary']
        print(f"\nErgebnisse länderspezifisches CNN-LSTM:")
        print(f"  Durchschnittlicher RMSE: {cnn_summary['mean_rmse']:.4f} ± {cnn_summary['std_rmse']:.4f}")
        print(f"  Durchschnittlicher MAE: {cnn_summary['mean_mae']:.4f} ± {cnn_summary['std_mae']:.4f}")
        print(f"  Durchschnittliche MAPE: {cnn_summary['mean_mape']:.2f}% ± {cnn_summary['std_mape']:.2f}%")

    # Fazit
    print(f"\nFazit:")
    print(f"Gemeinsames Training ist geeignet für:")
    print(f"  - Länder mit wenig Datenbestand, da Wissen geteilt wird")
    print(f"  - das Auffinden allgemeiner Muster")
    print(f"  - begrenzte Rechenressourcen")

    print(f"\nLänderspezifisches Training ist geeignet für:")
    print(f"  - Länder mit ausreichend Daten, für individualisierte Modelle")
    print(f"  - stark länderspezifische Muster")
    print(f"  - Anwendungen mit sehr hohen Genauigkeitsanforderungen")


def run_country_specific_comparison(train_val_dataset, test_dataset, train_val_countries,
                                    test_countries, input_size, save_dir="exports"):
    """Führt den länderspezifischen Trainingsvergleich aus"""
    Utils.set_seed(config.RANDOM_SEED)

    trainer = CountrySpecificTrainer(device=config.get_device())

    # Länderspezifisches LSTM-Training
    lstm_results = trainer.train_all_countries(
        train_val_dataset, test_dataset, train_val_countries, test_countries,
        model_type="LSTM", input_size=input_size
    )
    lstm_analysis = trainer.analyze_country_results(lstm_results, "LSTM")
    trainer.save_country_models(lstm_results, "LSTM", save_dir)

    # Länderspezifisches CNN-LSTM-Training
    cnn_results = trainer.train_all_countries(
        train_val_dataset, test_dataset, train_val_countries, test_countries,
        model_type="CNN-LSTM", input_size=input_size
    )
    cnn_analysis = trainer.analyze_country_results(cnn_results, "CNN-LSTM")
    trainer.save_country_models(cnn_results, "CNN-LSTM", save_dir)

    # Ergebnisse speichern
    results_summary = {
        'lstm_country_training': lstm_analysis,
        'cnn_country_training': cnn_analysis
    }

    summary_path = Path(save_dir) / "country_specific_training_results.json"
    import json
    with open(summary_path, 'w') as f:
        # Nicht serialisierbare Modellobjekte entfernen
        save_data = {}
        for key, analysis in results_summary.items():
            if analysis and 'details' in analysis:
                save_data[key] = {
                    'summary': analysis['summary'],
                    'country_details': analysis['details'].to_dict('records')
                }

        json.dump(save_data, f, indent=2, default=str)

    print(f"\nErgebnisse des länderspezifischen Trainings gespeichert unter: {summary_path}")

    return lstm_analysis, cnn_analysis


if __name__ == "__main__":
    print("Länderspezifischer Trainer geladen")
    print("Bitte die Funktion run_country_specific_comparison() für den Vergleich verwenden")
