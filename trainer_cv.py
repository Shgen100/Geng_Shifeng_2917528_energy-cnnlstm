"""
Zeitreihen-Cross-Validation-Trainer
Behandelt Verteilungsverschiebungen und dient zur Modellauswahl sowie Hyperparameterabstimmung
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score
from config import config
from utils import Utils
from matplotlib.offsetbox import AnchoredText

plt.rcParams.update({
    "font.family": "DejaVu Sans",  # lateinische Schrift ist ausreichend
    "axes.unicode_minus": False
})


class TimeSeriesTrainer:
    """Trainer für Zeitreihen-Cross-Validation"""

    def __init__(self, model, device=None, processor=None):
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.cv_results = []
        self.fold_models = []
        self.processor = processor

    def evaluate_model(self, data_loader, processor=None):
        """Bewertet das Modell auf einem Datensatz. Unterstützt Rücktransformation und Kennzahlen im Originalmaßstab."""
        self.model.eval()
        criterion = nn.MSELoss()

        all_predictions = []
        all_targets = []
        all_countries = []
        total_loss = 0.0

        with torch.no_grad():
            for batch_data in data_loader:
                # Entpacken: (X, y, country[, year])
                if len(batch_data) >= 3:
                    X, y, countries = batch_data[:3]
                else:
                    X, y = batch_data[:2]
                    countries = None

                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss = criterion(pred, y)
                total_loss += loss.item()

                all_predictions.append(pred.cpu().numpy().reshape(-1))
                all_targets.append(y.cpu().numpy().reshape(-1))

                if countries is not None:
                    if isinstance(countries, torch.Tensor):
                        countries = countries.cpu().numpy()
                    all_countries.extend([str(c) for c in np.ravel(countries)])

        # Standardisierungsraum
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Falls vorhanden: Rücktransformation je Land über processor
        if (processor is not None and
                hasattr(processor, "target_scalers") and processor.target_scalers and
                len(all_countries) == len(all_predictions)):
            preds_real = processor.inverse_transform_by_country(all_predictions, np.array(all_countries, dtype=object))
            trues_real = processor.inverse_transform_by_country(all_targets, np.array(all_countries, dtype=object))
        else:
            preds_real = all_predictions
            trues_real = all_targets

        # Kennzahlen im Originalmaßstab
        mae = float(np.mean(np.abs(preds_real - trues_real)))
        rmse = float(np.sqrt(np.mean((preds_real - trues_real) ** 2)))

        eps = 1e-8
        mask = np.abs(trues_real) > eps
        mape = float(np.mean(np.abs((preds_real[mask] - trues_real[mask]) / trues_real[mask])) * 100.0) if np.any(
            mask) else float("inf")

        return {
            'loss': total_loss / len(data_loader),
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }

    def train_single_fold(self, train_loader, val_loader, epochs=50, learning_rate=1e-4, verbose=False):
        """Trainiert einen einzelnen Fold"""
        Utils.set_seed()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.MSELoss()

        train_history = {'loss': [], 'mae': [], 'rmse': [], 'mape': []}
        val_history = {'loss': [], 'mae': [], 'rmse': [], 'mape': []}

        best_val_loss = float('inf')
        best_metrics = None
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            for batch_data in train_loader:
                X, y = batch_data[:2]
                X, y = X.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                pred = self.model(X)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

            # Bewertung
            train_metrics = self.evaluate_model(train_loader, processor=self.processor)
            val_metrics = self.evaluate_model(val_loader, processor=self.processor)

            for key, value in train_metrics.items():
                train_history[key].append(value)
            for key, value in val_metrics.items():
                val_history[key].append(value)

            # Early Stopping
            current_val_loss = val_metrics['loss']
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_metrics = val_metrics.copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 10:
                break

            if verbose and epoch % 10 == 0:
                print(f"  Epoche {epoch}: Train Loss={train_metrics['loss']:.4f}, "
                      f"Val Loss={val_metrics['loss']:.4f}, MAE={val_metrics['mae']:.4f}, "
                      f"RMSE={val_metrics['rmse']:.4f}, MAPE={val_metrics['mape']:.2f}%")

        return train_history, val_history, best_metrics


def create_time_series_cv_splits(X_windows, y_windows, target_years, n_splits=3):
    """Erstellt Zeitreihen-Cross-Validation-Splits"""
    # Indizes nach Zieljahr sortieren
    sorted_indices = np.argsort(target_years)
    sorted_years = target_years[sorted_indices]

    # Splitpunkte bestimmen
    unique_years = np.unique(sorted_years)
    total_years = len(unique_years)

    if total_years < n_splits + 1:
        print(f"Warnung: nur {total_years} Jahre vorhanden, Anzahl Splits auf {total_years - 1} reduziert")
        n_splits = total_years - 1

    splits = []

    for i in range(n_splits):
        # Jahresgrenzen für Training und Validierung
        train_start_year = unique_years[0]
        train_end_year = unique_years[int((i + 1) * total_years / (n_splits + 1))]
        val_start_year = train_end_year + 1
        val_end_year = unique_years[min(int((i + 2) * total_years / (n_splits + 1)), total_years - 1)]

        # Zugehörige Stichprobenindizes
        train_mask = (sorted_years >= train_start_year) & (sorted_years <= train_end_year)
        val_mask = (sorted_years >= val_start_year) & (sorted_years <= val_end_year)

        train_indices = sorted_indices[train_mask]
        val_indices = sorted_indices[val_mask]

        if len(train_indices) > 0 and len(val_indices) > 0:
            splits.append({
                'train_indices': train_indices,
                'val_indices': val_indices,
                'train_years': f"{train_start_year}-{train_end_year}",
                'val_years': f"{val_start_year}-{val_end_year}",
                'train_size': len(train_indices),
                'val_size': len(val_indices)
            })

    return splits


def run_time_series_cv(dataset, X_windows, y_windows, target_years, model_class,
                       input_size, n_splits=3, batch_size=32, processor=None):
    """Führt Zeitreihen-Cross-Validation mit Leckage-Schutz aus"""
    Utils.set_seed()

    # Schutz vor Datenleckagen: Testjahre entfernen
    if target_years.max() > config.TRAIN_END_YEAR:
        print(f"Warnung: CV-Daten enthalten Testjahre, max Jahr={target_years.max()}, Filter auf ≤{config.TRAIN_END_YEAR}")
        train_mask = target_years <= config.TRAIN_END_YEAR
        valid_idx = np.where(train_mask)[0]

        # Datensatz beschneiden
        dataset = Subset(dataset, valid_idx)

        # X, y und Jahre synchron filtern
        X_windows = X_windows[train_mask]
        y_windows = y_windows[train_mask]
        target_years = target_years[train_mask]

    # Ab hier dürfen keine Testjahre mehr enthalten sein
    assert target_years.max() <= config.TRAIN_END_YEAR, \
        f"CV-Daten enthalten weiterhin Testjahre, max Jahr: {target_years.max()} > {config.TRAIN_END_YEAR}"

    print(f"Starte Zeitreihen-Cross-Validation ({n_splits} Folds)...")
    print(f"Jahresbereich der CV-Daten: {target_years.min()}-{target_years.max()}")

    # Splits erzeugen
    splits = create_time_series_cv_splits(X_windows, y_windows, target_years, n_splits)
    cv_results = []

    for fold_idx, split in enumerate(splits):
        print(f"\nFold {fold_idx + 1}/{len(splits)}")
        print(f"  Trainingsjahre: {split['train_years']} ({split['train_size']} Stichproben)")
        print(f"  Validierungsjahre: {split['val_years']} ({split['val_size']} Stichproben)")

        # Teilmengen erzeugen
        train_subset = Subset(dataset, split['train_indices'])
        val_subset = Subset(dataset, split['val_indices'])

        # DataLoader
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Modell initialisieren
        model = model_class(input_size=input_size)
        trainer = TimeSeriesTrainer(model, processor=processor)

        # Training
        train_history, val_history, best_metrics = trainer.train_single_fold(
            train_loader, val_loader, epochs=50, learning_rate=1e-4, verbose=False
        )

        # Endgültige Trainingsmetriken berechnen
        final_train_metrics = trainer.evaluate_model(train_loader, processor=processor)

        # Ergebnisse speichern
        cv_results.append({
            'fold': fold_idx + 1,
            'train_years': split['train_years'],
            'val_years': split['val_years'],
            'train_size': split['train_size'],
            'val_size': split['val_size'],
            'train_history': train_history,
            'val_history': val_history,
            'best_val_metrics': best_metrics,
            'final_train_metrics': final_train_metrics,
            'model': model
        })

        print(f"  Beste Validierung: Loss={best_metrics['loss']:.4f}, MAE={best_metrics['mae']:.4f}, "
              f"RMSE={best_metrics['rmse']:.4f}, MAPE={best_metrics['mape']:.2f}%")

    return cv_results


def run_time_series_cv_comparison(dataset, target_years, n_splits=3, processor=None):
    """Führt die Zeitreihen-Cross-Validation durch und vergleicht Modelle. Mit Leckage-Schutz."""
    Utils.set_seed()
    from models import SimpleLSTM, SimpleCNNLSTM

    # Zu Beginn Testzeitraum entfernen
    print("[CV Datenfilter] Sicherstellen, dass keine Testsamples enthalten sind...")
    mask = target_years <= config.TRAIN_END_YEAR
    if mask.sum() < len(target_years):
        dataset = Subset(dataset, np.where(mask)[0])
        target_years = target_years[mask]
    assert target_years.max() <= config.TRAIN_END_YEAR, \
        f"CV-Daten enthalten Testjahre, max Jahr: {target_years.max()} > {config.TRAIN_END_YEAR}"
    print(f"CV-Jahre: {target_years.min()}–{target_years.max()} ; Stichproben: {len(dataset)}")

    print("=== Starte Vergleich mit Zeitreihen-Cross-Validation ===")

    # Dateninfo
    sample_batch = next(iter(DataLoader(dataset, batch_size=1)))
    X_sample = sample_batch[0]
    input_size = X_sample.shape[-1]

    print(f"Anzahl Eingangsmerkmale: {input_size}")
    print(f"Fensterlänge: {X_sample.shape[1]}")
    print(f"Gesamtzahl Stichproben: {len(dataset)}")

    # CV-Splits vorbereiten
    X_windows = torch.stack([dataset[i][0] for i in range(len(dataset))]).numpy()
    y_windows = torch.stack([dataset[i][1] for i in range(len(dataset))]).numpy().squeeze()

    # LSTM
    print("\n=== Zeitreihen-CV für LSTM ===")
    lstm_cv_results = run_time_series_cv(
        dataset, X_windows, y_windows, target_years,
        SimpleLSTM, input_size, n_splits=n_splits, batch_size=config.BATCH_SIZE,
        processor=processor
    )

    # CNN-LSTM
    print("\n=== Zeitreihen-CV für CNN-LSTM ===")
    cnn_lstm_cv_results = run_time_series_cv(
        dataset, X_windows, y_windows, target_years,
        SimpleCNNLSTM, input_size, n_splits=n_splits, batch_size=config.BATCH_SIZE,
        processor=processor
    )

    # Analyse
    lstm_analysis = analyze_cv_results(lstm_cv_results, "LSTM")
    cnn_lstm_analysis = analyze_cv_results(cnn_lstm_cv_results, "CNN-LSTM")

    # Visualisierung auf Deutsch
    plot_cv_comparison(lstm_cv_results, cnn_lstm_cv_results, lang="de")

    # Speichern
    save_cv_results_to_csv(lstm_cv_results, "LSTM")
    save_cv_results_to_csv(cnn_lstm_cv_results, "CNN-LSTM")

    print("\n=== Cross-Validation abgeschlossen (keine Test-Leckage) ===")
    print(f"LSTM Mittelwert Validierung: Loss={lstm_analysis['summary']['mean_val_loss']:.4f}±{lstm_analysis['summary']['std_val_loss']:.4f}")
    print(f"CNN-LSTM Mittelwert Validierung: Loss={cnn_lstm_analysis['summary']['mean_val_loss']:.4f}±{cnn_lstm_analysis['summary']['std_val_loss']:.4f}")

    return None, lstm_analysis, cnn_lstm_analysis


def analyze_cv_results(cv_results, model_name="Model"):
    """Analysiert die Cross-Validation-Ergebnisse"""
    print(f"\n=== {model_name} Ergebnisse der Zeitreihen-Cross-Validation ===")

    # Metriken sammeln
    metrics_names = ['loss', 'mae', 'rmse', 'mape']
    val_metrics = {metric: [result['best_val_metrics'][metric] for result in cv_results]
                   for metric in metrics_names}
    train_metrics = {metric: [result['final_train_metrics'][metric] for result in cv_results]
                     for metric in metrics_names}

    # Mittelwert und Standardabweichung
    print("Validierungsmetriken (Mittelwert ± Standardabweichung):")
    for metric in metrics_names:
        mean_val = np.mean(val_metrics[metric])
        std_val = np.std(val_metrics[metric])
        unit = "%" if metric == 'mape' else ""
        print(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}{unit}")

    print("\nTrainingsmetriken (Mittelwert ± Standardabweichung):")
    for metric in metrics_names:
        mean_train = np.mean(train_metrics[metric])
        std_train = np.std(train_metrics[metric])
        unit = "%" if metric == 'mape' else ""
        print(f"  {metric.upper()}: {mean_train:.4f} ± {std_train:.4f}{unit}")

    # Overfitting-Verhältnis
    print(f"\nOverfitting-Verhältnis (Validierung/Training):")
    overfitting_ratios = {}
    for metric in metrics_names:
        ratio = np.mean(val_metrics[metric]) / np.mean(train_metrics[metric])
        overfitting_ratios[metric] = ratio
        print(f"  {metric.upper()}: {ratio:.2f}")

    # Details je Fold
    print("\nFold-Details:")
    for result in cv_results:
        metrics = result['best_val_metrics']
        print(f"Fold {result['fold']}: {result['train_years']} -> {result['val_years']}")
        print(f"  Loss: {metrics['loss']:.4f}, MAE: {metrics['mae']:.4f}, "
              f"RMSE: {metrics['rmse']:.4f}, MAPE: {metrics['mape']:.2f}%")

    return {
        'val_metrics': val_metrics,
        'train_metrics': train_metrics,
        'overfitting_ratios': overfitting_ratios,
        'summary': {
            'mean_val_loss': np.mean(val_metrics['loss']),
            'std_val_loss': np.std(val_metrics['loss']),
            'mean_val_mae': np.mean(val_metrics['mae']),
            'mean_val_rmse': np.mean(val_metrics['rmse']),
            'mean_val_mape': np.mean(val_metrics['mape']),
        }
    }


def plot_cv_results(cv_results, model_name="Model", save_path=None):
    """Visualisiert Cross-Validation-Ergebnisse mit deutschen Beschriftungen"""
    n_folds = len(cv_results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Verlustkurven je Fold
    ax1 = axes[0, 0]
    for i, result in enumerate(cv_results):
        epochs = range(1, len(result['train_history']['loss']) + 1)
        ax1.plot(epochs, result['train_history']['loss'], '--', alpha=0.7, label=f'Fold {i + 1} Training')
        ax1.plot(epochs, result['val_history']['loss'], '-', alpha=0.7, label=f'Fold {i + 1} Validierung')
    ax1.set_xlabel('Epoche')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} Trainingskurven nach Fold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 2. Endmetriken im Vergleich
    ax2 = axes[0, 1]
    folds = [f"Fold {r['fold']}" for r in cv_results]
    metrics_to_plot = ['loss', 'mae', 'rmse']

    x = np.arange(len(folds))
    width = 0.25

    for i, metric in enumerate(metrics_to_plot):
        train_values = [r['final_train_metrics'][metric] for r in cv_results]
        val_values = [r['best_val_metrics'][metric] for r in cv_results]

        ax2.bar(x - width + i*width, train_values, width, label=f'Training {metric.upper()}', alpha=0.7)
        ax2.bar(x + i*width, val_values, width, label=f'Validierung {metric.upper()}', alpha=0.7)

    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Metrikwert')
    ax2.set_title(f'{model_name} Metriken nach Fold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(folds)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. MAPE separat
    ax3 = axes[0, 2]
    train_mape = [r['final_train_metrics']['mape'] for r in cv_results]
    val_mape = [r['best_val_metrics']['mape'] for r in cv_results]

    ax3.bar(x - width/2, train_mape, width, label='Training MAPE', alpha=0.7)
    ax3.bar(x + width/2, val_mape, width, label='Validierung MAPE', alpha=0.7)
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('MAPE (%)')
    ax3.set_title(f'{model_name} MAPE nach Fold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(folds)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Zeitliche Verteilung
    ax4 = axes[1, 0]
    for i, result in enumerate(cv_results):
        train_start, train_end = map(int, result['train_years'].split('-'))
        val_start, val_end = map(int, result['val_years'].split('-'))

        ax4.barh(i, train_end - train_start + 1, left=train_start,
                 alpha=0.7, label='Training' if i == 0 else "")
        ax4.barh(i, val_end - val_start + 1, left=val_start,
                 alpha=0.7, label='Validierung' if i == 0 else "")

    ax4.set_xlabel('Jahr')
    ax4.set_ylabel('Fold')
    ax4.set_title(f'{model_name} Zeitsplit-Visualisierung')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Boxplots der Validierungsmetriken
    ax5 = axes[1, 1]
    val_losses = [r['best_val_metrics']['loss'] for r in cv_results]
    val_maes = [r['best_val_metrics']['mae'] for r in cv_results]
    val_rmses = [r['best_val_metrics']['rmse'] for r in cv_results]

    ax5.boxplot([val_losses, val_maes, val_rmses], labels=['Loss', 'MAE', 'RMSE'])
    ax5.set_ylabel('Metrikwert')
    ax5.set_title(f'{model_name} Verteilung der Validierungsmetriken')
    ax5.grid(True, alpha=0.3)

    # 6. Overfitting-Analyse
    ax6 = axes[1, 2]
    metric_names = ['Loss', 'MAE', 'RMSE', 'MAPE']
    overfitting_ratios = []

    for metric in ['loss', 'mae', 'rmse', 'mape']:
        ratios = [r['best_val_metrics'][metric] / r['final_train_metrics'][metric] for r in cv_results]
        overfitting_ratios.append(np.mean(ratios))

    bars = ax6.bar(metric_names, overfitting_ratios, alpha=0.7)
    ax6.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Perfekte Anpassung')
    ax6.set_ylabel('Overfitting-Verhältnis (Val/Train)')
    ax6.set_title(f'{model_name} Overfitting-Analyse')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Zahlenlabels für die Verhältnisse
    for bar, ratio in zip(bars, overfitting_ratios):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{ratio:.2f}', ha='center', va='bottom')

    # Hinweis auf geschätztes R²
    try:
        avg_val_loss = np.mean([r['best_val_metrics']['loss'] for r in cv_results])
        r2_estimate = max(0, 1 - avg_val_loss)

        fig.text(0.02, 0.02, f'Geschätztes R² ≈ {r2_estimate:.3f}',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    except:
        pass

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CV-Grafik gespeichert unter: {save_path}")

    plt.show()


def save_cv_results_to_csv(cv_results, model_name, save_path="exports/cv_results.csv"):
    """Speichert CV-Ergebnisse als CSV"""
    results_data = []

    for result in cv_results:
        val_metrics = result['best_val_metrics']
        train_metrics = result['final_train_metrics']

        row = {
            'model': model_name,
            'fold': result['fold'],
            'train_years': result['train_years'],
            'val_years': result['val_years'],
            'train_size': result['train_size'],
            'val_size': result['val_size'],
            'val_loss': val_metrics['loss'],
            'val_mae': val_metrics['mae'],
            'val_rmse': val_metrics['rmse'],
            'val_mape': val_metrics['mape'],
            'train_loss': train_metrics['loss'],
            'train_mae': train_metrics['mae'],
            'train_rmse': train_metrics['rmse'],
            'train_mape': train_metrics['mape'],
            'overfitting_ratio_loss': val_metrics['loss'] / train_metrics['loss'],
            'overfitting_ratio_mae': val_metrics['mae'] / train_metrics['mae'],
            'overfitting_ratio_rmse': val_metrics['rmse'] / train_metrics['rmse'],
            'overfitting_ratio_mape': val_metrics['mape'] / train_metrics['mape'],
        }
        results_data.append(row)

    # DataFrame und Speichern
    df = pd.DataFrame(results_data)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Falls vorhanden: anhängen
    if Path(save_path).exists():
        existing_df = pd.read_csv(save_path)
        df = pd.concat([existing_df, df], ignore_index=True)

    df.to_csv(save_path, index=False)
    print(f"CV-Ergebnisse gespeichert unter: {save_path}")

    return df


def plot_cv_comparison(
        lstm_results,
        cnn_lstm_results,
        save_path="exports/figures/cv_comparison.png",
        lang="de"
):
    """
    Vergleicht die CV-Ergebnisse zweier Modelle und exportiert GENAU vier eigenständige Abbildungen:
      1) Loss je Fold
      2) MAE, RMSE und MAPE als Dreifachgrafik
      3) Trainingskurven des letzten Folds
      4) Vergleich der Durchschnittswerte mit R²-Hinweis
    (Alle anderen Plots wurden entfernt.)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from utils import Utils

    txt = {
        "fold": "Fold",
        "val_loss": "Validierungs-Loss",
        "loss_cmp": "Loss-Vergleich",
        "val_mae": "Validierungs-MAE",
        "mae_cmp": "MAE-Vergleich",
        "val_rmse": "Validierungs-RMSE",
        "rmse_cmp": "RMSE-Vergleich",
        "val_mape": "Validierungs-MAPE (%)",
        "mape_cmp": "MAPE-Vergleich",
        "epoch": "Epoche",
        "train_curve": "Trainingskurven (letzter Fold)",
        "metrics": "Metriken",
        "avg_cmp": "Durchschnittlicher Leistungsvergleich",
        "avg_value": "Durchschnitt",
        "legend_lstm": "LSTM",
        "legend_cnn": "CNN-LSTM",
    }

    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
    })

    # Metriken extrahieren
    lstm_val_losses = [r['best_val_metrics']['loss'] for r in lstm_results]
    cnn_val_losses  = [r['best_val_metrics']['loss'] for r in cnn_lstm_results]
    lstm_val_mae    = [r['best_val_metrics']['mae']  for r in lstm_results]
    cnn_val_mae     = [r['best_val_metrics']['mae']  for r in cnn_lstm_results]
    lstm_val_rmse   = [r['best_val_metrics']['rmse'] for r in lstm_results]
    cnn_val_rmse    = [r['best_val_metrics']['rmse'] for r in cnn_lstm_results]
    lstm_val_mape   = [r['best_val_metrics']['mape'] for r in lstm_results]
    cnn_val_mape    = [r['best_val_metrics']['mape'] for r in cnn_lstm_results]

    folds = np.arange(len(lstm_results))
    width = 0.35

    # Ausgabeordner
    base = Path(save_path)
    out_dir = base.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Loss je Fold
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.bar(folds - width/2, lstm_val_losses, width, label=txt["legend_lstm"], alpha=0.7)
    ax1.bar(folds + width/2, cnn_val_losses,  width, label=txt["legend_cnn"], alpha=0.7)
    ax1.set_xlabel(txt["fold"]); ax1.set_ylabel(txt["val_loss"])
    ax1.set_title(txt["loss_cmp"])
    ax1.set_xticks(folds); ax1.set_xticklabels([str(i) for i in folds])
    ax1.legend(); ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    Utils.save_figure(out_dir / "cv_loss_only", fig1)

    # 2) Dreifachgrafik MAE/RMSE/MAPE
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    triples = [
        (axes2[0], lstm_val_mae,  cnn_val_mae,  txt["val_mae"],  txt["mae_cmp"]),
        (axes2[1], lstm_val_rmse, cnn_val_rmse, txt["val_rmse"], txt["rmse_cmp"]),
        (axes2[2], lstm_val_mape, cnn_val_mape, txt["val_mape"], txt["mape_cmp"]),
    ]
    for ax, l_data, c_data, ylabel, title in triples:
        ax.bar(folds - width/2, l_data, width, label=txt["legend_lstm"], alpha=0.7)
        ax.bar(folds + width/2, c_data, width, label=txt["legend_cnn"], alpha=0.7)
        ax.set_xlabel(txt["fold"]); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.set_xticks(folds); ax.set_xticklabels([str(i) for i in folds])
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    Utils.save_figure(out_dir / "cv_metrics_triple", fig2)

    # 3) Validierungsverlustkurve (letzter Fold)
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    lstm_last = lstm_results[-1]
    cnn_last  = cnn_lstm_results[-1]
    ax3.plot(range(len(lstm_last['val_history']['loss'])),
             lstm_last['val_history']['loss'], label=f"{txt['legend_lstm']} Val", linewidth=2)
    ax3.plot(range(len(cnn_last['val_history']['loss'])),
             cnn_last['val_history']['loss'], label=f"{txt['legend_cnn']} Val", linewidth=2)
    ax3.set_xlabel(txt["epoch"]); ax3.set_ylabel(txt["val_loss"])
    ax3.set_title("Trainingskurven (letzter Fold)")
    ax3.legend(); ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    Utils.save_figure(out_dir / "cv_epoch_curve", fig3)

    # 4) Durchschnittliche Metriken und R²
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    metrics_names = ["Loss", "MAE", "RMSE", "MAPE"]
    lstm_means = [np.mean(lstm_val_losses), np.mean(lstm_val_mae),
                  np.mean(lstm_val_rmse), np.mean(lstm_val_mape)]
    cnn_means  = [np.mean(cnn_val_losses),  np.mean(cnn_val_mae),
                  np.mean(cnn_val_rmse),  np.mean(cnn_val_mape)]
    pos = np.arange(len(metrics_names))
    ax4.bar(pos - width/2, lstm_means, width, label=txt["legend_lstm"], alpha=0.7)
    ax4.bar(pos + width/2, cnn_means,  width, label=txt["legend_cnn"], alpha=0.7)
    ax4.set_xlabel("Metriken"); ax4.set_ylabel("Durchschnitt")
    ax4.set_title("Durchschnittlicher Leistungsvergleich")
    ax4.set_xticks(pos); ax4.set_xticklabels(metrics_names)
    ax4.legend(); ax4.grid(True, alpha=0.3)

    # R²-Hinweis
    try:
        lstm_r2_est = max(0.0, 1.0 - float(np.mean(lstm_val_losses)))
        cnn_r2_est  = max(0.0, 1.0 - float(np.mean(cnn_val_losses)))
        r2_text = ("Geschätztes $R^2$\n"
                   f"LSTM ≈ {lstm_r2_est:.3f}\n"
                   f"CNN-LSTM ≈ {cnn_r2_est:.3f}")
        ax4.text(0.015, 0.985, r2_text, transform=ax4.transAxes,
                 va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.35",
                           fc=(0.88, 1.00, 0.88), ec="0.3", alpha=0.95))
    except Exception as e:
        print(f"R^2-Hinweis konnte nicht hinzugefügt werden: {e}")

    plt.tight_layout()
    Utils.save_figure(out_dir / "cv_avg_comparison", fig4)

    # Standardeinstellungen wiederherstellen
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"Vier Abbildungen exportiert nach: {out_dir.resolve()}")



def run_time_series_cv_with_data_protection(
    dataset,
    target_years,
    n_splits=3,
    processor=None,
    boundary_year=None
):
    """
    Bequemer Einstiegspunkt mit Leckage-Schutz für CV.
    - boundary_year: obere Jahresgrenze (inklusive). Standard ist config.VAL_END_YEAR.
      Alternativ config.TRAIN_END_YEAR, je nachdem bis zu welchem Jahr die CV reichen soll.
    """
    if boundary_year is None:
        boundary_year = getattr(config, 'VAL_END_YEAR', 2022)

    # Vorabfilterung nach boundary_year
    from torch.utils.data import Subset
    mask = target_years <= boundary_year
    if mask.sum() < len(target_years):
        dataset = Subset(dataset, np.where(mask)[0])
        target_years = target_years[mask]

    # Zusätzliche Absicherung: keine Testjahre in der CV
    assert target_years.max() <= boundary_year, \
        f"CV-Daten überschreiten die Grenzjahre, max={target_years.max()} > boundary_year={boundary_year}"

    # Weiterverwendung der Vergleichsroutine mit zusätzlicher Absicherung
    return run_time_series_cv_comparison(dataset, target_years, n_splits=n_splits, processor=processor)
