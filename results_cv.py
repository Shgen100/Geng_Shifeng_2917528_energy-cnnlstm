"""
Modul zur Verwaltung von Cross-Validation-Ergebnissen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import Utils


def save_cv_results(cv_results, model_name, save_dir="exports"):
    """Speichert die CV-Ergebnisse eines einzelnen Modells als CSV"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    for r in cv_results:
        v = r['best_val_metrics']
        t = r['final_train_metrics']

        # Kompatibilität: sowohl 'loss' als auch 'mse' unterstützen
        v_loss = v.get('loss') or v.get('mse', 0)
        t_loss = t.get('loss') or t.get('mse', 0)

        rows.append({
            'Model': model_name,
            'Fold': r['fold'],
            'Train_Years': r['train_years'],
            'Val_Years': r['val_years'],
            'Train_Size': r['train_size'],
            'Val_Size': r['val_size'],
            # Validierungsmetriken
            'Val_Loss': v_loss,
            'Val_MAE': v['mae'],
            'Val_RMSE': v['rmse'],
            'Val_MAPE': v['mape'],
            # Trainingsmetriken
            'Train_Loss': t_loss,
            'Train_MAE': t['mae'],
            'Train_RMSE': t['rmse'],
            'Train_MAPE': t['mape'],
            # Overfitting-Verhältnisse
            'Overfit_Loss': v_loss/t_loss if t_loss else np.nan,
            'Overfit_MAE': v['mae']/t['mae'] if t['mae'] else np.nan,
            'Overfit_RMSE': v['rmse']/t['rmse'] if t['rmse'] else np.nan,
            'Overfit_MAPE': v['mape']/t['mape'] if t['mape'] else np.nan,
        })

    df = pd.DataFrame(rows)
    csv_path = Path(save_dir) / f"{model_name}_cv_results.csv"

    # Falls vorhanden: zusammenführen und Duplikate entfernen
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        df = pd.concat([old, df], ignore_index=True)

    # Duplikate nach Model, Fold und Zeiträumen entfernen
    df = df.drop_duplicates(subset=['Model', 'Fold', 'Train_Years', 'Val_Years'], keep='last')
    df.to_csv(csv_path, index=False)
    print(f"CV-Ergebnisse wurden gespeichert unter: {csv_path}")
    return df


def print_cv_summary(cv_results, model_name):
    """Gibt eine kompakte Zusammenfassung der CV-Ergebnisse aus"""
    print(f"\n=== Zusammenfassung der Cross-Validation für {model_name} ===")

    # Metriken extrahieren, kompatibel mit 'loss' bzw. 'mse'
    metrics = {}
    for metric in ['mae', 'rmse', 'mape']:
        metrics[metric] = [r['best_val_metrics'][metric] for r in cv_results]

    loss_values = []
    for r in cv_results:
        v = r['best_val_metrics']
        loss_values.append(v.get('loss') or v.get('mse', 0))
    metrics['loss'] = loss_values

    print("Validierungsmetriken (Mittelwert ± Standardabweichung):")
    print(f"  Loss:  {np.mean(metrics['loss']):.4f} ± {np.std(metrics['loss']):.4f}")
    print(f"  MAE:   {np.mean(metrics['mae']):.4f} ± {np.std(metrics['mae']):.4f}")
    print(f"  RMSE:  {np.mean(metrics['rmse']):.4f} ± {np.std(metrics['rmse']):.4f}")
    print(f"  MAPE:  {np.mean(metrics['mape']):.2f}% ± {np.std(metrics['mape']):.2f}%")

    print("\nErgebnisse je Fold:")
    for i, r in enumerate(cv_results, 1):
        m = r['best_val_metrics']
        loss_val = m.get('loss') or m.get('mse', 0)
        print(f"  Fold {i}: Loss={loss_val:.4f}, MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}, MAPE={m['mape']:.2f}%")


def plot_cv_metrics(cv_results, model_name, save_path=None):
    """Visualisiert die CV-Metriken eines einzelnen Modells"""
    Utils.setup_plotting_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    folds = [f"Fold {r['fold']}" for r in cv_results]

    # Metriken sammeln
    metrics_data = {}
    for metric in ['mae', 'rmse', 'mape']:
        metrics_data[metric] = [r['best_val_metrics'][metric] for r in cv_results]

    loss_data = []
    for r in cv_results:
        v = r['best_val_metrics']
        loss_data.append(v.get('loss') or v.get('mse', 0))
    metrics_data['loss'] = loss_data

    # Plotkonfiguration
    plot_configs = [
        (axes[0, 0], 'loss', 'Val-Verlust', 'Verlust'),
        (axes[0, 1], 'mae', 'Val-MAE', 'MAE'),
        (axes[1, 0], 'rmse', 'Val-RMSE', 'RMSE'),
        (axes[1, 1], 'mape', 'Val-MAPE', 'MAPE (%)')
    ]

    for ax, metric, title, ylabel in plot_configs:
        data = metrics_data[metric]
        bars = ax.bar(folds, data, alpha=0.8)
        ax.set_title(f'{model_name} - {title}')
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=45)

        # Zahlenbeschriftungen
        for b, v in zip(bars, data):
            txt = f"{v:.2f}%" if metric == 'mape' else f"{v:.3f}"
            ax.text(b.get_x() + b.get_width() / 2.0, b.get_height(), txt,
                    ha='center', va='bottom', fontsize=9)

    # Zusammenfassung am unteren Rand
    desc = (f"Loss {np.mean(metrics_data['loss']):.3f}±{np.std(metrics_data['loss']):.3f} | "
            f"MAE {np.mean(metrics_data['mae']):.3f}±{np.std(metrics_data['mae']):.3f} | "
            f"RMSE {np.mean(metrics_data['rmse']):.3f}±{np.std(metrics_data['rmse']):.3f} | "
            f"MAPE {np.mean(metrics_data['mape']):.2f}±{np.std(metrics_data['mape']):.2f}%")
    fig.text(0.5, 0.01, desc, ha='center', fontsize=11)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    if save_path:
        Utils.save_figure(save_path, fig)
        print(f"CV-Metrikdiagramm wurde gespeichert unter: {save_path}")
    plt.show()


def compare_models_cv(results_dict, save_path=None):
    """Vergleicht die CV-Ergebnisse mehrerer Modelle"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    model_names = list(results_dict.keys())

    plot_configs = [
        ('loss', 'Loss'),
        ('mae', 'MAE'),
        ('rmse', 'RMSE'),
        ('mape', 'MAPE (%)')
    ]

    for i, (metric, title) in enumerate(plot_configs):
        ax = axes[i // 2, i % 2]
        means, stds = [], []

        for name in model_names:
            # Kompatibel zu 'loss' bzw. 'mse'
            if metric == 'loss':
                vals = []
                for r in results_dict[name]:
                    v = r['best_val_metrics']
                    vals.append(v.get('loss') or v.get('mse', 0))
            else:
                vals = [r['best_val_metrics'][metric] for r in results_dict[name]]

            means.append(np.mean(vals))
            stds.append(np.std(vals))

        bars = ax.bar(model_names, means, yerr=stds, alpha=0.7, capsize=5)
        ax.set_title(f'CV-Vergleich - {title}')
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)

        # Zahlenbeschriftungen
        for b, m, s in zip(bars, means, stds):
            ax.text(b.get_x() + b.get_width() / 2.0, b.get_height() + s + 0.01,
                    f'{m:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Modellvergleichsdiagramm wurde gespeichert unter: {save_path}")
    plt.show()


def load_cv_results(csv_path):
    """Lädt zuvor gespeicherte CV-Ergebnisse"""
    p = Path(csv_path)
    if not p.exists():
        print(f"Datei nicht gefunden: {csv_path}")
        return None
    df = pd.read_csv(p)
    print(f"CV-Ergebnisse geladen: {df.shape[0]} Einträge")
    return df


def export_cv_summary(results_dict, save_path="exports/cv_summary.csv"):
    """Exportiert eine Zusammenfassung der CV-Ergebnisse"""
    rows = []
    for name, res in results_dict.items():
        # Metriken extrahieren
        metrics_data = {}
        for metric in ['mae', 'rmse', 'mape']:
            metrics_data[metric] = [r['best_val_metrics'][metric] for r in res]

        loss_data = []
        for r in res:
            v = r['best_val_metrics']
            loss_data.append(v.get('loss') or v.get('mse', 0))
        metrics_data['loss'] = loss_data

        rows.append({
            'Model': name,
            'N_Folds': len(res),
            'Avg_Loss': np.mean(metrics_data['loss']),
            'Std_Loss': np.std(metrics_data['loss']),
            'Avg_MAE': np.mean(metrics_data['mae']),
            'Std_MAE': np.std(metrics_data['mae']),
            'Avg_RMSE': np.mean(metrics_data['rmse']),
            'Std_RMSE': np.std(metrics_data['rmse']),
            'Avg_MAPE': np.mean(metrics_data['mape']),
            'Std_MAPE': np.std(metrics_data['mape']),
        })

    df = pd.DataFrame(rows)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"CV-Zusammenfassung wurde gespeichert unter: {save_path}")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    print("Modul zur Verwaltung der CV-Ergebnisse geladen")
