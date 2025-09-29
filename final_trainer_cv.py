# final_trainer_cv.py
"""
Finaler Trainer (kompakte, optimierte Fassung)
- Öffentliche Schnittstellen und Abhängigkeiten bleiben unverändert
- Doppelte bzw. überflüssige Hilfsfunktionen wurden zusammengeführt
- Auswertung strikt nur auf dem Testset, um Informationsleckagen zu vermeiden
- R²-Streudiagramm zur Visualisierung verfügbar
"""

import os
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Projektabhängigkeiten (unverändert beibehalten)
from utils import Utils
import config

# Versuch, benutzerdefinierten Trainer zu importieren (Name bleibt gleich; Fallback auf lokale Implementierung)
try:
    from final_trainer_cv import FinalModelTrainer  # gleichnamiger Import für externe Kompatibilität
except Exception:
    FinalModelTrainer = None

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False
})


# ========= Allgemeine Hilfsfunktionen (vereint, ohne Duplikate) =========
def _get_device():
    """Einheitliche Geräteauswahl, kompatibel zu config."""
    if hasattr(config, "get_device"):
        try:
            return config.get_device()
        except Exception:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_device(model: torch.nn.Module):
    """Ermittelt das aktuelle Geräteziel des Modells."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _to_col(x: np.ndarray):
    """Formatiert Eingaben als Spalte (N, 1), geeignet für sklearn-Scaler."""
    x = np.asarray(x)
    return x.reshape(-1, 1)


def _unwrap_batch(batch):
    """
    Unterstützt DataLoader-Ausgaben der Form (X, y) oder (X, y, countries).
    Rückgabe: X, y, countries oder None.
    """
    if len(batch) >= 3:
        return batch[0], batch[1], batch[2]
    return batch[0], batch[1], None


def _safe_numpy(t: torch.Tensor):
    """Konvertiert Tensor in NumPy-Array und glättet übliche Zeitreihenformen."""
    arr = t.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.reshape(-1)
    return arr.reshape(-1)


# ========= Robuste Kennzahlen =========
def wape(y_true, y_pred, eps=1e-8) -> float:
    num = np.sum(np.abs(y_pred - y_true))
    den = np.sum(np.abs(y_true)) + eps
    return float(num / den * 100.0)


def smape(y_true, y_pred, eps=1e-8) -> float:
    num = np.abs(y_pred - y_true)
    den = (np.abs(y_true) + np.abs(y_pred)) + eps
    return float(np.mean(2.0 * num / den) * 100.0)


def calculate_real_scale_metrics(targets, predictions):
    """
    Kennzahlen im Originalmaßstab berechnen, inkl. MAPE, WAPE, sMAPE und R².
    Bei MAPE werden Zielwerte nahe 0 maskiert, um numerische Ausreißer zu vermeiden.
    """
    y = np.asarray(targets).reshape(-1)
    yhat = np.asarray(predictions).reshape(-1)

    mae = float(np.mean(np.abs(yhat - y)))
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))

    eps = 1e-8
    mask = np.abs(y) > eps
    mape = float(np.mean(np.abs((yhat[mask] - y[mask]) / y[mask])) * 100.0) if np.any(mask) else float("inf")

    wape_val = wape(y, yhat, eps=eps)
    smape_val = smape(y, yhat, eps=eps)

    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "wape": wape_val,
        "smape": smape_val,
        "r2": r2,
        "mape_masked_ratio": float(np.mean(mask)) if len(mask) else 0.0,
    }


# ========= Auswertung mit korrekter Rücktransformation =========
def evaluate_model_with_proper_denormalization(model, test_loader, criterion, processor):
    """
    Bewertung ausschließlich auf dem Testset. Reihenfolge der Rücktransformation:
    1) bevorzugt länderspezifische target_scalers, wenn vorhanden
    2) Rückfall auf globale y_scaler, scaler_y oder scaler
    3) falls nichts vorhanden, werden standardisierte Werte verwendet

    Rückgabe:
        {
          "loss": float,
          "standardized_metrics": {...},
          "denormalized_metrics": {...},
          "predictions": np.ndarray,
          "targets": np.ndarray,
          "countries": list[str] oder None
        }
    """
    device = _model_device(model)
    model.eval()

    preds_std, gts_std, countries_all = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            X, y, countries = _unwrap_batch(batch)
            X, y = X.to(device), y.to(device)

            yhat = model(X)
            loss = criterion(yhat, y)
            total_loss += float(loss.item())

            preds_std.extend(_safe_numpy(yhat))
            gts_std.extend(_safe_numpy(y))

            if countries is not None:
                if isinstance(countries, torch.Tensor):
                    countries = countries.cpu().numpy()
                countries_all.extend([str(c) for c in np.ravel(countries)])

    preds_std = np.asarray(preds_std).reshape(-1)
    gts_std = np.asarray(gts_std).reshape(-1)

    # Basiskennzahlen im Standardisierungsraum, nur als Referenz
    std_metrics = Utils.calculate_all_metrics(
        torch.tensor(gts_std), torch.tensor(preds_std)
    )

    # Rücktransformation
    denorm_pred, denorm_true = None, None
    try:
        # länderspezifische Skalierer
        if hasattr(processor, "target_scalers") and processor.target_scalers:
            denorm_pred = processor.inverse_transform_by_country(preds_std, np.asarray(countries_all))
            denorm_true = processor.inverse_transform_by_country(gts_std, np.asarray(countries_all))

        # globale Skalierer
        if denorm_pred is None or denorm_true is None:
            y_scaler = getattr(processor, "y_scaler", None) or \
                       getattr(processor, "scaler_y", None) or \
                       getattr(processor, "scaler", None)
            if y_scaler is not None:
                denorm_pred = y_scaler.inverse_transform(_to_col(preds_std)).ravel()
                denorm_true = y_scaler.inverse_transform(_to_col(gts_std)).ravel()

        # ohne Skalierer
        if denorm_pred is None or denorm_true is None:
            denorm_pred, denorm_true = preds_std.copy(), gts_std.copy()

        # Kennzahlen im Originalmaßstab
        real_metrics = calculate_real_scale_metrics(denorm_true, denorm_pred)

        return {
            "loss": total_loss / max(1, len(test_loader)),
            "standardized_metrics": std_metrics,
            "denormalized_metrics": real_metrics,
            "predictions": denorm_pred,
            "targets": denorm_true,
            "countries": countries_all if len(countries_all) == len(denorm_pred) else None,
        }

    except Exception as e:
        # Fallback auf standardisierte Kennzahlen, falls Rücktransformation scheitert
        print(f"[WARN] Rücktransformation fehlgeschlagen; es werden standardisierte Kennzahlen verwendet: {e}")
        return {
            "loss": total_loss / max(1, len(test_loader)),
            "standardized_metrics": std_metrics,
            "denormalized_metrics": std_metrics,
            "predictions": preds_std,
            "targets": gts_std,
            "countries": countries_all if len(countries_all) == len(preds_std) else None,
        }


# ========= Trainer mit gleichbleibenden Namen/Signaturen =========
class FixedFinalModelTrainer:
    """
    Reparierte und komprimierte Endtrainer-Version.
    Auswahl des besten Modells anhand des CV-RMSE.
    Danach erneutes Training auf train+val (vom Aufrufer bereitgestellt).
    """

    def __init__(self, device):
        self.device = device

    # Signatur und Name unverändert
    def select_best_model(self, lstm_cv_results, cnn_cv_results):
        lstm_rmse = lstm_cv_results.get("summary", {}).get("mean_val_rmse", float("inf"))
        cnn_rmse = cnn_cv_results.get("summary", {}).get("mean_val_rmse", float("inf"))
        return ("LSTM", {"rmse": lstm_rmse}) if lstm_rmse <= cnn_rmse else ("CNN-LSTM", {"rmse": cnn_rmse})

    # Signatur und Name unverändert
    def train_final_model(self, model_name, train_dataset, input_size):
        from models import SimpleLSTM, SimpleCNNLSTM  # externe Abhängigkeiten und Namen unverändert

        if model_name == "LSTM":
            model = SimpleLSTM(
                input_size=input_size,
                hidden_size=getattr(config, "HIDDEN_SIZE", 64),
                num_layers=getattr(config, "NUM_LAYERS", 2),
                dropout=getattr(config, "DROPOUT_RATE", 0.2),
            )
        else:
            model = SimpleCNNLSTM(
                input_size=input_size,
                hidden_size=getattr(config, "HIDDEN_SIZE", 64),
                num_layers=getattr(config, "NUM_LAYERS", 2),
                dropout=getattr(config, "DROPOUT_RATE", 0.2),
                cnn_filters=getattr(config, "CNN_FILTERS", 32),
                kernel_size=getattr(config, "KERNEL_SIZE", 3),
            )

        model = model.to(self.device)

        batch_size = getattr(config, "BATCH_SIZE", 32)
        epochs = getattr(config, "EPOCHS", 150)
        lr = getattr(config, "LEARNING_RATE", 1e-3)
        shuffle_train = getattr(config, "SHUFFLE_TRAIN", True)
        drop_last = getattr(config, "DROP_LAST", False)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=bool(shuffle_train), drop_last=bool(drop_last)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        Utils.set_seed(getattr(config, "RANDOM_SEED", 42))

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                X, y, _ = _unwrap_batch(batch)
                X, y = X.to(self.device), y.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                pred = model(X)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())

            if (epoch + 1) % 10 == 0:
                print(f"[Final Train] Epoche {epoch + 1:03d}  Loss={epoch_loss / max(1, len(train_loader)):.4f}")

        return model

    # Signatur und Name unverändert
    def save_model(self, model, model_name, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "model_name": model_name}, model_path)
        print(f"[Save] Modell gespeichert: {model_path}")


def plot_prediction_scatter_with_r2(predictions, targets, model_name, save_path=None):
    """
    Streudiagramm Vorhersage vs. Istwerte mit R² anzeigen.
    Funktionssignatur bleibt unverändert. Bei save_path werden PNG und PDF abgelegt.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from sklearn.metrics import r2_score
    from utils import Utils

    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
    })

    r2 = r2_score(targets, predictions)

    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.scatter(targets, predictions, alpha=0.6, s=30)

    # Linie perfekter Vorhersage
    min_val = min(np.min(targets), np.min(predictions))
    max_val = max(np.max(targets), np.max(predictions))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Perfekte Vorhersage')

    # R²-Anzeige
    ax.text(
        0.05, 0.95, f"R² = {r2:.3f}",
        transform=ax.transAxes,
        fontsize=16, fontweight='bold',
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9, ec='black')
    )

    ax.set_xlabel('Tatsächliche Werte', fontweight='bold')
    ax.set_ylabel('Vorhergesagte Werte', fontweight='bold')
    ax.set_title(f'{model_name} – Vorhergesagte vs. tatsächliche Werte', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    plt.tight_layout()

    # Export als PNG und PDF
    if save_path:
        base = Path(save_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        stem = base.with_suffix('')

        try:
            Utils.save_figure(stem, fig)
        except Exception:
            fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor='white', edgecolor='none')
            fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor='white', edgecolor='none')

        print(f"[Save] Grafik gespeichert: {stem.with_suffix('.png')} und {stem.with_suffix('.pdf')}")

    plt.show()
    plt.rcParams.update(plt.rcParamsDefault)

    return r2



def run_final_training_fixed(lstm_cv_results, cnn_cv_results, train_val_dataset,
                             test_dataset, input_size, processor):
    """
    Reparierte und optimierte End-Trainingsroutine:
    - Modellauswahl auf Basis der CV-Ergebnisse
    - Erneutes Training auf train+val
    - Bewertung ausschließlich auf dem Testset
    - R²-Streudiagramm als Zusatz
    - Zusätzlich: Ablage der echten Vorhersagen unter artifacts/predictions.csv für spätere Länderanalysen
    """
    import os
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from torch import nn
    from torch.utils.data import DataLoader

    Utils.set_seed(getattr(config, "RANDOM_SEED", 42))
    device = _get_device()

    trainer = FinalModelTrainer(device) if FinalModelTrainer is not None else FixedFinalModelTrainer(device)

    # Modellauswahl
    best_model_name, cv_metrics = trainer.select_best_model(lstm_cv_results, cnn_cv_results)
    print("\n=== Modellauswahl ===")
    print(f"LSTM  CV-RMSE: {lstm_cv_results.get('summary', {}).get('mean_val_rmse', 'N/A')}")
    print(f"CNN-LSTM CV-RMSE: {cnn_cv_results.get('summary', {}).get('mean_val_rmse', 'N/A')}")
    print(f"Gewähltes Modell: {best_model_name}")

    # Finales Training auf train+val
    final_model = trainer.train_final_model(best_model_name, train_val_dataset, input_size)

    # Testbewertung
    batch_size = getattr(config, "BATCH_SIZE", 32)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.MSELoss()

    print("\n=== Endgültige Testbewertung ===")
    print(f"Anzahl Testsamples: {len(test_dataset)}")

    # Bewertung mit korrekter Rücktransformation
    results = evaluate_model_with_proper_denormalization(final_model, test_loader, criterion, processor)

    # Kennzahlen ausgeben
    std_m = results["standardized_metrics"]
    real_m = results["denormalized_metrics"]
    print(f"\n[Standardisierungsraum] Loss={results['loss']:.4f}, "
          f"MAE={std_m.get('mae', 0):.4f}, RMSE={std_m.get('rmse', 0):.4f}, "
          f"MAPE={std_m.get('mape', 0):.2f}%")
    print("[Originalmaßstab]")
    print(f"  MAE : {real_m['mae']:.4f}")
    print(f"  RMSE: {real_m['rmse']:.4f}")
    print(f"  MAPE: {real_m['mape']:.2f}%  (Maskenanteil: {real_m['mape_masked_ratio']:.3f})")
    print(f"  WAPE: {real_m['wape']:.2f}%")
    print(f"  sMAPE: {real_m['smape']:.2f}%")
    print(f"  R²  : {real_m['r2']:.4f}")

    # Vorhersagen für die Länderanalyse speichern
    try:
        preds = np.asarray(results["predictions"]).reshape(-1)
        trues = np.asarray(results["targets"]).reshape(-1)
        countries = results.get("countries")

        if countries is None or len(countries) != len(preds):
            print("[WARN] Länderinformationen fehlen oder sind uneinheitlich. Es wird 'Unknown' verwendet.")
            countries = ["Unknown"] * len(preds)

        df_out = pd.DataFrame({
            "country": countries,
            "y_pred": preds,
            "y_true": trues,
        })

        out_dir = Path("artifacts")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "predictions.csv"
        df_out.to_csv(out_path, index=False)
        print(f"Vorhersagen gespeichert: {out_path.resolve()}  (Zeilen: {len(df_out)})")
    except Exception as e:
        print(f"[WARN] Speichern der Vorhersagen unter artifacts/predictions.csv fehlgeschlagen: {e}")

    # Modell und Bericht speichern
    models_dir = getattr(config, "MODELS_DIR", "exports/models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"final_{best_model_name.lower().replace('-', '_')}.pt")
    trainer.save_model(final_model, best_model_name, model_path)

    # R²-Streudiagramm
    figures_dir = getattr(config, "FIGURES_DIR", "exports/figures")
    scatter_path = os.path.join(figures_dir, f"final_{best_model_name.lower().replace('-', '_')}_scatter_r2.png")
    print(f"\n=== R²-Streudiagramm ===")
    r2_value = plot_prediction_scatter_with_r2(
        predictions=results["predictions"],
        targets=results["targets"],
        model_name=best_model_name,
        save_path=scatter_path
    )
    print(f"Berechnetes R²: {r2_value:.4f}")

    _save_detailed_results(results, best_model_name, models_dir)

    return final_model, results["denormalized_metrics"]



def _save_detailed_results(results, model_name, output_dir):
    """Speichert einen detaillierten Bewertungsbericht. Interne Hilfsfunktion; externer Name bleibt save_detailed_results."""
    try:
        report = {
            "model_name": model_name,
            "test_loss": float(results["loss"]),
            "standardized_metrics": {
                k: float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in results["standardized_metrics"].items()
            },
            "denormalized_metrics": {
                k: float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in results["denormalized_metrics"].items()
            },
            "sample_count": int(len(results["predictions"])),
            "prediction_stats": {
                "min": float(np.min(results["predictions"])),
                "max": float(np.max(results["predictions"])),
                "mean": float(np.mean(results["predictions"])),
                "std": float(np.std(results["predictions"])),
            },
            "target_stats": {
                "min": float(np.min(results["targets"])),
                "max": float(np.max(results["targets"])),
                "mean": float(np.mean(results["targets"])),
                "std": float(np.std(results["targets"])),
            },
        }

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"final_{model_name.lower().replace('-', '_')}_report.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[Save] Bewertungsbericht gespeichert: {path}")
    except Exception as e:
        print(f"[WARN] Speichern des Bewertungsberichts fehlgeschlagen: {e}")


# Kompatibilität: externer Funktionsname bleibt erhalten
def save_detailed_results(results, model_name, output_dir):
    return _save_detailed_results(results, model_name, output_dir)


def run_final_training(lstm_cv_results, cnn_cv_results, train_val_dataset,
                       test_dataset, input_size, processor):
    """
    Kompatibler Einstiegspunkt, Funktionsname und Signatur bleiben unverändert.
    """
    return run_final_training_fixed(lstm_cv_results, cnn_cv_results, train_val_dataset,
                                    test_dataset, input_size, processor)


if __name__ == "__main__":
    print("Finaler Trainer (kompakte, optimierte Fassung) geladen")
    print("Alle externen Schnittstellen und Abhängigkeiten bleiben unverändert.")
    print("Doppelte Hilfsfunktionen entfernt und zusammengeführt.")
    print("Bewertung findet ausschließlich auf dem Testset statt.")
    print("Kennzahlen enthalten WAPE, sMAPE und R², Rücktransformation robust.")
    print("Neu: R²-Streudiagramm mit deutschen Beschriftungen.")
