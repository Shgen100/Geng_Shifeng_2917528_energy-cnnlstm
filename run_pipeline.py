#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py
End-to-End-Runner für das CNN-LSTM-Projekt zur Energieprognose.

Schritte:

CV-fähige Daten vorbereiten (Bereinigung, Skalierung, Sliding-Windows).

Zeitreihen-Cross-Validation: Vergleich LSTM vs. CNN-LSTM.

Klassische Splits vorbereiten (Train+Val vs. Test).

Finales Modell auf Train+Val trainieren und einmalige Bewertung auf dem Testset (ohne Leakage).

Optional: Länderebene-Analyse mithilfe der gespeicherten Vorhersagen.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Project modules (must be available on PYTHONPATH or in the same directory)
from config import config
from utils import Utils
from data_processor_cv import (
    complete_data_processing_for_cv,
    complete_data_processing,
    EnergyDataset,
)
from trainer_cv import run_time_series_cv_with_data_protection
from final_trainer_cv import run_final_training
from country_analysis import run_complete_country_analysis


def _ensure_dirs():
    """Create common output directories if missing."""
    for d in [config.OUTPUT_DIR, getattr(config, "FIGURES_DIR", "exports/figures"),
              getattr(config, "MODELS_DIR", "exports/models"), "artifacts"]:
        Path(d).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end pipeline for energy forecasting (LSTM vs. CNN-LSTM)."
    )
    p.add_argument("--csv", type=str, default=config.CSV_PATH,
                   help="Path to the OWID CSV (default: from config.CSV_PATH).")
    p.add_argument("--cv-folds", type=int, default=getattr(config, "CV_FOLDS", 5),
                   help="Number of CV folds (default: config.CV_FOLDS or 5).")
    p.add_argument("--skip-country-analysis", action="store_true",
                   help="Skip country-level analysis step.")
    p.add_argument("--seed", type=int, default=getattr(config, "RANDOM_SEED", 42),
                   help="Random seed (default: config.RANDOM_SEED or 42).")
    return p.parse_args()


def step1_cv_data(csv_path: str):
    print("\n[1/5] Preparing CV-ready data ...")
    processor_cv, cv_dataset, (X_cv, y_cv, years_cv, countries_cv) = complete_data_processing_for_cv(csv_path)
    print(f"  Windows prepared: X={X_cv.shape}, y={y_cv.shape}, years={years_cv.min()}–{years_cv.max()}")
    return processor_cv, cv_dataset, (X_cv, y_cv, years_cv, countries_cv)


def step2_cv_comparison(cv_dataset, years_cv, processor_cv, n_folds: int):
    print("\n[2/5] Running time-series cross-validation comparison (LSTM vs. CNN-LSTM) ...")
    _, lstm_cv, cnn_cv = run_time_series_cv_with_data_protection(
        dataset=cv_dataset,
        target_years=years_cv,
        n_splits=n_folds,
        processor=processor_cv
    )
    print("  CV comparison finished.")
    return lstm_cv, cnn_cv


def step3_classic_splits(csv_path: str):
    print("\n[3/5] Preparing classic splits (train+val vs. test) ...")
    processor, train_loader, val_loader, test_loader, (X_all, y_all, years_all, countries_all) = complete_data_processing(csv_path)
    print(f"  All windows: X={X_all.shape}, y={y_all.shape}")
    return processor, (X_all, y_all, years_all, countries_all)


def step4_final_training(lstm_cv, cnn_cv, arrays, processor):
    print("\n[4/5] Final training on train+val and one-shot evaluation on test ...")
    X_all, y_all, years_all, countries_all = arrays

    # Build datasets for final training
    mask_trainval = years_all <= config.VAL_END_YEAR
    mask_test = years_all > config.VAL_END_YEAR

    if not np.any(mask_test):
        raise RuntimeError("No test samples found (years > VAL_END_YEAR). Please check your config year boundaries.")

    trainval_dataset = EnergyDataset(X_all[mask_trainval], y_all[mask_trainval], countries_all[mask_trainval])
    test_dataset = EnergyDataset(X_all[mask_test], y_all[mask_test], countries_all[mask_test])

    input_size = X_all.shape[-1]
    final_model, test_metrics = run_final_training(
        lstm_cv, cnn_cv, trainval_dataset, test_dataset, input_size, processor
    )

    print("  Final model trained and evaluated on test.")
    print("  Test metrics (original scale):")
    for k, v in test_metrics.items():
        try:
            print(f"    {k}: {float(v):.6f}")
        except Exception:
            print(f"    {k}: {v}")

    return final_model


def step5_country_analysis(skip: bool):
    print("\n[5/5] Country-level analysis ...")
    if skip:
        print("  Skipped by user option.")
        return

    pred_path = Path("artifacts") / "predictions.csv"
    if not pred_path.exists():
        print("  No 'artifacts/predictions.csv' found. Skipping analysis.")
        return

    preds = pd.read_csv(pred_path)
    test_details = {
        "countries": preds["country"].tolist(),
        "predictions": preds["y_pred"].tolist(),
        "targets": preds["y_true"].tolist(),
    }

    # Use config.CSV_PATH to avoid hard-coded OS-specific paths
    run_complete_country_analysis(
        test_details=test_details,
        model_name="Final Model",
        csv_path=config.CSV_PATH,
        save_dir=config.OUTPUT_DIR
    )
    print("  Country analysis finished.")


def main():
    args = parse_args()
    Utils.set_seed(args.seed)
    _ensure_dirs()

    # Optional: ensure CSV path from CLI is used by config if desired. We don't override config.CSV_PATH here,
    # we pass the path explicitly to the data-processing steps to avoid changing user modules.
    csv_path = args.csv

    # Step 1: CV data
    processor_cv, cv_dataset, (_, _, years_cv, _) = step1_cv_data(csv_path)

    # Step 2: CV comparison
    lstm_cv, cnn_cv = step2_cv_comparison(cv_dataset, years_cv, processor_cv, args.cv_folds)

    # Step 3: classic splits for final training & test
    processor, arrays = step3_classic_splits(csv_path)

    # Step 4: final training + test (use the processor from classic pipeline for denormalization)
    final_model = step4_final_training(lstm_cv, cnn_cv, arrays, processor)

    # Step 5: country-level analysis (optional)
    step5_country_analysis(skip=args.skip_country_analysis)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print("\n[ERROR] Pipeline failed: ", str(e))
        traceback.print_exc()
        sys.exit(1)
