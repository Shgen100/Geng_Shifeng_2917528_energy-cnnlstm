# README

## Projektüberblick

Dieses Repository enthält den Code für ein Projekt zur Prognose des weltweiten Energieverbrauchs mit einem CNN LSTM. Die Pipeline umfasst Datenaufbereitung, zeitliche Cross Validation, Modellvergleich zwischen LSTM und CNN LSTM, finales Training sowie optionale Länderanalysen.
---
## Datengrundlage

* Quelle: Our World in Data, Zeitreihen 2000 bis 2024
---
## Ordnerstruktur
```
.
├── artifacts/               # Zwischenergebnisse und Vorhersagedateien
│   └── predictions.csv
├── exports/                 # Alle Ausgabeergebnisse
│   ├── figures/            # Generierte Abbildungen (PNG/PDF)
│   ├── models/             # Trainierte Modellgewichte
│   ├── bad_performance_countries.csv
│   ├── by_country_metrics.csv
│   ├── cv_results.csv
│   ├── gdp_top10_metrics.csv
│   └── good_performance_countries.csv
├── owid-energy_consumption_2000_2024.csv  # Rohdatendatei
├── run_pipeline.py          # Hauptausführungsskript der Pipeline
├── config.py               # Zentrale Konfigurationsdatei
├── models.py               # LSTM- und CNN-LSTM-Modelldefinitionen
├── utils.py                # Hilfsfunktionen und Visualisierung
├── artifacts_utils.py      # Funktionen zum Speichern und Laden
├── data_processor_cv.py    # Datenverarbeitung und Cross-Validation-Vorbereitung
├── trainer_cv.py           # Zeitreihen Cross-Validation Training
├── final_trainer_cv.py     # Finales Modelltraining und Test
├── country_analysis.py     # Analysen auf Länderebene
├── country_specific_trainer.py  # Optionales länderspezifisches Training
├── results_cv.py           # Zusammenfassung der Cross-Validation-Ergebnisse
└── README.md
```
---
## Wichtige Module kurz erklärt

* `config.py`
  Verzeichnisse, Zufallskeime, Hyperparameter, Fenstereinstellungen.
* `data_processor_cv.py`
  Bereinigung, Imputation, Skalierung, Konstruktion der Gleitfenster, Erstellung von CV Splits und klassischen Splits.
* `models.py`
  Definition von LSTM und CNN LSTM.
* `trainer_cv.py`
  Training und Evaluierung über zeitliche Folds, Erzeugung der CV Abbildungen.
* `final_trainer_cv.py`
  Training auf Train plus Val und einmalige Bewertung auf dem Testsegment ohne Informationsleckagen, R Quadrat Scatter Plot.
* `country_analysis.py`
  Analysen mit gespeicherten Vorhersagen, etwa GDP gegen RMSE und Gruppenvergleich.
* `artifacts_utils.py`, `utils.py`
  Speichern und Laden von Artefakten, Plots, Helfer für Pfade und Logs.
* `country_specific_trainer.py`
  Optionale Routinen für einzelnes Land mit angepasstem Training.
* `results_cv.py`
  Zusammenfassung und Export der Cross Validation Ergebnisse.
---
## Systemvoraussetzungen

* Python ab 3.10
* Empfohlene Pakete: numpy, pandas, scikit-learn, matplotlib, pytorch, tqdm
---
## Konfiguration

Alle zentralen Einstellungen befinden sich in `config.py`.
Wichtig sind unter anderem

* Pfade für Daten und Exporte
* Fensterlänge und Horizont
* Anzahl der Folds
* Trainingsparameter wie Epochen und Batch Größe

Änderungen in `config.py` greifen ohne weitere Anpassungen in den Trainingsskripten.
---
## Ausführung

### 1.End to End

Sobald `run_pipeline.py` vorhanden ist, kann die komplette Pipeline gestartet werden:

```
python run_pipeline.py
```

Der Runner führt folgende Schritte aus

1. Daten für die Cross Validation vorbereiten
2. Cross Validation mit LSTM und CNN LSTM
3. Klassische Splits erzeugen
4. Finales Training auf Train plus Val und einmalige Bewertung auf Test
5. Optional Länderanalyse mit gespeicherten Vorhersagen

### 2.Einzelschritte

Die Module können auch separat ausgeführt werden. Typischer Ablauf:

1. Datenaufbereitung

   ```
   python -c "import data_processor_cv as m; m.prepare_cv_data()"
   ```
2. Cross Validation

   ```
   python -c "import trainer_cv as m; m.run_time_series_cv()"
   ```
3. Finales Training und Testbewertung

   ```
   python -c 'import final_trainer_cv as m; m.train_and_eval_final()'
   ```
4. Länderanalyse optional

   ```
   python -c 'import country_analysis as m; m.run_all_country_analyses()'
   ```

---
## Ausgaben und Abbildungen

Ergebnisse werden in `exports/` gespeichert.

* Modelle und Metriken in `exports/models` und `exports/predictions`
* Abbildungen in `exports/figures`

Die Pipeline erzeugt genau sieben Abbildungen

1. Trainingskurven letzter Fold
2. Durchschnittlicher Leistungsvergleich mit R Quadrat Hinweis
3. R Quadrat Scatter auf Testdaten des finalen Modells
4. GDP gegen RMSE für die Top zehn Länder nach Wirtschaftsleistung
5. Durchschnittsvergleich guter und schwacher Ländergruppen
6. Loss Vergleich je Fold
7. Dreifachgrafik MAE RMSE MAPE je Fold
---
## Reproduzierbarkeit

* Setzen Sie feste Zufallskeime in `config.py`.
* Dokumentieren Sie die verwendete Version der Trainingsdaten.
* Bewahren Sie `exports/models` und `exports/predictions` für die Nachprüfung auf.
---
## Hinweise

* Die Skripte verändern keine Funktionsnamen, Klassennamen oder Variablennamen innerhalb der Module.
* Es findet keine Überschreibung der Rohdaten statt.
* Informationsleckagen werden vermieden. Das finale Modell sieht die Testjahre erst nach Abschluss der Modellwahl.
---
## Final-Training (automatische Modellwahl) + Testbewertung
```
from final_trainer_cv import run_final_training
```
## Nutzung der zuvor berechneten CV-Resultate:
```
final_model, test_metrics, test_details = run_final_training(
    lstm_cv, cnn_cv, train_val_dataset, test_dataset, input_size, processor
)

from country_analysis import run_complete_country_analysis
```
## test_details stammt aus der finalen Testbewertung (Preds/Targets/Countries)
```
country_metrics, gdp_data, top10_metrics, good_c, bad_c = run_complete_country_analysis(
    test_details,
    model_name="Final Model",
    csv_path="owid-energy_consumption_2000_2024.csv",
    save_dir="exports"
)
```
> **Outputs** (standardmäßig in `exports/`): Figuren (CV‑Vergleiche, R²‑Scatter, GDP‑Top‑10‑Analysen), CSV‑Zusammenfassungen, JSON‑Reports, gespeicherte Modelle.

---

## 6) wichtige Implementierungsdetails

- **Datenbereinigung & Standardisierung:** Feature‑Spalten werden pro Land **nur mit Trainingsjahren (≤ `TRAIN_END_YEAR`)** skaliert; danach wird derselbe Scaler auf Val/Test angewendet — schützt gegen **Informationsleckage**. (Siehe `EnergyDataProcessor.standardize_data_for_cv()` in `data_processor_cv.py`.)
- **Sliding Window Generierung:** Fenstergröße `WINDOW_SIZE`, Vorhersagehorizont `PREDICTION_HORIZON`; nur **konsekutive** Jahresfenster werden zugelassen.
- **Zeitreihen‑CV:** Splits sind **chronologisch**. CV nutzt ausschließlich Jahre **≤ `TRAIN_END_YEAR`**; `trainer_cv.py` filtert Testsamples explizit heraus. 
- **Bewertung in Originalskala:** Inference‑Outputs werden **länderweise** zurücktransformiert (Inverse‑Transform des jeweiligen Target‑Scalers), erst danach MAE/RMSE/MAPE berechnet.
- **Finales Modell:** Auswahl nach CV‑RMSE (LSTM vs. CNN–LSTM), Training auf Train+Val, **einmalige** Bewertung auf dem Testsegment; Speicherung des Modells & **R²‑Scatter‑Plot**.
- **Länder‑Analysen:** Aggregation der Testfehler pro Land, GDP‑Top‑10‑Vergleich, Good/Bad‑Gruppen nach MAPE; CSV‑Exports für Nachnutzung.

---

## 7) Reproduzierbarkeit & Exportpfade

- **Seeds:** über `utils.set_seed()`/`Utils.set_seed()` gesetzt.
- **Artefakte:** Scaler + Metadaten via `artifacts_utils.save_artifacts()` nach `exports/artifacts/`.
- **Exporte:** Figuren (PNG+PDF), CSV‑Zusammenfassungen (CV‑Metriken, by‑Country‑Metriken) und JSON‑Reports liegen in `exports/`.

---

## 8) Lizenz / Nutzung

Dieses Projekt ist akademisch orientiert. Bitte nennen/zitieren Sie die Datengrundlage (*Our World in Data*) und beachten Sie die jeweilige Lizenzierung der Daten. Für Code verwenden Sie bitte übliche akademische Zitierweisen in Ihrer Arbeit.

---
## Kontakt

Bei Fragen zur Ausführung oder zu den Pfaden wenden Sie sich bitte an:
Shifeng Geng (Heinrich-Heine-Universität)
E-Mail: shgen100@hhu.de
---
### Quickstart (ganz kurz)

```python
from data_processor_cv import complete_data_processing_for_cv
from trainer_cv import run_time_series_cv_with_data_protection
processor, cv_dataset, (X,y,yrs,ctys) = complete_data_processing_for_cv()
_, lstm_cv, cnn_cv = run_time_series_cv_with_data_protection(cv_dataset, yrs, 5, processor)
print(lstm_cv["summary"]); print(cnn_cv["summary"])
# → Danach final_trainer_cv.run_final_training(...) und country_analysis.run_complete_country_analysis(...)
```