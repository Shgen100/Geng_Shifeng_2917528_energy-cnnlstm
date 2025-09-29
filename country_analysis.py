"""
Länderbezogenes Analysemodul
Fasst die Prognoseleistung pro Land zusammen und unterstützt Auswertungen für BIP-Top-10 sowie Leistungsgruppen.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import Utils


def _set_large_font_style():
    """Größere Schrift für Abbildungen einstellen"""
    plt.rcParams.update({
        'font.size': 14,          # Grundschriftgröße
        'axes.titlesize': 16,     # Untertitel
        'axes.labelsize': 14,     # Achsenbeschriftungen
        'xtick.labelsize': 12,    # x-Achsen-Ticks
        'ytick.labelsize': 12,    # y-Achsen-Ticks
        'legend.fontsize': 12,    # Legende
        'figure.titlesize': 18,   # Haupttitel
    })


def _reset_font_style():
    """Schriftstil auf Standard zurücksetzen"""
    plt.rcParams.update(plt.rcParamsDefault)


def load_gdp_data(csv_path = "owid-energy_consumption_2000_2024.csv"):
    """BIP-Daten laden und das durchschnittliche BIP pro Land berechnen (Aggregationen wie Welt/Kontinente/Einkommensgruppen entfernen)"""
    print("BIP-Daten werden geladen ...")
    df = pd.read_csv(csv_path)

    # Durchschnittliches BIP pro Land berechnen
    gdp_by_country = (
        df.groupby("country")["gdp"]
        .mean()
        .sort_values(ascending=False)
        .dropna()
    )

    # Nur echte Länder behalten, Aggregationen/Regionen entfernen
    blacklist_exact = {
        "World", "Africa", "Asia", "Europe", "European Union (27)", "European Union (28)",
        "European Union (EU)", "North America", "South America", "Oceania",
        "High-income countries", "Upper-middle-income countries", "Lower-middle-income countries",
        "Low-income countries", "Other non-OECD Asia", "Other Africa", "Other Middle East",
        "Other CIS (excl. EU)", "Other non-OECD Europe"
    }
    blacklist_keywords = [
        "income", "Other ", "other ", "Region", "Europe (", "Asia (", "Africa (", "America",
        "Caribbean", "Middle East", "CIS", "World"
    ]
    def is_aggregate(name: str) -> bool:
        if name in blacklist_exact:
            return True
        low = name.lower()
        return any(k.lower() in low for k in blacklist_keywords)

    gdp_by_country = gdp_by_country[~gdp_by_country.index.map(is_aggregate)]

    print(f"Insgesamt {len(gdp_by_country)} Länder mit BIP-Daten (Aggregationen entfernt).")
    print("Top-10 nach BIP:")
    for i, (country, gdp) in enumerate(gdp_by_country.head(10).items(), 1):
        print(f"  {i:2d}. {country:<25} {gdp:>15,.0f}")

    return gdp_by_country

def calculate_country_metrics(test_details, model_name="Final Model"):
    """Pro-Land-Kennzahlen aus den Testergebnissen berechnen"""
    print(f"\nLänderspezifische Kennzahlen für {model_name} werden berechnet ...")

    if not test_details['countries']:
        print("Warnung: Keine Länderinformationen im Testdatensatz vorhanden.")
        return pd.DataFrame()

    # In DataFrame umwandeln
    results_df = pd.DataFrame({
        'country': test_details['countries'],
        'prediction': test_details['predictions'],
        'target': test_details['targets']
    })

    # Kennzahlen pro Land
    country_metrics = []

    for country in results_df['country'].unique():
        country_data = results_df[results_df['country'] == country]
        pred = np.array(country_data['prediction'])
        true = np.array(country_data['target'])

        # Metriken
        mae = np.mean(np.abs(pred - true))
        rmse = np.sqrt(np.mean((pred - true) ** 2))

        # MAPE (Division durch 0 vermeiden)
        mask = true != 0
        mape = np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100 if mask.sum() > 0 else np.nan

        country_metrics.append({
            'country': country,
            'samples': len(country_data),
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'mean_target': np.mean(true),
            'mean_prediction': np.mean(pred)
        })

    metrics_df = pd.DataFrame(country_metrics)
    print(f"Kennzahlen für {len(metrics_df)} Länder berechnet.")
    return metrics_df


def _plot_gdp_vs_rmse(ax, data):
    """Streudiagramm BIP vs. RMSE zeichnen (mit größerer Schrift)"""
    ax.scatter(data['gdp'], data['rmse'], alpha=0.7, s=120)
    for _, row in data.iterrows():
        ax.annotate(row['country'][:10], (row['gdp'], row['rmse']),
                   xytext=(5, 5), textcoords='offset points', fontsize=10)
    ax.set_xlabel('Durchschnittliches BIP (2000-2024)', fontweight='bold')
    ax.set_ylabel('RMSE', fontweight='bold')
    ax.set_title('BIP vs. Modell-RMSE (Top-10 nach BIP)', pad=15, fontweight='bold')
    ax.grid(True, alpha=0.3)


def _plot_country_metrics(ax, data):
    """Vergleich der Kennzahlen pro Land (größere Schrift)"""
    x = np.arange(len(data))
    width = 0.35

    bars1 = ax.bar(x - width / 2, data['mae'], width, label='MAE', alpha=0.8)
    bars2 = ax.bar(x + width / 2, data['rmse'], width, label='RMSE', alpha=0.8)

    ax.set_xlabel('Land', fontweight='bold')
    ax.set_ylabel('MAE / RMSE', fontweight='bold')
    ax.set_title('Modellleistung nach Land (BIP Top-10)', pad=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c[:8] for c in data['country']], rotation=45)
    ax.legend()

    # Zweite Achse: MAPE
    ax2 = ax.twinx()
    ax2.plot(x, data['mape'], marker='o', linewidth=3, markersize=8,
             label='MAPE (%)', color='red')
    ax2.set_ylabel('MAPE (%)', fontweight='bold')
    ax2.legend(loc='upper right')


def _plot_actual_vs_predicted(ax, data):
    """Istwert-gegen-Prognose zeichnen (größere Schrift)"""
    ax.scatter(data['mean_target'], data['mean_prediction'], alpha=0.7, s=120)

    # Referenzlinie y = x
    min_val = min(data['mean_target'].min(), data['mean_prediction'].min())
    max_val = max(data['mean_target'].max(), data['mean_prediction'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5,
            linewidth=3, label='Perfekte Prognose')

    for _, row in data.iterrows():
        ax.annotate(row['country'][:8], (row['mean_target'], row['mean_prediction']),
                   xytext=(5, 5), textcoords='offset points', fontsize=10)

    ax.set_xlabel('Mittelwert Istwert', fontweight='bold')
    ax.set_ylabel('Mittelwert Prognose', fontweight='bold')
    ax.set_title('Istwert vs. Prognose (BIP Top-10)', pad=15, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_sample_distribution(ax, data):
    """Verteilung der Testbeispiele pro Land (größere Schrift)"""
    bars = ax.bar(range(len(data)), data['samples'], alpha=0.7)
    ax.set_xlabel('Land', fontweight='bold')
    ax.set_ylabel('Anzahl der Testbeispiele', fontweight='bold')
    ax.set_title('Verteilung der Testbeispiele (BIP Top-10)', pad=15, fontweight='bold')
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([c[:8] for c in data['country']], rotation=45)

    # Wertebeschriftungen
    for bar, value in zip(bars, data['samples']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
               f'{value}', ha='center', va='bottom', fontsize=11, fontweight='bold')


def analyze_gdp_top10_performance(country_metrics, gdp_data, save_dir="exports"):
    """
    Analysiert die Modellleistung für BIP-Top-10-Länder, erzeugt nur GDP vs RMSE Streudiagramm
    Filtert automatisch 'World' und andere Aggregationen heraus
    """
    print(f"\nAnalysiere Modellleistung für BIP-Top-10-Länder...")

    # Aggregationen ausschließen (einschließlich World)
    blacklist = {
        "World", "Africa", "Asia", "Europe", "North America", "South America",
        "Oceania", "European Union", "High-income countries",
        "Upper-middle-income countries", "Lower-middle-income countries",
        "Low-income countries"
    }

    # Länder, die sowohl in Testdaten als auch in BIP-Daten vorhanden sind
    available = [
        c for c in gdp_data.index
        if c in set(country_metrics['country'].values) and c not in blacklist
    ]

    # Top-10 auswählen (nach BIP sortiert)
    target_n = 10
    pick = []
    for c in gdp_data.index:
        if c in available and c not in pick:
            pick.append(c)
        if len(pick) >= target_n:
            break

    if len(pick) == 0:
        print("Warnung: Keine geeigneten Länder gefunden")
        return None

    print(f"{len(pick)} Länder für Analyse ausgewählt (World und Aggregationen ausgeschlossen)")

    # Daten vorbereiten
    top_metrics = country_metrics[country_metrics['country'].isin(pick)].copy()
    top_metrics['gdp'] = top_metrics['country'].map(gdp_data)
    top_metrics = top_metrics.sort_values('gdp', ascending=False)

    # Diagramm erstellen
    _set_large_font_style()
    fig, ax = plt.subplots(figsize=(12, 9))

    # GDP vs RMSE Streudiagramm
    ax.scatter(top_metrics['gdp'], top_metrics['rmse'], alpha=0.7, s=120)
    for _, row in top_metrics.iterrows():
        ax.annotate(
            row['country'][:10],
            (row['gdp'], row['rmse']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10
        )

    ax.set_xlabel('Durchschnittliches BIP (2000-2024)', fontweight='bold')
    ax.set_ylabel('RMSE', fontweight='bold')
    ax.set_title('BIP vs. Modell-RMSE (Top-10 nach BIP)', pad=15, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Speichern (ohne Dateierweiterung)
    save_path = Path(save_dir) / "figures" / "gdp_top10_rmse_scatter"
    Utils.save_figure(save_path, fig)
    plt.show()
    plt.close(fig)

    _reset_font_style()
    print(f"Diagramm gespeichert: {save_path}")

    return top_metrics


def _plot_performance_boxplot(ax, good_countries, bad_countries):
    """Leistungsboxplot für Gruppen zeichnen (größere Schrift)"""
    data_to_plot = [good_countries['mape'], bad_countries['mape']]
    bp = ax.boxplot(data_to_plot, labels=['Gut (untere 25 %)', 'Schwach (obere 25 %)'],
                    patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel('MAPE (%)', fontweight='bold')
    ax.set_title('Leistungsgruppen (MAPE-Verteilung)', pad=15, fontweight='bold')
    ax.grid(True, alpha=0.3)


def _plot_metrics_comparison(ax, good_countries, bad_countries):
    """Vergleich der Durchschnittsmetriken der Gruppen (größere Schrift)"""
    metrics = ['MAE', 'RMSE', 'MAPE']
    good_means = [good_countries['mae'].mean(), good_countries['rmse'].mean(),
                  good_countries['mape'].mean()]
    bad_means = [bad_countries['mae'].mean(), bad_countries['rmse'].mean(),
                 bad_countries['mape'].mean()]

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width / 2, good_means, width, label='Gute Länder',
           color='lightgreen', alpha=0.7)
    ax.bar(x + width / 2, bad_means, width, label='Schwache Länder',
           color='lightcoral', alpha=0.7)

    ax.set_xlabel('Metriken', fontweight='bold')
    ax.set_ylabel('Durchschnittlicher Wert', fontweight='bold')
    ax.set_title('Vergleich der Durchschnittsleistung', pad=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_samples_comparison(ax, good_countries, bad_countries):
    """Vergleich der Stichprobengrößen der Gruppen (größere Schrift)"""
    sample_data = [good_countries['samples'], bad_countries['samples']]
    bp = ax.boxplot(sample_data, labels=['Gute Länder', 'Schwache Länder'],
                    patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel('Anzahl der Testbeispiele', fontweight='bold')
    ax.set_title('Verteilung der Stichprobengrößen', pad=15, fontweight='bold')
    ax.grid(True, alpha=0.3)


def _save_group_data(good_countries, bad_countries, save_dir, threshold_good, threshold_bad):
    """Gruppendaten speichern"""
    good_csv = Path(save_dir) / "good_performance_countries.csv"
    bad_csv = Path(save_dir) / "bad_performance_countries.csv"

    good_countries.to_csv(good_csv, index=False)
    bad_countries.to_csv(bad_csv, index=False)

    print(f"Daten der gut performenden Länder: {good_csv}")
    print(f"Daten der schwach performenden Länder: {bad_csv}")

    # Übersichtliche Listen ausgeben
    print(f"\nGute Länder (MAPE ≤ {threshold_good:.2f} %):")
    for _, row in good_countries.sort_values('mape').iterrows():
        print(f"  {row['country']:<20} MAPE: {row['mape']:6.2f} %")

    print(f"\nSchwache Länder (MAPE ≥ {threshold_bad:.2f} %):")
    for _, row in bad_countries.sort_values('mape', ascending=False).iterrows():
        print(f"  {row['country']:<20} MAPE: {row['mape']:6.2f} %")


def analyze_model_performance_groups(country_metrics, save_dir="exports"):
    """
    Analysiert Leistungsgruppen (Good vs Bad), erzeugt nur das Balkendiagramm zum Durchschnittsvergleich
    """
    print(f"\nAnalysiere Leistungsgruppen...")

    if len(country_metrics) == 0:
        print("Keine länderspezifischen Kennzahlen vorhanden")
        return None, None

    # Gruppierung nach MAPE: unteres Quartil = gut, oberes Quartil = schlecht
    mape_threshold_good = country_metrics['mape'].quantile(0.25)
    mape_threshold_bad = country_metrics['mape'].quantile(0.75)

    good_countries = country_metrics[
        country_metrics['mape'] <= mape_threshold_good
        ].copy()
    bad_countries = country_metrics[
        country_metrics['mape'] >= mape_threshold_bad
        ].copy()

    print(f"Gruppierungskriterium: gut ≤ {mape_threshold_good:.2f}%, schlecht ≥ {mape_threshold_bad:.2f}%")
    print(f"Gute Länder: {len(good_countries)}, Schlechte Länder: {len(bad_countries)}")

    # Diagramm erstellen
    _set_large_font_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    # Durchschnittswerte berechnen
    metrics = ['MAE', 'RMSE', 'MAPE']
    good_means = [
        good_countries['mae'].mean(),
        good_countries['rmse'].mean(),
        good_countries['mape'].mean()
    ]
    bad_means = [
        bad_countries['mae'].mean(),
        bad_countries['rmse'].mean(),
        bad_countries['mape'].mean()
    ]

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(
        x - width / 2, good_means, width,
        label='Good Countries', color='lightgreen', alpha=0.7
    )
    ax.bar(
        x + width / 2, bad_means, width,
        label='Bad Countries', color='lightcoral', alpha=0.7
    )

    ax.set_xlabel('Metriken', fontweight='bold')
    ax.set_ylabel('Durchschnittlicher Wert', fontweight='bold')
    ax.set_title('Durchschnittlicher Leistungsvergleich', pad=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Speichern (ohne Dateierweiterung)
    save_path = Path(save_dir) / "figures" / "performance_groups_avg_comparison"
    Utils.save_figure(save_path, fig)
    plt.show()
    plt.close(fig)

    _reset_font_style()
    print(f"Diagramm gespeichert: {save_path}")

    # Gruppendaten als CSV speichern
    _save_group_data(
        good_countries, bad_countries, save_dir,
        mape_threshold_good, mape_threshold_bad
    )

    return good_countries, bad_countries

def run_complete_country_analysis(test_details, model_name="Final Model",
                                  csv_path="owid-energy_consumption_2000_2024.csv",
                                  save_dir="exports"):
    """Vollständige länderspezifische Analyse ausführen"""
    print("Starte länderspezifische Gesamtanalyse.")
    print("=" * 50)

    # 1. BIP-Daten laden
    gdp_data = load_gdp_data(csv_path)

    # 2. Kennzahlen pro Land berechnen
    country_metrics = calculate_country_metrics(test_details, model_name)

    if len(country_metrics) == 0:
        print("Analyse nicht möglich: Es fehlen Länderinformationen.")
        return None, gdp_data, None, None, None

    # 3. Analyse der BIP-Top-10
    top10_metrics = analyze_gdp_top10_performance(country_metrics, gdp_data, save_dir)

    # 4. Analyse der Leistungsgruppen
    good_countries, bad_countries = analyze_model_performance_groups(country_metrics, save_dir)

    # 5. Vollständige Ländermetriken speichern
    all_metrics_path = Path(save_dir) / "by_country_metrics.csv"
    country_metrics.to_csv(all_metrics_path, index=False)
    print(f"Vollständige Ländermetriken gespeichert: {all_metrics_path}")

    print("\nLänderspezifische Analyse abgeschlossen.")
    print(f"Alle Ergebnisse gespeichert unter: {save_dir}")

    return country_metrics, gdp_data, top10_metrics, good_countries, bad_countries


if __name__ == "__main__":
    # Beispielhafter Test
    sample_test_details = {
        'countries': ['United States', 'China', 'Germany', 'Japan', 'India'] * 10,
        'predictions': np.random.normal(100, 20, 50),
        'targets': np.random.normal(95, 25, 50)
    }

    run_complete_country_analysis(sample_test_details, "Sample Model")
