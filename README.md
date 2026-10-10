# Partikelfilter: mehrgipflige Verteilungen mit Stichproben darstellen – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-particle-filter-demo.streamlit.app/)** -->

Sechstes Stück der Linie **Zustandsschätzung und Filter** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“.
Es führt zwei Stücke zusammen: [Erweiterter Kalman-Filter](https://github.com/sebastian-hanisch/ekf-demo) (kontinuierlicher, nichtlinearer Zustand, aber **ein** Gauß-Gipfel) und
[HMM-Dekodieren](https://github.com/sebastian-hanisch/hmm-decode-demo) (mehrere Gipfel, aber **diskreter, kleiner** Zustandsraum). Der Partikelfilter stellt die Verteilung als Wolke von
Stichproben dar. Im Unterschied zu den Fall-Demos zeigt die Demo ein einziges Verfahren an einem wachsenden Beispiel.

## Warum dieses Problem

Eine Bake, die nur die Entfernung misst, kann zwei spiegelbildliche Orte nicht unterscheiden. Ein Gauß-Filter legt seinen Mittelwert **dazwischen**, wo das Fahrzeug nicht sein kann.
Die Kernfrage: **Wann lohnt sich die Stichprobenwolke, und was kostet sie?** Die Antwort ist differenziert: sie hält beide Möglichkeiten offen, ist aber beim Positionsfehler nicht besser als der UKF
und braucht genug Partikel.

## Modell

Zustand $(x, y, v_x, v_y)$, Konstante-Geschwindigkeit-Modell. Straße bei $y = 60$ m, Bake im Ursprung, optional eine zweite bei $(0, 30)$. Messarten: nur Entfernung (zweideutig) oder Entfernung
und Peilung (eindeutig). Anfangswissen: $x$ eng, $y$ weit und symmetrisch um 0 (die Straßenseite ist unbekannt). Der Partikelfilter ist der einfache **Bootstrap-Filter** mit systematischem
Resampling unter einer effektiven Stichprobengröße von $N/2$. EKF und UKF wie im vorigen Stück. Formeln im 📐-Expander, Kern in `pfl_algorithm.py`.

## Methodik

Drei Experimente auf Knopfdruck, je über 30 feste Fahrten (`SWEEP_SEEDS`):

1. Partikelzahl 100 bis 10 000 bei eindeutiger Messung: Fehler und Rechenzeit gegen EKF und UKF.
2. Mehrdeutigkeit: wer findet die richtige Seite, je nachdem, wann die zweite Bake dazukommt.
3. Dimension: Anteil effektiver Partikel nach einem Gewichtungsschritt, gemessen und nach geschlossener Formel.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Median bzw. Anteil über 30 feste Fahrten, Entfernungsrauschen 3 m; `tests/test_claims.py` rechnet jede nach.

| Frage | Befund |
|---|---|
| Wie viele Partikel bei eindeutiger Messung? | Fehler **6,1 m** (100), **3,4 m** (300), **2,4 m** (1000), **2,3 m** (10 000); EKF **2,4 m**, UKF **2,2 m**. Ab etwa 1000 Partikeln ist der Partikelfilter auf dem Niveau der Gauß-Filter, bei einem Vielfachen der Rechenzeit. |
| Mehrdeutige Entfernungsmessung, zweite Bake ab Sekunde 30 | Richtige Seite gefunden: EKF **0 %**, UKF **97 %**, Partikelfilter (1000) **77 %**. |
| Und mit mehr Partikeln? | Partikelfilter: **47 %** (300), **77 %** (1000), **97 %** (3000). Mit wenigen Partikeln geht in manchen Fahrten der richtige Gipfel verloren, bevor die zweite Bake ihn braucht. |
| Wo liegt der Vorteil? | Vor der zweiten Bake liegen **82 %** der Wahrscheinlichkeit des Partikelfilters auf den beiden möglichen Orten (innerhalb 15 m), beim UKF **9 %**: der Gauß-Filter legt seine Masse dazwischen. |
| Wenn die zweite Bake nie kommt | Keiner findet die Seite zuverlässig: UKF **0 %**, Partikelfilter **13 %**; der Partikelfilter hält aber beide Möglichkeiten offen (Masse auf der „oberen“ Seite im Mittel **44 %**). |
| Dimension | Bei Prior-Streuung gleich Messrauschen bleiben von 1000 Partikeln nach **einem** Gewichten im Mittel **87 %** (1 Dimension), **56 %** (4), **32 %** (8), **11 %** (16) wirksam; bei dreimal schärferer Messung **40 %** (1), **2,6 %** (4), 3 von 1000 Partikeln (8) und praktisch **einer** in 32 Dimensionen. |

## Ehrliche Grenzen

- **Die Wahrheit ist das Modell.** Fahrt und Filter teilen dasselbe Modell und bekannte Rauschwerte; es ist kein echter Sensordatensatz im Spiel.
- **Beim Positionsfehler ist der Partikelfilter hier nicht besser als der UKF.** Der UKF findet die richtige Seite nach der zweiten Bake bei dieser Anordnung ebenso oft; der Vorteil liegt in der Darstellung vor der Auflösung.
- **Der Mittelwert einer zweigipfligen Wolke ist kein Ort.** Die Demo liest deshalb die Masse um die beiden Orte; bei „nie“ gibt es keinen sinnvollen Punktschätzer.
- **Bootstrap-Filter, nichts Besseres.** Bessere Vorschlagsverteilungen oder gleichzeitiges Lernen von Parametern sind nicht umgesetzt; die Dimensionsaussage gilt für den einfachen Filter.
- **Rechenzeiten sind Messungen dieses Rechners** und plattformabhängig; nur ihre Größenordnung trägt.
- **Seltene Fehlschläge brauchen mehr Fahrten.** Bei 30 Fahrten sind einzelne Prozentpunkte beim Anteil „gefunden“ nicht belastbar.

## Verifikation

- **Kern:** Messfunktion und Jacobi-Matrix, Spiegelsymmetrie der Entfernungsmessung, ESS, systematisches Resampling (Handrechnung, Zählung innerhalb einer Einheit der Erwartung), Neuziehen nur bei kleiner ESS (`tests/test_algorithm.py`).
- **Orakel** (`tests/test_oracle_particle_filter.py`, anderer Rechenweg):
  - Linear-gaußscher Grenzfall: 40 000 Partikel gegen den exakten Kalman-Filter.
  - UKF gleich EKF bei linearer Messung.
  - Eine Messung gegen den **exakten Posterior auf einem Gitter**, einschließlich der beiden Gipfel (je etwa die Hälfte der Masse).
  - Effektive Stichprobengröße gegen die geschlossene Formel.
  - Gipfelmasse gegen eine einfache Schleife; angezeigte Kennzahlen gegen unabhängige Neuberechnung.
- **Zahlen im README:** `tests/test_claims.py`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, Regler an Minimum und Maximum, jede Auswahloption, beide Messarten, Experiment-Knöpfe) und Footer-Test.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App (Skelett nach DEMO-PLAYBOOK §2) |
| `pfl_algorithm.py` | Kern: Messmodelle, EKF, UKF, Bootstrap-Partikelfilter, Resampling, ESS |
| `pfl_scenario.py` | Fahrt, Baken, Messungen (ein Generator, Regler skalieren denselben Zufall) |
| `pfl_evaluation.py` | Kennzahlen, Sweeps, Gipfelmasse, Dimensionsrechnung |
| `pfl_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `pfl_presets.py` | Permalink (Begrenzen/Einrasten), Presets, Zufalls-Seed-Button |
| `pfl_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `pfl_pdf_export.py` | PDF-Auswertung (fpdf2) |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit bzw. Laufzeit plus pytest |
| `tests/` | Einheiten, Orakel, Behauptungen, Presets, AppTest, Footer |

## Bewusst nicht umgesetzt

- **Bessere Vorschlagsverteilungen, Rao-Blackwellisierung, Ensemble-Filter** für hohe Dimension: nicht umgesetzt.
- **Echte Sensordaten, Straßennetz:** nicht eingebunden.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
