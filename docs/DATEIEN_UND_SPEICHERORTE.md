# Dateien: Herkunft, Formate und Speicherorte

Stand: 28.10.2026, mit dem vorhandenen Quellcode abgeglichen.

`<Projekt>` bezeichnet `D:\Python_Projects\NN_AFM_pyproj`. Relative Datenpfade
werden in [config_paths.py](../src/config/config_paths.py) aus dem Projektpfad
ermittelt. Externe Motor-CAD-/eduVPN-Pfade sind getrennt konfiguriert.

Installation und CLI stehen in der [README](../README.txt), die Verarbeitung im
[Workflow](WORKFLOW_DE.md), Testdateien und Prüfnachweise im [Testkonzept](TESTS_DE.md).
Diese Seite ist die zentrale Referenz für produktive Artefakte.

## 1. Eingaben und Simulationsergebnisse

| Datei / Muster | Ort | Erzeuger und Bedeutung |
|---|---|---|
| `base_first_data.mot` | `D:\MotorCAD` | Extern bereitgestelltes Basismodell; wird vom Projekt nicht erzeugt |
| Motor-CAD-EXE | Pfad aus `PATH_MCAD_EXE` | Externe Installation; Hash geht in Simulationsmetadaten ein |
| Externe Abhängigkeitsdateien | Einträge in `MCAD_DEPENDENCY_FILES` | Bei Konfiguration gehasht; Liste derzeit leer |
| `Results_EMagnetic_simulation_...csv` | `data/MCAD_Results` oder explizites `csv_paths`-Ziel | `MCADSimulation.prepare_output()` / `prepare_result_csv()` legt Kopfzeile unmittelbar vor der jeweiligen Kombination an; danach Ergebniszeilen |
| `<Dateiname>.csv.meta.json` | Neben der CSV | `write_metadata()` schreibt Version-2-Setup und Fingerprint; native Ergebnisse ohne `adoption` |
| Adoptiertes Sidecar | Neben unveränderter Alt-CSV | API `adopt_legacy_csv(..., write=True)` oder `main_adopt_legacy_csv.py`; zusätzlich CSV-Hash, Begründung, UTC-Zeit und historische Setup-Annahme |
| Private MOT-Kopie | `D:\MotorCAD\temp_files\simulation_run_<ID>\base_first_data_<zufall>\base_first_data.mot` | Laufordner durch Pipeline, privater Workerordner durch `create_temp_file_copy()`; Originaldateiname bleibt erhalten |
| Weitere Motor-CAD-Artefakte | Gegebenenfalls im privaten MOT-Arbeitsordner | Von Motor-CAD erzeugt; kein vollständiges Verzeichnis aller externen Dateiformate durch Python garantiert |
| Haupt-/Test-CSVs | `data/Training_data` | Bewusste Auswahl durch Benutzer; keine automatische Übergabe aus Simulation |

Ein Sidecar enthält aktuell Hashes von MOT und EXE, Integrationsversion, externe
Abhängigkeitsrevision/-hashes, Spaltenreihenfolge und Speicherpräzision. **Es enthält
keine Python-Implementierungshashes.** Der Simulationsfingerprint bindet diesen
Payload. Ein Adoptionsvermerk mit `original_setup_verified=False` dokumentiert
eine Annahme; er verifiziert nicht nachträglich das historische Setup.

Der direkte Adoptionseinstieg kann ein Sidecar ersetzen; die API lehnt vorhandene
Sidecars ab und schreibt standardmäßig nicht. Die API
`migrate_simulation_metadata()` kann bei gleichem Setup Metadatenversion 1 in 2
überführen. Sie migriert weder CSV-Spalten noch Messwerte.

### Ergebnisname lesen

Muster:

~~~text
Results_EMagnetic_simulation_<B>_<L>_<MAP_SIZE[0]>_GenType-v2_<Datum_Zeit_Zone>_<ID>.csv
~~~

| Bestandteil | Bedeutung |
|---|---|
| `B` | Grunddesignanzahl |
| `L` | Anzahl gemeinsam gesampelter Lambda-Designs |
| `MAP_SIZE[0]` | Größe der ersten Rasterachse; nicht Gesamtpunktzahl oder vollständige Rasterbeschreibung |
| `GenType-v2` | Aktuell unterstützter Generator |
| Datum, Zeit, `CET`/`CEST` | Jobnamenszeit in `Europe/Berlin` |
| `ID` | Achtstellige Lauf-ID |

CSV-Format: UTF-8, Semikolon, Dezimalkomma, vier Nachkommastellen für Floatwerte.
Die aktuelle Reihenfolge `CSV_PARAMETERS` umfasst 39 Spalten; vollständiges
Schema und aktueller Datenbestand stehen in der README. Alte 29-Spalten-Dateien
werden durch ein neues Sidecar allein nicht kompatibel.

`SimulationSeries.prepare_output()` prüft vorab Metadaten-Voraussetzungen;
erst vor der jeweiligen Kombination werden deren CSV und Sidecar vorbereitet.
Spätere Jobs erhalten damit nicht schon bei Serienstart leere Ergebnisdateien.
Cache-Suche berücksichtigt direkte Ergebnis-CSVs und ein explizites bestehendes
Ziel, keine Archivunterordner.

## 2. Optimierungsstudien

| Datei | Ort | Erzeuger und Inhalt |
|---|---|---|
| `study.journal` | `data/optimization/<Fingerprint>` | Optuna/`NNOptimizer`: Trialparameter, Architektur, Verluste, Kanal-/Verlustbereichsattribute und Studienkonfiguration |
| Journal-Lockdatei, etwa `study.journal.lock` | Neben dem Journal | `JournalFileOpenLock` koordiniert gemeinsame Zugriffe |
| Historisches `study.sqlite3` und Begleitdateien | Bestehende Studienordner | Frühere SQLite-Speicherung; keine automatische Übernahme in aktuelle Journals |

Der Studienname ist der Fingerprint. Jeder Kanal erhält eine eigene Studie.
`experiment_fingerprint` und `experiment_config` sind Studienattribute;
eine separate Konfigurations-JSON wird nicht erzeugt. Aktuelle Trials speichern
keine R²-Attribute und keine Netzgewichte.

Gleicher Fingerprint lädt die Studie und ergänzt Trials je Worker. Ein Journal
ersetzt weder Trainingsquellen noch Modellbundle. Aktive Journals, Locks oder
SQLite-Transaktionsdateien nicht bearbeiten oder löschen. Allein vorhandene
Dateiendungen belegen keine sichere Bereinigbarkeit.

## 3. Modellläufe und Exporte

[model_paths.py](../src/modules/neuronal_networks/model_data_io/model_paths.py)
bildet die Kategorien und Dateinamen. `<Lauf>` ist
`<JJJJMMTT_HHMMSS>_<achtstellige ID>`.

| Variante | Modelldatei |
|---|---|
| Referenzmodell | `data/trained_nn/Reference_model/<Lauf>/SurrogateModel.pt` |
| Kombimodell | `data/trained_nn/Combi_model/<Lauf>/SurrogateModel.pt` |
| CNN-Sammlung | `data/trained_nn/AutoencoderCollections/<Lauf>/AutoencoderCollection.pt` |

Zusätzlich schreibt `main_optimize.py` im selben Lauf:

| Datei | Erzeuger | Inhalt |
|---|---|---|
| `evaluation.pt` | `save_evaluation()` | Referenzen, Vorhersagen/Rekonstruktionen und Metriken je Datenteil, gebunden an Modellname und SHA-256 |
| `result_plots/<Kanal>_prediction_vs_sim_ref-model.png` | `plot_prediction_results()` | Sieben Rasterdiagramme |
| `result_plots/<Kanal>_prediction_vs_sim_ref-model.pdf` | `plot_prediction_results()` | Sieben Einzel-PDFs |
| `result_plots/prediction_report.pdf` | `create_prediction_report()` | Ein A4-Querformatblatt mit sieben Diagrammen, Kennzahlen, Architekturen und Trainingsinformationen |

Sieben Netze werden gemeinsam in einer Modelldatei gespeichert, nicht als sieben
separate Modelle. Modellbundle und Auswertung werden über temporäre Dateien
im Zielordner atomar ersetzt. Nach normalem Schreiben bleibt keine zweite
temporäre Modelldatei übrig. Ein späterer Fehler im Plotexport kann Modell und
Auswertung bereits vollständig hinterlassen.

### Format- und Herkunftsgrenzen

| Artefakt | Aktuelles Format und wesentliche Inhalte |
|---|---|
| `SurrogateModel.pt` | Surrogatformat 2, `model_type="surrogate_model"`; individuelle MLP-Architekturen/-Gewichte, optionale CNN-Decoder, Features/Kanäle, Geometriegrenzen, Achsen, Zielskalierung, Split-IDs, Quellenhashes, Seed, Metriken, Trainingsdauern und ggf. CNN-Herkunft |
| `AutoencoderCollection.pt` | Sammlungsformat 1, `model_type="autoencoder_collection"`; sieben CNN-Architekturen/-Gewichte, Achsen, Skalierung, Splits, Quellenhashes, Seed und Metriken |
| `evaluation.pt` | Auswertungsformat 2; `evaluation_kind="prediction"` oder `"reconstruction"` und `value_scale="physical"` |

Modelle werden auf CPU mit `weights_only=True` geladen und strukturell geprüft.
Ein altes `surrogate.pt` mit Surrogatformat 1 ist kein aktuelles Format-2-Bundle.
`load_bundle()` bietet keine automatische Migration; Umbenennen ersetzt sie nicht.
Auch aktuelle Formatnummern garantieren bei geänderter Feature-/Kanaldefinition
keine Kompatibilität mit dem heutigen Code.

Die Auswertung enthält für Training, Validierung und gegebenenfalls Test:

- Maschinen-IDs und **normalisierte** Geometrieeingaben;
- physikalisch skalierte Referenz- und Modellkennfelder mit Form `(N,S,Q,D,7)`;
- normalisierten MSE, R² je Kanal und Mittel sowie physikalischen MAE/MSE je Kanal.

Dazu kommen Achsen, Feature-/Kanalnamen, Quellenhashes und Modellname/-hash.
`value_scale="physical"` bezieht sich auf Kennfelder; die Geometrie bleibt normalisiert.
Die Datei bewahrt Referenzwerte für spätere Diagramme auf, ersetzt aber nicht die
vollständigen Original-CSVs samt Simulationsparametern und Herkunft.

### Erneuter Diagrammexport

`main_plot_results.py` liest `MODEL_FOLDER` aus dem Skript. Relative Ordner beziehen
sich auf `data/trained_nn`, zum Beispiel `Reference_model/<Lauf>`. Absolute Ordner
innerhalb dieses Wurzelpfads sind ebenfalls zulässig. Der Bericht liest den in
`evaluation.pt` gespeicherten Modellnamen und prüft dessen SHA-256.
Er verlangt Training/Validierung und genau sieben Kanäle.

Benötigt werden kompatible Auswertung, Modell und Arial; Original-CSVs oder neues
Training sind nicht erforderlich. Vorhandene Diagramme gleichen Namens werden
überschrieben. Das eingetragene historische `MODEL_FOLDER` vor Nutzung prüfen.
`data/plots` enthält Altbestände; es ist kein reguläres Ziel der heutigen Exporte.

## 4. Logs und Ablauf eines erfolgreichen Laufs

`main_mcad_sim` und `main_optimize` erzeugen je Aufruf:

~~~text
data/logs/<JJJJMMTT_HHMMSS_Mikrosekunden>_<ID>_application.log
~~~

Hauptprozess und Worker teilen während der Pipeline eine Datei über Queue-Logging;
davor/danach schreibt der Hauptprozess über Anwendungshandler. Es gibt keine
aktuellen getrennten INFO-/ERROR-TXT-Ausgaben. Reines `main_plot_results` richtet
nicht selbst dieselbe Anwendungslogdatei ein.

Bei einer Simulation mit zwei ausdrücklich gewählten Kombinationen entsteht:

1. Ein Log und ein temporärer Pipeline-Laufordner.
2. CSV/Sidecar der ersten Kombination unmittelbar vor ihrem Lauf.
3. Ergebniszeilen und private Workerordner je tatsächlich gestarteten Worker;
   neue Pools und Workerordner bei späteren Batches.
4. Danach CSV/Sidecar der zweiten Kombination und deren Berechnung.
5. Nach Bereinigung verbleiben CSVs, Sidecars und Log; der eigene Tempordner
   wird entfernt, soweit keine Bereinigungsfehler entgegenstehen.

Die getrennte Optimierung erzeugt eigene Logs und Kanalstudien. Nach Suche und
finalem Training **innerhalb der Pipeline** wird die fertige GUI geschlossen;
anschließend schreibt der Einstieg Modell, Auswertung, Einzelplots und Bericht.
Scheitert ein Schritt, sind später genannte Artefakte nicht automatisch vorhanden.

Eine gemeinsame ID über getrennte Simulation, Optimierung, Modell und Log wird
nicht durchgängig geführt. Quellenhashes, Studienattribute, Modellpfade und Logs
dienen zur Zuordnung.

## 5. Was nicht dauerhaft gespeichert wird

| Objekt | Verbleib bzw. Einschränkung |
|---|---|
| Vollständiger `DataLoader`-/Trainingszustand | Arbeitsspeicher; `evaluation.pt` persistiert ausgewählte Eingaben/Kennfelder, keinen wiederaufnehmbaren Trainingszustand |
| Berechnete Lambda-Spalten | Geometrietensoren enthalten Features; Original-CSV wird nicht umgeschrieben |
| Jede Trainingsepoche und Optimizerzustand | Keine automatische History-CSV oder fortsetzbarer Epochencheckpoint |
| Netzgewichte jedes Optuna-Trials | Keine dauerhafte Trial-Modelldatei |
| `ChannelOptimizationResult` | Python-Ergebnis je Kanal; Architektur/Verlust zusätzlich aus Studie und `metrics.search_trials` ableitbar |
| Vollständige Original-CSV/Python-Umgebung | Nicht als Kopie im Modellbundle |
| `predict_raw_geometry()`-Ergebnis | Rückgabetensor, kein automatischer CSV-Export |
| Separate `metrics.json` | Nicht erzeugt; Metriken in Bundles und Log |

## 6. Hilfsdateien und Aufbewahrung

`backups` enthält manuelle Sicherungen, keine reguläre Ablaufstation.
`__pycache__`, `.pytest_cache` und gegebenenfalls Ruff-Cache sind Werkzeugdaten.
`test-results/pytest.xml` entsteht nur bei ausdrücklich gewähltem JUnit-Export;
Testdateien liegen normalerweise im Betriebssystem-Tempordner.
`PATH_TEMP_DATA` und `PATH_DATA_MLP_RESEARCH_BASIC` sind keine regulären Ausgaben
der aktuellen Einstiege; `create_temp_csv_file()` ist ein separat aufrufbarer Helfer.

Die alten drei Ruff-Prüftexte und `README_TESTS_UND_DOKUMENTATION.md` sind beim
Abgleich nicht mehr vorhanden. `check_logging.py` ist ebenfalls kein aktuelles
Prüfskript; Loggingtests liegen unter `tests`.

Für nachvollziehbare Läufe gemeinsam aufbewahren: CSV und Sidecar, tatsächlich
verwendete Trainings-/Testquellen, Studien, Modell, Auswertung, Logs sowie Code-
und Umgebungsstand. Alte Modelle/SQLite-Studien/Backups sind nicht allein wegen
ihres Alters entbehrlich. Verwaiste Motor-CAD-Arbeitsordner erst nach Ende der
zugehörigen Prozesse gezielt prüfen. Es gibt keinen pauschalen Löschlauf über
fremde Tempordner.
