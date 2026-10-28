# Projektworkflow

Stand: 28.10.2026, mit dem vorhandenen Quellcode abgeglichen.

Dieses Dokument beschreibt die Verarbeitung und ihre fachlichen Grenzen.
Installation, aktuelle CLI-Optionen und konfigurierte Zahlenwerte stehen in der
[README](../README.txt). Die [Artefaktreferenz](DATEIEN_UND_SPEICHERORTE.md)
erklärt Erzeuger, Formate und Speicherorte; das [Testkonzept](TESTS_DE.md)
beschreibt vorhandene Prüfungen und noch offene Integrationsaufgaben.

## 1. Abläufe und Modellvarianten

Simulation und Modelloptimierung sind getrennte Aufrufe. `main_mcad_sim.py`
erzeugt Simulationsergebnisse. Eine bewusste Auswahl stellt anschließend passende
CSVs unter `data/Training_data` bereit; diese Übergabe ist nicht automatisiert.

`main_optimize.py` unterstützt drei Wege:

| Variante | Eingabe und Ausgabe | Zweck |
|---|---|---|
| Referenzmodell: MLP ohne CNN-Sammlung | Geometrie → sieben MLPs → sieben Kennfelder | Direkte Kennfeldvorhersage |
| CNN-Sammlung | Vorhandene Kennfelder → sieben Autoencoder → rekonstruierte Kennfelder | Kompression und Rekonstruktion |
| Kombimodell: MLP mit geladener CNN-Sammlung | Geometrie → sieben MLPs → eingefrorene CNN-Decoder → Kennfelder | Vorhersage über einen latenten Raum |

Jeder Kanal erhält eine eigene Architektur und eigene Gewichte. Eine CNN-Sammlung
allein sagt kein Kennfeld aus einer Geometrie voraus.

## 2. Simulation vorbereiten

Maßgeblich sind
[config_paths.py](../src/config/config_paths.py),
[config_mcad_parameters.py](../src/config/config_mcad_parameters.py) und
[main_mcad_sim.py](../src/executable_files/main_mcad_sim.py).

Vor einem Lauf sind Basismodell, Motor-CAD-EXE, Lizenzzugang, optional eduVPN und
der gewünschte Umfang der Kombinationen `(B,L)` zu prüfen. Der Einstieg stellt
keine CLI für diese Einstellungen bereit. Die aktuelle Liste und der daraus
folgende Umfang stehen in der README.

Der Generator unterstützt nur `v2`. `B` zählt gemeinsam gesampelte Luftspalt-/
Statorbohrungskombinationen, `L` gemeinsam gesampelte zwölf Lambda-Verhältnisse.
Daraus folgen `B * L` Geometrien und `B * L * prod(MAP_SIZE)` Anforderungen.
Latin-Hypercube-Sampling verwendet Seed 17 für die Grundmaße und Seed 42 für
die Lambdas. Die Pipeline enthält noch den Standard `gen_type="v1"` und dessen
Punktzählung; eigene Aufrufe müssen ausdrücklich `gen_type="v2"` verwenden.

Die zentrale Umrechnung zwischen zwölf Lambda-Verhältnissen und Abmessungen
steht in [lambda_geometry.py](../src/modules/geometry/lambda_geometry.py).
Der Generator setzt berechnete Geometrie- und Betriebsparameter in Motor-CAD.
`DEFAULT_VALUES` wird nicht vollständig als Konfiguration gesetzt; weitere
Einstellungen stammen aus der MOT-Basisdatei.

### 2.1 Serie und einzelne Aufträge

`SimulationPipeline` validiert Kombinationen und Zielpfade, erstellt Jobs,
Fortschritts-/Loggingqueues und beansprucht jede Instanz nur einmal.
Ein zweiter Start ist auch nach einem Fehlstart unzulässig.

`SimulationSeries.prepare_output()` prüft zunächst die Metadaten-Voraussetzungen.
Danach erstellt die Pipeline einen eigenen temporären Laufordner und startet
optional eduVPN. **CSV und Sidecar werden erst vor der jeweiligen Kombination
vorbereitet.** Spätere Aufträge haben zu diesem Zeitpunkt noch keine neue Ausgabe.

Die Kombinationen laufen nacheinander. Ein Batch umfasst
`effektive Workerzahl * prod(MAP_SIZE)` Anforderungen. Jeder Batch bekommt einen
neuen `ProcessPoolExecutor`. Jeder tatsächlich gestartete Worker öffnet eine eigene
Motor-CAD-Instanz mit privater MOT-Kopie unter unverändertem Dateinamen.
Das begrenzt die Lebensdauer der Instanzen aus Stabilitätsgründen.

Grenzbedingungen werden geprüft, danach gegebenenfalls die magnetische Rechnung
ausgeführt. Erfolgreiche Ergebnisse werden gespeichert; abgelehnte Geometrien
oder fehlgeschlagene Rechnungen können ohne Ergebniszeile enden. Unerwartete
Workerfehler werden weitergegeben. Die Zeilenfolge richtet sich nach der
Fertigstellung, nicht nach der Generatorreihenfolge.

### 2.2 Kompatible Punkte wiederverwenden

`MCADSimulation` durchsucht CSVs direkt unter `data/MCAD_Results` und ein
ausdrücklich gewähltes bestehendes Ziel. Unterordner werden nicht durchsucht.
Quellen müssen passende Metadatenversion 2, gültigen Fingerprint, exakte
Spaltenreihenfolge und endliche numerische Werte besitzen. Bei adoptierten
Dateien wird zusätzlich der gespeicherte CSV-Hash geprüft.

Mit absoluter Parametertoleranz `1e-4` wird unterschieden:

- `existing`: im eigenen Ziel vorhanden, keine zusätzliche Zeile;
- `copied`: aus anderer kompatibler Quelle ins Ziel übernommen;
- `simulated`: neu berechnet und gespeichert;
- `dropped`: bearbeitet, aber ohne erfolgreiche Ergebniszeile.

Das eigene Ziel hat Vorrang. Gleiche offene Anforderungen innerhalb eines Batches
werden nur einmal berechnet, im Fortschritt jedoch für alle Anforderungen gezählt.
Ein voller Fortschrittsbalken ist deshalb kein Nachweis ausschließlich erfolgreicher
Rechnungen. Fremde Quellen bleiben unverändert.

### 2.3 Was der Simulationsfingerprint abdeckt

[simulation_metadata.py](../src/modules/data_managment/simulation_metadata.py)
verwendet Metadatenversion 2. Erfasst werden MOT-/EXE-Hashes, Version der
Motor-CAD-Anbindung, deklarierte externe Revision und Dateihashes, CSV-Schema und
Speicherpräzision. `is_compatible_metadata()` verlangt identische Version-2-Payloads
und korrekt berechnete identische Fingerprints. Gerichtete Freigabelisten sind
nicht Bestandteil des aktuellen Verfahrens.

Quellcode-Hashes gehören nicht dazu. Reine Python-Formatierung ändert deshalb
diesen Fingerprint nicht. Fachliche Änderungen außerhalb erfasster Dateien müssen
über die Revision berücksichtigt werden. Nicht in `MCAD_DEPENDENCY_FILES` erfasste
Materialdateien werden nicht automatisch entdeckt oder gehasht.

## 3. Altdateien und Datenmigration

Drei unterschiedliche Vorgänge sind zu unterscheiden:

| Vorgang | Verhalten |
|---|---|
| `main_adopt_legacy_csv.py` | Liest `CSV_PATH`, `MOT_SHA256` und `REASON` aus dem Skript; schreibt direkt und kann ein vorhandenes Sidecar ersetzen |
| API `adopt_legacy_csv()` | Verlangt ausdrückliche Setup-Annahme, Begründung und passenden MOT-Hash; lehnt vorhandene Sidecars ab; schreibt erst mit `write=True` |
| API `migrate_simulation_metadata()` | Prüft Version 1 → 2 bei ansonsten gleichem Setup; entfernt alte Implementierungshashes; schreibt erst mit `write=True` |

Der aktuelle [Adoptionseinstieg](../src/executable_files/main_adopt_legacy_csv.py)
hat keine Dry-Run-CLI. Die CSV wird bei diesen Vorgängen nicht umgeschrieben.
`original_setup_verified=False` dokumentiert eine historische Annahme. Adoptierte
Dateien bleiben ausschließlich Cache-Quellen; ihr CSV-Hash muss unverändert passen.

**Metadatenmigration ergänzt keine Datenfelder und transformiert kein Kennfeld.**
Die beiden aktuell direkt im Trainingsordner liegenden CSVs vom 26.10.2026 haben
29 Spalten, das heutige Schema verlangt 39. Die zehn fehlenden Spalten stehen in
der README. Insbesondere werden `Slot_Corner_Radius` und `Tooth_Tip_Depth` zur
Rekonstruktion der neuen Lambda-Eingaben benötigt.

Für eine fachlich begründete Datenmigration wären folgende Schritte erforderlich:

1. Historisches CSV-Schema, Achsendefinition, Einheiten und tatsächliches
   Simulationssetup feststellen.
2. Für jedes fehlende Feld die historischen Werte nachweisen. Heutige
   `DEFAULT_VALUES` sind dafür allein kein Beleg.
3. Geometrie-/Lambda-Roundtrip, Achsen, vollständige Raster und unabhängige
   Maschinen nach der Transformation prüfen.
4. In eine neue Datei exportieren und Herkunft, Annahmen und Transformation
   dokumentieren; Originale samt Sidecars erhalten.

Dieser Ablauf ist eine Anforderung an eine künftige Migration, kein vorhandenes
automatisches Migrationswerkzeug. Alternativ sind neue kompatible Simulationen
auszuwählen.

## 4. Hauptquelle, Tests und Datenaufbereitung

`--csv` wählt einen reinen Dateinamen direkt unter `Training_data`. Der Einstieg
wählt alle anderen CSVs direkt in diesem Ordner alphabetisch als Testquellen.
Unterordner und Sidecars sind keine Datenquellen. Die Auswahl wird geloggt;
eine Option `--test-csv` gibt es nicht.

Im Kombimodus wird außerdem jede Datei mit dem Hash der ursprünglichen
CNN-Hauptquelle von den Testquellen ausgeschlossen. Dafür müssen das geladene
Bundle `metrics.training_csv` und den passenden Quellenhash enthalten. Ohne
weitere CSV wird kein Testsatz angelegt.

Der [Datenlader](../src/modules/neuronal_networks/manipulating_data.py):

1. Ermittelt CSV-Hashes, prüft benötigte Spalten und endliche Werte und rekonstruiert
   die zwölf Lambdas. Fehlende Spalten oder ungültige Geometrien führen zum Fehler.
2. Berechnet Leiterfläche, q-/d-Stromdichten und normierte Drehzahl. Die vorgesehene
   Formel ist `ShaftSpeed / (Stator_Lam_Dia * pi)`. Achsenreihenfolge:
   `Shaft_Speed_norm`, `RMSPhaseCurrent_Q_density`, `RMSPhaseCurrent_D_density`.
3. Ordnet Rundungsabweichungen nur dann einem Referenzwert zu, wenn dessen
   Zuordnung eindeutig durch die vierstellige CSV-Rundung erklärbar ist.
4. Gruppiert nach `MACHINE_DEFINITION`: `Airgap`, `Stator_Bore` und zwölf Lambdas.
   Fehlende, zusätzliche oder doppelte Rasterpunkte verwerfen die ganze Maschine.
   Es wird nicht interpoliert. Quellen ohne verbleibendes Kennfeld werden abgelehnt.
5. Erstellt Geometrien `(N,14)` und Kennfelder `(N,S,Q,D,7)`, aktuell `(N,5,5,5,7)`.
6. Teilt die Hauptquelle nach ganzen Maschinen ungefähr 80/20 in Training und
   Validierung, fest mit Seed 42; mindestens fünf vollständige Maschinen sind nötig.
   Externe Testquellen werden getrennt zusammengeführt.
7. Prüft exakte float32-Geometrie-IDs vor Normalisierung auf Überschneidungen
   zwischen Splits und auf doppelte Testmaschinen. Ähnliche Geometrien werden
   dadurch nicht als identisch erkannt.
8. Skaliert nur `Airgap`/`Stator_Bore` anhand der Konfigurationsgrenzen; Lambdas
   bleiben Verhältniszahlen. Zielminima und Spannweiten stammen ausschließlich
   aus Training; der Nenner ist mindestens `1e-8`.

Validierung und Test können nach Skalierung außerhalb `[0,1]` liegen. Split und
Shuffle nutzen Seed 42 unabhängig von `--seed`. CSV-Quellen werden nicht verändert;
Trainings-Sidecars werden nicht geprüft. `write_from_csv_in_csv()` kopiert auch
Kopfzeilen und ist kein validierender Datensatzexport.

**Offener Codepunkt:** `_recalculate_data()` und Generator verwenden den Faktor
`pi`, `_align_speed_grid()` berechnet Intervalle dagegen ohne ihn. CSV-Rundung
kann daher nicht konsistent abgeglichen werden. Der Code wurde für diese
Dokumentationsaktualisierung nicht korrigiert; siehe README, Abschnitt 10.

## 5. Kanalsuche und finales Training

`OptimizationPipline` lädt Daten bereits vor dem GUI-Start. Beim Kombimodell
übernimmt `AutoencoderCollection.prepare_data()` die CNN-Zielskalierung und prüft
Kompatibilität. Testmaschinen dürfen weder im CNN-Training noch bei dessen
Validierung verwendet worden sein.

`OptimizationService` optimiert die sieben Kanäle nacheinander. Je Kanal laufen
bis zu vier Suchworker; die effektive Anzahl ist auf verfügbare CPUs begrenzt.
Jeder Trial trainiert nur das Netz dieses Kanals:

| Zweig | Suchgegenstand | Zu minimierender Validierungsverlust |
|---|---|---|
| MLP | Versteckte Schichten und Neuronenzahlen | Vollständiger normalisierter Kennfeld-MSE |
| CNN | Steigende Featurezahlen und latente Dimension | Vollständiger normalisierter Rekonstruktions-MSE |
| MLP mit CNN | MLP-Schichten und Neuronenzahlen bei eingefrorener CNN | MSE an 3 × 3 × 3 Stützstellen des dekodierten Kennfelds |

`NNTrainer` arbeitet auf CPU mit Adam und dem gesamten Trainingsdatensatz je
Schritt. Early Stopping stellt den besten Validierungszustand wieder her;
nicht endliche Verluste oder Gradienten führen zum Abbruch. Testdaten beeinflussen
weder Architekturauswahl noch Early Stopping.

Der Optimierer verwendet je Kanalfingerprint eine Studie im `study.journal` mit
`JournalStorage`, `JournalFileBackend` und `JournalFileOpenLock`. In den Fingerprint
gehen Daten, Splits, Skalierung, Kanal, Modelltyp, Such-/Trainingsparameter,
Workerzahl, Seed, Paketversionen und ausgewählte Implementierungshashes ein,
bei Kombimodellen auch die festen CNN-Gewichte. Die Hashliste erfasst nicht jede
fachlich relevante Datei; die verbleibende Einschränkung steht in der README.

Gleicher Fingerprint ergänzt weitere Trials; `--trials` gilt **je Worker und
Kanal**, nicht als historische Gesamtsumme. `--worker-timeout` begrenzt die
Workerwartephase je Kanal, nicht Datenaufbereitung, Gesamtanwendung oder finales
Training. Budget und Zeitlimit gehören nicht zum Fingerprint. Historische
SQLite-Studien werden nicht automatisch in Journals migriert.
Studien speichern Architekturen und Verluste, keine Trialgewichte oder aktuellen
R²-Trialattribute. Parallelität und fortgesetzte TPE-Suche sind nicht als bitgleich
reproduzierbar zugesichert.

Nach allen Kanalsuchen initialisiert und trainiert der Service sieben neue Netze
mit den jeweils gewählten Architekturen. Gewichte des besten Trials werden nicht
dauerhaft übernommen. **Dieses finale Training geschieht innerhalb der Pipeline,
während die GUI noch läuft.** Nach regulärem Abschluss bleibt das Fenster sichtbar.
Erst dessen Schließen lässt `start()` mit `SurrogateModel` beziehungsweise
`AutoencoderCollection` zurückkehren.

## 6. Speichern, auswerten und erneut darstellen

Anschließend führt `main_optimize` aus:

1. Gemeinsames Modellbundle aller sieben Netze schreiben.
2. Bundle erneut laden und Validierungsvorhersage bzw. Rekonstruktion mit
   `torch.testing.assert_close` gegen das Modell vor dem Speichern vergleichen.
3. `evaluation.pt` mit Referenzen, Vorhersagen/Rekonstruktionen und Metriken schreiben.
4. Sieben PNGs, sieben Einzel-PDFs und den einseitigen PDF-Bericht erzeugen.

Modellkategorien, Formate, Hashbindung und Exportgrenzen stehen in der
[Artefaktreferenz](DATEIEN_UND_SPEICHERORTE.md). Scheitert ein späterer Plotexport,
können Modell und Auswertung bereits vollständig vorliegen. Erneutes Plotten
benötigt diese Artefakte und Arial, kein neues Training und keine Original-CSVs.
`MODEL_FOLDER` in `main_plot_results.py` ist noch auf einen historischen Lauf
gesetzt; vor Nutzung einen mit aktuellen Loadern kompatiblen Lauf auswählen.

R² wird je Kanal berechnet, das Gesamt-R² ist das ungewichtete Mittel der sieben
Werte. Ein konstanter Referenzkanal ergibt `None` und damit kein definiertes
Gesamt-R². Die Auswertung umfasst außerdem normalisierten Gesamt-MSE und
physikalischen MAE/MSE je Kanal. Fehler verschiedener Einheiten werden nicht
zu einem physikalischen Gesamtfehler vermischt. Kombimodelle werden abschließend
auf vollständigen Kennfeldern bewertet, trotz spärlicher Trainingsstützstellen.

## 7. Gespeicherte Modelle anwenden

Eine Vorhersage-CLI oder ein automatischer CSV-Export existiert nicht.
Dieses API-Beispiel gilt für aktuelle Surrogatbundles:

~~~python
from pathlib import Path

from torch import Tensor

from src.modules.neuronal_networks.model_data_io.model_io import (
    load_bundle,
    predict_raw_geometry,
)


def predict(model_path: Path, raw_geometry: Tensor) -> Tensor:
    model, checkpoint = load_bundle(model_path)
    return predict_raw_geometry(model, checkpoint, raw_geometry)
~~~

`raw_geometry` hat Form `(N,14)` in der Reihenfolge
`checkpoint["feature_names"]`. `Airgap` und `Stator_Bore` sind unskaliert,
die zwölf Lambdas müssen bereits berechnet sein. `lambdas_from_geometry()` ist
die zentrale Umrechnung aus Abmessungen; die Vorhersagefunktion liest keine
MOT-Datei. Eingaben bleiben unverändert, Ergebnisse sind physikalisch skaliert.

CNN-Sammlungen werden mit `load_autoencoders()` geladen. `reconstruct()` erwartet
vorhandene Kennfelder in der gespeicherten Normalisierung. Das ist Rekonstruktion,
keine Geometrievorhersage.

## 8. GUI, Abbruch und Wiederanlauf

Die GUI läuft im Hauptthread, der Service in einem Thread, Rechenworker in Prozessen.
Ereignisse und Logs werden über getrennte Queues transportiert. Die Simulation
zeigt Punkte der aktuellen Kombination und abgeschlossene Kombinationen.
Optimierung zählt abgeschlossene Trials über alle Kanäle; finale Trainings
haben nur einen Statushinweis, keinen Epochenbalken.

Schließen während Arbeit fordert Abbruch an und versteckt das Fenster während
der Bereinigung. Simulationsworker werden bei Fehler/Abbruch über `kill_workers()`
beendet; Windows-Jobobjekte schützen den Worker-Kindprozessbaum. Suchworker werden
beendet und bereinigt. `PipelineCancelledError` wird an den Aufrufer weitergereicht;
regulärer Modell-/Auswertungsexport entfällt. Finale Trainings prüfen Abbruch
zwischen Kanälen, nicht pro Epoche. Es fehlt ein allgemeiner Rechnungstimeout
für dauerhaft hängende Motor-CAD-Aufrufe.

Eigene temporäre Laufordner, Queues und Listener werden bereinigt. Bereits extern
gestartetes eduVPN bleibt bestehen. Bei hartem Abbruch können Reste zurückbleiben.

Gültige CSV-Zeilen und Studien bleiben erhalten. Neue Standard-Simulationsläufe
verwenden neue Zielnamen und können kompatible Punkte übernehmen. Explizite native
Ziele sind über `csv_paths` fortführbar; adoptierte Dateien bleiben Quellen.
Neue Suchaufrufe ergänzen passende Studien, nehmen aber kein abgebrochenes
Einzeltraining wieder auf. Eine neue Pipelineinstanz ist erforderlich.

`main_mcad_sim` und `main_optimize` schreiben je Aufruf ein gemeinsames INFO-Log
für Hauptprozess und Worker. Ein vollständiger realer Ablauf sowie fachliche
Modellgüte wurden mit dieser Dokumentationsänderung nicht neu getestet.
