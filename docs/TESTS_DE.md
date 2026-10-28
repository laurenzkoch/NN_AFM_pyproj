# Automatische Tests und Prüfnachweise

Stand: 28.10.2026, statischer Abgleich mit den vorhandenen Testdateien.

Die Tests verwenden echten Projektcode für Datenaufbereitung, Metriken, Modelle,
Training, Optuna-Journals, Simulationssteuerung und Logging. Motor-CAD-Start und
Tk-Fenster sind in der Testsitzung gesperrt oder an den Testgrenzen ersetzt.
Das belegt keine ausgeführte Motor-CAD-Rechnung und keine geprüfte GUI-Darstellung.

Installation und Abhängigkeiten stehen in der [README](../README.txt),
fachliche Abläufe im [Workflow](WORKFLOW_DE.md), Dateien und Aufbewahrung in der
[Artefaktreferenz](DATEIEN_UND_SPEICHERORTE.md).

## 1. Ausführung

Alle Befehle aus der Projektwurzel mit der Projektumgebung ausführen:

~~~powershell
conda activate Env_NN_AFM
Set-Location D:\Python_Projects\NN_AFM_pyproj
python -m pytest
~~~

`pytest.ini` setzt `testpaths=tests`, `pythonpath=.`, `-ra` und strikte Marker.
`pytest` und Ruff sind derzeit in `requirements.txt` enthalten. Eine separate
`requirements-dev.txt` existiert nicht. Windows-Prozessschutztests benötigen
Windows und `pywin32`; auf anderen Betriebssystemen wird der entsprechende Test
übersprungen.

Erst sammeln, ohne Testfunktionen auszuführen:

~~~powershell
python -m pytest --collect-only -q
~~~

Auch beim Sammeln werden Testmodule und `conftest.py` importiert. Eine dort
ermittelte Testfallzahl ist nur zusammen mit Code-/Umgebungsstand aussagekräftig.
Zwölf Dateien `test_*.py` sind nicht gleich zwölf Testfälle; Parametrisierungen
erzeugen weitere Fälle. Eine feste historische Angabe wie „58 Tests“ wird daher
nicht als aktueller Prüfnachweis geführt.

## 2. Testmatrix

Dateipfade beziehen sich auf [tests](../tests).

| Modul | Vorhandene Prüfinhalte | Grenze |
|---|---|---|
| [test_metadata.py](../tests/test_metadata.py) | API-Dry-Run, Adoption/Sidecar-Roundtrip, CSV-Hashmanipulation, ungültige/leere CSVs, Bestätigung, native/adoptierte Ziele, Fingerprints, Version-2-Setup und Ausschluss von Python-Dateihashes | Synthetische Metadaten; keine echten MOT-/EXE-Hashes erforderlich |
| [test_data_and_metrics.py](../tests/test_data_and_metrics.py) | Siebenkanaliger MSE/R², konstante Referenz, NaN/Inf/Formfehler, Training-only-Skalierung, unabhängige Tests, unveränderte CSVs, Nullnenner, fehlende/doppelte Punkte | Synthetische CSVs; deren Drehzahlerzeugung enthält noch nicht die aktuelle pi-Definition |
| [test_axis_alignment.py](../tests/test_axis_alignment.py) | Rundungsabgleich für Drehzahl und Stromdichte, Erhalt rasterfremder und mehrdeutiger Werte | Einzelne Helfer; Drehzahltests verwenden noch Quotient ohne pi |
| [test_lambda_geometry.py](../tests/test_lambda_geometry.py) | Zwölf Lambda-Verhältnisse als Geometrie-Roundtrip, verschiedene Pol-/Nutzahlen, CSV-Rundung und unabhängige Radiusprüfung | Algebraische Umrechnung, keine Motor-CAD- oder Fachvalidierung |
| [test_files_and_grid.py](../tests/test_files_and_grid.py) | Private MOT-Kopien, Namen/Inhalte, Bereinigung bei Kopierfehler, ungleiche Rasterachsen, ungültige Größen, CPU-Limit | Keine Motor-CAD-Instanz |
| [test_models.py](../tests/test_models.py) | Sieben Submodelle, Bundle-Roundtrip, ungültige Bundlefelder, unveränderter Vorhersageinput, unabhängige Kanalnetze, kurzes CPU-Training, Trainerparameter, Auswertungsexport mit/ohne Test | Synthetische Tensoren; keine Modellgüte auf realen Maschinen |
| [test_autoencoder_collection.py](../tests/test_autoencoder_collection.py) | CNN-Training/Rekonstruktion, Sammlungs-Roundtrip, Auswertung, Kombitraining mit eingefrorenen CNNs, Gewichts-/Herkunfts-/Trainingsdauererhalt | Kleine synthetische Netze; kein vollständiger CNN-/Kombi-CLI-Lauf |
| [test_optuna_worker.py](../tests/test_optuna_worker.py) | Zwei echte kurze Trials im Worker und vier Trials über zwei spawn-Prozesse, gemeinsames Journal, Architekturattribute und Fortschritt | Direkter Workeraufruf bzw. Workerprozesse; keine vollständige GUI-/Siebenkanal-Pipeline |
| [test_logging.py](../tests/test_logging.py) | Echter spawn-Worker über [logging_worker.py](../tests/logging_worker.py), INFO/ERROR, Phasen vor/während/nach Pipeline, eine Datei mit eindeutigen Markern | Kein Motor-CAD und keine GUI |
| [test_simulation_control.py](../tests/test_simulation_control.py) | Punktzählung, ungültige Kombinationen, einmaliger Start, Queuebereinigung, eigener Tempordner, sequentielle CSV-Erstellung, Fortschrittsreset, Pool je Batch, Future-Fehler | Pool/GUI an ausgewählten Grenzen ersetzt; v1-Punktzählung belegt keine Generatorunterstützung |
| [test_cancellation.py](../tests/test_cancellation.py) | Abbruchsignal, Poolbeendigung, keine weiteren Batches und Bereinigung gestarteter Suchworker | Ersetzte Prozess-/Poolgrenzen; kein Abbruch einer realen Motor-CAD-Rechnung |
| [test_worker_job.py](../tests/test_worker_job.py) | Echter Windows-Worker mit Kindprozess; Kind endet nach Workerende, unabhängiger Prozess bleibt | Windows/pywin32 erforderlich; keine Motor-CAD-Instanz |

Die Matrix beschreibt Testcode, keinen in dieser Änderung ausgeführten erfolgreichen
Testlauf und keinen Coverage-Prozentsatz.

## 3. Marker und gezielte Auswahl

Registriert sind:

| Marker | Tatsächliche Verwendung |
|---|---|
| `training` | Kleine CPU-Trainings in Modell-/CNN-Tests und beide Optuna-Workertests; der parallele Journaltest startet dabei auch Prozesse |
| `subprocess` | Loggingtest mit echtem spawn-Worker |
| Kein solcher Marker | Der Windows-Jobobjekttest startet ebenfalls echte Prozesse |

Daher bedeutet `-m "not training and not subprocess"` derzeit **nicht**, dass
kein Unterprozess startet. Eine explizite Auswahl ohne diese Trainings- und
Prozesstests lautet:

~~~powershell
python -m pytest -m "not training and not subprocess" --ignore=tests/test_worker_job.py
~~~

Weitere Eingrenzung:

~~~powershell
python -m pytest -x -vv
python -m pytest tests/test_metadata.py tests/test_simulation_control.py -vv
python -m pytest tests/test_axis_alignment.py tests/test_lambda_geometry.py -vv
python -m pytest -m training -vv
python -m pytest -m subprocess -vv
python -m pytest tests/test_worker_job.py -vv
~~~

`-m subprocess` allein erfasst weder den Windows-Jobtest noch den parallelen
Journaltest. Vorschlag für eine spätere Codeänderung: beide passend markieren;
die Dokumentation beschreibt bis dahin die vorhandenen Marker.

## 4. Isolation und Ressourcen

[conftest.py](../tests/conftest.py) sperrt `MotorCAD` und `set_motorcad_exe` auch
bei installiertem SDK. Fehlt das SDK, wird nur dessen Importgrenze ersetzt.
`tkinter.Tk` ist gesperrt; bei fehlender Tk-Runtime wird eine Importgrenze
bereitgestellt. Echte externe Starts sollen in diesen Tests fehlschlagen.

Dateierzeugende Tests verwenden temporäre Pfade; Produktionspfade in getesteten
Modulen werden nötigenfalls mit `monkeypatch` umgestellt. Bestehende
Simulations-/Trainingsdateien und deren Sidecars sind keine Testeingaben.
Logginghandler werden gesichert und wiederhergestellt; Queues, Prozesse und
Handles haben eigene Bereinigung.

Die Sitzungssperren des Hauptprozesses sind kein allgemeiner Schutz sämtlicher
beliebiger Unterprozesse. Die vorhandenen spawn-Tests haben ausdrücklich gewählte
Ziele für Training, Logging oder Prozessschutz. Neue echte Motor-CAD-/GUI-
Integrationstests gehören in eine getrennte, ausdrücklich gestartete Testsuite,
die diese Sperren nicht versehentlich erbt.

Ein Import-/DLL-/BLAS-Fehler ist von einer fehlgeschlagenen fachlichen Assertion
zu unterscheiden. Für einen belastbaren Befund die benutzte Umgebung mit erfassen;
eine Paketinstallation allein validiert deren Laufzeitverhalten nicht.

## 5. Quellcodeprüfung

Lesende Prüfungen ohne automatische Korrekturen:

~~~powershell
python -m ruff check src tests --select E,F,I,N,UP,ANN,G,LOG --no-cache
python -m ruff format src tests --check --no-cache
~~~

`pyproject.toml` legt das Ruff-Ziel `py314` fest. Die expliziten Regelgruppen
gehören zum obigen Aufruf, nicht zu einer umfassenden Regelkonfiguration im
Projekt. Annotationen und Tests ersetzen keine vollständige statische Typprüfung.

Formatierung von Python-Dateien verändert den aktuellen **Simulationsfingerprint
nicht**. Der **Optimierungsfingerprint** enthält dagegen ausgewählte Quellcode-
Hashes; Formatierung dieser Dateien kann eine neue Studie bewirken. Historische
Prüftexte sind kein Beleg für heutige Ruff-Befunde.

## 6. Offene Testgrenzen

- `_recalculate_data()` und Maschinengenerator verwenden
  `ShaftSpeed/(Stator_Lam_Dia*pi)`. `_align_speed_grid()` sowie Achsentests und
  CSV-Fixtures verwenden noch den früheren Quotienten ohne pi. Einzelne grüne
  Helfertests würden die fachliche Konsistenz dieses Gesamtwegs nicht belegen.
- Ein durchgängiger Test aller `main_optimize`-Schritte einschließlich automatischer
  Testauswahl, sieben Kanalsuchen, finalem Training, GUI und Export ist nicht vorhanden.
- Bundle-/Auswertungstests ersetzen keine automatisierte Prüfung der erzeugten
  Einzelplots und PDF-Berichte oder deren fachlicher Darstellung.
- Metadatenmigration und Daten-/Modellformatmigration sind verschiedene Aufgaben.
  Die vorhandenen Metadatentests sind kein Nachweis einer vollständigen
  Version-1-zu-2-Migrationssuite oder einer Migration alter CSV-/Modellbestände.
- Ein Abbruch im finalen Training wird zwischen Kanälen geprüft; ein Test zur
  sofortigen Reaktion innerhalb einer Epoche würde eine noch fehlende
  Implementierung voraussetzen.

Diese Grenzen sind statisch festgestellt. Sie behaupten keinen neu ausgeführten
oder fehlgeschlagenen Test.

## 7. Weiterhin notwendige reale Integration

1. Mit gültiger Lizenz, geprüftem MOT-/Materialsetup und aktuellem Schema mindestens
   zwei Simulationskombinationen rechnen; mindestens eine muss mehr als einen
   Batch umfassen. Bei vier Workern und 125 Rasterpunkten sind das mehr als 500
   Anforderungen.
2. Frische Worker-/Motor-CAD-Instanzen pro Batch, private Dateien, CSV-/Sidecar-
   Zeitpunkt, Wiederverwendung und beide GUI-Fortschritte prüfen.
3. Fehler und kontrollierten Abbruch in einer separaten Umgebung prüfen:
   Worker-/Kindprozessbereinigung, Tempordner und erhaltene Ergebnisse.
   Ein allgemeiner Timeout für hängendes Motor-CAD ist nicht implementiert.
4. Referenz-, CNN- und Kombimodus mit gültigen Daten durchführen: automatische
   Testauswahl, Herkunfts-/Überlappungsprüfung, Journalfortsetzung, finales
   Training, Bundle-Roundtrip, Auswertung und erneuter Berichtsexport.
5. Modellgüte auf unabhängigen Maschinen beurteilen; insbesondere CNN-
   Rekonstruktionsgüte von Geometrievorhersage unterscheiden und Kombimodelle
   auf vollständigen Kennfeldern bewerten.

Diese Punkte wurden für die Dokumentationsänderung nicht ausgeführt.

## 8. Prüfnachweis führen

Für künftige Testläufe hier oder in einem tatsächlich angelegten Prüfprotokoll
festhalten:

| Angabe | Erforderlicher Inhalt |
|---|---|
| Zeitpunkt und Stand | Datum, Commit sowie relevante uncommittete Änderungen |
| Umgebung | Interpreterpfad/-version, Betriebssystem, benötigte Paketversionen |
| Aufruf | Exakter Befehl einschließlich Marker, Dateiauswahl und Optionen |
| Ergebnis | Exitcode, bestanden/fehlgeschlagen/übersprungen und Grund der Auswahl |
| Umfang | Unit-/Prozess-/Trainingstest oder reale Integration; eingesetzte Ersatzgrenzen |
| Artefakte | Logs, gegebenenfalls JUnit-Datei und zugeordnete Daten-/Modellpfade |
| Grenzen | Nicht geprüfte externe Komponenten und verbleibende Befunde |

Optionaler JUnit-Export:

~~~powershell
New-Item -ItemType Directory -Force .\test-results | Out-Null
python -m pytest --junitxml=.\test-results\pytest.xml
~~~

Eine `docs/PRUEFPROTOKOLL.md` wird derzeit nicht vorausgesetzt oder verlinkt.
Für den Abgleich vom 28.10.2026 wurden Dokumentation und Quellcode lesend geprüft;
es gab keinen neuen pytest-, Ruff-, Trainings- oder Motor-CAD-Lauf. Erfolgreich
geprüft wurden 34 lokale Verweisziele, alle neun dokumentierten CLI-Optionen,
drei Optimierungsbeispiele, die Übereinstimmung der Matrix mit zwölf Testmodulen
und die Syntax des Python-API-Beispiels. Ein erneuter AST-Abgleich bestätigte die
Python-Syntax sowie das Schema mit 39 Spalten, 14 Features und sieben Kanälen.
Der Dateihashvergleich begrenzt die Änderungen auf README und die drei docs-Dateien.
