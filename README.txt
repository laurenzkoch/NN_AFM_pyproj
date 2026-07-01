# NN_AFM_pyproj

@JONAS
### Theoretisch sollte es funktionieren, wenn du src/executable_files/main_mcad_simulation.py ausführst.

### In src/config/config_process.py wird die Anzahl der Prozesse festgelegt, die für parallele Motor-CAD-Simulationen verwendet werden sollen. Falls weniger CPU-Kerne verfügbar sind, wird automatisch die maximal verfügbare Anzahl verwendet.
        Wenn es nicht so viele Kerne gibt, wie in der Konfiguration angegeben, wird die maximale Anzahl an verfügbaren Kernen verwendet.

### src/config/config_paths.py ...sind die Pfade angegeben

### src/executable_files/main_mcad_simulation.py ganz unten:

        with data_generating.MCADDataGenerator(start_eduvpn=True, init_mcad=False) as data_generator: ...

        start_eduvpn sollte = False sein, wenn du im Uni netzwerk bist; inity_mcad = False muss bleiben !!!

### base_first_data.mot ... Ist das Basis-file für die Simulation. Das muss im MotorCAD-Ordner liegen.


------der Rest ist KI generiert----------

Python-Projekt zur automatisierten Erzeugung, Speicherung, Auswertung und Visualisierung von Motor-CAD-Simulationsdaten für Axialflussmaschinen.

Das Projekt steuert Motor-CAD über die Python-Schnittstelle an, variiert definierte Eingangsparameter, prüft geometrische und elektrische Randbedingungen, führt magnetische Simulationen aus und speichert die Ergebnisse als CSV-Dateien. Zusätzlich sind Module für Datenmanagement, Plot-Erstellung und erste Machine-Learning-Auswertungen enthalten.

## Projektstruktur

NN_AFM_pyproj/
- data/
  - MCAD_Results/
  - plots/
- src/
  - config/
    - config_parameter_lists.py
    - config_paths.py
    - config_process.py
  - executable_files/
    - main.ipynb
    - main_mcad_simulation.py
    - main_plot_data.py
  - modules/
    - data_generating.py
    - data_managing.py
    - ml_model_trainig.py
    - plotting.py
    - process_managing.py
    - tests.py
- requirements.txt
- README.txt

## Überblick

Das Projekt besteht aus drei Hauptbereichen:

- Konfigurationsdateien für Pfade, Parameterlisten und Prozessanzahl
- ausführbare Skripte zur Simulation und Auswertung
- Module für Motor-CAD-Ansteuerung, Datenmanagement, Plot-Erstellung und Machine Learning

Die erzeugten Simulationsdaten werden im Ordner data/MCAD_Results gespeichert.

Die erzeugten Plots werden im Ordner data/plots gespeichert.

## Zentrale Dateien

### src/executable_files/main_mcad_simulation.py

Hauptskript zur Erzeugung von Motor-CAD-Simulationsdaten.

Das Skript definiert verschiedene Parameterstudien, startet bei Bedarf eduVPN, erzeugt einen Ergebnisordner mit Zeitstempel und führt die Simulationen aus.

Beim Start werden:

1. die Parameterstudien definiert,
2. ein Ergebnisordner mit aktuellem Zeitstempel erzeugt,
3. eduVPN gestartet, falls aktiviert,
4. Motor-CAD für die Simulationen initialisiert,
5. die Parameterwerte gesetzt,
6. Randbedingungen geprüft,
7. magnetische Berechnungen durchgeführt,
8. gültige Simulationsergebnisse als CSV gespeichert.

Die Ergebnisdateien werden gespeichert unter:

data/MCAD_Results/<Zeitstempel>/

Je nach Art der Parameterstudie werden Unterordner für einzelne oder kombinierte Parametervariationen erzeugt.

### src/executable_files/main_plot_data.py

Skript zur Visualisierung vorhandener CSV-Ergebnisdaten.

Das Skript liest Simulationsdaten ein, trennt Eingangs- und Ausgangsgrößen und erzeugt Scatterplots mit Matplotlib.

Die erzeugten Plots werden gespeichert unter:

data/plots/

### src/modules/data_generating.py

Enthält die zentrale Logik zur Motor-CAD-Datenerzeugung.

Aufgaben:

- Starten und Beenden von eduVPN
- Initialisieren von Motor-CAD
- Öffnen einer Motor-CAD-Basisdatei
- Setzen von Motor-CAD-Variablen
- Auslesen von Eingangs- und Ausgangsparametern
- Prüfen geometrischer und elektrischer Randbedingungen
- Ausführen magnetischer Berechnungen
- Parallele Simulation mehrerer Parameterkombinationen
- Schreiben der Simulationsergebnisse in CSV-Dateien

### src/modules/data_managing.py

Hilfsmodul für Datei- und Datenoperationen.

Aufgaben:

- Import von CSV-Dateien als Pandas-DataFrame
- Trennung von Eingangs- und Ausgangsdaten
- Aufteilen von Simulationswerten in mehrere Chunks
- Erstellen und Befüllen von CSV-Dateien
- Zusammenführen temporärer CSV-Dateien
- Erstellen und Löschen temporärer Motor-CAD-Dateien
- Löschen temporärer Ergebnisordner

CSV-Dateien werden mit Semikolon als Trennzeichen und Komma als Dezimaltrennzeichen verarbeitet.

### src/modules/process_managing.py

Hilfsmodul zum Umgang mit externen Prozessen.

Aufgaben:

- Starten externer Programme
- Prüfen, ob ein Prozess bereits läuft
- Warten auf Prozessstart
- Beenden gestarteter Prozesse
- Festlegen der maximal nutzbaren Worker-Anzahl anhand verfügbarer CPU-Kerne

Das Modul ist aktuell auf Windows ausgelegt.

### src/modules/plotting.py

Modul zur Erstellung von Diagrammen.

Unterstützt:

- 2D- und 3D-Scatterplots mit Plotly
- Matplotlib-Figuren
- Hinzufügen von Datenpunkten
- Hinzufügen von Textboxen
- Speichern von Plots als Bilddateien

### src/modules/ml_model_trainig.py

Modul für einfache Machine-Learning-Workflows.

Enthält Funktionen zum:

- Mischen und Aufteilen von Daten in Trainings- und Testdaten
- Trainieren eines übergebenen scikit-learn-Modells
- Bewerten des Modells mit R²-Score und Mean Squared Error

Das Modul ist für erste Regressions- und Modellvergleichsexperimente vorbereitet.

## Konfiguration

### src/config/config_parameter_lists.py

Enthält die verwendeten Motor-CAD-Parameter.

Die Datei definiert:

- Eingangsparameter, die aus Motor-CAD ausgelesen und gespeichert werden
- Ausgangsparameter, die nach der Simulation ausgewertet werden

Aktuelle Ausgangsgrößen:

- ShaftTorque
- SystemEfficiency
- ConductorLoss
- MagnetLoss
- StatorIronLoss_Total
- RotorIronLoss_Total
- WedgeLoss

### src/config/config_paths.py

Zentrale Pfadkonfiguration.

Enthält Pfade für:

- Motor-CAD-Daten
- Simulationsergebnisse
- Plot-Ausgaben
- eduVPN-Executable

Wichtige Pfade müssen lokal angepasst werden, insbesondere:

- PATH_MCAD_DATA
- PATH_eduVPN_EXE

Im Motor-CAD-Datenordner muss die Basisdatei liegen:

base_first_data.mot

Diese Datei dient als Ausgangspunkt für die Parameterstudien.

### src/config/config_process.py

Konfiguration der parallelen Verarbeitung.

Hier wird festgelegt, wie viele Prozesse maximal für parallele Motor-CAD-Simulationen verwendet werden sollen.

Falls weniger CPU-Kerne verfügbar sind, wird automatisch die maximal verfügbare Anzahl verwendet.

## Voraussetzungen

- Windows
- Python 3.14
- Motor-CAD
- gültige Motor-CAD-Lizenz
- eduVPN, falls für den Lizenzzugriff erforderlich
- Python-Pakete aus requirements.txt

## Python-Abhängigkeiten

Die wichtigsten Abhängigkeiten sind:

- pandas
- matplotlib
- ansys-motorcad-core

Zusätzlich werden im Projekt unter anderem verwendet:

- numpy
- plotly
- scikit-learn

## Konfiguration vor dem Start

Vor dem ersten Ausführen sollten die Pfade in folgender Datei geprüft und angepasst werden:

src/config/config_paths.py

Besonders wichtig sind:

- der Pfad zum Motor-CAD-Datenordner
- der Pfad zur eduVPN-Executable

Im Motor-CAD-Datenordner muss die Basisdatei liegen:

base_first_data.mot

## Simulation ausführen

Die Motor-CAD-Datenerzeugung wird über folgendes Skript gestartet:

src/executable_files/main_mcad_simulation.py

Nach erfolgreichem Start werden die Simulationsergebnisse automatisch als CSV-Dateien gespeichert.

Die Ergebnisdateien liegen anschließend unter:

data/MCAD_Results/<Zeitstempel>/

## Plots erzeugen

Zur Visualisierung vorhandener Simulationsergebnisse wird folgendes Skript verwendet:

src/executable_files/main_plot_data.py

Die erzeugten Plotdateien werden im Plot-Ordner gespeichert:

data/plots/

## CSV-Format

Die erzeugten CSV-Dateien verwenden:

- Semikolon als Trennzeichen
- Komma als Dezimaltrennzeichen
- UTF-8-Encoding

Die Spalten bestehen aus:

1. Eingangsparametern
2. Ausgangsparametern

## Randbedingungen der Simulation

Vor jeder Motor-CAD-Berechnung werden Plausibilitätsprüfungen durchgeführt.

Ungültige Designs werden übersprungen und nicht in die Ergebnisdatei geschrieben.

Geprüft werden unter anderem:

- Statorbohrung kleiner als Statordurchmesser
- Rotor- und Magnetgeometrie
- ausreichende Motorlänge
- Slot-Tiefe kleiner als Statorblechlänge
- Wickelbarkeit anhand Slotzahl, Polzahl und Phasenzahl
- zulässige Slotbreite
- positive aktive Magnetlänge

Wenn eine Bedingung verletzt wird, wird eine Warnung ausgegeben und der Designpunkt nicht simuliert.

## Parallele Verarbeitung

Die Simulationen werden parallelisiert.

Die Anzahl der Worker wird in folgender Datei konfiguriert:

src/config/config_process.py

Temporäre Zwischenergebnisse werden während der Simulation in temporäre CSV-Dateien geschrieben und anschließend zu einer finalen Ergebnisdatei zusammengeführt.

## Machine Learning

Das Projekt enthält ein vorbereitetes Modul für Machine-Learning-Auswertungen:

src/modules/ml_model_trainig.py

Typischer Ablauf:

1. CSV-Datei laden
2. Eingangs- und Ausgangsdaten trennen
3. Modell trainieren
4. Modell bewerten

Das Modul ist für erste Regressions- und Modellvergleichsexperimente vorbereitet.

## Tests

Es existiert ein erstes Testmodul:

src/modules/tests.py

Die Tests beziehen sich teilweise noch auf ältere archivierte Module. Vor produktiver Nutzung sollten die Tests an die aktuelle Modulstruktur angepasst werden.

## Hinweise

- Das Projekt ist aktuell stark auf eine Windows-Umgebung ausgelegt.
- Motor-CAD und eduVPN müssen lokal korrekt installiert und erreichbar sein.
- Pfade zu externen Programmen und Motor-CAD-Dateien müssen in config_paths.py angepasst werden.
- Parallele Motor-CAD-Prozesse können hohe CPU- und Lizenzressourcen benötigen.
- Ungültige Designs werden automatisch übersprungen.

## Aktueller Funktionsumfang

- Motor-CAD-Ansteuerung
- automatische Parameterstudien
- geometrische und elektrische Plausibilitätsprüfung
- parallele Simulation
- CSV-Export
- Plot-Erzeugung
- vorbereitete Machine-Learning-Auswertung

