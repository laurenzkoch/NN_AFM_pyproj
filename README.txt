NN_AFM_pyproj
=============

Automatisierte Motor-CAD-Simulationen und Surrogatmodelle für Axialflussmaschinen
Dokumentationsstand: 16.10.2026

Das Projekt erzeugt Maschinengeometrien und Betriebspunktkennfelder, berechnet
sie mit Ansys Motor-CAD und speichert erfolgreiche Ergebnisse als CSV. Aus
vollständigen Kennfeldern werden PyTorch-Tensoren aufgebaut. Optuna sucht eine
MLP-Architektur zur Vorhersage ganzer Kennfelder aus der Maschinengeometrie.

Aktuelle Abläufe:

    main_mcad_sim.py
        -> Folge von (Basisdesigns, Lambda-Designs)-Kombinationen
        -> Maschinen- und Betriebspunktgenerator je Kombination
        -> parallele elektromagnetische Motor-CAD-Simulation
        -> data/MCAD_Results/*.csv

    main_optimize.py
        -> CSV-Dateien in data/Training_data
        -> Bereinigung, Tensoren, Train-/Validierungs-/Testaufteilung
        -> Optuna-Suche mit sieben MLP-Teilmodellen je Versuch
        -> Validierungs-MSE, R² und experimentbezogene SQLite-Studie
        -> finales Training der besten Architektur und externe Testauswertung
        -> Modellbundle mit Gewichten, Skalierung und Herkunftsdaten

Beide Abläufe verwenden die gemeinsame Pipeline mit Tkinter-Fortschrittsfenster
und Logging. Die Optimierung benötigt keine laufende Motor-CAD-Anwendung.
Die Simulationsanbindung ist vom gemeinsamen Pipeline-Modul getrennt.
main_optimize.py besitzt Kommandozeilenoptionen; Simulationsumfang und
Trainings-/Suchraumvorgaben werden weiterhin im Code geändert. adopt_legacy_csv.py
ist ein zusätzlicher CLI-Einstieg zur ausdrücklich bestätigten Übernahme alter
Ergebnisdateien als Cache-Quelle.

Die zuvor dokumentierten Fehler bei Dienstaufruf, Studienanlage, Workerstart,
MSE-Rückgabe, Parameterweitergabe und Queue-/Batch-Verarbeitung sind im aktuellen
Quellcode korrigiert. Diese README wurde statisch mit dem Projekt abgeglichen;
Simulation, Optimierung und Datenübernahme wurden dafür nicht gestartet.


1. Projektstruktur
------------------

NN_AFM_pyproj/
|-- README.txt
|-- environment.yml                 Conda-Umgebung Env_NN_AFM
|-- requirements.txt                Gemeinsame Python-Paketliste
|-- pyproject.toml                  Ruff-Zielversion Python 3.14
|-- check_logging.py                Prüfskript für gemeinsames Prozess-Logging
|-- data/
|   |-- MCAD_Results/                Simulations-CSV; neue Läufe mit .meta.json
|   |-- Training_data/              Explizit ausgewählte Trainings-/Test-CSV
|   |-- optimization/               Vorgesehen: Studie je Experimentfingerprint
|   |-- logs/                       Pipeline- und Simulationsprotokolle
|   |-- plots/                      Vorhandene Diagramme früherer Versuche
|   '-- trained_nn/                 Modellbundles und ältere Trainingsartefakte
'-- src/
    |-- config/
    |   |-- config_paths.py          Datei-, Programm- und Ausgabepfade
    |   |-- config_logging.py        Queue-basiertes Logging
    |   '-- config_mcad_parameters.py
    |                               CSV-Spalten, Merkmale, Raster und Bereiche
    |-- executable_files/
    |   |-- main_mcad_sim.py         Simulation starten
    |   |-- main_optimize.py         MLP-Architektur mit Optuna optimieren
    |   |-- machine_test.py          Geometrie-Randbedingungen prüfen
    |   '-- adopt_legacy_csv.py      Alte CSV prüfen und Herkunftsannahme erfassen
    '-- modules/
        |-- Pipeline_and_GUI/       Gemeinsame Pipeline, Ereignisse, Tkinter-GUI
        |-- data_managment/         CSV, Dateien, Experimente, Simulationsmetadaten
        |-- simulation/             Simulationspipeline und Prozesspool
        |   '-- generators/         Maschinenentwürfe und Betriebspunktraster
        |-- process/                Windows-Prozesse, eduVPN, CPU-Ressourcen
        |   '-- motor_cad_processes/ Motor-CAD-Instanzen und Randbedingungen
        |-- neuronal_networks/      Daten, Training, Surrogatmodelle, Modell-I/O
        |   |-- NN_types/           MLP und Conv3dAutoencoder
        |   '-- optimization/       Optuna-Worker und OptimizationPipline
        '-- plotting/               Matplotlib-, Plotly- und Heatmap-Helfer

Die Schreibweisen data_managment, neuronal_networks und optimization_pipline.py
entsprechen dem Code. main_nn.py, main_create_plots.py und der frühere
scikit-learn-MLP sind im aktuellen Arbeitsstand entfernt.


2. Umgebung einrichten
----------------------

Die Simulation und Prozesssteuerung sind auf Windows ausgelegt. Die konfigurierte
PyCharm-Umgebung ist Env_NN_AFM mit Python 3.14.4. Die Versionsvorgaben in den
Umgebungsdateien entsprechen dem am 12.10.2026 geprüften Paketstand.
Sie sind keine vollständige Sperrdatei aller indirekten Abhängigkeiten.

Aus der Projektwurzel in einer für Conda eingerichteten PowerShell:

    conda env create -f environment.yml
    conda activate Env_NN_AFM

Eine bestehende Umgebung aktualisieren:

    conda env update -n Env_NN_AFM -f environment.yml

Conda installiert Python, Tk, Zeitzonendaten sowie die aufgeführten numerischen,
Plot- und HTTP-Pakete. Der pip-Abschnitt bindet requirements.txt ein und ergänzt
unter anderem PyTorch, PyMotorCAD, Optuna und seaborn. Doppelt aufgeführte
Python-Pakete haben in beiden Dateien dieselben Versionsvorgaben.

Alternativ in einer bereits eingerichteten Python-3.14.4-Umgebung mit Tkinter:

    python -m pip install -r requirements.txt

Python, Tkinter, Motor-CAD und eduVPN werden nicht über requirements.txt
installiert. sqlite3 für die Optuna-Datenbank gehört zur Python-Standardbibliothek.
Das Python-Paket tzdata liefert unter Windows die Daten für zoneinfo.

Pakete und Aufgaben:

    numpy, pandas          Numerik, CSV-Verarbeitung und Tensorvorbereitung
    scipy                  Latin-Hypercube-Sampling und Nut-/Polpaar-Zuordnung
    torch                  MLP, Autoencoder, CPU-Training und Tensoren
    optuna                 Architektur-Suche und persistente SQLite-Studie
    ansys-motorcad-core    Motor-CAD-Schnittstelle
    matplotlib, plotly,
    seaborn                Diagramm- und Heatmap-Helfer
    requests, idna,
    charset-normalizer     Explizit gepflegter HTTP-Unterbau für PyMotorCAD
    tzdata                 Zeitzone Europe/Berlin unter Windows

scikit-learn wird vom aktuellen Quellcode nicht importiert und ist deshalb kein
Eintrag mehr in den Umgebungsdateien. Historische Diagramme können weiterhin
Ergebnisse früherer scikit-learn-Versuche enthalten.

Importpfade und Startverzeichnis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Die aktuellen Projektmodule verwenden src.config/src.modules. Alle folgenden
Startbefehle werden aus der Projektwurzel ausgeführt; dafür ist keine zusätzliche
PYTHONPATH-Einstellung erforderlich. Bei direktem Start einer einzelnen Datei aus
einem anderen Arbeitsverzeichnis muss die Projektwurzel im Suchpfad liegen.
Die PyCharm-Konfiguration markiert Projektwurzel und src als Quellverzeichnisse
und verwendet Env_NN_AFM. Die Studiendatenbank wird über einen absoluten,
projektbezogenen Pfad angebunden; optimization.db im Arbeitsverzeichnis wird
vom aktuellen Optimierer nicht verwendet.

Externe Voraussetzungen für Simulationen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Benötigt werden eine lokale Motor-CAD-Installation mit verfügbarer Lizenz, eine
passende .mot-Basisdatei und ein funktionierender Zugang zum Lizenzserver.
Gegebenenfalls wird eduVPN für das Lizenznetz benötigt. Diese Anwendungen und
die Basisdatei sind nicht Teil der Python-Umgebung; die Basisdatei liegt außerhalb
des Projekts.


3. Zentrale Konfiguration
-------------------------

src/config/config_paths.py enthält aktuell:

    PATH_MCAD_DATA        D:\MotorCAD
    PATH_MCAD_BASEFILE    D:\MotorCAD\base_first_data.mot
    PATH_MCAD_EXE         C:\Program Files\AnsysMotorCAD\v252\Motor-CAD_2025_2_1.exe
    PATH_eduVPN_EXE       C:\Program Files\eduVPN\Core\eduVPN.Client.exe

Diese Werte an die lokale Installation anpassen. Datenpfade werden relativ zum
Projekt ermittelt:

    PATH_DATA_MCAD_RESULTS    data/MCAD_Results
    PATH_TRAINING_DATA        data/Training_data
    PATH_LOGS                 data/logs
    PATH_DATA_PLOTS           data/plots
    PATH_DATA_TRAINED_NN      data/trained_nn

PATH_TEMP_FILES bezeichnet temp_files. Jeder Simulationslauf erhält darunter
einen eigenen Ordner D:\MotorCAD\temp_files\simulation_run_<ID>. Darin bekommt
jede Motor-CAD-Instanz ein eigenes Unterverzeichnis mit einer Kopie der Basisdatei
unter deren ursprünglichem Dateinamen. PATH_TEMP_DATA und
PATH_DATA_MLP_RESEARCH_BASIC sind definiert, aber keine regulären Ausgabeziele
der aktuellen Startskripte.

src/config/config_mcad_parameters.py legt CSV-Spalten, MACHINE_DEFINITION,
AXIS_LIST, MAP_SIZE, DEFAULT_VALUES und die Samplingbereiche fest:

    MAP_SIZE              (5, 5, 5), also 125 Betriebspunkte je Geometrie
    SLOT_NUMBER           3 ... 30, Kandidaten in Schritten von 3
    POLE_NUMBER           2 ... 20, Kandidaten in Schritten von 2
    AIRGAP                0,25 ... 5
    STATOR_BORE            11 ... 100
    SHAFT_SPEED            1 ... 5000
    D_CURRENT_DENSITY      -50 ... 0
    Q_CURRENT_DENSITY      5 ... 100

    LAMBDA_1_1            0,2 ... 0,8
    LAMBDA_1_2            0,5 ... 0,9
    LAMBDA_1_3            0,2 ... 1,0
    LAMBDA_2_1            0,3 ... 0,7
    LAMBDA_2_2            0,1 ... 1,0
    LAMBDA_2_3            0,6 ... 1,0
    LAMBDA_3_1            0,0 ... 0,95
    LAMBDA_3_2            0,1 ... 0,4
    LAMBDA_3_3            0,2 ... 1,0

Die neun Lambda-Werte beschreiben Geometrieverhältnisse. Maßgeblich sind die
Berechnung in AFMMachineGenerator._calculate_geometric und die Rekonstruktion
in DataLoader._recalculate_data; einzelne ältere Codekommentare weichen davon ab.
DEFAULT_VALUES wird nicht vollständig in Motor-CAD gesetzt. Nicht vom Generator
überschriebene Wicklungs-, Material- und Maschinenwerte stammen aus der Basisdatei.
Die DEFAULT_VALUES-Schlüssel bestimmen unter anderem die ausgelesenen Prüfwerte.

Simulationsherkunft: config_paths.py enthält zusätzlich
MCAD_DEPENDENCY_REVISION="materials_and_solver_2026_10_13" und
MCAD_DEPENDENCY_FILES=(). Externe Material-/Solverdateien können in der Dateiliste
für eine SHA-256-Prüfung erfasst werden. Nicht aufgeführte externe Abhängigkeiten
werden nur durch die manuell gepflegte Revision repräsentiert; diese bei einer
entsprechenden Änderung aktualisieren.


4. Motor-CAD-Simulation
-----------------------

Einstieg: src/executable_files/main_mcad_sim.py

    python -m src.executable_files.main_mcad_sim

Aktueller SimulationPipeline-Aufruf:

    design_combinations=[(1, 2), (1, 3)]
    start_eduvpn=False
    gen_type="v2"
    num_workers=4

SimulationPipeline baut aus den Kombinationen SimulationJob-Objekte auf.
SimulationSeries berechnet die Aufträge nacheinander; jede Kombination erhält
eine eigene Ergebnis-CSV. Die Liste darf nicht leer sein und muss aus eindeutigen
(B, L)-Tupeln mit positiven ganzen Zahlen bestehen. B darf die Zahl verfügbarer
Nut-/Polpaare nicht übersteigen.

MAP_SIZE wird zentral konfiguriert. num_workers wird auf verfügbare CPUs und je
Auftrag auf die Punktzahl begrenzt. Eine Batch umfasst Workerzahl * 125 Punkte.
Jeder Batch verwendet einen eigenen Prozesspool; jeder Worker öffnet eine
temporäre Kopie des Basismodells in Motor-CAD.

B = num_base_designs, L = num_lam_designs, P = Produkt der MAP_SIZE-Werte:

    Variante    Geometrien      Vorgesehene Betriebspunkte
    v1          B * L^3         B * L^3 * P
    v2          B * L           B * L   * P

v1 kombiniert drei gesampelte Lambda-Gruppen. v2 sampelt alle neun Lambda-Werte
gemeinsam. Beide verwenden Latin-Hypercube-Sampling mit festen Seeds. Nut-/Polpaare
werden aus zulässigen Kombinationen gewählt und gemeinsam den Samplingzielen
zugeordnet. Das Betriebspunktkennfeld ist ein äquidistantes kartesisches Raster.
Die aktuelle v2-Startkonfiguration umfasst zwei Aufträge mit 250 und 375 Punkten,
also insgesamt 625 generierte Anforderungen. Bei gleichen bekannten Punkten
können Ergebnisse kopiert oder wiederverwendet werden.

Die gemeinsame Pipeline startet den Dienst automatisch und zeigt den Fortschritt
im Tkinter-Fenster. Es gibt keinen Startknopf und kein autostart-Argument mehr.
Die Oberfläche zeigt den Punktfortschritt des aktuellen Auftrags und einen
zusätzlichen Gesamtfortschritt für abgeschlossene Kombinationen. Der Punktwert
umfasst simulierte, verworfene, vorhandene und kopierte Anforderungen und wird
für jede neue Kombination zurückgesetzt.

Schließen während eines Laufs blendet das Fenster aus und hält die Ereignisschleife
bis zum Dienstende aktiv; es bricht die Arbeit nicht ab. Die Queue wird auch bei
vorübergehend fehlenden Ereignissen weiter abgefragt. Nach terminalem Ereignis,
beendetem Dienst-Thread und geleerter Queue wird der Abschluss markiert. Wurde
Schließen bereits angefordert, kehrt die Pipeline dann zurück. Andernfalls nach
dem Abschluss das Fenster schließen, damit nachgelagerte Schritte weiterlaufen.

CSV-Ausgabe und Wiederverwendung
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Die Zieldatei erhält standardmäßig den Namen:

    Results_EMagnetic_simulation_<B>_<L>_<MAP_SIZE[0]>_GenType-<v1|v2>_<Datum_Zeit_Zone>_<Lauf-ID>.csv

Optional legt csv_paths genau einen Zielpfad je Kombination fest. Relative Pfade
werden unter data/MCAD_Results aufgelöst; absolute Pfade sind ebenfalls möglich.
Alle Ziele müssen verschieden sein und auf .csv enden. Eine bestehende Datei
wird nur mit exakt aktuellem Fingerprint und passendem Inhalt fortgeführt.
Neue Ausgaben erhalten zusätzlich eine Begleitdatei <Dateiname>.csv.meta.json.
Die Simulation liest kompatible CSV-Dateien direkt aus dem Ergebnisordner, ohne
Unterordner; ein ausdrücklich gewähltes vorhandenes Ziel außerhalb dieses Ordners
wird zusätzlich einbezogen. Quellen ohne gültige Metadaten werden übersprungen.
Eine bestehende Zieldatei ohne passende Metadaten führt beim Vorbereiten zu einem
Fehler. Für eine neue CSV darf keine verwaiste .csv.meta.json-Datei existieren.

Der Fingerprint umfasst SHA-256-Werte der .mot-Basisdatei und Motor-CAD-EXE,
die Version von ansys-motorcad-core, die externe Abhängigkeitsrevision und
eingetragene Abhängigkeitsdateien sowie CSV-Schema, Speicherpräzision und relevante
Konfigurations-/Generator-/Prüfmodule einschließlich simulation.py. Die Begleitdatei
wird auf JSON-Grundstruktur und ihren Payload-Hash geprüft. Wiederverwendungsquellen
müssen kompatible Metadaten, dieselbe Spaltenreihenfolge und ausschließlich endliche
numerische Werte besitzen. Die Herkunft wird vor dem Vorbereiten aller Aufträge
und beim Aufbau der einzelnen Simulation ermittelt; dabei werden Basismodell,
EXE und konfigurierte Abhängigkeitsdateien gelesen.

is_compatible_metadata akzeptiert identische Fingerprints oder ein ausdrücklich
geprüftes, gerichtetes Paar in REVIEWED_COMPATIBLE_FINGERPRINTS. Bei einem solchen
Paar müssen zusätzlich alle Payload-Felder außer implementation_hashes gleich
sein. Die Liste enthält derzeit einen Übergang vom früheren zum bereinigten Stand.
Das ist keine allgemeine Freigabe für abweichenden Quellcode. Diese Ausnahme gilt
für Cache-Quellen; bestehende Zieldateien benötigen weiterhin exakte Übereinstimmung.

Generierte Eingabeparameter werden mit absoluter Toleranz 1e-4 verglichen.
Ein Treffer in der Zieldatei wird als existing gezählt; ein Treffer in einer
anderen Datei wird als copied übernommen. Die Zieldatei hat beim Suchen Vorrang;
weitere Quellen und Zeilen werden in fester Reihenfolge geprüft. _append_result()
aktualisiert sowohl den Datenbestand im Speicher als auch den bekannten-Punkte-
Cache. Die Batch-Verarbeitung enthält außerdem eine Zusammenfassung gleicher
Anforderungen: Ein identischer Punkt wird je Batch nur einmal eingereicht, aber
für alle zugehörigen Anforderungen im Fortschritt berücksichtigt. Einreichung
und Ergebnisverarbeitung sind getrennt; alle abgeschlossenen Futures werden
ausgewertet. Ein Batch ausschließlich bekannter Punkte benötigt keine neuen
Simulationsaufträge.

Die sechs direkt unter data/MCAD_Results vorhandenen Altdateien besitzen inzwischen
Metadaten mit adoption.kind="legacy_assumption". Ihre ursprünglichen Setups sind
damit ausdrücklich nicht verifiziert. Der Metadatenleser prüft zusätzlich den
gespeicherten SHA-256-Wert der übernommenen CSV; spätere Änderungen werden erkannt.
Übernommene Dateien dürfen nur als Cache-Quelle dienen und nicht als Ziel
weitergeschrieben werden. Die Kopien unter Training_data benötigen für den
DataLoader keine Simulationsmetadaten.

Randbedingungen werden vor der Magnetfeldrechnung geprüft. Erfolgreiche Punkte
werden geschrieben; ungültige Designs und abgefangene Motor-CAD-Rechenfehler
werden als dropped gezählt. Andere Worker-Fehler beenden den Dienst mit Exception.

SimulationPipeline.start() entfernt abschließend ausschließlich den für diesen
Lauf erzeugten simulation_run_-Ordner. Andere Läufe und fremde Dateien unter
temp_files bleiben erhalten. Bereinigungsfehler werden protokolliert. Motor-CAD-
Worker registrieren einen multiprocessing-Finalizer und räumen ihre Instanz auch
nach Initialisierungs-/Rechenfehlern auf. Bei start_eduvpn=True wird eine laufende
eduVPN-Anwendung beendet und neu gestartet; der gestartete Prozessbaum wird
abschließend beendet. Ein laufender eduVPN-Prozess bestätigt keine erfolgreiche
VPN- oder Lizenzverbindung.

Alte Ergebnisdateien ausdrücklich übernehmen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Einstieg: src/executable_files/adopt_legacy_csv.py

    python -m src.executable_files.adopt_legacy_csv PFAD_ZUR_ALTEN_CSV --mot-sha256 SHA256_DER_MOT --reason "Begründung der Setup-Annahme" --accept-setup-assumption

Die Platzhalter durch den tatsächlichen CSV-Pfad und den bestätigten SHA-256-Wert
der zugehörigen MOT-Datei ersetzen. Ohne --write werden nur Prüfungen durchgeführt;
erst zusätzliches --write schreibt die .csv.meta.json-Datei. Die CSV bleibt dabei
unverändert. --accept-setup-assumption und eine nicht leere Begründung sind nötig.
Der angegebene MOT-Hash muss dem aktuell konfigurierten Basismodell entsprechen.
Auch EXE und konfigurierte externe Abhängigkeiten müssen lesbar sein, weil das
Skript die aktuellen Metadaten aufbaut.

adopt_legacy_csv prüft die exakte Spaltenfolge, numerische/endliche Werte,
mindestens eine Ergebniszeile und einen unveränderten CSV-Hash während der Prüfung.
Eine vorhandene Begleitdatei wird nicht überschrieben. Der Übernahmevermerk enthält
Begründung, UTC-Zeit, CSV-Hash und original_setup_verified=False. Das Skript belegt
damit nicht nachträglich, mit welcher MOT-Datei, EXE oder welchem Solver die
historischen Ergebnisse tatsächlich erzeugt wurden. Die Übernahme hält die
ausdrücklich akzeptierte Setup-Annahme fest.


5. CSV-Daten und Optimierung starten
------------------------------------

CSV-Format: UTF-8, Semikolon als Spaltentrenner, Komma als Dezimaltrennzeichen,
vier Nachkommastellen bei geschriebenen Floatwerten. Jede erfolgreiche Rechnung
liefert eine Zeile. Die Reihenfolge hängt von der Fertigstellung der Worker ab.

CSV_PARAMETERS umfasst 29 Spalten in dieser Reihenfolge:

    Geometrie (16):
    Pole_Number, Slot_Number, Airgap, Stator_Bore, Stator_Lam_Dia,
    AFM_D_Rotor, Magnet_Arc_[ED], Magnet_Length, Magnet_Thickness,
    Slot_Depth, Slot_Width, Slot_Opening, Magnet_Embed_Depth,
    Rotor_Lam_Length, Stator_Lam_Length, Motor_Length

    Betriebspunkt (6):
    ShaftSpeed, RMSPhaseCurrent_D, RMSPhaseCurrent_Q,
    RMSCurrentDensity, PeakCurrent, PhaseAdvance

    Ergebnisse (7), zugleich OUTPUT_NEURON_LIST:
    ShaftTorque, SystemEfficiency, BMean_AirGap, ConductorLoss,
    MagnetLoss, StatorIronLoss_Total, RotorIronLoss_Total

Direkt in data/Training_data liegen am Dokumentationsdatum folgende Dateien:

    Dateiname                                                               Datenzeilen
    Results_EMagnetic_simulation_1_10_5_GenType-v2_04.10.2026_19-31-19_CEST.csv     1.250
    Results_EMagnetic_simulation_1_20_5_GenType-v2_04.10.2026_19-33-27_CEST.csv     2.375
    Results_EMagnetic_simulation_1_30_5_GenType-v2_04.10.2026_19-41-31_CEST.csv     3.500
    Results_EMagnetic_simulation_1_40_5_GenType-v2_04.10.2026_20-06-28_CEST.csv         0
    Results_EMagnetic_simulation_1_50_5_GenType-v2_06.10.2026_17-00-36_CEST.csv     6.125

Die Datei mit null Datenzeilen enthält nur den Header und ist weder als Haupt-
noch als Testquelle geeignet. data/MCAD_Results enthält dieselben v2-Dateinamen,
aber die dortige 1_40_5-Datei hat inzwischen 5.000 Datenzeilen; die Trainingskopie
wurde nicht entsprechend aktualisiert. Zusätzlich liegt dort der v1-Datensatz
Results_EMagnetic_simulation_3_3_5_GenType-v1_18.09.2026_13-52-26_CEST.csv mit
10.000 Datenzeilen. Die Zeilenzahl allein bestätigt keine gültigen Kennfelder.
Ältere Dateien in Unterordnern besitzen teilweise andere Spaltenschemata.

Einstieg: src/executable_files/main_optimize.py

Die Hauptdatei wird über das erforderliche Argument --csv gewählt. Es gibt keine
fest eingetragene, fehlende Trainingsdatei mehr. Beispiel mit vorhandener CSV:

    python -m src.executable_files.main_optimize --csv Results_EMagnetic_simulation_1_10_5_GenType-v2_04.10.2026_19-31-19_CEST.csv

Trainingsdaten werden nicht automatisch aus data/MCAD_Results übernommen.
Für eigene Versuche passende Ergebnisdateien in data/Training_data bereitstellen.
DataLoader lädt ausschließlich die Hauptdatei und ausdrücklich angegebene
Testdateien. Jede Quelle muss genau einmal als .csv-Dateiname ohne Verzeichnispfad
angegeben werden. Die anderen Dateien im Ordner werden nicht automatisch geladen.

Kommandozeilenoptionen:

    --csv DATEI             Erforderliche Hauptquelle unter data/Training_data
    --test-csv DATEI        Optionale externe Testquelle; mehrfach verwendbar
    --seed ZAHL             Standard 42
    --trials ZAHL           Standard 100; positive Ganzzahl
    --worker-timeout SEK    Optionales Zeitbudget für Optimierungsworker

--test-csv nur für unabhängige Geometrien verwenden. Überlappende Maschinen
zwischen Training/Validierung/Test und doppelte Maschinen zwischen Testquellen
werden abgelehnt. --seed und --worker-timeout werden vom Einstieg an die Pipeline
und weiter an NNOptimizer übergeben. Das Worker-Zeitbudget gilt für den gesamten
Workerlauf der Suche; Datenaufbereitung und finales Training liegen außerhalb.
Bei Überschreitung wird TimeoutError ausgelöst und die Workerbereinigung gestartet.

Aktuelle Parameter:

    max_num_neuron_per_layer=200
    max_num_layers=3
    neuron_step=5
    learning_rate=0.001
    patience=50
    num_epochs=2000
    weight_decay=0.001
    n_trials=100

Die Optimierungspipeline fordert intern einen Worker an. n_trials gilt je Worker;
vorgesehen sind standardmäßig 100 Architekturversuche mit 700 MLP-Trainings und
anschließend sieben zusätzlichen Teilmodell-Trainings für das finale Modell.
Die Datenaufbereitung erfolgt bereits beim Aufbau der Pipeline, vor dem GUI-Start.


6. Datenaufbereitung und Modelle
--------------------------------

DataLoader rekonstruiert die Leiterfläche aus PeakCurrent und RMSCurrentDensity,
berechnet d-/q-Stromdichten und rundet sie auf Schritte von 0,05. Außerdem werden
die neun Lambda-Verhältnisse aus den Geometriespalten berechnet.

Vor diesen Berechnungen werden NaN/unendliche Quellwerte und nicht positive
Nennergrößen abgelehnt. Dazu gehören PeakCurrent, RMSCurrentDensity, Stator_Bore,
Stator_Lam_Dia, Magnet_Length, Slot_Depth und Rotor_Lam_Length. Radiale Breite und
radiale Breite abzüglich Magnet_Thickness müssen ebenfalls positiv sein.

MACHINE_DEFINITION legt die 13 Geometriemerkmale und deren Reihenfolge fest:

    Slot_Number, Pole_Number, Airgap, Stator_Bore,
    lambda_1_1, lambda_1_2, lambda_1_3,
    lambda_2_1, lambda_2_2, lambda_2_3,
    lambda_3_1, lambda_3_2, lambda_3_3

Kennfeldachsen in AXIS_LIST:

    ShaftSpeed, RMSPhaseCurrent_D_density, RMSPhaseCurrent_Q_density

DataLoader baut ein festes Referenzraster aus MAP_SIZE und den konfigurierten
Drehzahl-/Stromdichtebereichen auf. Rekonstruierte Stromdichteachsen werden auf
Schritte von 0,05 gerundet. Jede Maschine muss genau dieses kartesische Raster
abdecken: Doppelte Punkte sowie fehlende oder zusätzliche Achsenkombinationen
entfernen die ganze Maschine. Auch eine Datei ohne verbliebenes gültiges Kennfeld
wird abgelehnt. NaN-Werte im Kennfeldtensor führen zu einem Fehler; fehlende Werte
werden nicht interpoliert. Die Achsen werden nicht mehr aus jeder CSV neu bestimmt.

Tensorformen:

    geometry_tensor    (N, 13)
    map_tensor         (N, 5, 5, 5, 7)

Die Hauptdatei wird nach ganzen Maschinen mit Seed 42 aufgeteilt: ca. 80 Prozent
Training, 20 Prozent Validierung, mindestens fünf vollständige Maschinen.
Nur die expliziten test_filenames werden zu einem zusätzlichen Testsatz verbunden.
Ohne Testdateien ist test=None. Geometrie-IDs werden aus den unnormalisierten
float32-Eingaben gebildet; Überschneidungen der drei Datenteile und Duplikate
zwischen Testquellen werden geprüft. Dies ist eine exakte Identitätsprüfung,
keine geometrische Ähnlichkeitsprüfung. Die Architekturwahl verwendet weiterhin
ausschließlich die Validierung. Der Einstieg wertet das finale Modell anschließend
auf der Validierung und, falls vorhanden, dem unabhängigen Testdatensatz aus.

Normalisierung:

- Die ersten vier Geometriemerkmale werden anhand der festen Grenzen aus
  GEO_PARAMETERS_MIN_MAX_DICT auf (Wert - Minimum) / (Maximum - Minimum) skaliert.
- Die neun Lambda-Merkmale bleiben als geometrische Verhältnisse erhalten.
- Kennfelder werden je Ergebniskanal mit Minimum und Maximum der Trainingsdaten
  skaliert. Validierung und Test verwenden dieselben Trainingsgrenzen.

Werte außerhalb der Trainingsgrenzen können außerhalb von [0, 1] liegen.
TrainValidationTestDataTensors enthält jetzt map_axes, target_minimum,
target_denominator, split_machine_ids und source_hashes neben den drei Datenteilen.
Damit sind Rückskalierung, Quellenprüfung und Speicherung der Aufteilung möglich.
target_denominator ist die auf mindestens 1e-8 begrenzte Trainingsspannweite.

MLP-Surrogatmodell und Architektur-Suche
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optuna wählt ein bis drei versteckte Schichten mit jeweils 5, 10, ..., 200
Neuronen. Die Schichten verwenden ReLU; die Ausgabeschicht ist linear.
Jedes MLP erhält 13 Geometriemerkmale und liefert 125 Werte, die auf
(N, 5, 5, 5, 1) umgeformt werden. Die Betriebspunkte sind durch das feste Raster
vorgegeben und keine zusätzlichen MLP-Eingaben.

SurrogateModelTrainer trainiert für jede der sieben Ergebnisgrößen ein eigenes
Teilmodell mit derselben Architektur. SurrogateModel.get_prediction verbindet
die Ausgaben in OUTPUT_NEURON_LIST-Reihenfolge zu einem siebenkanaligen Kennfeld.

NNTrainer arbeitet auf der CPU mit Adam, MSE und einem vollständigen Datensatz
je Optimierungsschritt. Es gibt im aktuellen Trainer keine Mini-Batch-Aufteilung.
Early Stopping verwendet patience=50 und eine Mindestverbesserung von 1e-6;
abschließend werden die Gewichte mit dem besten Validierungsverlust geladen.
Split und Shuffle verwenden weiterhin fest Seed 42, unabhängig von --seed.
Der Optimierer enthält Seeds für Python, NumPy, PyTorch und den TPE-Sampler;
das Objective setzt einen Trial-abhängigen PyTorch-Seed. Das finale Training setzt
die drei Seeds anhand von --seed neu. Die aktuelle Weitergabe dieser Werte zwischen
Pipeline und Worker ist eingerichtet. Ein fortgesetzter TPE-Lauf wird mit dem Seed
neu initialisiert; eine bitgleiche Folge gegenüber einem ununterbrochenen Lauf
wird damit nicht zugesichert.

Der Trainer prüft passende, nicht leere Ziel-/Vorhersagetensoren und endliche
Verluste. Nicht endliche Gradienten brechen das Training vor dem Optimiererschritt
mit FloatingPointError ab. Early-Stopping-Meldungen gehen an das gemeinsame Logging.

Optuna minimiert MSE über die normalisierten Validierungskennfelder. Zusätzlich
wird R² je Kanal berechnet und ungewichtet über die sieben Kanäle gemittelt.
calc_r_squared_report liefert für konstante Referenzkanäle None und dann auch
für den Gesamtmittelwert None. Das Objective speichert r2_mean und r2_per_channel.
Die strengere Hilfsfunktion calc_r_squared löst bei undefiniertem Gesamt-R²
weiterhin ValueError aus. Kennzahlenprüfungen lehnen leere Daten, unterschiedliche
Tensorformen und nicht endliche Referenz-/Vorhersagewerte ab.

Optionaler Autoencoder
~~~~~~~~~~~~~~~~~~~~~~

Conv3dAutoencoder und ein zusätzlicher Trainingszweig sind weiterhin vorhanden.
Wird SurrogateModelTrainer ein passender einkanaliger Autoencoder übergeben,
trainiert er zuerst je Kanal die Kennfeldrekonstruktion, friert den Autoencoder
ein und trainiert das MLP für dessen latenten Raum. Dabei bewertet der Trainer
3 x 3 x 3 Stützstellen des dekodierten Kennfelds. MLP-Ausgabe und latent_dim
müssen übereinstimmen. Der aktuelle Optuna-Ablauf übergibt keinen Autoencoder
und trainiert die MLPs direkt auf vollständige Kennfelder.


7. Ergebnisse, Logging und vorhandene Artefakte
------------------------------------------------

NNOptimizer berechnet einen Experimentfingerprint und verwendet dafür:

    data/optimization/<Fingerprint>/study.sqlite3
    Studienname: layer_<Fingerprint>

In den Fingerprint gehen Quell-CSV-Hashes, Split-IDs, Rasterachsen, Normalisierung,
Merkmals-/Kanalreihenfolge, Workerzahl, Trainingsparameter, Suchraum, Seed,
Python-/Paketversionen und Hashes ausgewählter Implementierungsdateien ein.
Eine Änderung dieser Angaben erzeugt einen anderen Studienpfad. Die Trialzahl
und das Worker-Zeitbudget sind keine Bestandteile dieses Fingerprints.
_prepare_study() legt die Studie mit load_if_exists=True an oder öffnet sie.
Sie speichert experiment_fingerprint und experiment_config als Studienattribute.
Vorhandene Trials ohne Experimentkennung oder mit abweichendem Fingerprint werden
abgelehnt. Die Worker laden anschließend diese vorbereitete Studie. Bei gleichem
Fingerprint können weitere Trials ergänzt werden; das beste Ergebnis stammt
dann aus allen Trials der Studie, einschließlich früherer Aufrufe.

OptimizationPipline.start() liefert ein OptimizationResult mit hidden_layers,
mse, r_squared und trial_number. main_optimize.py verwendet die ausgewählte
Architektur für ein neues finales Surrogatmodell und erstellt Kennzahlen für
Suche, finale Validierung und gegebenenfalls den externen Testdatensatz.
evaluate_surrogate liefert normalisierten MSE, R² pro Kanal und als Mittelwert,
sowie MAE und MSE je Kanal in physikalischer Skalierung. Die Skalierung der Fehler
verwendet target_denominator; es werden keine Einheiten über Kanäle hinweg gemittelt.

Bei erfolgreichem Durchlauf speichert der Einstieg:

    data/trained_nn/<JJJJMMTT_HHMMSS>_<achtstellige ID>/surrogate.pt

save_bundle schreibt zunächst eine temporäre Datei und ersetzt das Ziel atomar.
Das Bundle enthält Formatversion, Modelltyp direct_mlp_ensemble, ReLU-Aktivierung,
Architektur, sieben CPU-state_dicts, Merkmals-/Kanalnamen, Geometriegrenzen,
Kennfeldachsen, Zielskalierung, Split-IDs, Quellenhashes, Seed und Kennzahlen.
Derzeit unterstützt dieses Format ausschließlich direkte MLP-Teilmodelle aus
Linear- und ReLU-Schichten, keine Autoencoder oder anderen Aktivierungen.
Die SQLite-Studie selbst enthält keine Netzgewichte.

load_bundle(path) lädt auf der CPU mit weights_only=True und prüft Pflichtfelder,
Format, Modelltyp, Aktivierung und Anzahl der Kanalzustände. Merkmals-, Kanal- und
Achsenreihenfolge müssen zur aktuellen Konfiguration passen. Zielskalierung muss
endlich, richtig dimensioniert und beim Nenner strikt positiv sein. Kennfeldachsen
müssen endlich, eindimensional, nicht leer und streng steigend sein; ihre Längen
müssen zu output_dims passen. Anschließend werden die MLPs im Auswertungsmodus
aufgebaut und die Gewichte mit strict=True geladen.
predict_raw_geometry(model, checkpoint, geometry) erwartet unnormalisierte
Geometrien als (N, 13) in der gespeicherten Merkmalsreihenfolge, normalisiert die
vier Grundmerkmale und liefert rückskalierte physikalische Kennfelder.
Der Einstieg lädt sein Bundle nach dem Speichern erneut und vergleicht die
Validierungsvorhersagen mit torch.testing.assert_close.

Unter data/trained_nn liegen noch data.pkl, trained_mlp.pt und
trained_autoencoder.pt. data/plots enthält historische Verlustkurven,
Vorhersagevergleiche und weitere Versuchsauswertungen. Die entfernten Startskripte
main_nn.py und main_create_plots.py erzeugen diese Artefakte derzeit nicht mehr.
Ältere Artefakte sind nicht automatisch mit dem neuen Bundleformat kompatibel.
Am Dokumentationsdatum liegen noch keine neuen Lauf-Unterordner mit surrogate.pt
und kein data/optimization-Verzeichnis vor. Trainingskurven und Vergleichsdiagramme
werden vom aktuellen Optimierungseinstieg weiterhin nicht automatisch erstellt.

main_mcad_sim.py und main_optimize.py rufen configure_application_logging() vor
dem Aufbau der Pipeline auf. Konsole und eine gemeinsame UTF-8-Datei erhalten
Meldungen ab INFO, einschließlich Warnungen und Fehlern:

    data/logs/<JJJJMMTT_HHMMSS_Mikrosekunden>_<achtstellige ID>_application.log

setup_main_logging() verwendet während der Pipeline dieselbe Datei weiter und
richtet den QueueListener ein. Ohne zuvor eingerichteten FileHandler legt es
selbst eine application.log-Datei an. Motor-CAD- und Optuna-Worker leiten ihre
Logmeldungen an die gemeinsame Queue weiter; Optuna propagiert an das Python-
Logging und deaktiviert seinen separaten Standardhandler. Auch Konstruktion,
finales Training, Modellpfad und Kennzahlen des Optimierungseinstiegs landen in
der Anwendungsdatei. Die früheren *_info.txt und *_error.txt bleiben historische
Dateien und werden durch diese beiden Startskripte nicht mehr erzeugt.

Nach dem Dienstende schließt die Pipeline GUI, Listener, dessen Handler und
Queues und stellt den vorherigen Root-Logger-Zustand wieder her. So können die
Schritte nach der Suchpipeline in dieselbe Anwendungsdatei schreiben. Dienstfehler
werden nach dem Warten auf den Dienst-Thread erneut ausgelöst. Pipeline-Instanzen
können nur einmal laufen; für einen weiteren Lauf eine neue Instanz erzeugen.


8. Hilfsmodule und Geometrieprüfung
-----------------------------------

plotting/matplotlib_plotting.py bietet Linien- und Scatterplots, Legenden,
Textfelder und PNG-Export mit 300 dpi. plotting/plotly_plotting.py verwendet den
Renderer plotly_mimetype und benötigt eine Umgebung, die ihn darstellen kann.
plotting/heatmap.py bildet Mittelwerte für mehrfache x/y-Paare, interpoliert
Lücken linear und unterstützt logarithmische Farbskalen für positive Werte.
Diese Helfer werden von den aktuellen Startskripten nicht automatisch aufgerufen.

Das Prüfskript wird aus der Projektwurzel gestartet:

    python -m src.executable_files.machine_test

Es prüft drei Kombinationen mit B=1 und L=3, 4, 5 über den v2-Generator.
Mit MAP_SIZE=(5,5,5) sind das insgesamt 1.500 Parameterprüfungen. Generatorwerte
werden mit DEFAULT_VALUES ergänzt und an test_boundary_conditions übergeben.
Es startet keine Motor-CAD-Instanz und schreibt keine Ergebnis-CSV. Wegen des
importierten Prüfmoduls muss ansys-motorcad-core dennoch installiert sein.

Die abschließende Ausgabe Expected operating points verwendet nun die v2-Formel
und zeigt korrekt 1.500 Punkte. Das Prüfskript und die drei anderen Einstiege
besitzen einen __main__-Guard; die Aktionen werden erst beim Skriptstart ausgeführt.

check_logging.py prüft das gemeinsame Logging mit einem Spawn-Worker in einem
temporären Verzeichnis. Es erwartet genau eine Logdatei, jede Testmeldung genau
einmal und Einträge vor, während sowie nach der Pipeline-Loggingphase:

    python check_logging.py

Eine umfassende automatisierte Testsuite für Simulation, Datenaufbereitung,
Optimierung und Modellpersistenz ist derzeit nicht eingerichtet. pyproject.toml
setzt für Ruff target-version="py314". Die zusätzlichen Prüfberichte und ZIP-
Archive im Projektstamm sind Bestandsartefakte; die Startskripte laden sie nicht.


9. Fehlerbehebung und aktuelle Grenzen
--------------------------------------

Fehlendes Argument --csv / FileNotFoundError beim Optimierungsstart
    --csv ist erforderlich und muss exakt zu einer Datei unter Training_data
    passen. Nur einen .csv-Dateinamen angeben; Abschnitt 5 enthält ein Beispiel.

ModuleNotFoundError für src oder ein Python-Paket
    Aus der Projektwurzel starten und den Interpreter Env_NN_AFM verwenden.
    environment.yml bzw. requirements.txt mit diesem Interpreter installieren.

Fehlende CSV-Spalten oder Fehler bei Tensorachsen
    Hauptquelle und ausdrücklich gewählte Testdateien prüfen. MAP_SIZE,
    AXIS_LIST, vollständiges Referenzraster und Spaltenschema müssen passen.
    Leere oder nach Bereinigung leere Dateien werden abgelehnt. Ältere Daten
    werden nicht automatisch migriert.

Overlapping machines / Duplicate machines between external test sources
    Unabhängige Geometrien für --test-csv auswählen. Umbenennen oder Kopieren
    einer CSV beseitigt die Überschneidung nicht; die Prüfung verwendet die
    exakten Geometrie-IDs vor der Normalisierung.

Zu wenige vollständige Maschinen
    Die Hauptdatei benötigt mindestens fünf gültige Kennfelder nach Bereinigung.
    Doppelte oder fehlende Betriebspunkte entfernen die ganze Maschine.

Motor-CAD oder Basismodell fehlt / keine Berechnung möglich
    Externe Pfade, Lizenzzugang und Basismodell prüfen. Die automatische
    eduVPN-Steuerung bestätigt nur den Programmstart. Details zu verworfenen
    Designs und Rechenfehlern stehen in Konsole und Simulationsprotokollen.

Fehlende Metadaten oder unpassendes Simulationssetup
    Neue CSV-Ausgaben benötigen ihre .csv.meta.json-Begleitdatei. Eine bestehende
    Zieldatei ohne exakt passenden Fingerprint wird abgelehnt. Für Cache-Quellen
    gilt zusätzlich die geprüfte Kompatibilitätsliste. Basismodell, EXE,
    Abhängigkeitsrevision und Fingerprint prüfen. Reine Quellcodeformatierung
    verändert SHA-256-Werte ebenfalls; eine Kompatibilitätsfreigabe im Code
    gilt nur für das ausdrücklich hinterlegte, gerichtete Fingerprintpaar.

Imported legacy files can only be used as a cache source
    Übernommene Dateien mit adoption-Vermerk nicht als csv_paths-Ziel angeben.
    Eine neue Zieldatei wählen; passende alte Ergebnisse können kopiert werden.

The imported CSV file was subsequently amended
    Der aktuelle Inhalt stimmt nicht mehr mit adoption.csv_sha256 überein.
    Die ursprüngliche Datei und den Übernahmevermerk prüfen. Das Setzen eines
    neuen Hashes ersetzt keine erneute Beurteilung der Herkunftsannahme.

Alte Datei lässt sich nicht übernehmen
    --accept-setup-assumption, nicht leere --reason-Angabe und bestätigten
    MOT-Hash prüfen. Die CSV benötigt den exakten Header und endliche Werte
    sowie mindestens eine Datenzeile. Vorhandene Metadaten werden nicht
    überschrieben. Ohne --write bleibt die Prüfung schreibfrei.

Optimierung vermischt frühere und aktuelle Versuche
    Den Fingerprint und data/optimization/<Fingerprint>/study.sqlite3 prüfen.
    Aktuelle Studien sind an Daten, Skalierung, Suchraum und Implementierung
    gebunden. Die frühere optimization.db wird nicht mehr verwendet.

Fenster ausgeblendet / Fortschritt oder Abschluss bleibt hängen
    Schließen während der Arbeit ist kein Abbruchsignal. Die Pipeline wartet
    auf Dienst-Thread und Abschlussereignis. Ohne vorherige Schließanforderung
    nach regulärem Abschluss das sichtbare Fenster schließen. Aktuelle Details
    und Workerfehler stehen in der gemeinsamen application.log-Datei.

Worker-Zeitbudget überschritten
    --worker-timeout gilt für die Suchworker insgesamt, nicht je Trial und
    nicht für das finale Training. Die Bereinigung beendet laufende Prozesse,
    wartet auf deren Ende und eskaliert bei Bedarf auf kill(). Bei Fehlern
    können nicht abgeschlossene Trials in der Studie verbleiben.

FloatingPointError im Training
    Quellen, Normalisierung, Modellvorhersagen und Lernrate prüfen. Der Trainer
    lehnt nicht endliche Verluste und Gradienten ab; ein ungültiger Gradient
    wird nicht durch einen Optimiererschritt weiterverarbeitet.

Modellbundle wird abgelehnt
    Aktuelle Merkmals-, Kanal- und Achsenreihenfolge sowie Skalierung und
    Ausgabedimensionen mit dem gespeicherten Bundle vergleichen. Das aktuelle
    Format unterstützt nur direkte MLPs aus Linear-/ReLU-Schichten.

Fehlende neue .pt-Dateien oder Diagramme nach der Optimierung
    Bei erfolgreichem Gesamtlauf wird surrogate.pt in einem neuen Laufordner
    gespeichert und erneut geladen. Nach der Suche läuft zunächst das finale
    Training. Modellpfad und Kennzahlen stehen im Anwendungslog. Neue Diagramme
    sind weiterhin kein Bestandteil des Einstiegs.
