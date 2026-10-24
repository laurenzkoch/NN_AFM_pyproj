NN_AFM_pyproj
=============

Automatisierte Motor-CAD-Simulationen und Surrogatmodelle für Axialflussmaschinen
Dokumentationsstand: 20.10.2026

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
        -> evaluation.pt mit Referenzwerten und Vorhersagen
        -> Einzelplots als PNG/PDF und einseitiger PDF-Bericht

Beide Abläufe verwenden die gemeinsame Pipeline mit Tkinter-Fortschrittsfenster
und Logging. Die Optimierung benötigt keine laufende Motor-CAD-Anwendung.
Die Simulationsanbindung ist vom gemeinsamen Pipeline-Modul getrennt.
main_optimize.py besitzt Kommandozeilenoptionen; Simulationsumfang und
Trainings-/Suchraumvorgaben werden weiterhin im Code geändert.
main_adopt_legacy_csv.py übernimmt den im Skript konfigurierten CSV-Stand als
Cache-Quelle. main_plot_results.py erzeugt Diagramme und Bericht erneut aus
gespeicherten Auswertungsdaten.

Diese README wurde statisch mit dem aktuellen Quellcode und Dateibestand
abgeglichen. Simulation, Training, Datenübernahme und Tests wurden dafür nicht
gestartet. Vorhandene Laufartefakte sind in Abschnitt 7 beschrieben.


1. Projektstruktur
------------------

NN_AFM_pyproj/
|-- README.txt
|-- environment.yml                 Conda-Umgebung Env_NN_AFM
|-- requirements.txt                Gemeinsame Python-Paketliste
|-- pyproject.toml                  Ruff-Zielversion Python 3.14
|-- pytest.ini                      Testpfade und Marker
|-- tests/                          Regressionstests mit synthetischen Daten
|-- docs/                           Ergänzende Workflow-, Datei- und Testdokumente
|-- data/
|   |-- MCAD_Results/                Simulations-CSV; neue Läufe mit .meta.json
|   |-- Training_data/              Hauptquelle und automatisch gewählte Test-CSV
|   |-- optimization/               SQLite-Studie je Experimentfingerprint
|   |-- logs/                       Pipeline- und Simulationsprotokolle
|   |-- plots/                      Historische Diagramme; neue Plots im Laufordner
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
    |   |-- main_adopt_legacy_csv.py  Metadaten der konfigurierten CSV neu schreiben
    |   '-- main_plot_results.py     Gespeicherten Modelllauf neu darstellen
    '-- modules/
        |-- Pipeline_and_GUI/       Pipeline, Ereignisse, GUI und Abbruchsignal
        |-- data_managment/         CSV, Dateien, Experimente, Simulationsmetadaten
        |-- simulation/             Simulationspipeline und Prozesspool
        |   '-- generators/         Maschinenentwürfe und Betriebspunktraster
        |-- process/                Windows-Prozesse, eduVPN, CPU-/Worker-Schutz
        |   '-- motor_cad_processes/ Motor-CAD-Instanzen und Randbedingungen
        |-- neuronal_networks/      Daten, Training, Surrogatmodelle, Modell-I/O
        |   |-- NN_types/           MLP und Conv3dAutoencoder
        |   '-- optimization/       Optuna-Worker und OptimizationPipline
        '-- plotting/               Gemeinsame Plotbasis, Einzelplots und PDF-Bericht

Die Schreibweisen data_managment, neuronal_networks und optimization_pipline.py
entsprechen dem Code. Die früheren Einstiege main_nn.py, main_create_plots.py
und adopt_legacy_csv.py sowie die alten allgemeinen Plothelfer sind entfernt.
Der aktuelle Optimierungsworkflow verwendet ausschließlich PyTorch.


2. Umgebung einrichten
----------------------

Die Simulation und Prozesssteuerung sind auf Windows ausgelegt. Die konfigurierte
PyCharm-Umgebung ist Env_NN_AFM mit Python 3.14.4. Die Versionsvorgaben in den
Umgebungsdateien beschreiben den aktuellen Paketbedarf, einschließlich NumPy
2.5.3 und SciPy 1.18.1. Ihre Kopfkommentare nennen noch den älteren Stand vom
12.10.2026. Sie sind keine vollständige Sperrdatei aller indirekten Abhängigkeiten.

Aus der Projektwurzel in einer für Conda eingerichteten PowerShell:

    conda env create -f environment.yml
    conda activate Env_NN_AFM

Eine bestehende Umgebung aktualisieren:

    conda env update -n Env_NN_AFM -f environment.yml

Conda installiert Python, Tk, Zeitzonendaten sowie die aufgeführten numerischen,
Plot- und HTTP-Pakete. Der pip-Abschnitt bindet requirements.txt ein und ergänzt
unter anderem PyTorch, PyMotorCAD, Optuna und seaborn. Doppelt aufgeführte
Python-Pakete sind kompatibel vorgegeben; bei idna erlaubt Conda >=3,<4,
während requirements.txt die Version 3.15 festlegt. Die Conda-Datei enthält
außerdem nomkl und blas=*=openblas für den BLAS-Unterbau.

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
    matplotlib             Einzelplots und PDF-Bericht mit Arial
    plotly, seaborn         Weiterhin in der Paketliste; keine aktiven Plothelfer
    pywin32                Windows-Jobobjekte für Motor-CAD-Worker und Kindprozesse
    types-pywin32          Typinformationen für die Windows-Anbindung
    pytest, ruff           Regressionstests und Quellcodeprüfung
    requests, idna,
    charset-normalizer     Explizit gepflegter HTTP-Unterbau für PyMotorCAD
    tzdata                 Zeitzone Europe/Berlin unter Windows

scikit-learn wird vom aktuellen Quellcode nicht importiert und ist deshalb kein
Eintrag mehr in den Umgebungsdateien. Historische Diagramme können weiterhin
Ergebnisse früherer scikit-learn-Versuche enthalten.

Die Diagrammfunktionen verlangen die installierte Schrift Arial ohne automatischen
Fallback. Sie wird nicht durch requirements.txt bereitgestellt. Python 3.14 ist
auch für ProcessPoolExecutor.kill_workers() im Abbruchpfad vorgesehen.

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

    design_combinations=[(B, L) for B in (1, 5, 10, 15)
                         for L in range(10, 300, 10)]
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
Die aktuelle v2-Startkonfiguration umfasst 116 Aufträge mit zusammen
16.856.250 generierten Anforderungen vor Wiederverwendung und Verwerfung.
Es werden 134.850 Geometrien angefordert; der erste Auftrag umfasst 1.250 Punkte,
der letzte 543.750. Bei gleichen bekannten Punkten können Ergebnisse kopiert
oder wiederverwendet werden. Den Umfang vor einem Start in main_mcad_sim.py prüfen.

Die gemeinsame Pipeline startet den Dienst automatisch und zeigt den Fortschritt
im Tkinter-Fenster. Es gibt keinen Startknopf und kein autostart-Argument mehr.
Die Oberfläche zeigt den Punktfortschritt des aktuellen Auftrags und einen
zusätzlichen Gesamtfortschritt für abgeschlossene Kombinationen. Der Punktwert
umfasst simulierte, verworfene, vorhandene und kopierte Anforderungen und wird
für jede neue Kombination zurückgesetzt.

Schließen während eines Laufs fordert jetzt einen Abbruch an, blendet das Fenster
aus und lässt die Ereignisschleife während der Bereinigung weiterlaufen.
CancellationToken wird an Serie, Simulation bzw. Optimierer weitergegeben.
Der Simulationspool beendet bei Abbruch oder Fehler seine Worker mit kill_workers();
der Optimierer beendet und bereinigt seine Suchprozesse. Ein CANCELLED-Ereignis
markiert den Abbruch, anschließend wird PipelineCancelledError weitergegeben.
Ein abgebrochener Suchlauf startet kein finales Training und keinen Export.

Bei regulärem Abschluss bleibt das Fenster sichtbar. Es muss geschlossen werden,
damit start() zurückkehrt und das finale Training in main_optimize.py beginnt.
Dieses anschließende Training und der Export haben kein eigenes Abbruchfenster.

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

Der Simulationsfingerprint verwendet Metadatenversion 2. Er umfasst SHA-256-Werte
der .mot-Basisdatei und Motor-CAD-EXE, die Version von ansys-motorcad-core, die
externe Abhängigkeitsrevision und eingetragene Abhängigkeitsdateien sowie CSV-Schema
und Speicherpräzision. Quellcodedatei-Hashes gehören nicht mehr dazu; reine
Formatierung ändert diesen Fingerprint deshalb nicht. Änderungen der fachlichen
Simulationsannahmen müssen über Modell, Abhängigkeitsdateien oder Revision
abgebildet werden. Die Begleitdatei wird auf JSON-Grundstruktur und ihren Payload-
Hash geprüft. Wiederverwendungsquellen müssen kompatible Metadaten, dieselbe
Spaltenreihenfolge und ausschließlich endliche
numerische Werte besitzen. Die Herkunft wird vor dem Vorbereiten aller Aufträge
und beim Aufbau der einzelnen Simulation ermittelt; dabei werden Basismodell,
EXE und konfigurierte Abhängigkeitsdateien gelesen.

is_compatible_metadata akzeptiert ausschließlich Version 2 mit vollständig gleichem
Payload und dessen korrekt berechnetem Fingerprint. Die frühere Liste gerichteter
Kompatibilitätsfreigaben ist entfernt. migrate_simulation_metadata() kann vorhandene
Version-1-Sidecars nach Prüfung auf Version 2 umstellen: Nur implementation_hashes
und die Versionsnummer dürfen sich unterscheiden, alle Setup-Angaben müssen passen.
Ein vorhandener adoption-Vermerk bleibt erhalten. Standardmäßig prüft die Funktion
nur; erst write=True schreibt geänderte Metadaten atomar. Die CSV wird dabei nicht
verändert. Diese Funktion ist eine API und besitzt keinen eigenen CLI-Einstieg.

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

Die zehn direkt unter data/MCAD_Results vorhandenen Altdateien besitzen inzwischen
Metadaten mit adoption.kind="legacy_assumption". Ihre ursprünglichen Setups sind
damit ausdrücklich nicht verifiziert. Der Metadatenleser prüft zusätzlich den
gespeicherten SHA-256-Wert der übernommenen CSV; spätere Änderungen werden erkannt.
Übernommene Dateien dürfen nur als Cache-Quelle dienen und nicht als Ziel
weitergeschrieben werden. Die Kopien unter Training_data benötigen für den
DataLoader keine Simulationsmetadaten, besitzen im aktuellen Bestand aber ebenfalls
Version-2-Begleitdateien.

Randbedingungen werden vor der Magnetfeldrechnung geprüft. Erfolgreiche Punkte
werden geschrieben; ungültige Designs und abgefangene Motor-CAD-Rechenfehler
werden als dropped gezählt. Andere Worker-Fehler beenden den Dienst mit Exception.

SimulationPipeline.start() entfernt abschließend ausschließlich den für diesen
Lauf erzeugten simulation_run_-Ordner. Andere Läufe und fremde Dateien unter
temp_files bleiben erhalten. Bereinigungsfehler werden protokolliert. Motor-CAD-
Worker registrieren einen multiprocessing-Finalizer und räumen ihre Instanz auch
nach Initialisierungs-/Rechenfehlern auf. Unter Windows bindet worker_job.py jeden
Motor-CAD-Worker an ein eigenes Jobobjekt mit KILL_ON_JOB_CLOSE. Beim Beenden des
Workers werden dadurch seine Kindprozesse ebenfalls beendet; unabhängige Prozesse
gehören nicht zu diesem Job. Dafür muss pywin32 installiert sein.
Bei start_eduvpn=True wird eine laufende eduVPN-Anwendung beendet und neu
gestartet; der gestartete Prozessbaum wird
abschließend beendet. Ein laufender eduVPN-Prozess bestätigt keine erfolgreiche
VPN- oder Lizenzverbindung.

Alte Ergebnisdateien ausdrücklich übernehmen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Einstieg: src/executable_files/main_adopt_legacy_csv.py

    python -m src.executable_files.main_adopt_legacy_csv

Vorher CSV_PATH, MOT_SHA256 und REASON im Skript prüfen. CSV_PATH verweist aktuell
auf Results_EMagnetic_simulation_10_10_5_GenType-v2_12.10.2026_21-45-45_CEST.csv
unter MCAD_Results. Der konfigurierte MOT-Hash muss zum aktuellen Basismodell passen.
Auch EXE und konfigurierte externe Abhängigkeiten müssen lesbar sein.

Der Einstieg besitzt keine CLI-Optionen und keine Vorschau: Er validiert Header,
endliche Werte, mindestens eine Ergebniszeile und den während der Prüfung
unveränderten CSV-Hash, schreibt die Metadaten direkt neu und liest sie zur
Kontrolle zurück. Eine vorhandene Begleitdatei kann dabei ersetzt werden.
Die CSV selbst bleibt unverändert. Der adoption-Vermerk enthält Begründung,
UTC-Zeit, CSV-Hash und original_setup_verified=False. Die historische Herkunft
wird damit weiterhin als Annahme festgehalten.

Die separate API adopt_legacy_csv() ist weiterhin vorhanden. Sie verlangt
accept_setup_assumption=True, eine nicht leere Begründung und einen passenden
confirmed_mot_hash; sie überschreibt keine vorhandene Begleitdatei und schreibt
erst bei write=True. Diesen API-Modus nicht mit dem direkten Einstieg verwechseln.


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

Direkt in data/Training_data liegen am Dokumentationsdatum:

    Dateiname                                                               Datenzeilen
    Results_EMagnetic_simulation_1_10_5_GenType-v2_04.10.2026_19-31-19_CEST.csv     1.250
    Results_EMagnetic_simulation_1_20_5_GenType-v2_04.10.2026_19-33-27_CEST.csv     2.375
    Results_EMagnetic_simulation_1_30_5_GenType-v2_04.10.2026_19-41-31_CEST.csv     3.500
    Results_EMagnetic_simulation_1_40_5_GenType-v2_04.10.2026_20-06-28_CEST.csv     5.000
    Results_EMagnetic_simulation_1_50_5_GenType-v2_06.10.2026_17-00-36_CEST.csv     6.125

Die Trainingskopie der 1_40_5-Datei ist inzwischen gefüllt. MCAD_Results enthält
zusätzlich unter anderem den v1-Datensatz vom 18.09.2026 mit 10.000 Zeilen sowie
die v2-Dateien 10_1_5, 19_1_5, 5_5_5 und 10_10_5. Zeilenzahlen allein bestätigen
keine gültigen oder unabhängigen Kennfelder. Unterordner werden nicht automatisch
nach Trainings- oder Testdaten durchsucht.

Einstieg: src/executable_files/main_optimize.py

    python -m src.executable_files.main_optimize --csv Results_EMagnetic_simulation_1_10_5_GenType-v2_04.10.2026_19-31-19_CEST.csv

--csv wählt die Hauptquelle unter data/Training_data. Der Einstieg wählt alle
anderen .csv-Dateien direkt in diesem Ordner automatisch und alphabetisch sortiert
als Testquellen. Die Hauptdatei wird anhand ihres aufgelösten Pfads ausgeschlossen;
.csv.meta.json-Dateien werden ignoriert. Hauptquelle und Testauswahl werden geloggt.
Die frühere Option --test-csv entfällt.

Der DataLoader selbst erhält weiterhin eine explizite test_filenames-Liste; die
Automatik sitzt im Startskript. Trainingsdateien werden nicht aus MCAD_Results
kopiert. Für einen Versuch deshalb nur passende, unabhängige CSV-Dateien direkt
in Training_data bereitstellen. Dateien ohne gültige Kennfelder, überschneidende
Maschinen oder doppelte Maschinen zwischen Testquellen werden abgelehnt und
nicht stillschweigend übersprungen. Ohne weitere CSV ist test=None.

Kommandozeilenoptionen:

    --csv DATEI             Erforderlicher .csv-Dateiname ohne Verzeichnispfad
    --seed ZAHL             Standard 42
    --trials ZAHL           Standard 1000; Zahl der Trials je Suchworker
    --worker-timeout SEK    Optionales Gesamtzeitbudget für die Suchworker

Aktuelle Trainings- und Suchraumvorgaben in main_optimize.py:

    max_num_neuron_per_layer=200
    max_num_layers=3
    neuron_step=5
    num_kernel=4
    learning_rate=0.001
    patience=100
    num_epochs=2000
    weight_decay=0.001

Die Workerzahl wird auf die verfügbaren CPUs begrenzt. Bei vier Workern ergeben
1.000 Trials je Worker insgesamt 4.000 Architekturversuche mit je sieben MLP-
Teilmodellen, also 28.000 Teilmodell-Trainings. Danach folgen sieben zusätzliche
Trainings für das finale Modell der besten Architektur. --trials verändert die
Zahl je Worker, nicht die Summe aller Prozesse. Die Fortschrittsanzeige zählt
abgeschlossene Trials; ihr Maximum ist n_trials * Workerzahl.

--seed wird an Suchworker und finales Training weitergegeben. --worker-timeout
begrenzt den gesamten Suchworker-Lauf einschließlich Prozessstart, nicht einzelne
Trials, Datenaufbereitung oder finales Training. Eine Überschreitung beendet die
Suche mit TimeoutError und löst die Workerbereinigung aus. Die Datenaufbereitung
findet bereits beim Aufbau der Pipeline statt, vor dem GUI-Start.

Nach regulärem Suchende das Fortschrittsfenster schließen. Erst danach trainiert
main_optimize.py das finale Modell, wertet Validierung und Testdaten aus, speichert
das Modell, lädt es zur Vorhersagekontrolle erneut und erzeugt Auswertungsdatei,
Einzelplots und PDF-Bericht. Ein Abbruch der Suche überspringt diese Folgeschritte.


6. Datenaufbereitung und Modelle
--------------------------------

DataLoader prüft das CSV-Schema sowie endliche Quellwerte. PeakCurrent,
RMSCurrentDensity, Stator_Bore, Stator_Lam_Dia, Magnet_Length, Slot_Depth und
Rotor_Lam_Length müssen positiv sein. Radiale Breite und radiale Breite abzüglich
Magnet_Thickness müssen ebenfalls positiv sein. Daraus werden Leiterfläche,
d-/q-Stromdichten und die neun Lambda-Verhältnisse berechnet.

MACHINE_DEFINITION legt die 13 Geometriemerkmale in dieser Reihenfolge fest:

    Slot_Number, Pole_Number, Airgap, Stator_Bore,
    lambda_1_1, lambda_1_2, lambda_1_3,
    lambda_2_1, lambda_2_2, lambda_2_3,
    lambda_3_1, lambda_3_2, lambda_3_3

Kennfeldachsen in AXIS_LIST:

    ShaftSpeed, RMSPhaseCurrent_D_density, RMSPhaseCurrent_Q_density

Das Referenzraster stammt aus MAP_SIZE und den konfigurierten Drehzahl-/
Stromdichtebereichen. Rekonstruierte Stromdichten werden nicht mehr pauschal auf
0,05 gerundet. _align_current_density_grid() prüft, ob die CSV-Rundung auf vier
Nachkommastellen die Abweichung von einem Rasterwert erklären kann: Aus den
Rundungsintervallen von Phasenstrom, Stromdichte und Spitzenstrom wird ein
möglicher Wertebereich bestimmt. Nur bei genau einem passenden Referenzwert
wird der Wert diesem Rasterpunkt zugeordnet. Andere Abweichungen bleiben bestehen.

Jede Maschine muss das vollständige kartesische Referenzraster besitzen. Doppelte
Punkte sowie fehlende oder zusätzliche Achsenkombinationen entfernen die ganze
Maschine. Dateien ohne verbliebenes gültiges Kennfeld werden abgelehnt.
Fehlende Kennfeldwerte werden nicht interpoliert.

Tensorformen beim aktuellen Raster:

    geometry_tensor    (N, 13)
    map_tensor         (N, 5, 5, 5, 7)

Die Hauptdatei wird nach ganzen Maschinen mit festem Seed 42 in ca. 80 Prozent
Training und 20 Prozent Validierung geteilt. Mindestens fünf vollständige
Maschinen werden benötigt. Die übrigen, vom Einstieg ausgewählten Dateien werden
zum Testsatz verbunden. Geometrie-IDs basieren auf den unnormalisierten float32-
Eingaben. Zwischen Training, Validierung und Test sind keine identischen IDs
erlaubt; auch Testquellen untereinander dürfen keine doppelten Maschinen enthalten.
Diese Prüfung erkennt exakte Identität, keine geometrische Ähnlichkeit.

Normalisierung:

- Die vier Grundmerkmale werden mit GEO_PARAMETERS_MIN_MAX_DICT skaliert:
  (Wert - Minimum) / (Maximum - Minimum).
- Die neun Lambda-Verhältnisse bleiben unverändert.
- Kennfelder werden je Kanal mit Minimum und Maximum ausschließlich der
  Trainingsdaten skaliert. Validierung und Test verwenden dieselben Werte.

Werte außerhalb der Trainingsgrenzen können außerhalb von [0, 1] liegen.
TrainValidationTestDataTensors enthält neben den Datenteilen map_axes,
target_minimum, target_denominator, split_machine_ids und source_hashes.
Der Nenner ist die auf mindestens 1e-8 begrenzte Trainingsspannweite.

MLP-Surrogatmodell und Architektur-Suche
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optuna wählt ein bis drei versteckte Schichten mit jeweils 5, 10, ..., 200
Neuronen. Versteckte Schichten verwenden ReLU, die Ausgabeschicht ist linear.
Jedes MLP erhält 13 Geometriemerkmale und liefert 125 Werte, umgeformt auf
(N, 5, 5, 5, 1). Betriebspunkte sind durch das feste Raster vorgegeben und keine
zusätzlichen MLP-Eingaben. SurrogateModelTrainer trainiert je Ergebnisgröße ein
eigenes Teilmodell derselben Architektur. SurrogateModel.get_prediction verbindet
die sieben Ausgaben in OUTPUT_NEURON_LIST-Reihenfolge.

NNTrainer arbeitet auf der CPU mit Adam und MSE. Pro Optimierungsschritt wird
der gesamte Trainingsdatensatz verwendet; es gibt keine Mini-Batch-Aufteilung.
main_optimize.py setzt patience=100; der allgemeine NNTrainer-Standard bleibt 50.
Early Stopping verlangt eine Verbesserung von mindestens 1e-6 und stellt die
Gewichte mit dem besten Validierungsverlust wieder her. Leere oder unpassende
Ziel-/Vorhersagetensoren werden abgelehnt. Nicht endliche Verluste oder Gradienten
führen zu FloatingPointError; ungültige Gradienten erhalten keinen Optimiererschritt.

Split und Shuffle verwenden fest Seed 42, unabhängig von --seed. Die Suchworker
setzen Python-, NumPy-, PyTorch- und TPE-Seeds; das Objective setzt einen
Trial-abhängigen PyTorch-Seed. Jeder Suchworker verwendet einen PyTorch-Thread.
Das finale Training setzt die drei Zufallsseeds erneut anhand von --seed.
Parallele Trialreihenfolge und fortgesetzte TPE-Läufe sind damit nicht als
bitgleich reproduzierbar zugesichert.

Optuna minimiert den MSE normalisierter Validierungskennfelder. Testdaten fließen
nicht in die Architekturwahl ein. R² wird je Kanal und als ungewichteter Mittelwert
berechnet. Ein konstanter Referenzkanal ergibt im R2Report None, dann ist auch das
Gesamtmittel None. Die strengere Funktion calc_r_squared löst in diesem Fall
ValueError aus. Metrikfunktionen prüfen Tensorformen, leere und nicht endliche Werte.

Optionaler Autoencoder
~~~~~~~~~~~~~~~~~~~~~~

Conv3dAutoencoder und der zusätzliche Trainingszweig sind weiterhin vorhanden.
Ein passend konfigurierter einkanaliger Autoencoder wird zuerst je Kanal auf
Kennfeldrekonstruktion trainiert und dann eingefroren. Das MLP lernt dessen
latenten Raum; bewertet werden 3 x 3 x 3 Stützstellen des dekodierten Kennfelds.
MLP-Ausgabe und latent_dim müssen passen. Der aktive Optuna-Ablauf verwendet
keinen Autoencoder, sondern direkte MLP-Vorhersagen vollständiger Kennfelder.


7. Studien, Modelle und Auswertungsdateien
------------------------------------------

NNOptimizer verwendet:

    data/optimization/<Fingerprint>/study.sqlite3
    Studienname: <Fingerprint>

Der Fingerprint bindet Quellenhashes, Split-IDs, Achsen, Skalierung, Merkmals-/
Kanalreihenfolge, Workerzahl, Trainingsparameter, Suchraum, Seed, Python-/
Paketversionen und ausgewählte Implementierungsdateien. Trialzahl und Worker-
Zeitbudget gehören nicht dazu. Anders als beim Simulationsfingerprint enthalten
Optimierungsfingerprints weiterhin Quellcode-Hashes und ändern sich bei Formatierung.

_prepare_study() legt die Studie mit load_if_exists=True an oder öffnet sie.
experiment_fingerprint und experiment_config werden als Studienattribute abgelegt.
Bestehende Trials ohne Experimentkennung oder mit fremdem Fingerprint werden
abgelehnt. Bei gleichem Fingerprint werden weitere Trials ergänzt; die beste
Architektur kann aus einem früheren Aufruf derselben Studie stammen.
OptimizationPipline.start() liefert hidden_layers, mse, r_squared und trial_number.

Ein erfolgreicher main_optimize-Lauf erzeugt einen eigenen Ordner:

    data/trained_nn/<JJJJMMTT_HHMMSS>_<achtstellige ID>/
    |-- surrogate.pt
    |-- evaluation.pt
    '-- result_plots/
        |-- <Kanal>_prediction_vs_sim_ref-model.png
        |-- <Kanal>_prediction_vs_sim_ref-model.pdf
        '-- prediction_report.pdf

Modellbundle
~~~~~~~~~~~~

save_bundle() schreibt surrogate.pt über eine temporäre Datei atomar. Es enthält
Architektur, sieben CPU-state_dicts, Merkmals-/Kanalnamen, Geometriegrenzen,
Achsen, Zielskalierung, Split-IDs, Quellenhashes, Seed und Kennzahlen einschließlich
training_config. Das Format unterstützt direkte MLPs aus Linear-/ReLU-Schichten,
keine Autoencoder. Die SQLite-Studie selbst enthält keine Netzgewichte.

load_bundle() lädt auf der CPU mit weights_only=True, prüft Pflichtfelder,
Format, Modelltyp, Merkmals-/Kanal-/Achsenreihenfolge, endliche Skalierungswerte
mit positivem Nenner und zur Achsengröße passende Ausgabedimensionen.
Netzgewichte werden mit strict=True
übernommen. Der Einstieg vergleicht anschließend die Validierungsvorhersagen des
neu geladenen Modells mit denen des gespeicherten Modells per assert_close.
predict_raw_geometry() erwartet unnormalisierte Eingaben (N, 13) in gespeicherter
Merkmalsreihenfolge und liefert Kennfelder in physikalischer Skalierung.

evaluate_surrogate() liefert normalisierten MSE, R² je Kanal und als Mittelwert
sowie MAE und MSE je Kanal in physikalischer Skalierung. Die Fehler werden mit
target_denominator rückskaliert; Größen verschiedener Einheiten werden dabei
nicht zu einem gemeinsamen physikalischen Fehlerwert vermischt.

Gespeicherte Auswertung
~~~~~~~~~~~~~~~~~~~~~~

save_evaluation() schreibt evaluation.pt neben surrogate.pt. Gespeichert werden
für Training, Validierung und gegebenenfalls Test:

- Maschinen-IDs und normalisierte Geometrieeingaben,
- Modellvorhersagen und Motor-CAD-Referenzkennfelder in physikalischer Skalierung,
- Kennzahlen je Datenteil.

Das Bundle enthält außerdem Achsen, Merkmals-/Kanalnamen, Quellenhashes und
Dateinamen samt SHA-256 des zugehörigen Modells. Tensorformen, Maschinenzahlen,
Skalierung und endliche Werte werden geprüft; die Modelldatei darf sich während
der Auswertung nicht verändern. Die Datei wird atomar ersetzt. Die Kennfeldtensoren
haben die Form (N, S, D, Q, 7). Die Geometrieeingaben bleiben normalisiert, obwohl
value_scale="physical" die gespeicherten Kennfeldwerte bezeichnet.

Einzelplots und PDF-Bericht
~~~~~~~~~~~~~~~~~~~~~~~~~~

plot_prediction_results() liest evaluation.pt und erzeugt für jeden der sieben
Kanäle einen PNG- und PDF-Vergleich mit Motor-CAD, insgesamt 14 Einzeldateien.
Dargestellt werden sämtliche Kennfeldpunkte, getrennt nach Training, Validierung
und Test, mit idealer Vergleichsgerade, Maschinen-/Punktzahlen, R² und MAE.
Einheiten: Drehmoment Nm, Wirkungsgrad %, Luftspaltflussdichte T und Verluste W.

create_prediction_report() erzeugt zusätzlich prediction_report.pdf: eine Seite
im A4-Querformat mit sieben Diagrammen, Datensatzübersicht, mittlerem R² sowie
MLP-Architektur und gespeicherten Trainingsparametern. Der Bericht prüft den
SHA-256-Wert von surrogate.pt gegen evaluation.pt. Er benötigt genau sieben
Ergebniskanäle und mindestens Training/Validierung; ein fehlender Testdatensatz
wird als nicht vorhanden gekennzeichnet. Beide Ausgaben verwenden Arial und
300 dpi für Rasterbestandteile. Vorhandene Ausgabedateien gleichen Namens werden
beim erneuten Plotten überschrieben.

Plots ohne neues Training erzeugen:

    python -m src.executable_files.main_plot_results

Vorher MODEL_FOLDER in src/executable_files/main_plot_results.py einstellen.
Aktuell ist 20261020_161109_f1353af2 eingetragen. Erwartet wird nur der Ordnername
unter trained_nn, kein vollständiger Pfad. Dieser vorhandene Lauf enthält
surrogate.pt, evaluation.pt, 14 Einzelplots und prediction_report.pdf.
Es werden keine ursprünglichen CSV-Dateien für das erneute Plotten benötigt.

Unter data/optimization liegen bereits mehrere Studienordner. trained_nn enthält
fünf Laufordner vom 18. bis 20.10.2026 und ältere Einzelartefakte data.pkl,
trained_mlp.pt sowie trained_autoencoder.pt. Die alten Einzelartefakte sind keine
neuen Bundles und werden vom aktuellen Plot-Einstieg nicht automatisch migriert.
Historische Diagramme bleiben unter data/plots; neue Ausgaben liegen im Laufordner.


8. Logging, Prüfskripte und Tests
---------------------------------

main_mcad_sim.py und main_optimize.py richten vor der Pipeline Anwendungslogging
ein. Konsole und gemeinsame UTF-8-Datei erhalten Meldungen ab INFO:

    data/logs/<JJJJMMTT_HHMMSS_Mikrosekunden>_<achtstellige ID>_application.log

setup_main_logging() verwendet während der Pipeline dieselbe Datei mit einem
QueueListener weiter. Motor-CAD- und Optuna-Worker senden ihre Logs an diese
Queue. Optuna propagiert an das Python-Logging und deaktiviert seinen separaten
Standardhandler. Haupt-/Testdateiauswahl, Early Stopping, Modell-/Auswertungspfade
und die formatierten Kennzahlen landen ebenfalls im Log. Nach Pipelineende wird
der vorherige Loggerzustand wiederhergestellt, sodass finales Training und Export
in dieselbe Datei schreiben. Alte *_info.txt und *_error.txt sind Bestandsdateien.

Die Pipeline schließt nach Dienstende GUI, Listener, dessen Handler und Queues.
Dienstfehler und Abbruch werden danach an den Aufrufer weitergegeben. Jede
Pipelineinstanz kann nur einmal laufen. Das finale Training und die Plotfunktionen
werden nach der Suchpipeline ausgeführt und haben keine eigene Fortschritts-GUI.

Geometrieprüfung:

    python -m src.executable_files.machine_test

Das Skript prüft den v2-Generator für (B,L)=(1,3),(1,4),(1,5), insgesamt
1.500 Betriebspunkte bei MAP_SIZE=(5,5,5). Generatorwerte werden mit DEFAULT_VALUES
zusammengeführt und an test_boundary_conditions() übergeben. Motor-CAD wird nicht
gestartet und es wird keine Ergebnis-CSV geschrieben. Der Paketimport der
Motor-CAD-Prüfung benötigt dennoch ansys-motorcad-core. Das Skript besitzt einen
__main__-Guard und startet seine Prüfung nicht beim Import.

Automatisierte Regressionstests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pytest.ini setzt testpaths=tests und pythonpath=.; die Tests werden aus der
Projektwurzel mit dem Projektinterpreter gestartet:

    python -m pytest
    python -m pytest -m "not training and not subprocess"
    python -m pytest tests/test_cancellation.py tests/test_worker_job.py

Die neun Testmodule behandeln Daten/Metriken, Dateien/Raster, Metadaten,
Modelle/CPU-Training, Optuna, Logging, Simulationssteuerung, Abbruch und den
Windows-Worker-Prozessbaum. Tests verwenden temporäre Dateien und synthetische
Daten; Motor-CAD-Start und Tk-Fenster werden über Testgrenzen gesperrt oder ersetzt.
Einzelne Tests starten echte Python-Unterprozesse bzw. kurzes CPU-Training.
Der Windows-Jobobjekttest benötigt Windows und pywin32. Echte Motor-CAD-Rechnungen,
Lizenzzugang, GUI-Darstellung und fachliche Modellgüte bleiben Integrationsaufgaben.
Für die README-Aktualisierung wurde kein neuer Testlauf ausgeführt.

pytest und Ruff stehen aktuell in requirements.txt. pyproject.toml setzt für Ruff
target-version="py314". Ergänzende Dokumente:

    docs/WORKFLOW_DE.md
    docs/DATEIEN_UND_SPEICHERORTE.md
    docs/TESTS_DE.md
    README_TESTS_UND_DOKUMENTATION.md

Diese Dokumente beschreiben teilweise einen älteren Stand und verweisen unter
anderem auf eine derzeit nicht vorhandene requirements-dev.txt. Für aktuelle
Startoptionen und Ausgabedateien diese README und den Quellcode heranziehen.
Der frühere Einzeltest check_logging.py ist entfernt; Loggingtests liegen in tests.


9. Fehlerbehebung und aktuelle Grenzen
--------------------------------------

Fehlendes Argument --csv / FileNotFoundError
    Einen .csv-Dateinamen direkt unter Training_data angeben. Hauptquelle und
    automatisch geloggte Testauswahl prüfen. Das Skript durchsucht keine
    Trainingsunterordner; --test-csv ist keine aktuelle CLI-Option mehr.

Fehlende CSV-Spalten / No complete reference-grid maps
    Auch jede automatisch gewählte Testquelle prüfen. Leere Dateien oder
    Dateien ohne vollständige Referenzkennfelder werden nicht übersprungen.
    MAP_SIZE, Achsen und Schema müssen passen. Die Stromdichtezuordnung
    korrigiert nur eindeutig durch CSV-Rundung erklärbare Abweichungen.

Overlapping machines / Duplicate machines between external test sources
    Den Trainingsordner auf unabhängige CSV-Quellen beschränken. Umbenennen
    oder Kopieren einer Datei beseitigt gleiche Geometrie-IDs nicht.

Zu wenige vollständige Maschinen
    Die Hauptquelle benötigt mindestens fünf gültige Kennfelder nach Bereinigung.
    Doppelte, fehlende oder zusätzliche Betriebspunkte entfernen ganze Maschinen.

ModuleNotFoundError / DLL- oder BLAS-Importfehler
    Projektwurzel und Interpreter Env_NN_AFM prüfen. Die Conda-Datei legt
    OpenBLAS/nomkl fest; die Python-Pakete mit derselben Umgebung installieren.
    Für den Windows-Worker-Schutz wird pywin32 benötigt.

Motor-CAD oder Basismodell fehlt / Rechnung scheitert
    Externe Pfade, Lizenzzugang und Basisdatei prüfen. eduVPN-Prozessstart ist
    kein Nachweis einer funktionierenden Lizenzverbindung. Details stehen im Log.

Unpassende Metadaten
    Cache-Quellen benötigen Metadatenversion 2 mit gleichem Setup und Fingerprint.
    Alte Version-1-Sidecars können nur bei gleicher Setup-Konfiguration über
    migrate_simulation_metadata() geprüft und migriert werden. Quellcode-Hashes
    sind kein Bestandteil des neuen Simulationsfingerprints; die alte Liste
    gerichteter Fingerprintfreigaben wird nicht mehr verwendet.

Imported legacy files can only be used as a cache source
    CSV-Dateien mit adoption-Vermerk nicht als Ausgabeziel fortschreiben.
    Eine neue Zieldatei wählen. Der gespeicherte CSV-Hash muss unverändert passen.

Altdatei-Metadaten neu schreiben
    CSV_PATH, MOT_SHA256 und REASON in main_adopt_legacy_csv.py prüfen. Der
    Einstieg schreibt direkt und kann vorhandene Sidecars ersetzen. Die API
    adopt_legacy_csv() prüft dagegen standardmäßig ohne Schreiben und lehnt
    vorhandene Sidecars ab. Beide erfassen eine historische Setup-Annahme.

Fenster während der Arbeit geschlossen
    Das fordert einen Abbruch an und blendet das Fenster während der Bereinigung
    aus. PipelineCancelledError wird weitergereicht; der Suchabbruch erzeugt kein
    finales Modell. Bereits geschriebene Simulationszeilen und Studien bleiben
    bestehen. Nach regulärem Suchabschluss dagegen das Fenster schließen, um
    finales Training und Export fortzusetzen.

Worker-Zeitbudget überschritten
    --worker-timeout gilt für die Suchworker insgesamt. Laufende Prozesse werden
    beendet und bereinigt. Nicht abgeschlossene Trials können in der SQLite-
    Studie verbleiben. Bei gleichem Experiment werden weitere Trials ergänzt;
    eine Wiederaufnahme einzelner abgebrochener Trials ist nicht implementiert.

FloatingPointError im Training
    Daten, Normalisierung, Lernrate und Modellvorhersagen prüfen. Nicht endliche
    Verluste oder Gradienten führen zum Abbruch vor dem nächsten Optimiererschritt.

Modellbundle oder Bericht wird abgelehnt
    Merkmal-/Kanal-/Achsenfolge, Skalierung und Dimensionen müssen passen.
    prediction_report.pdf benötigt die zur evaluation.pt gehörende surrogate.pt
    mit identischem SHA-256 sowie genau sieben Ausgabekanäle.

Arial oder evaluation.pt fehlt / keine neuen Diagramme
    Nach Suchende zunächst finales Training und Export abwarten. evaluation.pt
    muss neben surrogate.pt liegen. Arial muss installiert sein; ein Fontfallback
    ist deaktiviert. Scheitert die Plotphase, können Modell und Auswertungsdatei
    bereits vorhanden sein. Nach Korrektur main_plot_results.py für diesen Lauf
    starten; neues Training ist dafür nicht erforderlich.
