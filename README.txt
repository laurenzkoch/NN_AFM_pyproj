NN_AFM_pyproj
=============

Motor-CAD-Simulationen und neuronale Ersatzmodelle für Axialflussmaschinen
Abgleich mit Quellcode, Tests und Konfiguration: 28.10.2026

Das Projekt erzeugt Maschinengeometrien und Betriebspunkte, berechnet sie mit
Ansys Motor-CAD und schreibt Ergebnisse als CSV. PyTorch verarbeitet vollständige
Kennfelder. Optuna sucht für jede der sieben Ergebnisgrößen eine eigene
Netzarchitektur. main_optimize.py unterstützt drei Varianten:

    Referenzmodell   Geometrie -> MLP -> vollständiges Kennfeld je Kanal
    CNN-Sammlung    Kennfeld -> Encoder -> latenter Raum -> Decoder -> Kennfeld
    Kombimodell     Geometrie -> MLP -> eingefrorener CNN-Decoder -> Kennfeld

Eine CNN-Sammlung rekonstruiert vorhandene Kennfelder. Erst Referenz- und
Kombimodell sagen Kennfelder aus Geometrien voraus. Simulation und Training sind
getrennt; Daten werden nicht automatisch nach Training_data kopiert.

Aktuell vor einem produktiven Lauf beachten:

- Beide direkt in Training_data vorhandenen CSVs haben 29 statt der erforderlichen
  39 Spalten und werden vom heutigen Datenlader abgelehnt (Abschnitt 5).
- Der Drehzahl-Rundungsabgleich weicht von der aktuellen Formel mit pi ab
  (Abschnitt 10). Die Startbeispiele setzen passende Daten voraus.
- Workflow, Artefaktreferenz und Testkonzept in docs sind aktualisiert.
  Abschnitt 11 beschreibt ihre Zuständigkeiten und noch offene Prüfaufgaben.

Der erste Projektabgleich erfasste alle 82 Python-Dateien unter src und tests
statisch und syntaktisch per AST sowie Konfiguration, Paketversionen und CSV-
Kopfzeilen. Bei der anschließenden Dokumentationsaktualisierung wurden diese
README und die drei Dateien in docs geändert; Code und Daten blieben unverändert.
Verweise, Testmatrix und dokumentierte CLI wurden erneut geprüft. Kein pytest-,
Ruff-, Trainings- oder Simulationslauf wurde dafür gestartet. Binäre Modelle,
Diagramme und Sicherungen wurden nicht fachlich neu validiert.


1. Struktur und Einstiegspunkte
--------------------------------

    src/config/                     Pfade, Parameter und Logging
    src/executable_files/           Startskripte
    src/modules/geometry/           Lambda-Definitionen und Geometrieumrechnung
    src/modules/simulation/         Serie, Generatoren und Prozesspool
    src/modules/process/            Motor-CAD, eduVPN und Windows-Prozessschutz
    src/modules/Pipeline_and_GUI/   Ablauf, Tk-GUI, Ereignisse und Abbruch
    src/modules/data_managment/     CSV, Dateien, Hashes und Metadaten
    src/modules/neuronal_networks/  Daten, Training, Modelle, Optimierung und I/O
    src/modules/plotting/           Einzelplots und PDF-Bericht
    tests/                          Zwölf Testmodule und gemeinsame Helfer
    docs/                           Workflow, Speicherorte und Testkonzept

Startskripte unter src/executable_files:

    main_mcad_sim.py          Konfigurierte Simulationsserie
    main_optimize.py          Referenzmodell, CNN-Sammlung oder Kombimodell
    main_plot_results.py      Gespeicherte Auswertung erneut darstellen
    main_adopt_legacy_csv.py  Metadaten einer konfigurierten Alt-CSV schreiben
    machine_test.py           Generatorgeometrien ohne Motor-CAD-Start prüfen

Alle Befehle gelten aus der Projektwurzel in PowerShell:

    Set-Location D:\Python_Projects\NN_AFM_pyproj

Imports verwenden src.config/src.modules; keine zusätzliche PYTHONPATH-Einstellung
für Modulstarts nötig. data_managment, neuronal_networks und
optimization_pipline.py sind die tatsächlichen Schreibweisen.


2. Umgebung und Abgleich der Abhängigkeiten
-------------------------------------------

PyCharm verwendet Env_NN_AFM mit Python 3.14.4:

    D:\Miniconda\envs\Env_NN_AFM\python.exe

Simulation und Windows-Jobobjekte benötigen Windows. Tkinter stellt die GUI
bereit; die Plotfunktionen verlangen eine installierte Arial-Schrift ohne
Fontfallback. Motor-CAD, Lizenz, .mot-Basisdatei und gegebenenfalls eduVPN werden
separat bereitgestellt. Python 3.14 wird unter anderem für
ProcessPoolExecutor.kill_workers() verwendet.

Neue Conda-Umgebung in einer für Conda eingerichteten PowerShell:

    conda env create -f environment.yml
    conda activate Env_NN_AFM

Bestehende Umgebung aktualisieren:

    conda env update -n Env_NN_AFM -f environment.yml

Alternativ in einer passenden Python-3.14.4-Umgebung mit Tkinter:

    python -m pip install -r requirements.txt

environment.yml installiert Python, pip, Tk, Zeitzonendaten sowie aufgeführte
numerische, Plot- und HTTP-Pakete mit Conda. Der pip-Abschnitt bindet
requirements.txt ein. nomkl und blas=*=openblas legen den Conda-BLAS-Unterbau
fest; reine pip-Installation bildet diese Konfiguration nicht nach.

Überlappende Vorgaben:

    Paket                   requirements.txt       Conda-Eintrag in YAML
    numpy                   ==2.5.3                ==2.5.3
    scipy                   ==1.18.1               ==1.18.1
    pandas                  ==3.0.2                =3.0.2
    matplotlib              ==3.10.9               =3.10.9
    plotly                  ==6.7.0                =6.7.0
    requests                ==2.34.2               =2.34.2
    idna                    ==3.15                 >=3,<4
    charset-normalizer      ==3.4.7                =3.4.7

Diese Vorgaben sind vereinbar. Bei idna schränkt requirements den weiteren
Conda-Bereich ein. Python-tzdata==2026.2 wird unter Windows über requirements
installiert; der Conda-Eintrag tzdata ist eine eigene Systemdaten-Abhängigkeit.
Python und Tk gehören nicht in requirements.txt.

Weitere requirements-Einträge werden über den pip-Abschnitt eingebunden:

    torch==2.13.0                Netze und CPU-Training
    ansys-motorcad-core==0.8.6    Motor-CAD-Anbindung
    optuna==5.0.0                Architektur-Suche und JournalStorage
    seaborn==0.13.2              Aktuell ohne Import im Projekt
    pytest>=8,<10                Regressionstests
    ruff>=0.14,<1                Quellcode- und Formatprüfung
    pywin32, nur Windows         Jobobjekte und Windows-Testhelfer
    types-pywin32, nur Windows   Typinformationen für pywin32

Für die direkten Imports in src und tests fehlt kein Python-Paket. Alle festen
Pins stimmen mit installierten Distributionsversionen überein. pytest 9.1.1 und
Ruff 0.16.8 erfüllen die Bereiche; pywin32 312 und types-pywin32
312.0.0.20261006 sind vorhanden. Das ist ein lokaler Abgleich, kein Nachweis einer
frischen Installation oder vollständiger Konfliktfreiheit indirekter Pakete.

Indirekte Pakete wie psutil/openpyxl für PyMotorCAD, fonttools für Matplotlib
und SQLAlchemy/PyYAML für Optuna deklarieren die jeweiligen Pakete selbst.
Für direkte Imports sind keine zusätzlichen Einträge nötig. SQLAlchemy bleibt
Optuna-Abhängigkeit trotz Journal-Speicher im Projekt. requests/idna/
charset-normalizer sind der explizit gepflegte HTTP-Unterbau. sqlite3, tkinter
und zoneinfo sind Standardbibliothek; scikit-learn wird nicht importiert.

Vorschläge für die unveränderten Umgebungsdateien:

- plotly aus requirements.txt UND environment.yml und seaborn aus requirements
  entfernen, sofern frühere externe Auswertungsskripte nicht unterstützt werden
  sollen. Beide werden im aktuellen Projektcode nicht importiert.
- pytest, Ruff und types-pywin32 optional in Entwicklungsabhängigkeiten auslagern.
  pywin32 bleibt Laufzeitabhängigkeit. requirements-dev.txt existiert derzeit
  nicht; aktuelle Installationsbefehle verwenden requirements.txt.
- Gemeinsame Pins zentral pflegen und bewusst doppelte Conda-Vorgaben synchron
  halten. Für exakte Wiederherstellung einen Lock-/Umgebungsexport vorsehen:
  Werkzeugbereiche, Windows- und indirekte Pakete sind nicht komplett fixiert.
- Kopfkommentare mit Datum 12.10.2026 sowie den überholten SQLite-Kommentar zur
  Optuna-Speicherung aktualisieren; der Projektcode verwendet JournalStorage.


3. Konfiguration und Geometrieschema
------------------------------------

Externe Pfade in src/config/config_paths.py:

    PATH_MCAD_DATA        D:\MotorCAD
    PATH_MCAD_BASEFILE    D:\MotorCAD\base_first_data.mot
    PATH_MCAD_EXE         C:\Program Files\AnsysMotorCAD\v252\Motor-CAD_2025_2_1.exe
    PATH_eduVPN_EXE       C:\Program Files\eduVPN\Core\eduVPN.Client.exe

data-Pfade werden aus dem Projektpfad ermittelt. Jeder Simulationslauf erhält
unter D:\MotorCAD\temp_files einen eigenen Ordner, jeder Worker darin eine
private MOT-Kopie mit unverändertem Originaldateinamen.

Aktuelle Vorgaben in config_mcad_parameters.py:

    MAP_SIZE              (5,5,5), 125 Betriebspunkte je Kennfeld
    SLOT_NUMBER           18, fest
    POLE_NUMBER           12, fest
    AIRGAP                0,75 ... 3
    STATOR_BORE           11 ... 100
    SHAFT_SPEED_NORM      0,05 ... 1
    Q_CURRENT_DENSITY     1 ... 25
    D_CURRENT_DENSITY     -10 ... 0

Achsenreihenfolge in Generator, Datenlader und MAP_SIZE: normierte Drehzahl, q, d.
AXIS_LIST: Shaft_Speed_norm, RMSPhaseCurrent_Q_density, RMSPhaseCurrent_D_density.
Generator und DataLoader._recalculate_data berechnen:

    Shaft_Speed_norm = ShaftSpeed / (Stator_Lam_Dia * pi)

Drehzahlstützstellen: 0,05; 0,2875; 0,525; 0,7625; 1.
q-Stützstellen: 1; 7; 13; 19; 25. d-Stützstellen: -10; -7,5; -5; -2,5; 0.
Zum abweichenden Rundungsabgleich siehe Abschnitt 10.

Zwölf Lambdas sind zentral in geometry/lambda_geometry.py definiert und werden
in beide Richtungen umgerechnet. LAMBDA_BOUNDS enthält:

    lambda_1     D_Si / D_Sa                             0,2 ... 0,8
    lambda_2     Magnet_Arc_[ED] / 180                   0,5 ... 0,9
    lambda_3     g / l_PM                               0,2 ... 1
    lambda_4     s*b_S / (pi*D_Si)                      0,3 ... 0,7
    lambda_5     b_S / d_S                              0,1 ... 1
    lambda_6     h_PM / ((D_Sa-D_Si)/2)                  0,6 ... 1
    lambda_7     d_emb / (radiale Breite-h_PM)           0 ... 0,95
    lambda_8     2*p*(l_S-d_S)/(pi*(D_Sa+D_Si))          0,1 ... 0,4
    lambda_9     l_PM / l_R                             0,2 ... 1
    lambda_10    Slot_Opening / Slot_Width               0,1 ... 1
    lambda_11    Tooth_Tip_Depth / Slot_Depth            0 ... 0,3
    lambda_12    Slot_Corner_Radius / (Slot_Width/2)      0 ... 1

s und p sind hier die Codewerte Slot_Number und Pole_Number. LambdaValues verlangt
lambda_6 strikt kleiner als 1; 1 ist die obere Samplinggrenze.
MACHINE_DEFINITION in geometry/feature_schema.py umfasst 14 Eingaben:

    Airgap, Stator_Bore, lambda_1, ..., lambda_12

Nur Airgap/Stator_Bore werden nach Konfigurationsgrenzen skaliert, Lambdas bleiben
unverändert. Pol-/Nutzahl sind keine MLP-Eingaben und im Sampling fest.
DEFAULT_VALUES wird nicht vollständig in Motor-CAD gesetzt. Weitere Maschinen-,
Wicklungs- und Materialwerte stammen aus der MOT-Basisdatei; DEFAULT_VALUES
benennt unter anderem ausgelesene Prüfwerte.


4. Simulation und Metadaten
----------------------------

Nach Prüfung von Pfaden, Lizenz und gewünschtem Umfang:

    python -m src.executable_files.main_mcad_sim

Aktuelle Kombinationen (B,L) in main_mcad_sim.py:

    (1,100), (1,200), (1,300), (100,1), (200,1), (300,1),
    (100,100), (200,200), (300,300)

gen_type="v2", num_workers=4, start_eduvpn=False; keine CLI-Optionen.
B zählt Luftspalt-/Statorbohrungskombinationen, L gemeinsam gesampelte zwölf Lambdas.
Latin-Hypercube-Seeds: 17 für B, 42 für L.

    Geometrien je Kombination:   B * L
    Betriebspunktanforderungen:  B * L * prod(MAP_SIZE)
    Gesamte Startliste:          141.200 Geometrien, 17.650.000 Punkte

Anforderungen vor Wiederverwendung/Verwerfung, keine garantierten Ergebniszeilen.
Generator unterstützt nur v2. Pipeline enthält noch v1-Punktzählung und Standard
gen_type="v1"; eigene API-Aufrufe müssen v2 setzen.

Kombinationen laufen nacheinander. SimulationSeries.prepare_output() prüft vorab
Metadaten-Voraussetzungen. CSV und Sidecar entstehen erst unmittelbar vor der
jeweiligen Kombination, nicht für die ganze Serie vorab. Workerzahl wird auf
CPUs und Auftragspunktzahl begrenzt. Ein Batch umfasst workers * 125 Punkte.
Jeder Batch erhält einen neuen ProcessPoolExecutor und jeder gestartete Worker
eine eigene Motor-CAD-Instanz zur Begrenzung ihrer Lebensdauer aus Stabilitätsgründen.

Standardname:

    Results_EMagnetic_simulation_<B>_<L>_<MAP_SIZE[0]>_GenType-v2_<Datum_Zeit_Zone>_<ID>.csv

Die Rasterzahl ist nicht die Gesamtpunktzahl. Sidecar: <Dateiname>.csv.meta.json.
Parametertoleranz für bekannte Punkte: absolut 1e-4.

    existing    Im eigenen Ziel vorhanden, keine neue Zeile
    copied      Aus anderer kompatibler Quelle übernommen
    simulated   Neu berechnet und gespeichert
    dropped     Abgelehnte Geometrie/Rechnung, keine Ergebniszeile

Gleiche offene Punkte je Batch werden einmal berechnet, aber für alle Anforderungen
gezählt. Zeilenfolge hängt von Workerfertigstellung ab. Cache-Suche liest nur CSVs
direkt in MCAD_Results und ein explizit gewähltes bestehendes Ziel, keine Archive.

Metadatenversion 2 bindet MOT-/EXE-Hashes, ansys-motorcad-core-Version, externe
Abhängigkeitsrevision/-dateihashes sowie CSV-Schema und vier Nachkommastellen.
Revision: materials_and_solver_2026_10_13; MCAD_DEPENDENCY_FILES ist leer.
Indirekte Materialdateien werden nicht entdeckt. Quellcode-Hashes gehören nicht
zum Simulationsfingerprint. Fachliche Änderungen außerhalb erfasster Dateien
müssen über die Revision berücksichtigt werden. is_compatible_metadata verlangt
identische Version-2-Payloads und gültige identische Fingerprints; keine gerichteten
Freigabelisten mehr. Native kompatible Ziele sind über csv_paths fortführbar;
Standardaufrufe erzeugen neue Namen. Ungültige Quellen werden übersprungen,
ungültige Ziele führen zum Fehler. CSV und Sidecar gemeinsam aufbewahren.

Altdatei-Einstieg:

    python -m src.executable_files.main_adopt_legacy_csv

CSV_PATH, MOT_SHA256 und REASON werden im Skript eingestellt. Es schreibt direkt,
kann Sidecars ersetzen und hat keine Dry-Run-CLI. Die CSV bleibt unverändert.
adoption speichert CSV-Hash, Begründung, UTC-Zeit und original_setup_verified=False:
historische Annahme, keine nachträglich verifizierte Herkunft.

Separate API adopt_legacy_csv(): verlangt Setup-Annahme, Begründung, passenden
MOT-Hash; lehnt vorhandene Sidecars ab und schreibt erst mit write=True.
Adoptierte Dateien sind nur Cache-Quellen; gespeicherter CSV-Hash muss passen.
migrate_simulation_metadata() prüft Version 1 -> 2 bei ansonsten gleichem Setup
und schreibt erst mit write=True. Das ergänzt keine fehlenden CSV-Spalten.


5. CSV-Auswahl und Datenaufbereitung
------------------------------------

Format: UTF-8, Semikolon, Dezimalkomma, vier Nachkommastellen für Floatausgaben.
CSV_PARAMETERS umfasst aktuell 39 Spalten:

    Geometrie/Wicklung (26):
    Pole_Number, Slot_Number, Airgap, Stator_Bore, Stator_Lam_Dia,
    AFM_D_Rotor, Magnet_Arc_[ED], Magnet_Length, Magnet_Thickness,
    CircumferentialSegments, Slot_Depth, Slot_Width, Slot_Opening,
    Slot_Corner_Radius, Tooth_Tip_Depth, Magnet_Embed_Depth,
    Rotor_Lam_Length, Stator_Lam_Length, Motor_Length, MagPhases,
    MagTurnsConductor, MagThrow, ParallelPaths, WindingLayers,
    MagWindingOffset, NumberOfCoils

    Betriebspunkt (6):
    ShaftSpeed, RMSPhaseCurrent_D, RMSPhaseCurrent_Q,
    RMSCurrentDensity, PeakCurrent, PhaseAdvance

    Ergebnisse (7), zugleich Kanalreihenfolge:
    ShaftTorque, SystemEfficiency, BMean_AirGap, ConductorLoss,
    MagnetLoss, StatorIronLoss_Total, RotorIronLoss_Total

Training verlangt alle Spalten und ordnet sie intern; Zusatzspalten werden nicht
übernommen. Simulation prüft auch die exakte Reihenfolge.

Direkt in Training_data liegen beim Abgleich:

    Results_EMagnetic_simulation_1_10_5_GenType-v2_26.10.2026_16-14-42_CET_99d7fc38.csv
    Results_EMagnetic_simulation_1_20_5_GenType-v2_26.10.2026_16-14-42_CET_99d7fc38.csv

Kopfzeilen und Sidecars haben noch 29 Spalten. Es fehlen CircumferentialSegments,
Slot_Corner_Radius, Tooth_Tip_Depth, MagPhases, MagTurnsConductor, MagThrow,
ParallelPaths, WindingLayers, MagWindingOffset und NumberOfCoils. Radius/Zahntiefe
sind für neue Lambdas nötig. Vorschlag: kompatible neue Daten oder belegte Migration
mit historisch nachgewiesenen Werten vorsehen. Fehlendes nicht pauschal mit heutigen
DEFAULT_VALUES ergänzen. Sidecar-Migration allein behebt diese Inkompatibilität nicht.

--csv wählt eine Hauptdatei direkt unter Training_data. Alle anderen CSVs dort
werden alphabetisch als unabhängige Testquellen gewählt. Keine Unterordner/Sidecars
als Datenquelle. --test-csv existiert nicht mehr. Im Kombimodus werden Dateien
mit dem Hash der ursprünglichen CNN-Hauptquelle zusätzlich von Tests ausgeschlossen.
Auswahl wird geloggt; ohne weitere CSV ist test=None. Nur bewusst ausgewählte
kompatible Quellen im Ordner belassen. Umbenennen beseitigt keine Überschneidungen.
write_from_csv_in_csv() kopiert auch Kopfzeilen und ist kein validierender Export.

DataLoader.load():

1. Hash, Spalten und endliche Werte prüfen, Geometrie validieren, zwölf Lambdas
   rekonstruieren. Leiterfläche aus PeakCurrent/RMSCurrentDensity, q-/d-Stromdichte
   und normierte Drehzahl berechnen.
2. Drehzahl-/Stromdichte-Rundungsabgleich auf Referenzraster anwenden; nur eindeutig
   durch CSV-Rundung erklärbare Zuordnungen sind vorgesehen.
3. Nach MACHINE_DEFINITION gruppieren. Doppelte, fehlende oder zusätzliche Punkte
   verwerfen das ganze Kennfeld. Keine Interpolation. Leere Restquellen scheitern.
4. Tensoren: geometry_tensor (N,14), map_tensor (N,5,5,5,7).
   Kennfelddimensionen: Maschine, normierte Drehzahl, q, d, Ergebniskanal.
5. Hauptquelle nach ganzen Maschinen ca. 80/20 teilen, fest Seed 42; mindestens
   fünf vollständige Maschinen. Testquellen verbinden. Exakte Geometrie-IDs dürfen
   sich zwischen Splits nicht überschneiden, doppelte Testmaschinen scheitern.
   IDs basieren auf unnormalisierten float32-Eingaben, keine Ähnlichkeitsprüfung.
6. Airgap/Stator_Bore nach Konfigurationsgrenzen skalieren. Zielwerte je Kanal mit
   ausschließlich Trainingsminima/-spannweiten normalisieren, Nenner >=1e-8.

Validation/Test können außerhalb [0,1] liegen. Split/Shuffle nutzen Seed 42
unabhängig von --seed. CSVs bleiben unverändert; Trainings-Sidecars werden nicht geprüft.


6. main_optimize.py: drei Modellvarianten
-----------------------------------------

Platzhalter durch vorhandene schema-/rasterkompatible Dateien ersetzen.
Referenzmodell mit kleinem Suchumfang:

    python -m src.executable_files.main_optimize --csv "TRAIN.csv" --network-type MLP --trials 1 --max_layer 1 --max_neuron 10 --unit_step 10

CNN-Sammlung:

    python -m src.executable_files.main_optimize --csv "CNN_TRAIN.csv" --network-type CNN --trials 1 --max_layer 1 --max_neuron 10 --unit_step 10

Kombimodell:

    python -m src.executable_files.main_optimize --csv "MLP_TRAIN.csv" --network-type MLP --autoencoders "data\trained_nn\AutoencoderCollections\LAUF\AutoencoderCollection.pt" --trials 1 --max_layer 1 --max_neuron 10 --unit_step 10

Argumente:

    --csv DATEI              Pflicht: CSV-Dateiname ohne Verzeichnispfad
    --network-type MLP|CNN   Standard MLP
    --autoencoders PFAD      Optionale CNN-Sammlung, nur mit MLP
    --max_layer ZAHL         Maximale versteckte Schichtzahl, Standard 5
    --max_neuron ZAHL        Maximale Neuronen-/Featurezahl, Standard 500
    --unit_step ZAHL         Kandidaten-Schrittweite, Standard 10
    --seed ZAHL              Standard 42
    --trials ZAHL            Weitere Trials je Worker UND Kanal, Standard 4000
    --worker-timeout SEK     Workerzeitbudget je Kanal, optional

Unterstriche gehören zu --max_layer, --max_neuron und --unit_step.
Fest im Einstieg: learning_rate=0.001, weight_decay=0.001, num_epochs=2000,
patience=100, num_kernel=4. Workerzahl wird auf CPUs begrenzt. Sieben Kanäle
laufen nacheinander, pro Kanal bis zu vier Suchworker. Bei vier Workern bedeutet
--trials 1 insgesamt 28 Suchtrainings plus sieben finale Trainings. Standard
4000 plant 112.000 Suchtrainings plus sieben finale Trainings. Ein kleines
Trialbudget begrenzt nicht die 2000 möglichen Epochen je Training.

MLP sucht 1 bis max_layer ReLU-Schichten mit unit_step, 2*unit_step, ... bis
max_neuron Neuronen, Ausgabe linear. CNN sucht strikt steigende Featurezahlen
und latente Dimension 1 bis 50, einkanalige Conv3d-Encoder/Decoder, Kernel 3.
MLP bewertet vollständigen normalisierten Validierungs-MSE, CNN vollständigen
Rekonstruktions-MSE. Kombimodell: CNNs eingefroren, MLPs liefern latente Decoder-
eingaben. Verlust für Suche/Training verwendet 3*3*3 Stützstellen (Anfang/Mitte/
Ende jeder Achse); Endauswertung verwendet vollständige Kennfelder.

Kombidaten werden auf gespeicherte CNN-Zielskalierung umgerechnet. Achsen, Form
und Herkunft werden geprüft. Testmaschinen dürfen weder im CNN-Training noch
in dessen Validierung vorkommen. metrics.training_csv und ihr Quellenhash müssen
die CNN-Hauptquelle identifizieren. Kombibundle enthält auch CNN-Gewichte/Herkunft.

NNTrainer: CPU, Adam, kompletter Trainingsdatensatz pro Schritt. Early Stopping
verlangt mindestens 1e-6 Verbesserung und stellt den besten Validierungszustand
wieder her. Nicht endliche Verluste/Gradienten brechen ab. Testdaten steuern weder
Architekturwahl noch Early Stopping.

Nach allen Kanalsuchen erstellt OptimizationService neue Netze mit den jeweils
besten Architekturen und trainiert sie INNERHALB der Pipeline. Trialgewichte
werden nicht dauerhaft übernommen. Finale Trainings laufen noch mit geöffnetem
Fortschrittsfenster. Nach regulärem Ende Fenster schließen; pipeline.start()
liefert das trainierte SurrogateModel oder die AutoencoderCollection. Danach
folgen Speichern, erneutes Laden zur Kontrolle, Auswertung und Diagrammexport.

--worker-timeout begrenzt die Suchworkerwartephase pro Kanal inklusive Prozessstart,
nicht Gesamtlauf, Einzeltrial oder finale Trainings. Abbruch wird im finalen
Training zwischen Kanälen geprüft, nicht innerhalb der NNTrainer-Epochenschleife.


7. Studien, Modelle, Auswertung und Anwendung
---------------------------------------------

Aktueller Optuna-Speicher:

    data/optimization/<Fingerprint>/study.journal
    Studienname: <Fingerprint>

Worker teilen JournalStorage mit JournalFileBackend und JournalFileOpenLock.
Kanalangabe erzeugt einen Fingerprint je Kanal. Er bindet Datenhashes, Splits,
Achsen, Skalierung, Features/Kanäle, Modelltyp, ggf. feste CNN-Gewichte, Suchraum,
Training, Workerzahl, Seed, Python-/Paketversionen und ausgewählte Quelldateien.
Trialbudget/Zeitlimit gehören nicht dazu. experiment_fingerprint/experiment_config
stehen in Studienattributen; keine separate Konfigurations-JSON. Gleicher
Fingerprint ergänzt Trials; beste Architekturen können aus früheren Aufrufen
stammen. Studien speichern keine Netzgewichte. R² ist kein aktuelles Trialattribut;
ChannelOptimizationResult.r_squared wird auf None gesetzt. Die ältere Klasse
OptimizationResult ist nicht der aktuelle Pipeline-Rückgabewert.

Historische study.sqlite3-Dateien werden nicht automatisch ins Journal migriert.
Abgebrochene Einzeltrainings werden nicht fortgesetzt. Parallele TPE-Suche und
Fortsetzungen sind nicht als bitgleich reproduzierbar zugesichert. Anders als
Simulationsfingerprints enthalten Optimierungsfingerprints Quellcode-Hashes.

Neue Modellpfade:

    data/trained_nn/Reference_model/<Zeitstempel>_<ID>/SurrogateModel.pt
    data/trained_nn/Combi_model/<Zeitstempel>_<ID>/SurrogateModel.pt
    data/trained_nn/AutoencoderCollections/<Zeitstempel>_<ID>/AutoencoderCollection.pt

Jeder vollständig exportierte Lauf enthält daneben:

    evaluation.pt
    result_plots/<Kanal>_prediction_vs_sim_ref-model.png  (7 Dateien)
    result_plots/<Kanal>_prediction_vs_sim_ref-model.pdf  (7 Dateien)
    result_plots/prediction_report.pdf

Modelle/evaluation.pt werden atomar über temporäre Dateien ersetzt. Bei späterem
Plotfehler können beide vollständig vorliegen. Sieben Netze werden gemeinsam
gespeichert. SurrogateModel.pt: Format 2 mit individueller Architektur/Gewichten
je Kanal, optionalen CNN-Decodern, Features, Kanälen, Geometriegrenzen, Achsen,
Zielskalierung, Split-IDs, CSV-Hashes, Seed, Metriken und Trainingsdauern.
AutoencoderCollection.pt: Sammlungsformat 1. Loader nutzen CPU, weights_only=True
und Struktur-/Gewichtsprüfungen. Alte surrogate.pt-Bundles mit Surrogatformat 1
unterstützt load_bundle() nicht; keine automatische Migration.

evaluation.pt hat Format 2: prediction oder reconstruction. Es enthält
Maschinen-IDs, normalisierte Geometrieeingaben, physikalische Referenz-/Modell-
kennfelder je Split, Metriken, Achsen, Kanäle, Datenhashes und Modellname/-hash.
value_scale="physical" gilt für Kennfelder; Geometrie bleibt normalisiert.
Metriken: normalisierter Gesamt-MSE, R² je Kanal und ungewichtetes Mittel,
physikalischer MAE/MSE je Kanal. Konstante Referenzkanäle ergeben R²=None und
kein definiertes Gesamt-R². Unterschiedliche Einheiten werden nicht zu einem
physikalischen Gesamtfehler vermischt.

Einzelplots und A4-Querformatbericht zeigen Referenz gegen Vorhersage bzw.
Rekonstruktion. Bericht prüft Modellhash, verlangt sieben Kanäle und Training/
Validation, kennzeichnet fehlende Tests und zeigt Kanalarchitekturen,
Trainingsparameter und Summe der Netztrainingszeiten, nicht Gesamtprogrammlaufzeit.

Erneut plotten:

    python -m src.executable_files.main_plot_results

Vorher MODEL_FOLDER auf kompatiblen Lauf einstellen, etwa Reference_model/LAUF.
Auch absolute Ordner innerhalb PATH_DATA_TRAINED_NN sind erlaubt. evaluation.pt
und Modell müssen zusammenliegen; CSVs sind nicht nötig. Plots werden überschrieben.
Aktuell zeigt MODEL_FOLDER noch auf historischen Lauf 20261020_161109_f1353af2
mit surrogate.pt. Formate gegen heutige Loader prüfen; alte Diagramme belegen
keine heutige Ladekompatibilität.

Geometrievorhersage per API:

    from pathlib import Path
    from src.modules.neuronal_networks.model_data_io.model_io import (
        load_bundle, predict_raw_geometry,
    )
    model, checkpoint = load_bundle(Path("PFAD/ZU/SurrogateModel.pt"))
    # raw_geometry: (N,14), Reihenfolge aus checkpoint["feature_names"]
    prediction = predict_raw_geometry(model, checkpoint, raw_geometry)

Airgap/Stator_Bore in Originaleinheiten, zwölf Lambdas bereits berechnet übergeben,
etwa mit lambdas_from_geometry(). Eingaben bleiben unverändert; Ausgabe sind
physikalische Kennfelder. Keine eigene Vorhersage-CLI/CSV-Ausgabe.
CNN-Sammlungen: load_autoencoders() und reconstruct() auf vorhandenen, anhand
gespeicherter Zielskalierung normalisierten Kennfeldern.


8. GUI, Logging und Abbruch
----------------------------

GUI im Hauptthread, Dienst in einem Thread, Worker in Prozessen. PipelineEvent
transportiert Fortschritt über Queues. Pipelineinstanzen sind einmal startbar.
Simulation zählt simulated/dropped/existing/copied und abgeschlossene Kombinationen.
Voller Balken bedeutet bearbeitet. Optimierungsmaximum: 7*trials*effektive Workerzahl.
Finale Trainings haben keinen eigenen Epochenbalken.

Schließen während Arbeit fordert Abbruch an und versteckt GUI während Bereinigung.
Simulation nutzt kill_workers(); Windows-Jobobjekte schützen Worker-Kindprozesse.
Optimierung beendet/bereinigt Suchprozesse. PipelineCancelledError wird weitergereicht;
regulärer Modell-/Auswertungsexport entfällt. Vorhandene CSV-Zeilen/Studien bleiben.
Finales Kanaltraining kann Abbruch bis zu seinem Ende verzögern. Nach regulärem
Ende Fenster schließen, damit Export folgt. Eigener Tempordner, Queues und Listener
werden bereinigt; extern gestartetes eduVPN bleibt. Kein allgemeiner Motor-CAD-
Rechnungstimeout.

main_mcad_sim/main_optimize loggen ab INFO auf Konsole und je Aufruf in eine Datei:

    data/logs/<Zeitstempel_mit_Mikrosekunden>_<ID>_application.log

UTF-8; Worker/Hauptprozess verwenden während Pipeline QueueListener, davor/danach
Anwendungshandler. Keine aktuellen getrennten *_info.txt-/ *_error.txt-Ausgaben.


9. Prüfungen und Tests
----------------------

    python -m src.executable_files.machine_test

Prüft v2 für (1,3),(1,4),(1,5), insgesamt 1500 Betriebspunkte ohne Motor-CAD-Start/
Ergebnis-CSV; Randbedingungsimport braucht trotzdem ansys-motorcad-core.

    python -m pytest

Die aktuelle Matrix der zwölf Testmodule, gezielte Aufrufe, Isolation,
Ruff-Befehle und Anforderungen an Prüfnachweise stehen in docs/TESTS_DE.md.
Dort sind auch Markergrenzen dokumentiert: Der Windows-Jobtest startet
Unterprozesse ohne subprocess-Marker; der parallele Journaltest trägt nur training.
Der Ausdruck -m "not training and not subprocess" allein schließt daher nicht
alle Unterprozesse aus.

Reale Motor-CAD-/Lizenz-/GUI-Läufe und fachliche Modellgüte bleiben
Integrationsaufgaben. Achsentests und CSV-Fixtures nutzen noch die frühere
Drehzahlformel ohne pi. Für diese Dokumentationsänderung wurde kein neuer
Regressionstest-Erfolgsstatus erhoben; statische Prüfungen ersetzen keinen Testlauf.


10. Aktuelle Grenzen und Codevorschläge (Code unverändert)
----------------------------------------------------------

Drehzahl-Rundungsabgleich
    Generator/_recalculate_data verwenden ShaftSpeed/(D*pi), _align_speed_grid
    Intervalle aus ShaftSpeed/D. Rundung wird nicht konsistent abgeglichen;
    passende Kennfelder können verworfen werden. pi auch in Intervallgrenzen
    verwenden und Tests/CSV-Fixtures an dieselbe Definition anpassen.

Altes Trainingsschema
    Zwei CSVs haben 29 statt 39 Spalten. Neue Daten oder belegte Migration vorsehen.
    Auch jede automatische Testquelle muss passen. Fehlende Spalten, leere
    Quellen und fehlende gültige Kennfelder brechen ab.

Generatorstandard
    Pipeline standardmäßig v1, Generator nur v2. Standard, Typangaben und
    Validierung vereinheitlichen, damit ungültige Modi früh scheitern.

Experimentherkunft
    In der Optimierungs-Hashliste fehlen unter anderem lambda_geometry.py,
    feature_schema.py, surrogate_model.py und finaler Trainingsservice.
    Abgeleitete Werte decken manche, nicht jede Verhaltensänderung ab.
    Fachlich relevante Implementierung vollständig versionieren oder hashen.

Alte Modelle/Plotvoreinstellung
    Kompatiblen Lauf auswählen oder Migration prüfen. Umbenennen allein macht
    alte Bundles nicht mit aktuellen Features und Formaten kompatibel.

Abbruch im finalen Training
    Für schnelle Reaktion CancellationToken in der Epochenschleife prüfen.
    Suchworkerzeitlimit begrenzt diesen Schritt nicht.

Testmarker
    Windows-Jobtest und parallelen Journaltest passend mit subprocess markieren.
    Bis dahin die explizite Auswahl in docs/TESTS_DE.md verwenden.

Diese Codevorschläge sind weiterhin offen und statisch festgestellt. Die
Dokumentationsaktualisierung behebt keine der genannten Implementierungsgrenzen.


11. Dokumentation: Zuständigkeiten und verbleibende Aufgaben
-----------------------------------------------------------

Die Aktualisierung der drei docs-Dateien ist umgesetzt. Veraltete Aussagen zu
v1, CSV-Erstellungszeitpunkt, Metadatenfreigaben, Testauswahl, Features/Achsen,
SQLite, Kanalarchitekturen, finalem Training und Exporten wurden korrigiert.
Die Zuständigkeiten sind:

    README.txt                         Einstieg, Installation, CLI und Grenzen
    docs/WORKFLOW_DE.md                 Vertiefter fachlicher Ablauf,
                                       CNN-/Kombimodus, Datenmigration und Anwendung
    docs/DATEIEN_UND_SPEICHERORTE.md     Artefakte, Formate, Herkunft und Aufbewahrung
    docs/TESTS_DE.md                    Zwölf Testmodule, Marker, Isolation,
                                       Integrationsaufgaben und Prüfnachweise

Installation und aktuelle CLI-Vorgaben bleiben zentral in der README.
Der Workflow verweist darauf; die Testmatrix wird zentral in TESTS_DE.md gepflegt.
Die Artefaktreferenz unterscheidet persistierte Auswertungsdaten vom vollständigen
Trainingszustand. Der Workflow enthält Anforderungen an eine belegte Migration;
ein automatisches Daten-/Modellmigrationswerkzeug existiert weiterhin nicht.

Offen bleiben ausgeführte Integrations- und Güteprüfungen sowie die in Abschnitt 10
genannten Codeänderungen. TESTS_DE.md enthält ein Schema für datierte Prüfnachweise;
ein aktueller erfolgreicher Testlauf wird nicht behauptet. Eine separate
PRUEFPROTOKOLL.md ist derzeit nicht vorhanden und wird nicht vorausgesetzt.

Die frühere Einbauanleitung README_TESTS_UND_DOKUMENTATION.md ist im Arbeitsbaum
nicht mehr vorhanden. Der frühere Archivierungs-/Entfernungsvorschlag ist erledigt.


12. TXT-Dateien und Aufbewahrung
--------------------------------

Aktuell sind README.txt und requirements.txt vorhanden und weiterhin notwendig:
diese Projektanleitung sowie die von environment.yml eingebundene Paketliste.

formatpruefung.txt, konventionspruefung.txt und restpruefung.txt sind im Arbeitsbaum
nicht mehr vorhanden. Die Entfernungsvorschläge sind damit erledigt; diese
Dateien wurden nicht erneut angelegt. Künftige Prüfausgaben nur nach tatsächlichem
Lauf mit Datum, Befehl, Code-/Umgebungsstand und Ergebnis sichern.

CSV samt Sidecar, Trainingsquellen, Modelle, evaluation.pt, Studien, Logs, Code-
und Umgebungsstand aufbewahren. Bundles ersetzen vollständige CSV-Quellen/Umgebung
nicht. Alte Modelle, SQLite/Journals und Backups sind nicht allein wegen Alters
überflüssig. Nach Herkunft und Wiederherstellbarkeit entscheiden; aktive Studien/
Prozessdateien nicht ändern. Details stehen in docs/DATEIEN_UND_SPEICHERORTE.md.
