NN_AFM_pyproj
=============

Automatisierte Motor-CAD-Simulationen und neuronale Netze für Axialflussmaschinen
Dokumentationsstand: 24.09.2026

Das Projekt erzeugt parametrierte Maschinengeometrien und Betriebspunktkennfelder,
berechnet sie mit Ansys Motor-CAD und speichert die Ergebnisse als CSV. Anschließend
lassen sich die Kennfelder mit PyTorch aufbereiten, ein MLP zur Vorhersage der
Simulationsergebnisse und ein 3D-Autoencoder zur Kennfeldrekonstruktion trainieren.

Der aktive Ablauf ist:

    Maschinen- und Betriebspunktgenerator
        -> parallele elektromagnetische Motor-CAD-Simulation
        -> CSV-Ergebnisse
        -> vollständige Kennfelder und Normalisierung
        -> MLP / 3D-Autoencoder
        -> Modellartefakte und Vergleichsdiagramme

Simulation, Training und Diagrammerstellung werden getrennt gestartet.
Für Training und Auswertung vorhandener Daten muss Motor-CAD nicht laufen.


1. Projektstruktur
------------------

NN_AFM_pyproj/
|-- README.txt
|-- environment.yml                 Conda-Umgebung mit Versionsvorgaben
|-- requirements.txt                Gemeinsame Python-Paketliste mit Versionen
|-- data/
|   |-- MCAD_Results/                Simulations-CSV und ältere Datenbestände
|   |-- plots/                      Trainingskurven und Auswertungsdiagramme
|   '-- trained_nn/                 Aufbereitete Daten und trainierte Modelle
'-- src/
    |-- config/
    |   |-- config_paths.py         Datei-, Programm- und Ausgabepfade
    |   '-- config_mcad_parameters.py
    |                               Parameter, Wertebereiche und NN-Merkmale
    |-- executable_files/
    |   |-- main_mcad_sim.py         Simulation starten
    |   |-- main_nn.py               MLP und Autoencoder trainieren
    |   |-- main_create_plots.py     Gespeicherte Modelle auswerten
    |   '-- machine_test.py         Geometrie-Randbedingungen prüfen
    '-- modules/
        |-- data_managment/         CSV-, Datei- und Tempdateioperationen
        |-- simulation/             Pipeline, Prozesspool, Tkinter-Oberfläche
        |   '-- generators/         Maschinenentwürfe und Betriebspunktraster
        |-- process/                Windows-Prozesse, eduVPN, CPU-Ressourcen
        |   '-- motor_cad_processes/ Motor-CAD-Instanzen und Randbedingungen
        |-- neuronal_networks/      Tensoren, Modelle, Training, Vergleichsplots
        '-- plotting/               Matplotlib-, Plotly- und Heatmap-Helfer

Die Verzeichnisnamen "data_managment" und "neuronal_networks" entsprechen genau
der vorhandenen Schreibweise im Code. Die älteren Unterordner in data/ sind
gespeicherte Versuche; die aktuellen Startskripte durchsuchen sie nicht automatisch.


2. Umgebung einrichten
-----------------------

Die Simulations- und Prozesssteuerung ist auf Windows ausgelegt. Die Datei
environment.yml beschreibt die Conda-Umgebung "Env_NN_AFM" mit Python 3.14.4.

Wichtige Pakete und ihre Aufgaben:

- NumPy, pandas: numerische Datenverarbeitung und CSV-Import
- SciPy: Latin-Hypercube-Sampling und Zuordnung zulässiger Nut-/Polpaare
- ansys-motorcad-core: Python-Schnittstelle zu Motor-CAD
- PyTorch (torch): Tensorverarbeitung, MLP und 3D-Autoencoder
- scikit-learn: zusätzlicher MLP-Regressionsworkflow
- Matplotlib, Plotly, seaborn: Diagramme und Heatmaps
- Tk/Tkinter: Fortschrittsfenster der Simulation
- tzdata: Zeitzonendaten für Zeitstempel in Europe/Berlin

Aus dem Projektordner in einer für Conda eingerichteten PowerShell:

    conda env create -f environment.yml
    conda activate Env_NN_AFM

environment.yml bindet requirements.txt über den pip-Abschnitt direkt ein.
requirements.txt enthält alle Python-Pakete für Simulation, Datenaufbereitung,
Training und Auswertung. Die Versionsvorgaben für die wissenschaftlichen Pakete
und requests sind mit der eingerichteten Umgebung abgeglichen.

Conda installiert zuerst Python, Tk und die aufgeführten wissenschaftlichen
Pakete sowie den HTTP-Unterbau für PyMotorCAD. Die anschließende Installation
aus requirements.txt ergänzt die übrigen Python-Pakete. Gemeinsam aufgeführte
Pakete verwenden dieselben Versionsvorgaben. Bei Versionsänderungen müssen diese
Vorgaben in beiden Dateien übereinstimmen.

Für eine bestehende Conda-Umgebung, ebenfalls aus der Projektwurzel:

    conda env update -n Env_NN_AFM -f environment.yml

Alternativ können die Python-Pakete in einer bereits eingerichteten Umgebung
mit Python 3.14.4 installiert werden:

    python -m pip install -r requirements.txt

Dieser alternative Weg setzt eine Python-Installation mit Tk/Tkinter voraus;
Python und Tk werden nicht über requirements.txt installiert. Condas tzdata
liefert System-Zeitzonendaten; der gleichnamige Eintrag in requirements.txt
installiert das Python-Paket für zoneinfo unter Windows. Die HTTP-Pakete
requests, idna und charset-normalizer unterstützen die Motor-CAD-Schnittstelle.
Indirekte Abhängigkeiten lösen die Paketmanager auf; nicht alle davon sind
auf eine feste Version gesetzt.

Importpfade vorbereiten
~~~~~~~~~~~~~~~~~~~~~~

Im Code werden sowohl "src.config"/"src.modules" als auch "config"/"modules"
importiert. Deshalb müssen Projektwurzel UND src im Python-Suchpfad liegen.

Die folgenden Befehle einmal pro PowerShell-Sitzung im Projektordner ausführen:

    $env:PYTHONPATH = (Get-Location).Path + ";" + (Join-Path (Get-Location).Path "src")
    New-Item -ItemType Directory -Force -Path .\data\trained_nn | Out-Null

Die Startbefehle in dieser README werden ebenfalls aus der Projektwurzel
ausgeführt. Es gibt derzeit keine Kommandozeilenoptionen; Einstellungen werden
direkt in den Startskripten bzw. Konfigurationsdateien geändert.

In PyCharm den Interpreter der Umgebung Env_NN_AFM wählen und Projektwurzel
sowie src zum Suchpfad der Run-Konfiguration hinzufügen. Die vorhandene
.idea-Konfiguration referenziert bereits diese Umgebung.

Externe Voraussetzungen für Simulationen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Zusätzlich werden benötigt:

- eine lokale Motor-CAD-Installation mit verfügbarer Lizenz,
- eine passende Axialflussmaschinen-Basisdatei im .mot-Format,
- eine bereits funktionierende Verbindung zum Lizenzserver,
- gegebenenfalls eduVPN für den Zugang zum Lizenznetz.

Motor-CAD, die Lizenz und eduVPN werden nicht durch environment.yml installiert.
Die Basisdatei liegt außerhalb des Projekts und ist nicht Bestandteil des
vorliegenden Projektordners.


3. Zentrale Konfiguration
-------------------------

Dateipfade: src/config/config_paths.py

    PATH_MCAD_DATA        D:\MotorCAD
    PATH_MCAD_BASEFILE    D:\MotorCAD\base_first_data.mot
    PATH_MCAD_EXE         C:\Program Files\AnsysMotorCAD\v252\Motor-CAD_2025_2_1.exe
    PATH_eduVPN_EXE       C:\Program Files\eduVPN\Core\eduVPN.Client.exe

Diese Werte müssen zur lokalen Installation passen.

Die Datenpfade werden relativ zum Speicherort des Projekts ermittelt:

    PATH_DATA_MCAD_RESULTS    data/MCAD_Results
    PATH_DATA_PLOTS           data/plots
    PATH_DATA_TRAINED_NN      data/trained_nn

PATH_TEMP_FILES ist auf "temp_files" gesetzt. Temporäre Kopien des Motor-CAD-
Basismodells entstehen standardmäßig in D:\MotorCAD\temp_files.

PATH_TEMP_DATA und PATH_DATA_MLP_RESEARCH_BASIC sind ebenfalls definiert, werden
aber von den aktuellen Hauptabläufen nicht als reguläre Ausgabeziele verwendet.

Parameter: src/config/config_mcad_parameters.py

Diese Datei enthält die CSV-Spaltenlisten, die NN-Merkmale, DEFAULT_VALUES sowie
die Bereiche für Geometrie- und Betriebspunktsampling. Die neun Lambda-Parameter
beschreiben geometrische Verhältnisse; ihre Definitionen stehen bei den jeweiligen
LAMBDA-Konstanten und in der Geometrieberechnung des Generators.

Aktuelle Samplingbereiche:

    AIRGAP                 0,25 ... 5
    STATOR_BORE             11 ... 100
    SHAFT_SPEED              1 ... 5000
    D_CURRENT_DENSITY      -50 ... 0
    Q_CURRENT_DENSITY        5 ... 100

    LAMBDA_1_1            0,2  ... 0,8
    LAMBDA_1_2            0,5  ... 0,9
    LAMBDA_1_3            0,2  ... 1,0
    LAMBDA_2_1            0,3  ... 0,7
    LAMBDA_2_2            0,1  ... 1,0
    LAMBDA_2_3            0,6  ... 1,0
    LAMBDA_3_1            0,0  ... 0,95
    LAMBDA_3_2            0,1  ... 0,4
    LAMBDA_3_3            0,2  ... 1,0

DEFAULT_VALUES wird bei der Simulation nicht vollständig in Motor-CAD gesetzt.
Die Schlüssel dienen unter anderem zum Auslesen der Randbedingungsparameter.
Nicht vom Generator gesetzte Einstellungen stammen aus dem .mot-Basismodell,
etwa Wicklungs-, Material- und weitere Maschinenparameter. Eine Änderung an
DEFAULT_VALUES allein konfiguriert deshalb nicht das gesamte Motor-CAD-Modell.


4. Motor-CAD-Simulation ausführen
--------------------------------

Vor dem Start die externen Pfade und das Basismodell prüfen. Umfang und Verhalten
werden in src/executable_files/main_mcad_sim.py eingestellt.

Aktuell liegen die Generatoren unter src/modules/simulation/generators. Die
Imports in pipeline.py und machine_generator.py verweisen noch auf den alten
Pfad src.modules.generators. Solange diese Imports nicht angepasst sind, scheitert
der Start mit ModuleNotFoundError. Siehe Abschnitt 9.

Aktuelle Werte des dortigen SimulationPipeline-Aufrufs:

    num_base_designs=1
    num_lam_designs=5
    num_operation_points_per_dim=5
    num_kernel=4
    autostart=True
    start_eduvpn=False
    gen_type="v2"

Start:

    python -m src.executable_files.main_mcad_sim

Dies öffnet das Tkinter-Fortschrittsfenster und startet die Simulation automatisch.
Mit autostart=False erfolgt der Start über die Schaltfläche "Start".

Generatorvarianten und Umfang
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

B = num_base_designs
L = num_lam_designs
O = num_operation_points_per_dim

    Variante    Geometrien      Vorgesehene Betriebspunkte
    v1          B * L^3         B * L^3 * O^3
    v2          B * L           B * L   * O^3

v1 kombiniert drei getrennt gesampelte Gruppen mit jeweils drei Lambda-Werten.
v2 sampelt alle neun Lambda-Werte gemeinsam. Beide verwenden Latin-Hypercube-
Sampling mit festen Seeds. Das Betriebspunktkennfeld ist ein äquidistantes
kartesisches Raster mit O Werten je Achse.

Die aktuelle Startkonfiguration erzeugt somit 5 Maschinengeometrien mit jeweils
125 Betriebspunkten, insgesamt 625 geplante Simulationen. Verworfene Punkte
verringern die Zahl der gespeicherten CSV-Zeilen.

Ein Basisdesign kombiniert Nut-/Polzahl, Luftspalt und Statorbohrung. Der Generator
wählt unterschiedliche zulässige Nut-/Polpaare aus Nuten 3 ... 30 und Polen 2 ... 20.
Die Anzahl der Basisdesigns ist durch die verfügbaren gültigen Paare begrenzt.

Ablauf und Parallelisierung
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Die Pipeline erzeugt die Ergebnisdatei und verarbeitet je Geometrie einen
  Batch mit O^3 Betriebspunkten.
- Für jeden Batch wird ein Prozesspool erstellt. Jeder Worker öffnet eine eigene,
  sichtbare Motor-CAD-Instanz mit einer temporären Kopie des Basismodells.
- Die Zahl der Worker wird auf die gewünschte Anzahl, verfügbare CPUs und die
  Gesamtpunktzahl begrenzt. Verfügbare Motor-CAD-Lizenzen werden nicht geprüft.
- Der Generator setzt Geometrie, Drehzahl, resultierende RMS-Stromdichte und
  Phasenwinkel. Bei geänderter Polzahl wird außerdem die Luftspaltvernetzung
  angepasst.
- Vor der magnetischen Berechnung werden Geometrie, Wickelbarkeit und
  Betriebspunktgrenzen geprüft.
- Nur erfolgreiche Berechnungen werden gespeichert. Ungültige Geometrien und
  fehlgeschlagene magnetische Berechnungen erscheinen als "Dropped".
- Fehler beim Setzen oder Lesen von Motor-CAD-Variablen können den Lauf abbrechen.
- Die Oberfläche zählt simulierte, verworfene und bereits vorhandene Punkte.

Die Randbedingungen prüfen unter anderem Durchmesser und Wellenabstände,
Magnetüberdeckung, axiale Einbaulänge, Nutabmessungen, Wickelbarkeit und einen
Phasenwinkel im Bereich 0 <= PhaseAdvance < 90.

Das Fenster bleibt nach "Finished" geöffnet. Nach dem Schließen wird der
Hauptablauf beendet und die Bereinigung aufgerufen. Schließen während einer
laufenden Simulation ist kein Abbruch: Der Hintergrundthread arbeitet weiter,
und das Programm wartet auf seinen Abschluss.

Ergebnisdatei und Fortsetzen
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Neue Ergebnisse liegen direkt unter data/MCAD_Results, mit diesem Namensschema:

    Results_EMagnetic_simulation_<B>_<L>_<O>_GenType-<v1|v2>_<Zeitstempel>.csv

Der Zeitstempel verwendet Europe/Berlin, z. B. TT.MM.JJJJ_HH-MM-SS_CEST.
Die Zahlen B, L und O im Dateinamen beschreiben die Generatorkonfiguration;
sie sind nicht die drei Dimensionen des Kennfelds.

Zum Fortsetzen eine vorhandene Datei im Pipeline-Aufruf angeben:

    csv_path="Results_EMagnetic_simulation_...csv"

Ein relativer Dateiname wird unter data/MCAD_Results aufgelöst. Vorhandene Dateien
werden nicht überschrieben; neue erfolgreiche Ergebnisse werden angehängt.
Parametersätze, die innerhalb einer absoluten Toleranz von 1e-4 je generiertem
Parameter bereits vorkommen, werden übersprungen.

Für das Fortsetzen müssen CSV-Schema, Generatorparameter und Basismodell
zusammenpassen. Änderungen an Materialien oder anderen Basismodell-Einstellungen
werden durch diese Dublettenprüfung nicht erkannt.

Temporäre Dateien und eduVPN
~~~~~~~~~~~~~~~~~~~~~~~~~~~

SimulationPipeline.stop() löscht den gesamten Ordner
PATH_MCAD_DATA / PATH_TEMP_FILES rekursiv. Dort ausschließlich temporäre
Simulationsdateien ablegen. Im aktuellen Startskript ist der Aufruf von stop()
nach einer Exception aus start() nicht garantiert.

Bei start_eduvpn=True wird eine bereits laufende eduVPN-Anwendung beendet und
neu gestartet; beim Stop wird der gestartete Prozessbaum beendet. Die Prüfung
erkennt nur den laufenden Prozess, keine erfolgreiche VPN- oder Lizenzverbindung.


5. CSV-Datenformat und vorhandene Daten
---------------------------------------

Die aktuellen Ergebnisdateien verwenden:

- UTF-8
- Semikolon als Spaltentrennzeichen
- Komma als Dezimaltrennzeichen
- vier Nachkommastellen für geschriebene Floatwerte
- eine Kopfzeile und anschließend eine Zeile je erfolgreichem Betriebspunkt

Die Zeilenreihenfolge hängt von der Fertigstellung der parallelen Worker ab.

SIM_INPUT_PARAMETERS legt trotz seines Namens die gesamte Reihenfolge fest:
16 Geometrieparameter + 6 Betriebsparameter + 7 Ergebnisgrößen = 29 Spalten.

Geometrieparameter:

    AFM_D_Rotor, Stator_Lam_Dia, Stator_Bore, Airgap, Pole_Number,
    Slot_Number, Slot_Width, Slot_Opening, Slot_Depth, Motor_Length,
    Rotor_Lam_Length, Stator_Lam_Length, Magnet_Length, Magnet_Embed_Depth,
    Magnet_Thickness, Magnet_Arc_[ED]

Betriebsparameter:

    ShaftSpeed, RMSPhaseCurrent_D, RMSPhaseCurrent_Q,
    RMSCurrentDensity, PeakCurrent, PhaseAdvance

Ergebnisgrößen / OUTPUT_NEURON_LIST, in Modellreihenfolge:

    ShaftTorque
    SystemEfficiency
    BMean_AirGap
    ConductorLoss
    MagnetLoss
    StatorIronLoss_Total
    RotorIronLoss_Total

Lambda-Werte und d-/q-Stromdichten sind abgeleitete Trainingsmerkmale.
Sie werden von DataTensor berechnet und stehen nicht als eigene Spalten in
diesem CSV-Exportschema.

Die derzeit direkt in data/MCAD_Results vorhandenen Hauptdateien sind:

    Results_EMagnetic_simulation_3_3_5_GenType-v1_18.09.2026_13-52-26_CEST.csv
        10.000 Datenzeilen; 80 vollständige Geometrien mit je 125 Punkten.
        Der Dateiname in main_nn.py muss vor dem Training angepasst werden.

    Results_EMagnetic_simulation_2_3_5_GenType-v2_22.09.2026_19-36-34_CEST.csv
        750 Datenzeilen; 6 vollständige Geometrien mit je 125 Punkten.

Beide Dateien besitzen ein 5 x 5 x 5-Raster. Ältere CSV-Dateien in den
Unterordnern können abweichende Spalten oder unvollständige Kennfelder enthalten.


6. Datenaufbereitung und neuronale Netze
----------------------------------------

Einstieg: src/executable_files/main_nn.py

Vor dem Training den Dateinamen in der Variablen "file" setzen. DataTensor liest
diese Datei aus data/MCAD_Results. Das Skript wählt nicht automatisch die neueste
Simulation und führt vorhandene CSV-Dateien nicht zusammen.

Der aktuell im Skript eingetragene Name enthält noch kein "GenType-v1" und
existiert nicht mehr unter data/MCAD_Results. Vor dem Start muss die Zuweisung
für den vorhandenen v1-Datensatz deshalb lauten:

    file = "Results_EMagnetic_simulation_3_3_5_GenType-v1_18.09.2026_13-52-26_CEST.csv"

Mit dem unveränderten alten Dateinamen bricht das Training mit FileNotFoundError
ab. Alternativ kann der vorhandene v2-Datensatz ausgewählt werden.

Start nach der Umgebungseinrichtung aus Abschnitt 2:

    python -m src.executable_files.main_nn

Das Skript bereitet die Daten auf, trainiert zuerst das MLP und anschließend den
3D-Autoencoder. Beide Trainings laufen aktuell ausdrücklich auf der CPU.

Datenaufbereitung mit DataTensor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. CSV einlesen und aus PeakCurrent und RMSCurrentDensity die Leiterfläche
   berechnen.
2. d-/q-Stromdichten aus den Phasenströmen rekonstruieren und auf Schritte von
   0,05 runden; die neun Lambda-Verhältnisse aus der Geometrie berechnen.
3. Maschinen nach MACHINE_DEFINITION gruppieren:
   Slot_Number, Pole_Number, Airgap, Stator_Bore und die neun Lambda-Werte.
4. Die globalen Kennfeldachsen aus den vorhandenen Betriebspunkten bestimmen.
5. Ganze Maschinen entfernen, wenn Betriebspunkte doppelt vorkommen oder das
   Kennfeld nicht vollständig ist. Fehlende Simulationen werden nicht interpoliert.
6. Geometrie-, Betriebspunkt- und Ergebnistensoren erzeugen. NaN-Werte im
   Ergebnistensor führen zu einer Fehlermeldung.
7. Ganze Maschinen zufällig in Training und Validierung aufteilen:
   ca. 80 % / 20 %, Split-Seed 42; mindestens fünf vollständige Maschinen.
8. Alle Merkmale mit Mittelwert und Standardabweichung der Trainingsdaten
   standardisieren. Dieselben Werte werden für die Validierung verwendet.

Es gibt im aktiven PyTorch-Ablauf keinen zusätzlichen, unabhängigen Testsatz.
Der feste Split-Seed macht die Datenaufteilung reproduzierbar; Initialisierung
und Training der Netze sind nicht vollständig durch einen globalen Seed fixiert.

Tensorform und Merkmalsreihenfolge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Geometrie:       (N, S, D, Q, 13)
    Betriebspunkte:  (N, S, D, Q,  3)
    Ergebnisse:     (N, S, D, Q,  7)

N bezeichnet die Maschinenzahl, S/D/Q die Anzahl der Werte auf den Drehzahl-,
d-Stromdichte- und q-Stromdichteachsen. Die Geometrie wird über das Kennfeld wiederholt.

DataTensor verwendet standardmäßig OP_NEURONS als label_axis. Damit stimmen
Kennfeldachsen und Betriebsmerkmale mit folgender Reihenfolge überein:

    ShaftSpeed, RMSPhaseCurrent_D_density, RMSPhaseCurrent_Q_density

Das MLP erhält die Merkmale in der Reihenfolge INPUT_NEURON: zuerst die 13
Geometriemerkmale, danach diese drei Betriebsmerkmale. Bei explizit abweichender
label_axis ist deren Reihenfolge maßgeblich.

Bereits gespeicherte data.pkl-Dateien behalten ihre damalige Achsenreihenfolge.
Das Laden baut die Tensoren nicht neu auf. Für die aktuelle Reihenfolge Daten
neu aufbereiten und beide Modelle gemeinsam neu trainieren; ältere Datenobjekte
und Modelle nicht mit neu angeordneten Eingaben mischen.

MLP
~~~

Das MLP erhält je Betriebspunkt 13 Geometriemerkmale und 3 Betriebsmerkmale
und liefert die 7 Ergebnisgrößen. Die aktuelle Architektur in main_nn.py ist:

    16 Eingänge -> 64 -> 32 -> 16 -> 7 Ausgänge

Die versteckten Schichten verwenden ReLU; die Ausgabeschicht ist linear.

3D-Autoencoder
~~~~~~~~~~~~~

Der Conv3dAutoencoder erhält das vollständige simulierte Ergebniskennfeld,
komprimiert es und rekonstruiert es. Im Startskript eingestellt sind:

    map_shape=(5, 5, 5)
    features=(7, 14, 21)
    latent_dim=15
    kernel_size=3
    activation=ReLU

Ein anderer Rasterumfang erfordert eine passende Änderung von map_shape sowohl
beim Modellaufbau als auch in der gespeicherten TrainedCNNAutoencoder-Konfiguration
in main_nn.py. Beide Angaben müssen übereinstimmen.
Der Autoencoder bekommt im aktiven Ablauf Ergebniskennfelder als Eingabe;
er ist für sich allein kein Prädiktor aus der Maschinengeometrie.

Training mit NNTrainer
~~~~~~~~~~~~~~~~~~~~~

Aktuelle Standardwerte:

    Gerät                  CPU
    Batchgröße             1000 Maschinen
    Optimierer             Adam
    Lernrate               0,001
    Weight Decay           0,001
    Maximale Epochen        2000
    Verlustfunktion        MSE auf normalisierten Ergebnissen
    Early-Stopping-Geduld   50 Epochen

Nach Early Stopping bzw. Erreichen der Epochenzahl werden die Gewichte mit dem
besten Validierungsverlust wiederhergestellt.

Speicherorte
~~~~~~~~~~~~

    data/trained_nn/data.pkl
        DataTensor-Objekt einschließlich Datenaufteilung und Normalisierungswerten.

    data/trained_nn/trained_mlp.pt
        MLP-Gewichte und Architekturkonfiguration.

    data/trained_nn/trained_autoencoder.pt
        Autoencoder-Gewichte und Architekturkonfiguration.

    data/plots/mse_loss.png
        Trainings- und Validierungsverlust.

data/trained_nn muss vor dem Start existieren. Der Plotordner wird bei Bedarf
angelegt. Diese Dateinamen sind fest; erneute Läufe überschreiben die Artefakte.
Da beide Trainings dieselbe mse_loss.png schreiben, bleibt nach einem vollständigen
Lauf die Verlustkurve des zuletzt trainierten Autoencoders erhalten.

Datenobjekt und beide Modelle sollten aus demselben Trainingslauf stammen, damit
Skalierung, Merkmalsreihenfolge und Architektur zueinander passen.


7. Modelle auswerten und Diagramme erzeugen
-------------------------------------------

Einstieg: src/executable_files/main_create_plots.py

    python -m src.executable_files.main_create_plots

Das Skript lädt data.pkl, trained_mlp.pt und trained_autoencoder.pt aus
data/trained_nn, rekonstruiert beide Modelle auf der CPU und erzeugt für jede
der sieben Ergebnisgrößen einen Vergleich zwischen Simulation und Modellausgabe.

Die Werte werden in die ursprüngliche Skalierung zurückgerechnet. Die Diagramme
zeigen Trainings- und Validierungsmaschinen getrennt und enthalten:

- eine ideale Vergleichsgerade,
- Pearson-Korrelation R,
- Bestimmtheitsmaß R²,
- minimalen und maximalen absoluten Fehler je Datenteil.

Ausgaben mit 300 dpi unter data/plots:

    <Ergebnisgröße>_prediction_vs_sim_MLP.png
    <Ergebnisgröße>_prediction_vs_sim_Conv3dAutoencoder.png

Ein vollständiger Lauf erzeugt damit 14 Vergleichsdiagramme. Vorhandene Dateien
mit denselben Namen werden überschrieben.

Aktuell werden alle Betriebspunkte dargestellt. Das Skript berechnet zwar einen
Referenzbetriebspunkt mit ref_op_idx=(2, 2, 2), der Plot verwendet diesen aber
nicht als Filter. Kleinere Raster können bereits beim Zugriff auf diesen Index
scheitern. Die Einheit von BMean_AirGap wird im Plot derzeit fälschlich als [W]
beschriftet.

Die Autoencoder-Diagramme bewerten die Rekonstruktion eingegebener Kennfelder;
die MLP-Diagramme bewerten Vorhersagen aus Geometrie und Betriebspunkt.


8. Weitere Module und experimentelle Teile
------------------------------------------

src/modules/neuronal_networks/mlp_with_sklearn.py
    Separater MLPRegressor-Workflow mit StandardScaler für Ein- und Ausgaben,
    zeilenweiser Train/Test-Aufteilung und R²-/MSE-Auswertung.
    Dieser Workflow wird von main_nn.py nicht gestartet.

src/modules/neuronal_networks/nn_training_torch.py
    Enthält zusätzlich einen Trainingszweig mit einem bereits trainierten,
    eingefrorenen Autoencoder. Der zugehörige Beispielblock in main_nn.py ist
    auskommentiert und kein fertig eingerichteter dritter Trainingsablauf.

src/modules/plotting/matplotlib_plotting.py
    Hilfen für Linien-, Scatterplots, Legenden, Textfelder und PNG-Export.

src/modules/plotting/plotly_plotting.py
    Interaktive 2D-/3D-Scatterplots mit dem Renderer "plotly_mimetype";
    vorgesehen für eine Umgebung, die diesen Renderer anzeigen kann.

src/modules/plotting/heatmap.py
    Heatmaps mit Mittelwertbildung bei mehrfachen x/y-Paaren, linearer
    Interpolation von Lücken und optionaler logarithmischer Farbskala.

src/modules/data_managment/
    Gemeinsame CSV- und Dateifunktionen sowie temporäre Dateikopien.
    Die aktuelle Simulationspipeline schreibt erfolgreiche Ergebnisse direkt
    in die Ziel-CSV; ein Zusammenführen temporärer Worker-CSV gehört nicht
    zum aktiven Ablauf.


9. Bekannte Grenzen und Fehlerbehebung
--------------------------------------

"ModuleNotFoundError: No module named 'config' / 'modules' / 'src'"
    Aus der Projektwurzel starten und sowohl Projektwurzel als auch src in den
    Python-Suchpfad aufnehmen, wie in Abschnitt 2 beschrieben.

"ModuleNotFoundError: No module named 'src.modules.generators'"
    Der Generatorordner liegt jetzt unter src/modules/simulation/generators.
    In src/modules/simulation/pipeline.py und src/executable_files/machine_test.py
    muss der Import von AFMMachineGenerator auf
    src.modules.simulation.generators.machine_generator zeigen. In
    src/modules/simulation/generators/machine_generator.py muss map_generator
    aus src.modules.simulation.generators.map_generator importiert werden.
    Diese noch bestehenden Codeabweichungen sind durch Paketinstallationen oder
    den zusätzlichen Suchpfad allein nicht behoben.

Fehlendes Python-Paket
    Den tatsächlich gewählten Interpreter und die Installation aus environment.yml
    bzw. requirements.txt prüfen. Beide verwenden dieselben Python-Anforderungen;
    Tk/Tkinter gehört zur Python-/Conda-Installation.

Trainings-CSV wird nicht gefunden
    Den Dateinamen in main_nn.py an die vorhandene Datei mit "GenType-v1"
    anpassen, wie in Abschnitt 6 beschrieben.

Motor-CAD oder Basismodell wird nicht gefunden
    PATH_MCAD_EXE und PATH_MCAD_BASEFILE in config_paths.py prüfen.

Motor-CAD startet, kann aber nicht rechnen
    Lizenzzugang und Basismodell prüfen. Die automatische eduVPN-Steuerung
    bestätigt nur den Programmstart. Rechenfehler und verletzte Randbedingungen
    werden im Terminal ausgegeben.

Fehler beim Schließen des Fensters vor dem Simulationsstart
    Bei autostart=False führt das Schließen vor dem ersten Klick auf "Start"
    derzeit zu einem RuntimeError: Die Pipeline wartet auf einen noch nicht
    gestarteten Thread. Der anschließende stop()-Aufruf wird dann nicht erreicht.

Zu wenige vollständige Maschinen für DataTensor
    Mindestens fünf gültige, vollständige Kennfelder werden benötigt.
    Verworfene Simulationspunkte oder doppelte Betriebspunkte können dazu führen,
    dass ganze Maschinen bei der Aufbereitung entfernt werden.

KeyError beim CSV-Import oder der Datenaufbereitung
    Die Spaltennamen mit config_mcad_parameters.py vergleichen. Alte
    Ergebnisdateien werden nicht automatisch auf das aktuelle Schema migriert.

Tensorgrößen passen nicht zusammen
    Kennfeldraster, Autoencoder-map_shape, Kanalzahl und gespeicherte Artefakte
    aufeinander abstimmen. Das aktuelle Hauptskript erwartet für den Autoencoder
    ein 5 x 5 x 5-Raster.

Auswertung zeigt unerwartete Werte nach erneutem Training
    Prüfen, ob data.pkl und beide .pt-Dateien aus demselben Lauf stammen.
    Ein abgebrochener Trainingslauf kann einen gemischten Artefaktstand hinterlassen.

Prüfskript und Tests
~~~~~~~~~~~~~~~~~~~

Eine automatisierte Testsuite mit eigenständiger Testkonfiguration ist derzeit
nicht vorhanden. src/executable_files/machine_test.py prüft generierte Geometrien
mit get_generator(_type="v1") und test_boundary_conditions().

Der _type-Aufruf ist korrigiert. Nach der Verschiebung des Generatorordners
verhindern jedoch noch die oben beschriebenen alten Imports den Start.

Start nach der Umgebungseinrichtung aus Abschnitt 2 und Korrektur der Imports:

    python -m src.executable_files.machine_test

Das Skript durchläuft neun Kombinationen mit B=3...5, L=3...5 und O=5. Daraus
ergeben sich insgesamt 324.000 Parameterprüfungen einschließlich Betriebspunkten.
Generierte Werte werden mit DEFAULT_VALUES ergänzt. Verletzte Randbedingungen
und die Zahl verworfener Designs werden im Terminal ausgegeben.

Es wird keine Motor-CAD-Instanz gestartet und keine Ergebnis-CSV geschrieben.
ansys-motorcad-core muss wegen des importierten Prüfmoduls dennoch installiert
sein. Die Prüfung mit DEFAULT_VALUES ersetzt keine Kontrolle des tatsächlichen
Motor-CAD-Basismodells, der Lizenzverbindung oder des NN-Trainings.

Die Startskripte für Training, Auswertung und das Geometrie-Prüfskript enthalten
Code auf Modulebene; bereits ihr Import führt die jeweiligen Aktionen aus.
Sie sind als ausführbare Einstiegspunkte zu verwenden.
