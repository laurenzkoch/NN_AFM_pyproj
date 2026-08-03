#list with all output parameters (str)
OUTPUT_NEURON_LIST = [
    "ShaftTorque",
    "SystemEfficiency",
    "ConductorLoss",
    "MagnetLoss",
    "StatorIronLoss_Total",
    "RotorIronLoss_Total",
    #"WedgeLoss",
]

INPUT_NEURON_LIST = [
    #"TorquePointsPerCycle",
    #"AirgapMeshPoints_layers",
    #"AirgapMeshPoints_mesh",
    "Stator_Lam_Dia",
    "Stator_Bore",
    "AFM_D_Rotor",
    "Shaft_Dia",
    "Shaft_Hole_Diameter",
    "Magnet_Embed_Depth",
    "Magnet_Thickness",
    "Motor_Length",
    "Stator_Lam_Length",
    "Rotor_Lam_Length",
    "Airgap",
    "Magnet_Length",
    "Tooth_Tip_Depth",
    "Slot_Opening",
    "Slot_Depth",
    "Slot_Width",
    "Slot_Number",
    "Pole_Number",
    "MagPhases",
    "PeakCurrent",
    "ShaftSpeed",
    "PhaseAdvance",
]

PEAK_CURRENT: tuple[float, float] = (1, 1000)
PHASE_ADVANCE: tuple[float, float] = (0, 370)
SHAFT_SPEED: tuple[float, float] = (1, 10000)

STATOR_LAM_LENGTH: tuple[float, float] = (1, 300)
ROTOR_LAM_LENGTH: tuple[float, float] = (1, 300)
MAGNET_LENGTH: tuple[float, float] = (1, 300)
AIRGAP: tuple[float, float] = (0.01, 100)
FACTORES_SLOT_DEPTH: tuple[float, float] = (0.01, 0.99)

ROTOR_DIAMETER: tuple[float, float] = (10, 1000)
FACTORS_SHAFT_DIA: tuple[float, float] = (0.1, 0.9)
FACTORS_MAGNET_TICKNESS: tuple[float, ...] = (0.1, 0.99)
FACTORS_EMBED_DEPTH: tuple[float, ...] = (0.0, 1.0)
FACTORS_SLOT_WIDTH: tuple[float, ...] = (0.1, 0.99)
FACTORS_SLOT_OPENING: tuple[float, ...] = (0.1, 0.99)