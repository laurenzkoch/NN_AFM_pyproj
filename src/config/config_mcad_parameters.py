

OUTPUT_NEURON_LIST = [
    "ShaftTorque",
    "SystemEfficiency",
    "BMean_AirGap",
    "ConductorLoss",
    "MagnetLoss",
    "StatorIronLoss_Total",
    "RotorIronLoss_Total",
    #"WedgeLoss",
]

INPUT_NEURON_LIST = [

    "AFM_D_Rotor",
    "Stator_Lam_Dia",
    "Stator_Bore",
    "Airgap",
    "Pole_Number",

    "Slot_Number",
    "Slot_Width",
    "Slot_Opening",
    "Slot_Depth",

    "Motor_Length",
    "Rotor_Lam_Length",
    "Stator_Lam_Length",

    "Magnet_Length",
    "Magnet_Embed_Depth",
    "Magnet_Thickness",
    "CircumferentialSegments",
    "Magnet_Arc_[ED]",
]

OP_PARAMETERS = [
    "ShaftSpeed",
    "RMSPhaseCurrent_D",
    "RMSPhaseCurrent_Q",
    #"PeakCurrent",
    #"PhaseAdvance",
]

DEFAULT_VALUES = {

    "AFM_D_Rotor": 150,
    "Stator_Lam_Dia": 150,
    "Stator_Bore": 70,
    "Airgap": 1,
    "Pole_Number": 12,

    "Slot_Number": 18,
    "Slot_Width": 7,
    "Slot_Opening": 3,
    "Slot_Depth": 18,
    "Slot_Corner_Radius": 0,
    "Tooth_Tip_Depth": 1,
    "Tooth_Tip_Angle": 30,

    "Motor_Length": 220,
    "Rotor_Lam_Length": 20,
    "Stator_Lam_Length": 50,

    "Magnet_Length": 5,
    "Magnet_Embed_Depth": 5,
    "Magnet_Thickness": 40,
    "CircumferentialSegments": 1,
    "Magnet_Arc_[ED]": 140,
    "Banding_Thickness": 0,

    "MagPhases": 3,
    "MagTurnsConductor": 50,
    "MagThrow": 1,
    "ParallelPaths": 1,
    "WindingLayers": 2,
    "MagWindingOffset": 1,
    "NumberOfCoils": 6,

    "Shaft_Dia": 10,
    "Shaft_Hole_Diameter": 0,
    "Shaft_Extension_[F]": 30,
    "Shaft_Extension_[R]": 10,

    "RMSCurrentDensity": 7.575,
    "PeakCurrent": 5,
    "ShaftSpeed": 3000,
    "PhaseAdvance": 0,
    "DCBusVoltage": 800,
}

D_CURRENT_DENSITY: tuple[float, float] = (-1000, 0)
Q_CURRENT_DENSITY: tuple[float, float] = (1, 1000)
#PHASE_ADVANCE: tuple[float, float] = (0, 85)
SHAFT_SPEED: tuple[float, float] = (1, 5000)

AIRGAP: tuple[float, float] = (0.25, 5)
STATOR_BORE: tuple[float, float] = (DEFAULT_VALUES["Shaft_Dia"]+1, 200)
SLOT_DEPTH: tuple[float, float] = (4, 100)




