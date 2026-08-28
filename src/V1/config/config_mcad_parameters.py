

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
    #"TorquePointsPerCycle",
    #"AirgapMeshPoints_layers",
    #"AirgapMeshPoints_mesh",
    "Stator_Lam_Dia",
    "Stator_Bore",
    "Slot_Opening",

    "AFM_D_Rotor",
    "Magnet_Embed_Depth",
    "Magnet_Thickness",
    "CircumferentialSegments",
    "Banding_Thickness",
    "Shaft_Dia",
    "Shaft_Hole_Diameter",

    "Motor_Length",
    "Stator_Lam_Length",
    "Magnet_Length",
    "Airgap",
    "Rotor_Lam_Length",
    "Shaft_Extension_[F]",
    "Shaft_Extension_[R]",

    "Slot_Number",
    "Slot_Width",
    "Slot_Depth",
    "Slot_Corner_Radius",
    "Tooth_Tip_Depth",
    "Tooth_Tip_Angle",
    "Pole_Number",

    "MagPhases",
    "PeakCurrent",
    "ShaftSpeed",
    "PhaseAdvance",
]

DEFAULT_VALUES = {
    "Stator_Lam_Dia": 150,
    "Stator_Bore": 70,
    "Slot_Opening": 3,

    "AFM_D_Rotor": 150,
    "Magnet_Embed_Depth": 5,
    "Magnet_Thickness": 40,
    "CircumferentialSegments": 1,
    "Banding_Thickness": 0,
    "Shaft_Dia": 50,
    "Shaft_Hole_Diameter": 0,

    "Motor_Length": 220,
    "Stator_Lam_Length": 50,
    "Magnet_Length": 5,
    "Airgap": 1,
    "Rotor_Lam_Length": 150,
    "Shaft_Extension_[F]": 30,
    "Shaft_Extension_[R]": 10,

    "Slot_Number": 18,
    "Slot_Width": 7,
    "Slot_Depth": 18,
    "Slot_Corner_Radius": 0,
    "Tooth_Tip_Depth": 1,
    "Tooth_Tip_Angle": 30,
    "Pole_Number": 12,
    "Magnet_Arc_[ED]": 140,

    "MagPhases": 3,
    "PeakCurrent": 5,
    "ShaftSpeed": 3000,
    "PhaseAdvance": 0,
    "Ke_Fundamental": 1.762,
    "DCBusVoltage": 800,
    "ArmatureWindingResistancePh": 1.675,
}

PEAK_CURRENT: tuple[float, float] = (1, 1000)
PHASE_ADVANCE: tuple[float, float] = (0, 85)
SHAFT_SPEED: tuple[float, float] = (1, 10000)

STATOR_LAM_LENGTH: tuple[float, float] = (DEFAULT_VALUES["Slot_Depth"]+2, 300)
ROTOR_LAM_LENGTH: tuple[float, float] = (1, 300)
MAGNET_LENGTH: tuple[float, float] = (1, 300)
AIRGAP: tuple[float, float] = (0.5, 100)

ROTOR_DIAMETER: tuple[float, float] = (DEFAULT_VALUES["Stator_Bore"]+10, 1000)




